#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "csi_model.h"
#include "preprocess_config.h"    
#ifndef CSI_SMOOTH
#define CSI_SMOOTH          3      
#endif
#ifndef CSI_HOLD_SECONDS
#define CSI_HOLD_SECONDS    30     
#endif
#ifndef TENSOR_ARENA_BYTES
#define TENSOR_ARENA_BYTES  (64 * 1024)   
#endif
#define SMOOTH_MAX          16
#define QUEUE_LEN           32

static const char *TAG = "csi_model";

extern const uint8_t model_start[] asm("_binary_cnn1d_int8_tflite_start");

typedef struct {
    float v[PREP_NUM_BINS];
} bin_row_t;

static QueueHandle_t s_queue;
static volatile uint32_t s_dropped;
static volatile uint32_t s_skipped;
static int s_needed;                 
static int s_motion_label = -1;      
static uint8_t *s_arena;
static tflite::MicroInterpreter *s_interp;
static TfLiteTensor *s_in;
static TfLiteTensor *s_out;

static float s_rows[PREP_WINDOW][PREP_NUM_BINS];  

extern "C" void csi_model_feed(const int8_t *buf, int len, float gain)
{
    if (!s_queue || !buf || len < 2 || (len & 1)) {
        return;
    }
    if (len / 2 < s_needed) {
        s_skipped++;
        return;
    }

    bin_row_t item;
    float total = 0.0f;
    for (int b = 0; b < PREP_NUM_BINS; b++) {
        float sum = 0.0f;
        const int k0 = PREP_BIN_OFF[b], k1 = PREP_BIN_OFF[b + 1];
        for (int k = k0; k < k1; k++) {
            const int i = PREP_BIN_IDX[k];
            const float im = (float)(int16_t)(gain * buf[2 * i]);
            const float re = (float)(int16_t)(gain * buf[2 * i + 1]);
            sum += sqrtf(im * im + re * re);
        }
        item.v[b] = sum / (float)(k1 - k0);
        total += item.v[b];
    }
    float mean = total / PREP_NUM_BINS;
    if (mean < 1e-6f) {
        mean = 1e-6f;
    }
    for (int b = 0; b < PREP_NUM_BINS; b++) {
        item.v[b] /= mean;          
    }

    if (xQueueSend(s_queue, &item, 0) != pdTRUE) {
        s_dropped++;
    }
}

static void infer_task(void *arg)
{
    bin_row_t item;
    int head = 0, filled = 0, since = 0;

    float recent[SMOOTH_MAX][PREP_NUM_LABELS];
    int recent_n = 0, recent_i = 0;
    const int smooth = (CSI_SMOOTH < 1) ? 1 : (CSI_SMOOTH > SMOOTH_MAX ? SMOOTH_MAX : CSI_SMOOTH);

    int shown = -1;
    bool present = false;
    int64_t last_motion_us = -1;
    const int64_t hold_us = (int64_t)CSI_HOLD_SECONDS * 1000000LL;
    const bool use_presence = (CSI_HOLD_SECONDS > 0) && (s_motion_label >= 0);

    const bool in_int8 = (s_in->type == kTfLiteInt8);
    const float in_scale = s_in->params.scale;
    const int in_zero = s_in->params.zero_point;
    const bool out_int8 = (s_out->type == kTfLiteInt8);
    const float out_scale = s_out->params.scale;
    const int out_zero = s_out->params.zero_point;

    ESP_LOGI(TAG, "Window: %d packets, new prediction every %d. Waiting for CSI...", PREP_WINDOW, PREP_STRIDE);
    if (use_presence) {
        ESP_LOGI(TAG, "Presence: PRESENT from the first '%s' until %d s pass without one.",
                 PREP_LABELS[s_motion_label], CSI_HOLD_SECONDS);
    }

    for (;;) {
        xQueueReceive(s_queue, &item, portMAX_DELAY);

        memcpy(s_rows[head], item.v, sizeof(item.v));
        head = (head + 1) % PREP_WINDOW;
        if (filled < PREP_WINDOW) {
            filled++;
        }
        since++;
        if (filled < PREP_WINDOW || since < PREP_STRIDE) {
            continue;
        }
        since = 0;

        int8_t *in_q = s_in->data.int8;
        float *in_f = s_in->data.f;
        for (int b = 0; b < PREP_NUM_BINS; b++) {
            float mean = 0.0f;
            for (int t = 0; t < PREP_WINDOW; t++) {
                mean += s_rows[(head + t) % PREP_WINDOW][b];
            }
            mean /= PREP_WINDOW;
            if (mean < 1e-6f) {
                mean = 1e-6f;
            }
            for (int t = 0; t < PREP_WINDOW; t++) {
                const float r = s_rows[(head + t) % PREP_WINDOW][b];
                float x = PREP_DETREND ? (r / mean - 1.0f) : (r - 1.0f);
                x /= PREP_SCALE;
                const int idx = t * PREP_NUM_BINS + b;
                if (in_int8) {
                    int q = (int)lroundf(x / in_scale + in_zero);
                    if (q < -128) q = -128;
                    if (q > 127) q = 127;
                    in_q[idx] = (int8_t)q;
                } else {
                    in_f[idx] = x;
                }
            }
        }

        const int64_t t0 = esp_timer_get_time();
        if (s_interp->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            continue;
        }
        const float took_ms = (esp_timer_get_time() - t0) / 1000.0f;

        for (int l = 0; l < PREP_NUM_LABELS; l++) {
            recent[recent_i][l] = out_int8 ? ((float)s_out->data.int8[l] - out_zero) * out_scale
                                           : s_out->data.f[l];
        }
        recent_i = (recent_i + 1) % smooth;
        if (recent_n < smooth) {
            recent_n++;
        }

        float probs[PREP_NUM_LABELS] = {0};
        for (int n = 0; n < recent_n; n++) {
            for (int l = 0; l < PREP_NUM_LABELS; l++) {
                probs[l] += recent[n][l] / recent_n;
            }
        }
        int best = 0;
        for (int l = 1; l < PREP_NUM_LABELS; l++) {
            if (probs[l] > probs[best]) {
                best = l;
            }
        }

        const int64_t now = esp_timer_get_time();
        if (use_presence) {
            if (best == s_motion_label) {
                last_motion_us = now;
            }
            const bool is_present = (last_motion_us >= 0) && (now - last_motion_us <= hold_us);
            if (is_present != present) {
                present = is_present;
                if (present) {
                    ESP_LOGI(TAG, ">>> PRESENT (movement detected)");
                } else {
                    ESP_LOGI(TAG, ">>> ABSENT  (no movement for %d s)", CSI_HOLD_SECONDS);
                }
            }
        }

        char detail[96] = {0};
        int pos = 0;
        for (int l = 0; l < PREP_NUM_LABELS && pos < (int)sizeof(detail) - 1; l++) {
            pos += snprintf(detail + pos, sizeof(detail) - pos, "%s %3.0f%%  ", PREP_LABELS[l], probs[l] * 100.0f);
        }

        if (best != shown) {
            shown = best;
            ESP_LOGI(TAG, "%-8s (%s) %.1f ms", PREP_LABELS[best], detail, took_ms);
        } else {
            ESP_LOGD(TAG, "%-8s (%s) %.1f ms", PREP_LABELS[best], detail, took_ms);
        }

        if (s_dropped || s_skipped) {
            ESP_LOGW(TAG, "dropped %u rows (queue full), skipped %u short packets",
                     (unsigned)s_dropped, (unsigned)s_skipped);
            s_dropped = 0;
            s_skipped = 0;
        }
    }
}

extern "C" esp_err_t csi_model_init(void)
{
    s_needed = 0;
    for (int k = 0; k < PREP_BIN_OFF[PREP_NUM_BINS]; k++) {
        if (PREP_BIN_IDX[k] + 1 > s_needed) {
            s_needed = PREP_BIN_IDX[k] + 1;
        }
    }

    const char *candidates[] = {"moving", "person"};
    for (int c = 0; c < 2 && s_motion_label < 0; c++) {
        for (int l = 0; l < PREP_NUM_LABELS; l++) {
            if (strcmp(PREP_LABELS[l], candidates[c]) == 0) {
                s_motion_label = l;
                break;
            }
        }
    }

    const tflite::Model *model = tflite::GetModel(model_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %u != supported %d", (unsigned)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    s_arena = (uint8_t *)heap_caps_aligned_alloc(16, TENSOR_ARENA_BYTES, MALLOC_CAP_8BIT);
    if (!s_arena) {
        ESP_LOGE(TAG, "Could not allocate %d byte tensor arena", TENSOR_ARENA_BYTES);
        return ESP_ERR_NO_MEM;
    }

    static tflite::AllOpsResolver resolver;
    s_interp = new tflite::MicroInterpreter(model, resolver, s_arena, TENSOR_ARENA_BYTES);
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed - increase TENSOR_ARENA_BYTES");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Tensor arena used: %u of %d bytes", (unsigned)s_interp->arena_used_bytes(), TENSOR_ARENA_BYTES);

    s_in = s_interp->input(0);
    s_out = s_interp->output(0);

    if (s_in->dims->size < 3 || s_in->dims->data[1] != PREP_WINDOW || s_in->dims->data[2] != PREP_NUM_BINS) {
        ESP_LOGE(TAG, "Model input shape does not match preprocess_config.h (%d x %d). "
                      "Use files from the same training run.", PREP_WINDOW, PREP_NUM_BINS);
        return ESP_ERR_INVALID_ARG;
    }
    if (s_out->dims->data[s_out->dims->size - 1] != PREP_NUM_LABELS) {
        ESP_LOGE(TAG, "Model has %d outputs but preprocess_config.h has %d labels",
                 s_out->dims->data[s_out->dims->size - 1], PREP_NUM_LABELS);
        return ESP_ERR_INVALID_ARG;
    }
    if ((s_in->type != kTfLiteInt8 && s_in->type != kTfLiteFloat32) ||
        (s_out->type != kTfLiteInt8 && s_out->type != kTfLiteFloat32)) {
        ESP_LOGE(TAG, "Unsupported tensor type (need int8 or float32)");
        return ESP_ERR_NOT_SUPPORTED;
    }

    s_queue = xQueueCreate(QUEUE_LEN, sizeof(bin_row_t));
    if (!s_queue) {
        return ESP_ERR_NO_MEM;
    }
    if (xTaskCreate(infer_task, "csi_infer", 8192, NULL, 4, NULL) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
