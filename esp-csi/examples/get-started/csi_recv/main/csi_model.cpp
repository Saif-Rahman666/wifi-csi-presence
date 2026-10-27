/*
 * On-device port of src/deploy/predict_live.py
 *
 * Pipeline (same arithmetic as the Python script):
 *   CSI packet -> amplitudes -> average into bins -> divide by packet mean   (csi_model_feed, CSI callback)
 *   rows queue -> sliding window -> detrend/scale -> int8 quantize           (inference task)
 *   TFLite Micro invoke -> dequantize -> smoothing -> argmax -> presence hold
 */

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
#include "esp_system.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "csi_model.h"
#include "ble_alert.h"
#include "preprocess_config.h"      /* generated from preprocess.json by tools/gen_preprocess_h.py */

/* ---- tunables (same meaning as the Python arguments) ---- */
#ifndef CSI_SMOOTH
#define CSI_SMOOTH          3       /* --smooth: average probabilities of this many windows (1 = off), max 16 */
#endif
#ifndef CSI_HOLD_SECONDS
#define CSI_HOLD_SECONDS    30      /* --hold: PRESENT until this many seconds without movement (0 = off) */
#endif
#ifndef TENSOR_ARENA_BYTES
#define TENSOR_ARENA_BYTES  (64 * 1024)   /* check the "arena used" log line and shrink/grow */
#endif
#ifndef CSI_RUN_MODEL
#define CSI_RUN_MODEL       1       /* 0 = benchmark baseline: capture only, no preprocessing, no model */
#endif
#ifndef CSI_STATS_PERIOD_S
#define CSI_STATS_PERIOD_S  10      /* print one stats line this often (0 = off) */
#endif
#ifndef CSI_TASK_STATS_EVERY
#define CSI_TASK_STATS_EVERY 3      /* print the FreeRTOS per-task CPU table every N stats lines, if enabled */
#endif
#define SMOOTH_MAX          16
#define QUEUE_LEN           32

static const char *TAG = "csi_model";

/* The .tflite file, embedded by EMBED_FILES in main/CMakeLists.txt */
extern const uint8_t model_start[] asm("_binary_cnn1d_int8_tflite_start");

typedef struct {
    float v[PREP_NUM_BINS];
} bin_row_t;

static QueueHandle_t s_queue;
static volatile uint32_t s_dropped;
static volatile uint32_t s_skipped;
static int s_needed;                 /* subcarriers a packet must have */
static int s_motion_label = -1;      /* index of "moving"/"person" label, -1 = presence off */

static uint8_t *s_arena;
static tflite::MicroInterpreter *s_interp;
static TfLiteTensor *s_in;
static TfLiteTensor *s_out;

static float s_rows[PREP_WINDOW][PREP_NUM_BINS];   /* circular window of binned rows */

/* ---- benchmark counters: cheap to update, read once per stats period ---- */
static volatile uint32_t s_packets;          /* CSI packets from the sender */
static volatile uint64_t s_feed_us;          /* time spent in csi_model_feed (Wi-Fi task) */
static volatile uint32_t s_dropped_total;    /* rows lost because the queue was full */
static uint64_t s_build_us;                  /* time building the input window (inference task) */
static uint64_t s_invoke_us;                 /* time inside Invoke() */
static uint32_t s_invoke_max_us;
static uint32_t s_inferences;
static size_t s_arena_used;
static TaskHandle_t s_infer_handle;

/* ------------------------------------------------------------------------- */
/* Runs in the Wi-Fi task: keep it short. Only the subcarriers the bins use are computed. */
static void feed_impl(const int8_t *buf, int len, float gain);

extern "C" void csi_model_feed(const int8_t *buf, int len, float gain)
{
    const int64_t t0 = esp_timer_get_time();
    s_packets = s_packets + 1;
#if CSI_RUN_MODEL
    feed_impl(buf, len, gain);
#endif
    s_feed_us = s_feed_us + (uint64_t)(esp_timer_get_time() - t0);
}

static void feed_impl(const int8_t *buf, int len, float gain)
{
    if (!s_queue || !buf || len < 2 || (len & 1)) {
        return;
    }
    if (len / 2 < s_needed) {
        s_skipped = s_skipped + 1;
        return;
    }

    bin_row_t item;
    float total = 0.0f;
    for (int b = 0; b < PREP_NUM_BINS; b++) {
        float sum = 0.0f;
        const int k0 = PREP_BIN_OFF[b], k1 = PREP_BIN_OFF[b + 1];
        for (int k = k0; k < k1; k++) {
            const int i = PREP_BIN_IDX[k];
            /* stored as imaginary, real pairs; gain applied and truncated like the printed CSV */
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
        item.v[b] /= mean;           /* remove the packet's gain */
    }

    if (xQueueSend(s_queue, &item, 0) != pdTRUE) {
        s_dropped = s_dropped + 1;
        s_dropped_total = s_dropped_total + 1;
    }
}

/* ------------------------------------------------------------------------- */
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

        /* Build the input [1, window, bins, 1]. Oldest row is at `head` once the window is full. */
        const int64_t tb = esp_timer_get_time();
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
        s_build_us += (uint64_t)(t0 - tb);
        if (s_interp->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            continue;
        }
        const uint32_t took_us = (uint32_t)(esp_timer_get_time() - t0);
        const float took_ms = took_us / 1000.0f;
        s_invoke_us += took_us;
        s_inferences++;
        if (took_us > s_invoke_max_us) {
            s_invoke_max_us = took_us;
        }

        /* Dequantize output into the smoothing buffer */
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

        /* Presence: movement sets it, it expires after `hold` seconds without movement */
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
                ble_alert_set(present ? 1 : 0);     /* 2 s BLE burst to the PC */
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

/* ------------------------------------------------------------------------- */
/* One line every CSI_STATS_PERIOD_S seconds. Lowest priority, so it never delays inference. */
static void stats_task(void *arg)
{
    const int period_s = CSI_STATS_PERIOD_S;
    uint32_t last_pk = 0, last_n = 0, last_drop = 0, line = 0;
    uint64_t last_feed = 0, last_build = 0, last_invoke = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(period_s * 1000));
        const uint32_t pk = s_packets, n = s_inferences, drop = s_dropped_total;
        const uint64_t feed = s_feed_us, build = s_build_us, invoke = s_invoke_us;
        const double window_us = period_s * 1e6;
        const uint32_t dn = n - last_n;
        ESP_LOGI("stats", "%.1f pkt/s | %u inferences, %.2f ms avg, %.2f ms max | CPU: model %.1f%%, "
                 "window build %.1f%%, per-packet %.1f%% | dropped %u | heap free %u, min %u, largest %u | "
                 "infer stack left %u B | arena %u B",
                 (pk - last_pk) / (double)period_s, (unsigned)dn,
                 dn ? (invoke - last_invoke) / 1000.0 / dn : 0.0, s_invoke_max_us / 1000.0,
                 100.0 * (invoke - last_invoke) / window_us, 100.0 * (build - last_build) / window_us,
                 100.0 * (feed - last_feed) / window_us, (unsigned)(drop - last_drop),
                 (unsigned)esp_get_free_heap_size(), (unsigned)esp_get_minimum_free_heap_size(),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                 s_infer_handle ? (unsigned)uxTaskGetStackHighWaterMark(s_infer_handle) : 0u,
                 (unsigned)s_arena_used);
        last_pk = pk; last_n = n; last_drop = drop;
        last_feed = feed; last_build = build; last_invoke = invoke;
        s_invoke_max_us = 0;
#if (configGENERATE_RUN_TIME_STATS == 1) && (configUSE_STATS_FORMATTING_FUNCTIONS == 1)
        if (CSI_TASK_STATS_EVERY > 0 && ++line % CSI_TASK_STATS_EVERY == 0) {
            static char table[1024];
            vTaskGetRunTimeStats(table);
            printf("task             time        %%\n%s", table);
        }
#else
        (void)line;
#endif
    }
}

static void start_stats(void)
{
    if (CSI_STATS_PERIOD_S > 0) {
        xTaskCreate(stats_task, "stats", 3072, NULL, 1, NULL);
    }
}

extern "C" esp_err_t csi_model_init(void)
{
#if !CSI_RUN_MODEL
    ESP_LOGW(TAG, "CSI_RUN_MODEL = 0: benchmark baseline, packets are counted but no model runs");
    s_queue = NULL;
    start_stats();
    return ESP_OK;
#endif
    /* Highest subcarrier index a packet must contain */
    s_needed = 0;
    for (int k = 0; k < PREP_BIN_OFF[PREP_NUM_BINS]; k++) {
        if (PREP_BIN_IDX[k] + 1 > s_needed) {
            s_needed = PREP_BIN_IDX[k] + 1;
        }
    }

    /* Presence uses "moving" if present, otherwise "person" (same as Python) */
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

    /* AllOpsResolver is easiest for bring-up; swap for MicroMutableOpResolver later to save flash. */
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    s_interp = new tflite::MicroInterpreter(model, resolver, s_arena, TENSOR_ARENA_BYTES);
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed - increase TENSOR_ARENA_BYTES");
        return ESP_ERR_NO_MEM;
    }
    s_arena_used = s_interp->arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena used: %u of %d bytes", (unsigned)s_arena_used, TENSOR_ARENA_BYTES);

    s_in = s_interp->input(0);
    s_out = s_interp->output(0);

    /* Same sanity check as Python: model input must match preprocess.json */
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
    if (xTaskCreate(infer_task, "csi_infer", 8192, NULL, 4, &s_infer_handle) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    start_stats();
    return ESP_OK;
}


