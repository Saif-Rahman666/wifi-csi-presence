#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Load the TFLite model, allocate tensors, start the inference task. Call once before CSI is enabled. */
esp_err_t csi_model_init(void);

/**
 * Feed one CSI packet (call from the CSI callback; it is cheap and never blocks).
 * buf  = info->buf (int8 pairs: imaginary, real)
 * len  = info->len in bytes
 * gain = compensate_gain from esp_csi_gain_ctrl (1.0f if unused)
 */
void csi_model_feed(const int8_t *buf, int len, float gain);

#ifdef __cplusplus
}
#endif
