#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t csi_model_init(void);

void csi_model_feed(const int8_t *buf, int len, float gain);

#ifdef __cplusplus
}
#endif
