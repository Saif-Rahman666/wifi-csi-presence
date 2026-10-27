#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Start the BLE stack (silent). Call once, after nvs_flash_init() and before Wi-Fi starts. */
void ble_alert_init(void);

/** Announce a presence change: advertises a 2 s BLE burst with the new state (1 = present, 0 = absent). */
void ble_alert_set(uint8_t present);

#ifdef __cplusplus
}
#endif
