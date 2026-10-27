/*
 * Presence alerts over Bluetooth LE.
 *
 * The BLE stack stays silent. When the presence state changes, the board advertises a
 * beacon every 100 ms for 2 s, then stops, so the shared radio goes back to Wi-Fi/CSI.
 * No pairing or connection: any PC or phone scanning nearby sees the beacon.
 *
 * Beacon contents (manufacturer data): 0xFFFF (company ID reserved for testing), 'C', state, counter.
 * The counter increases on every change, so a receiver can tell new alerts from repeats.
 * Listener on the PC: tools/ble_listen.py
 */

#include "esp_log.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"

#include "ble_alert.h"

#define BURST_MS        2000    /* how long each alert is advertised */
#define INTERVAL_UNITS  160     /* advertising interval in 0.625 ms units: 160 = 100 ms */

static const char *TAG = "ble_alert";
static uint8_t s_state = 0;     /* 0 = absent, 1 = present */
static uint8_t s_seq = 0;       /* increases on every change */

static void adv_burst(void)
{
    uint8_t mfg[5] = {0xFF, 0xFF, 'C', s_state, s_seq};
    struct ble_hs_adv_fields f = {0};
    f.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    f.name = (uint8_t *)"CSI-SAR";
    f.name_len = 7;
    f.name_is_complete = 1;
    f.mfg_data = mfg;
    f.mfg_data_len = sizeof mfg;

    ble_gap_adv_stop();                         /* end a burst that may still be running */
    int rc = ble_gap_adv_set_fields(&f);
    if (rc != 0) {
        ESP_LOGW(TAG, "set_fields failed: %d", rc);
        return;
    }

    struct ble_gap_adv_params p = {0};
    p.conn_mode = BLE_GAP_CONN_MODE_NON;        /* beacon only */
    p.disc_mode = BLE_GAP_DISC_MODE_GEN;
    p.itvl_min = INTERVAL_UNITS;
    p.itvl_max = INTERVAL_UNITS;
    rc = ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BURST_MS, &p, NULL, NULL);
    if (rc != 0) {
        ESP_LOGW(TAG, "adv_start failed: %d", rc);
    } else {
        ESP_LOGI(TAG, "BLE alert: %s (#%u)", s_state ? "PRESENT" : "ABSENT", s_seq);
    }
}

static void on_sync(void)
{
    ble_hs_util_ensure_addr(0);                 /* stay silent until the first change */
}

static void host_task(void *arg)
{
    nimble_port_run();                          /* returns only when nimble_port_stop() is called */
    nimble_port_freertos_deinit();
}

void ble_alert_init(void)
{
    if (nimble_port_init() != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: BLE alerts disabled");
        return;
    }
    ble_hs_cfg.sync_cb = on_sync;
    nimble_port_freertos_init(host_task);
}

void ble_alert_set(uint8_t present)
{
    s_state = present ? 1 : 0;
    s_seq++;
    if (ble_hs_synced()) {
        adv_burst();
    }
}
