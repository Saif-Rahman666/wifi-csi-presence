# Wi-Fi CSI-Based Person Detection for Search and Rescue

Tiny Machine Learning (ML733N) HT2026 — Group 1, Sub-Group 5
Md Saif Ur Rahman · Timothy Nordahl · John Martin Kodakanady
Supervisor: Jing Zhan

Device-free human presence detection from Wi-Fi Channel State Information,
with a quantized classifier deployed on an ESP32-C3.

## Hardware
- ESP32-C3-DevKitM-1 — receiver and on-device inference
- ESP32 DevKit V1 — fixed-rate transmitter

## Layout
| Path | Contents |
|---|---|
| `firmware/csi_recv_router/` | Single-board capture via router ping |
| `firmware/csi_send/`, `firmware/csi_recv/` | Two-board capture at fixed rate |
| `tools/` | Serial capture, parsing and live viewer |
| `data/` | Recordings — not in git, see data/README.md |
| `src/` | Preprocessing, training, LiteRT conversion |
| `notebooks/` | Exploration and results |
| `models/final/` | Deployed .tflite models |
| `docs/` | Project plan, seminars, report |

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r tools/requirements.txt
    . ~/projects/tinyml/esp/esp-idf/export.sh   # for firmware work

## Capture
    cd firmware/csi_recv_router
    idf.py set-target esp32c3
    idf.py menuconfig          # set Wi-Fi SSID and password
    idf.py -p /dev/cu.usbserial-XXXX flash monitor
