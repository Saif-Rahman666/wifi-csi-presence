"""Print presence alerts that the ESP32-C3 receiver sends over Bluetooth LE.

The board advertises a 2 s burst named CSI-SAR whenever presence changes.
No pairing needed: just keep this running on a PC with Bluetooth on.

Example:
    python tools/ble_listen.py

Needs:  python -m pip install bleak
"""

import asyncio
import time

from bleak import BleakScanner

TEST_COMPANY_ID = 0xFFFF
last_seq = None


def on_advert(device, adv):
    global last_seq
    data = adv.manufacturer_data.get(TEST_COMPANY_ID)
    if not data or len(data) < 3 or data[0] != ord("C"):
        return
    state, seq = data[1], data[2]
    if seq == last_seq:
        return                      # repeat of an alert already shown
    last_seq = seq
    print(f"{time.strftime('%H:%M:%S')}  {'PRESENT' if state else 'ABSENT ':8}"
          f"  alert #{seq}  from {device.address}  (BLE RSSI {adv.rssi} dBm)", flush=True)


async def main():
    print("Listening for CSI-SAR alerts... Ctrl+C to stop.")
    async with BleakScanner(on_advert):
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
