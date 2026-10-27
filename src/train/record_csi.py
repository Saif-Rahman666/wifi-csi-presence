import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import serial  

LABELS = ["empty", "still", "moving"]
REPO = Path(__file__).resolve().parents[1]        


def firmware_commit():
    """Short hash of the repository's current commit, or None outside git."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def is_complete_csi_line(line):
    """A full line starts with CSI_DATA and ends with the closing ]" of the array."""
    return line.startswith("CSI_DATA") and line.rstrip().endswith(']"')


def main():
    parser = argparse.ArgumentParser(description="Record labelled CSI to a CSV file")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM8")
    parser.add_argument("--room", required=True, help="room name, e.g. kitchen")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--seconds", type=float, default=120, help="recording length")
    parser.add_argument("--delay", type=float, default=15,
                        help="seconds to wait before recording (time to leave or get in place)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out", default=str(REPO / "data" / "raw"), help="root folder for recordings")
    parser.add_argument("--subject", help="who was in the room, e.g. john")
    parser.add_argument("--distance", type=float, help="distance between the boards in metres")
    parser.add_argument("--positions", help="where the boards stood, in words")
    parser.add_argument("--obstruction", help="anything between the boards, e.g. 'none' or 'wooden door'")
    parser.add_argument("--notes", help="anything else worth knowing about this recording")
    args = parser.parse_args()

    started_at = datetime.now()
    folder = os.path.join(args.out, args.room, started_at.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{args.label}_{started_at.strftime('%H%M%S')}.csv")

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as err:
        sys.exit(f"Could not open {args.port}: {err}\n"
                 "Close any monitor or graph window that is using the port.")

    print(f"Opened {args.port}. Waiting for CSI (the board may restart and reconnect)...")

    header = None
    first_csi = None    
    record_start = None 
    saved = 0
    dropped = 0
    last_report = 0.0

    try:
        with port, open(path, "w", newline="") as out:
            while True:
                raw = port.readline()
                now = time.time()

                if first_csi is None and not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()

                if header is None and line.startswith("type,"):
                    header = line
                    continue

                if line.startswith("CSI_DATA"):
                    if first_csi is None:
                        first_csi = now
                        print(f"CSI is arriving. Recording starts in {args.delay:.0f} s.")

                    if record_start is None:
                        if now - first_csi < args.delay:
                            continue 
                        record_start = now
                        last_report = now
                        if header:
                            out.write(header + "\n")
                        print(f"RECORDING '{args.label}' in '{args.room}' "
                              f"for {args.seconds:.0f} s...")

                    if is_complete_csi_line(line):
                        out.write(line + "\n")
                        saved += 1
                    else:
                        dropped += 1

                if record_start is not None:
                    elapsed = now - record_start
                    if now - last_report >= 5:
                        print(f"  {elapsed:5.0f} s   {saved} packets   "
                              f"{saved / elapsed:5.1f} packets/s")
                        last_report = now
                    if elapsed >= args.seconds:
                        break
    except KeyboardInterrupt:
        print("\nStopped early.")

    if record_start is None or saved == 0:
        if os.path.exists(path):
            os.remove(path)
        sys.exit("No CSI was recorded. Is the sender powered, and is this the receiver's port?")

    duration = time.time() - record_start
    print(f"\nSaved {saved} packets in {duration:.0f} s "
          f"({saved / duration:.1f} packets/s), {dropped} incomplete lines skipped.")

    meta = {
        "label": args.label, "room": args.room, "date": started_at.strftime("%Y-%m-%d"),
        "start_time": started_at.strftime("%H:%M:%S"), "duration_s": round(duration, 1),
        "packets": saved, "packets_per_s": round(saved / duration, 1), "incomplete_lines": dropped,
        "subject": args.subject, "distance_m": args.distance, "board_positions": args.positions,
        "obstruction": args.obstruction, "notes": args.notes,
        "firmware_commit": firmware_commit(), "port": args.port, "baud": args.baud,
    }
    with open(path[:-4] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if header is None:
        print("Note: no header line was seen (the board did not restart). "
              "The file has data rows only, which is fine.")
    print(f"File: {path}")


if __name__ == "__main__":
    main()
