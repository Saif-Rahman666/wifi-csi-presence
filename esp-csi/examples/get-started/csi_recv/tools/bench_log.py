"""Log the ESP32-C3 benchmark output and summarise it.

Reads the serial port for a fixed time, saves everything to a log file, picks out the
'stats:' lines the firmware prints every 10 s, skips the warm-up, and prints the averages.

Example:
    python tools/bench_log.py --port COM8 --minutes 3 --name cnn1d_w50_f16-32
    python tools/bench_log.py --port COM8 --minutes 3 --name baseline_no_model

Close idf.py monitor first: only one program can use the port.
Needs:  python -m pip install pyserial
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import serial

REPO = Path(__file__).resolve().parents[1]
NUM = r"([\d.]+)"
FIELDS = {
    "pkt_per_s": rf"{NUM} pkt/s",
    "inferences": rf"\| (\d+) inferences",
    "infer_ms_avg": rf"inferences, {NUM} ms avg",
    "infer_ms_max": rf"ms avg, {NUM} ms max",
    "cpu_model_pct": rf"model {NUM}%",
    "cpu_window_pct": rf"window build {NUM}%",
    "cpu_packet_pct": rf"per-packet {NUM}%",
    "dropped": r"dropped (\d+)",
    "heap_free": r"heap free (\d+)",
    "heap_min": r"min (\d+)",
    "heap_largest": r"largest (\d+)",
    "stack_left": r"stack left (\d+)",
    "arena_used": r"arena (\d+)",
}


def parse(line):
    row = {}
    for key, pattern in FIELDS.items():
        m = re.search(pattern, line)
        if m:
            row[key] = float(m.group(1))
    return row


def main():
    parser = argparse.ArgumentParser(description="Log and summarise the firmware's benchmark stats")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM8 or /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--name", required=True, help="name of this run, e.g. cnn1d_w100_f8-16")
    parser.add_argument("--skip", type=int, default=2, help="stats lines to ignore at the start (warm-up)")
    parser.add_argument("--out", default=str(REPO / "results" / "bench"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.name}.log"
    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as err:
        sys.exit(f"Could not open {args.port}: {err}\nClose idf.py monitor or any other program using it.")

    print(f"Logging {args.port} for {args.minutes:g} min to {log_path} ... (Ctrl+C stops early)")
    rows, end = [], time.time() + args.minutes * 60
    with port, open(log_path, "w", encoding="utf-8") as log:
        try:
            while time.time() < end:
                line = port.readline().decode("utf-8", errors="ignore").rstrip()
                if not line:
                    continue
                log.write(line + "\n")
                if "stats:" in line or line.startswith("stats"):
                    rows.append(parse(line))
                    print("  " + line[line.find("stats"):][:150])
                elif "Tensor arena used" in line or "PRESENT" in line or "ABSENT" in line or "BLE alert" in line:
                    print("  " + line.strip()[:150])
        except KeyboardInterrupt:
            print("Stopped early.")

    used = rows[args.skip:]
    if not used:
        sys.exit(f"Not enough stats lines ({len(rows)}). Is the benchmark firmware flashed and CSI arriving?")
    keys = [k for k in FIELDS if any(k in r for r in used)]
    summary = {"run": args.name, "stats_lines": len(used)}
    for k in keys:
        vals = [r[k] for r in used if k in r]
        summary[k] = round(sum(vals) / len(vals), 2)
    for k in ("infer_ms_max", "dropped"):
        if k in keys:
            summary[k + "_worst"] = max(r.get(k, 0) for r in used)
    if "heap_min" in keys:
        summary["heap_min_lowest"] = min(r["heap_min"] for r in used if "heap_min" in r)

    print(f"\nSummary of {len(used)} stats lines ({args.skip} warm-up lines skipped):")
    for k, v in summary.items():
        print(f"  {k:20} {v}")
    table = out / "bench_summary.csv"
    exists = table.exists()
    with open(table, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f"\nAppended to {table}   (one row per run: compare them in Excel)")


if __name__ == "__main__":
    main()
