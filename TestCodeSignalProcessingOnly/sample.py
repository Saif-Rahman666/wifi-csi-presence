import argparse
import sys
import time
from collections import deque

import numpy as np
import serial  

def parse_amplitudes(line):
    """Return per-subcarrier amplitudes from one CSI_DATA line, or None if it is broken."""
    start = line.find("[")
    end = line.rfind("]")
    if not line.startswith("CSI_DATA") or start < 0 or end < start:
        return None
    try:
        values = np.array(line[start + 1:end].split(","), dtype=np.float32)
    except ValueError:
        return None
    if values.size < 2 or values.size % 2:
        return None
    imag, real = values[0::2], values[1::2]  
    return np.sqrt(imag * imag + real * real)


def smooth_rows(rows, k):
    """Moving average over k consecutive packets, for a (packets x subcarriers) block."""
    if k <= 1:
        return rows
    total = np.cumsum(rows, axis=0)
    total[k:] = total[k:] - total[:-k]
    return total[k - 1:] / k


def motion_score(window):
    """Average fluctuation over time across subcarriers, for a (packets x subcarriers) block."""
    return float(np.std(window, axis=0).mean())


def main():
    parser = argparse.ArgumentParser(description="Threshold motion detector for ESP32 CSI")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM8")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--window", type=int, default=100,
                        help="packets per window (100 = 1 s at 100 packets/s)")
    parser.add_argument("--calib", type=float, default=8,
                        help="seconds of still room used to learn the quiet level")
    parser.add_argument("--smooth", type=int, default=5,
                        help="packets averaged together to suppress noise (1 = off)")
    parser.add_argument("--factor", type=float, default=1.5,
                        help="motion when score > quiet level x factor (lower = more sensitive)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="set the threshold directly and ignore --factor")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="seconds to keep showing MOTION after the score drops")
    args = parser.parse_args()

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as err:
        sys.exit(f"Could not open {args.port}: {err}\n"
                 "Close any monitor or graph window that is using the port.")

    live = sys.stdout.isatty()
    print(f"Opened {args.port}. Waiting for CSI...")

    n_sub = None            
    calib_rows = []       
    calib_start = None
    mask = None             
    threshold = None
    window = deque(maxlen=args.window)
    recent = deque(maxlen=max(1, args.smooth))   
    since_update = 0
    state = None
    last_motion = 0.0
    last_packet = time.time()

    try:
        with port:
            while True:
                raw = port.readline()
                if not raw:
                    if calib_start is not None and time.time() - last_packet > 5:
                        sys.exit("\nNo CSI for 5 s. Is the transmitter powered?")
                    continue
                amp = parse_amplitudes(raw.decode("utf-8", errors="ignore").strip())
                if amp is None:
                    continue
                if n_sub is None:
                    n_sub = amp.size
                if amp.size != n_sub:
                    continue                     
                now = last_packet = time.time()
              
                if threshold is None:
                    if calib_start is None:
                        calib_start = now
                        print(f"CSI is arriving ({n_sub} subcarriers per packet).")
                        print(f"Calibrating for {args.calib:.0f} s: keep the room STILL...")
                    calib_rows.append(amp)
                    if now - calib_start < args.calib:
                        continue

                    data = np.array(calib_rows)
                    if len(data) < args.window * 2:
                        sys.exit(f"Only {len(data)} packets in {args.calib:.0f} s: too few. "
                                 "Use a longer --calib or a smaller --window.")
                    mean_amp = data.mean(axis=0)
                    mask = mean_amp > 0.1 * np.median(mean_amp[mean_amp > 0])
                    norm = data[:, mask]
                    norm = norm / norm.mean(axis=1, keepdims=True)
                    norm = smooth_rows(norm, args.smooth)
                    scores = [motion_score(norm[i:i + args.window])
                              for i in range(0, len(norm) - args.window, 10)]
                    quiet = float(np.median(scores))
                    threshold = args.threshold if args.threshold is not None else quiet * args.factor
                    rate = len(data) / (now - calib_start)
                    print(f"Done. {rate:.0f} packets/s, {int(mask.sum())} active subcarriers.")
                    print(f"Quiet level {quiet:.4f}  ->  threshold {threshold:.4f}")
                    print("Now move around. Ctrl+C to stop.\n")
                    continue

                row = amp[mask]
                level = row.mean()
                if level <= 0:
                    continue
                recent.append(row / level)
                window.append(np.mean(recent, axis=0))
                since_update += 1
                if len(window) < args.window or since_update < 10:
                    continue
                since_update = 0

                score = motion_score(np.array(window))
                if score > threshold:
                    last_motion = now
                new_state = "MOTION" if now - last_motion <= args.hold else "still"

                if new_state != state:
                    state = new_state
                    if live:
                        sys.stdout.write("\r" + " " * 78 + "\r")
                    print(f"{time.strftime('%H:%M:%S')}  {state:6}  (score {score:.4f})")
                if live:
                    bar = "#" * min(40, int(20 * score / threshold))
                    sys.stdout.write(f"\r  {state:6}  score {score:.4f}  |{bar:<40}|")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
