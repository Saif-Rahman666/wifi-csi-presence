"""Run a trained CSI model live on the PC, on the stream from the ESP32 receiver.

Reads CSI from the serial port, repeats the preprocessing saved by train_csi.py
(models/checkpoints/preprocess.json), runs the int8 TFLite model on each window and prints
the predicted label. This is the same arithmetic the ESP32 will do later.

Example:
    python src/deploy/predict_live.py --port COM8
    python src/deploy/predict_live.py --port COM8 --model models/checkpoints/lenet_int8.tflite

Needs:  python -m pip install tensorflow numpy pyserial
Close idf.py monitor and any other script using the port first.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import serial
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # src/deploy/ -> repository root
CHECKPOINTS = REPO / "models" / "checkpoints"


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
    imag, real = values[0::2], values[1::2]   # stored as imaginary, real pairs
    return np.sqrt(imag * imag + real * real)


def load_interpreter(path):
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=path)
    except (ImportError, AttributeError):
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def main():
    parser = argparse.ArgumentParser(description="Live CSI classification on the PC")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM8")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--model", default=str(CHECKPOINTS / "cnn1d_int8.tflite"), help="a .tflite file from train_csi.py")
    parser.add_argument("--prep", default=str(CHECKPOINTS / "preprocess.json"), help="preprocess.json from the same training run")
    parser.add_argument("--smooth", type=int, default=3,
                        help="average the probabilities of this many windows (1 = off)")
    parser.add_argument("--hold", type=float, default=30,
                        help="report PRESENT from the first movement until this many seconds "
                             "have passed without movement (0 = off)")
    args = parser.parse_args()

    for path in (args.model, args.prep):
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}\nRun src/train/train_csi.py first, or pass --model and --prep.")
    with open(args.prep) as f:
        prep = json.load(f)
    window, stride = prep["window_packets"], prep["stride_packets"]
    bins = [np.array(b) for b in prep["subcarrier_bins"]]
    labels, scale, detrend = prep["labels"], prep["scale"], prep["detrend"]
    needed = max(int(b.max()) for b in bins) + 1      # subcarriers a packet must have

    interpreter = load_interpreter(args.model)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if tuple(inp["shape"][1:3]) != (window, len(bins)):
        sys.exit(f"{args.model} expects input {tuple(inp['shape'][1:3])}, but {args.prep} "
                 f"describes {(window, len(bins))}. Use files from the same training run.")

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as err:
        sys.exit(f"Could not open {args.port}: {err}\n"
                 "Close any monitor or other script that is using the port.")

    # Presence: a still person looks like an empty room to the model, so presence is
    # inferred from movement and held for a while afterwards.
    motion_label = next((l for l in ("moving", "person") if l in labels), None)
    use_presence = args.hold > 0 and motion_label is not None
    last_motion, present = None, False

    live = sys.stdout.isatty()
    print(f"Model: {args.model}   labels: {', '.join(labels)}")
    print(f"Window: {window} packets, new prediction every {stride}. Waiting for CSI... Ctrl+C to stop.\n")
    if use_presence:
        print(f"Presence: PRESENT from the first '{motion_label}' until {args.hold:.0f} s pass without one.\n")

    rows = deque(maxlen=window)
    recent = deque(maxlen=max(1, args.smooth))
    since = 0
    shown = None
    counts = Counter()
    skipped = 0
    started = None
    packets = 0
    present_changes = 0

    try:
        with port:
            while True:
                raw = port.readline()
                if not raw:
                    continue
                amp = parse_amplitudes(raw.decode("utf-8", errors="ignore").strip())
                if amp is None:
                    continue
                if amp.size < needed:
                    skipped += 1
                    continue
                packets += 1
                started = started or time.time()

                # Same steps as training: average into columns, remove the packet's gain.
                binned = np.array([amp[b].mean() for b in bins], dtype=np.float32)
                rows.append(binned / max(float(binned.mean()), 1e-6))
                since += 1
                if len(rows) < window or since < stride:
                    continue
                since = 0

                x = np.array(rows, dtype=np.float32)
                if detrend:
                    x = x / np.maximum(x.mean(axis=0, keepdims=True), 1e-6) - 1.0
                else:
                    x = x - 1.0
                x = (x / scale)[None, :, :, None]

                if inp["dtype"] == np.int8:
                    s, z = inp["quantization"]
                    x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
                t0 = time.perf_counter()
                interpreter.set_tensor(inp["index"], x)
                interpreter.invoke()
                y = interpreter.get_tensor(out["index"])[0].astype(np.float32)
                took = (time.perf_counter() - t0) * 1000
                if out["dtype"] == np.int8:
                    s, z = out["quantization"]
                    y = (y - z) * s

                recent.append(y)
                probs = np.mean(recent, axis=0)
                best = int(np.argmax(probs))
                counts[labels[best]] += 1
                detail = "  ".join(f"{l} {p:4.0%}" for l, p in zip(labels, probs))

                now = time.time()
                if use_presence:
                    if labels[best] == motion_label:
                        last_motion = now
                    is_present = last_motion is not None and now - last_motion <= args.hold
                    if is_present != present:
                        present = is_present
                        if live:
                            sys.stdout.write("\r" + " " * 96 + "\r")
                        print(f"{time.strftime('%H:%M:%S')}  >>> " + ("PRESENT (movement detected)" if present
                              else f"ABSENT  (no movement for {args.hold:.0f} s)"))
                        present_changes += 1

                if labels[best] != shown:
                    shown = labels[best]
                    if live:
                        sys.stdout.write("\r" + " " * 96 + "\r")
                    print(f"{time.strftime('%H:%M:%S')}  {shown.upper():7} ({detail})")
                if live:
                    state = ("PRESENT" if present else "ABSENT ") + " | " if use_presence else ""
                    sys.stdout.write(f"\r  {state}{shown.upper():7} {detail}   {took:.1f} ms on this PC")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    total = sum(counts.values())
    if total:
        rate = packets / max(time.time() - started, 1e-6)
        print(f"\n\n{total} predictions at {rate:.0f} packets/s:  "
              + ",  ".join(f"{l} {counts[l] / total:.0%}" for l in labels))
    else:
        print("\nNo predictions were made. Is the sender powered?")
    if skipped:
        print(f"{skipped} packets were skipped because they had fewer subcarriers than the model was trained on.")


if __name__ == "__main__":
    main()
