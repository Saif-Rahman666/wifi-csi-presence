import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

LABEL_ORDER = ["empty", "still", "moving"]
REPO = Path(__file__).resolve().parents[1]         

CHECKPOINTS = REPO / "models" / "checkpoints"
RSSI_FIELD = 3                     


def load_recording(path):
    """Read one CSV and return amplitudes with shape (packets, subcarriers)."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("CSI_DATA"):
                continue
            start, end = line.find("["), line.rfind("]")
            if start < 0 or end < start:
                continue
            try:
                values = np.array(line[start + 1:end].split(","), dtype=np.float32)
            except ValueError:
                continue
            if values.size >= 2 and values.size % 2 == 0:
                rows.append(values)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    usual = Counter(r.size for r in rows).most_common(1)[0][0] 
    data = np.array([r for r in rows if r.size == usual])
    imag, real = data[:, 0::2], data[:, 1::2]                 
    return np.sqrt(imag * imag + real * real)


def find_recordings(folder, labels=None, quiet=False):
    """Find recordings in a folder and its subfolders, as a list of (room, label, path).

    Two naming schemes are understood:
      <room>/<date>/<label>_<time>.csv      the layout in data/README.md, written by tools/record_csi.py
      <room>_<label>_<date-time>.csv        the older flat names
    """
    labels = labels or LABEL_ORDER
    found = []
    for path in sorted(Path(folder).rglob("*.csv")):
        parts = path.stem.rsplit("_", 2)
        first = path.stem.split("_", 1)[0]
        if len(parts) == 3 and parts[1] in labels:
            found.append((parts[0], parts[1], path))
        elif first in labels:
            room_dir = path.parent
            if room_dir.name[:1].isdigit():     
                room_dir = room_dir.parent
            found.append((room_dir.name, first, path))
        elif not quiet:
            print(f"  skipping {path}: cannot tell its room and label from the name")
    return found


def load_interpreter(path):
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(path))
    except (ImportError, AttributeError):
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def mean_rssi(path):
    values = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("CSI_DATA"):
                try:
                    values.append(float(line.split(",", RSSI_FIELD + 1)[RSSI_FIELD]))
                except (ValueError, IndexError):
                    pass
    return float(np.mean(values)) if values else None


def read_meta(path):
    meta = path.with_name(path.stem + "_meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except (OSError, ValueError):
            pass
    return {}


def make_windows(amp, prep):
    """Same steps as training and predict_live.py: bins, packet gain, window, detrend, scale."""
    bins = [np.array(b) for b in prep["subcarrier_bins"]]
    window, stride = prep["window_packets"], prep["stride_packets"]
    rows = np.stack([amp[:, b].mean(axis=1) for b in bins], axis=1)
    rows = rows / np.maximum(rows.mean(axis=1, keepdims=True), 1e-6)
    if len(rows) < window:
        return np.zeros((0, window, len(bins)), dtype=np.float32)
    out = np.stack([rows[s:s + window] for s in range(0, len(rows) - window + 1, stride)]).astype(np.float32)
    if prep["detrend"]:
        out = out / np.maximum(out.mean(axis=1, keepdims=True), 1e-6) - 1.0
    else:
        out = out - 1.0
    return out / prep["scale"]


def predict(interpreter, windows):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    preds = np.empty(len(windows), dtype=int)
    for i, w in enumerate(windows):
        x = w[None, :, :, None]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
    return preds


def pct(v):
    return "   -  " if v is None else f"{v * 100:5.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Score a trained model on new recordings, per scenario")
    parser.add_argument("--data", default=str(REPO / "data"), help="folder with the scenario recordings (searched recursively)")
    parser.add_argument("--model", default=str(CHECKPOINTS / "cnn1d_int8.tflite"), help="a .tflite from train_csi.py")
    parser.add_argument("--prep", default=str(CHECKPOINTS / "preprocess.json"), help="preprocess.json from the same run")
    parser.add_argument("--out", default=str(REPO / "results"), help="folder for the results table and chart")
    parser.add_argument("--per-file", action="store_true", help="also print one line per recording")
    parser.add_argument("--by-recording", action="store_true",
                        help="also score every recording on its own and every setup (same notes) on its own; "
                             "saves recording_results.csv / setup_results.csv and a chart")
    args = parser.parse_args()

    for path in (args.model, args.prep):
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}\nTrain first (tools/train_csi.py), or pass --model and --prep.")
    prep = json.loads(Path(args.prep).read_text())
    labels = prep["labels"]
    motion = next((l for l in ("moving", "person") if l in labels), None)
    if motion is None:
        sys.exit(f"The model's labels {labels} have no 'moving' or 'person' class to score.")
    needed = max(max(b) for b in prep["subcarrier_bins"]) + 1

    interpreter = load_interpreter(args.model)
    shape = tuple(interpreter.get_input_details()[0]["shape"][1:3])
    if shape != (prep["window_packets"], len(prep["subcarrier_bins"])):
        sys.exit(f"{args.model} expects {shape} but {args.prep} describes "
                 f"{(prep['window_packets'], len(prep['subcarrier_bins']))}. Use files from the same run.")

    recordings = find_recordings(args.data)
    if not recordings:
        sys.exit(f"No recordings found in '{args.data}'.")
    print(f"Model: {args.model}  ({', '.join(labels)}, {prep['window_packets']}-packet windows)")
    print(f"Scoring {len(recordings)} recordings from '{args.data}'...\n")

    scen = defaultdict(lambda: {"true": [], "pred": [], "rssi": [], "pps": [], "distance": None, "obstruction": None,
                                 "notes": []})
    records = []                                      
    for room, label, path in recordings:
        amp = load_recording(path)
        if amp.size == 0 or amp.shape[1] < needed:
            print(f"  skipping {path}: unreadable or too few subcarriers")
            continue
        windows = make_windows(amp, prep)
        if len(windows) == 0:
            print(f"  skipping {path}: shorter than one window")
            continue
        preds = predict(interpreter, windows)
        s = scen[room]
        s["true"] += [label] * len(preds)
        s["pred"] += [labels[p] for p in preds]
        rssi = mean_rssi(path)
        if rssi is not None:
            s["rssi"].append(rssi)
        meta = read_meta(path)
        if meta.get("packets_per_s"):
            s["pps"].append(meta["packets_per_s"])
        s["distance"] = s["distance"] if s["distance"] is not None else meta.get("distance_m")
        s["obstruction"] = s["obstruction"] or meta.get("obstruction")
        if meta.get("notes") and meta["notes"] not in s["notes"]:
            s["notes"].append(meta["notes"])
        records.append({"scenario": room, "label": label, "file": path.name,
                        "start_time": meta.get("start_time") or "", "notes": (meta.get("notes") or "").strip(),
                        "rssi": rssi, "pred": [labels[p] for p in preds]})
        if args.per_file:
            called = np.mean([labels[p] == motion for p in preds])
            print(f"  {room:16} {label:7} {path.name:28} {len(preds):4} windows, called moving {called:5.1%}")

    results = []
    for room in sorted(scen):
        s = scen[room]
        true, pred = np.array(s["true"]), np.array(s["pred"])
        is_moving, said_moving = true == "moving", pred == motion
        quiet = ~is_moving
        moving_found = float(said_moving[is_moving].mean()) if is_moving.any() else None
        quiet_ok = float((~said_moving[quiet]).mean()) if quiet.any() else None
        parts = [v for v in (moving_found, quiet_ok) if v is not None]
        three = float((pred == true).mean()) if set(true) <= set(labels) and len(labels) == 3 else None
        results.append({
            "scenario": room, "obstruction": s["obstruction"] or "", "distance_m": s["distance"],
            "rssi_dbm": round(float(np.mean(s["rssi"])), 1) if s["rssi"] else None,
            "packets_per_s": round(float(np.mean(s["pps"])), 1) if s["pps"] else None,
            "windows": len(true), "moving_windows": int(is_moving.sum()), "quiet_windows": int(quiet.sum()),
            "moving_found": moving_found, "quiet_called_quiet": quiet_ok,
            "balanced": float(np.mean(parts)) if len(parts) == 2 else None, "accuracy_3class": three,
            "notes": " | ".join(s["notes"]),
        })

    print(f"\n{'scenario':16}{'obstruction':16}{'dist':>6}{'RSSI':>7}{'windows':>9}"
          f"{'moving found':>14}{'quiet ok':>10}{'balanced':>10}{'3-class':>9}")
    for r in results:
        dist = f"{r['distance_m']:g} m" if r["distance_m"] is not None else "-"
        rssi = f"{r['rssi_dbm']:.0f}" if r["rssi_dbm"] is not None else "-"
        print(f"{r['scenario'][:15]:16}{r['obstruction'][:15]:16}{dist:>6}{rssi:>7}{r['windows']:>9}"
              f"{pct(r['moving_found']):>14}{pct(r['quiet_called_quiet']):>10}{pct(r['balanced']):>10}"
              f"{pct(r['accuracy_3class']):>9}")
    noted = [r for r in results if r["notes"]]
    if noted:
        print("\nNotes from the recordings:")
        for r in noted:
            print(f"  {r['scenario']:16}{r['notes']}")
    print("\nmoving found = recall of moving;  quiet ok = empty/still windows not called moving;  "
          "balanced = their average (chance 50%).")
    missing = [r["scenario"] for r in results if r["balanced"] is None]
    if missing:
        print(f"No balanced score for {', '.join(missing)}: record both 'empty' and 'moving' there.")

    os.makedirs(args.out, exist_ok=True)
    if args.by_recording:
        by_recording(records, motion, labels, Path(args.out))
    table = Path(args.out) / "scenario_results.csv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nTable saved to {table}")
    plot(results, Path(args.out) / "scenario_results.png")


def score(true, pred, motion, labels):
    true, pred = np.array(true), np.array(pred)
    is_moving, said_moving = true == "moving", pred == motion
    moving_found = float(said_moving[is_moving].mean()) if is_moving.any() else None
    quiet_ok = float((~said_moving[~is_moving]).mean()) if (~is_moving).any() else None
    balanced = (moving_found + quiet_ok) / 2 if moving_found is not None and quiet_ok is not None else None
    three = float((pred == true).mean()) if set(true) <= set(labels) and len(labels) == 3 else None
    return moving_found, quiet_ok, balanced, three


def by_recording(records, motion, labels, out):
    """Score each recording alone, and each setup (scenario + identical notes) alone."""
    rows = []
    for r in sorted(records, key=lambda r: (r["scenario"], r["notes"], r["label"], r["start_time"])):
        mf, qo, _, three = score([r["label"]] * len(r["pred"]), r["pred"], motion, labels)
        called = float(np.mean([p == motion for p in r["pred"]]))
        rows.append({"scenario": r["scenario"], "label": r["label"], "file": r["file"],
                     "start_time": r["start_time"], "windows": len(r["pred"]),
                     "rssi_dbm": round(r["rssi"], 1) if r["rssi"] is not None else None,
                     "called_moving": called,
                     "correct_motion": called if r["label"] == "moving" else 1 - called,
                     "correct_3class": three, "notes": r["notes"]})
    print("\nPer recording  (correct = share of windows judged right as motion vs no motion)")
    print(f"  {'scenario':14}{'label':8}{'windows':>8}{'RSSI':>6}{'called moving':>15}{'correct':>9}  notes")
    for r in rows:
        rssi = f"{r['rssi_dbm']:.0f}" if r["rssi_dbm"] is not None else "-"
        print(f"  {r['scenario'][:13]:14}{r['label']:8}{r['windows']:>8}{rssi:>6}"
              f"{pct(r['called_moving']):>15}{pct(r['correct_motion']):>9}  {r['notes']}")

    groups = defaultdict(list)
    for r in records:
        groups[(r["scenario"], r["notes"])].append(r)
    setups = []
    for (scenario, notes), recs in sorted(groups.items()):
        true = [x["label"] for x in recs for _ in x["pred"]]
        pred = [p for x in recs for p in x["pred"]]
        mf, qo, bal, three = score(true, pred, motion, labels)
        setups.append({"scenario": scenario, "notes": notes, "recordings": len(recs),
                       "labels": "+".join(sorted({x["label"] for x in recs})), "windows": len(pred),
                       "moving_found": mf, "quiet_called_quiet": qo, "balanced": bal, "accuracy_3class": three})
    print("\nPer setup  (recordings with identical notes in the same scenario)")
    print(f"  {'scenario':14}{'labels':15}{'windows':>8}{'moving found':>14}{'quiet ok':>10}{'balanced':>10}  notes")
    for s in setups:
        print(f"  {s['scenario'][:13]:14}{s['labels'][:14]:15}{s['windows']:>8}{pct(s['moving_found']):>14}"
              f"{pct(s['quiet_called_quiet']):>10}{pct(s['balanced']):>10}  {s['notes']}")
    print("  balanced needs both a moving and a quiet (empty/still) recording with the same notes.")

    for name, data in (("recording_results.csv", rows), ("setup_results.csv", setups)):
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    print(f"\nSaved {out / 'recording_results.csv'} and {out / 'setup_results.csv'}")
    plot_recordings(rows, out / "recording_results.png")


def plot_recordings(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ink, muted, grid = "#2b2b29", "#8a8a85", "#e4e3df"
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(rows))[::-1]
    colours = ["#1f5fd0" if r["label"] == "moving" else "#e0a100" for r in rows]
    ax.barh(y, [r["correct_motion"] * 100 for r in rows], color=colours, height=0.62)
    for yi, r in zip(y, rows):
        ax.text(r["correct_motion"] * 100 + 1, yi, f"{r['correct_motion'] * 100:.0f}%", va="center",
                fontsize=8, color=ink)
    names = [f"{r['scenario']} | {r['label']} | {r['notes'][:60]}" for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=ink)
    ax.axvline(50, color=muted, linestyle="--", linewidth=1)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Windows judged right as motion / no motion (%)", color=ink)
    ax.set_title("Each scenario recording on its own   (blue: moving, amber: still/empty)", loc="left",
                 color=ink, fontsize=11)
    ax.grid(axis="x", color=grid, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=140)
    except OSError:
        import time
        path = path.with_name(f"recording_results_{time.strftime('%H%M%S')}.png")
        fig.savefig(path, dpi=140)
    print(f"Chart saved to {path}")


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, so no chart was saved)")
        return
    rows = [r for r in results if r["moving_found"] is not None or r["quiet_called_quiet"] is not None]
    dist = sorted([r for r in rows if r["distance_m"] is not None and r["balanced"] is not None
                   and r["obstruction"].lower() in ("", "none")], key=lambda r: r["distance_m"])
    panels = 2 if len(dist) >= 2 else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 4.6), facecolor="#fcfcfb", squeeze=False)
    ink, muted, grid = "#2b2b29", "#8a8a85", "#e4e3df"
    colours = {"moving found": "#1f5fd0", "quiet called quiet": "#e0a100"}

    ax = axes[0][0]
    x = np.arange(len(rows))
    for i, (name, key) in enumerate((("moving found", "moving_found"), ("quiet called quiet", "quiet_called_quiet"))):
        vals = [(r[key] or 0) * 100 for r in rows]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=colours[name], label=name)
        for b, v, r in zip(bars, vals, rows):
            if r[key] is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=ink)
    ax.set_xticks(x)
    ax.set_xticklabels([r["scenario"] for r in rows], rotation=30, ha="right", color=ink)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of windows", color=ink)
    ax.set_title("Detection per scenario", loc="left", color=ink, pad=26)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)

    if panels == 2:
        ax2 = axes[0][1]
        d = [r["distance_m"] for r in dist]
        ax2.plot(d, [r["balanced"] * 100 for r in dist], marker="o", color="#1f5fd0", linewidth=2)
        for r in dist:
            ax2.annotate(f"{r['balanced'] * 100:.0f}%", (r["distance_m"], r["balanced"] * 100),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=ink)
        ax2.axhline(50, color=muted, linestyle="--", linewidth=1)
        ax2.text(d[0], 52, "chance", color=muted, fontsize=8)
        ax2.set_ylim(0, 110)
        ax2.set_xlabel("Distance between boards (m), line of sight", color=ink)
        ax2.set_ylabel("Balanced accuracy (%)", color=ink)
        ax2.set_title("Detection against distance", loc="left", color=ink, pad=26)

    for row in axes:
        for a in row:
            a.set_facecolor("#fcfcfb")
            a.grid(axis="y", color=grid, linewidth=1)
            a.set_axisbelow(True)
            a.tick_params(colors=muted, length=0)
            for side in ("top", "right", "left"):
                a.spines[side].set_visible(False)
            a.spines["bottom"].set_color(grid)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=140)
    except OSError:                               
        import time
        path = path.with_name(f"scenario_results_{time.strftime('%H%M%S')}.png")
        fig.savefig(path, dpi=140)
    print(f"Chart saved to {path}")


if __name__ == "__main__":
    main()
