"""Train and score many model settings in one go, and collect the results in one table.

For every combination of model, window, stride and (for cnn1d) filters it:
  1. runs leave-one-room-out: trains on all rooms but one, tests on the one left out,
     once per room (tools/train_csi.py --test-room <room>);
  2. optionally trains once on every room (--test-room none) and scores that model on the
     scenario recordings with tools/evaluate.py.
Finished runs are kept in results/sweep/<setting>/ and reused when you run the sweep again,
so an interrupted sweep continues where it stopped (--fresh retrains everything).

Examples:
    python tools/sweep.py --data data/raw --models cnn1d --windows 50 100 --filters 8,16 16,32 32,64 16,32,32
    python tools/sweep.py --data data/raw --models mlp lenet cnn1d --windows 50 100 --scenarios data/scenarios

Each training run takes a minute or two on a laptop CPU: the number of runs is printed first.
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]          # tools/ -> repository root
LABELS = ("empty", "still", "moving")

TRAIN = REPO / "tools" / "train_csi.py"
EVALUATE = REPO / "tools" / "evaluate.py"
N_BINS = 64


def find_rooms(folder):
    """Room names in a data folder: <room>/<date>/<label>_<time>.csv or <room>_<label>_<stamp>.csv."""
    rooms = set()
    for path in Path(folder).rglob("*.csv"):
        parts = path.stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1] in LABELS:
            rooms.add(parts[0])
        elif path.stem.split("_", 1)[0] in LABELS:
            room_dir = path.parent
            if room_dir.name[:1].isdigit():          # skip the date folder
                room_dir = room_dir.parent
            rooms.add(room_dir.name)
    return sorted(rooms)


def macs(model, window, filters, n_bins=N_BINS, n_classes=3):
    """Multiply-adds for one window (the work per prediction)."""
    if model == "mlp":
        return window * n_bins * 32 + 32 * 16 + 16 * n_classes
    if model == "lenet":
        w, b = window, n_bins
        total = w * b * 8 * 25
        w, b = w // 2, b // 2
        total += w * b * 16 * 9 * 8
        w, b = w // 2, b // 2
        total += w * b * 32 * 9 * 16
        w, b = w // 2, b // 2
        return total + w * b * 32 * 32 + 32 * n_classes
    steps = window - 4
    total = steps * filters[0] * 5 * n_bins
    steps //= 2
    for prev, f in zip(filters, filters[1:]):
        steps -= 4
        total += steps * f * 5 * prev
    return total + filters[-1] * n_classes


def scores(table, labels):
    """3-class accuracy and motion-vs-no-motion scores from a confusion matrix."""
    t = np.array(table)
    acc = float(np.trace(t) / t.sum()) if t.sum() else None
    motion = next((labels.index(l) for l in ("moving", "person") if l in labels), None)
    if motion is None:
        return acc, None, None, None
    quiet = [i for i in range(len(labels)) if i != motion]
    moving_found = float(t[motion, motion] / t[motion].sum()) if t[motion].sum() else None
    q_total = t[quiet].sum()
    quiet_ok = float(1 - t[quiet][:, motion].sum() / q_total) if q_total else None
    balanced = (moving_found + quiet_ok) / 2 if moving_found is not None and quiet_ok is not None else None
    return acc, moving_found, quiet_ok, balanced


def train(cfg, data, test_room, out, epochs, fresh, log):
    """Run train_csi.py once; return the saved model entry and labels, or None if it failed."""
    prep_path = out / "preprocess.json"
    if prep_path.exists() and not fresh:
        info = json.loads(prep_path.read_text())
        return info["models"][0], info["labels"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRAIN), "--data", str(data), "--model", cfg["model"],
           "--window", str(cfg["window"]), "--stride", str(cfg["stride"]),
           "--filters", cfg["filters"], "--test-room", test_room, "--epochs", str(epochs),
           "--out", str(out)]
    with open(out / "train.log", "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not prep_path.exists():
        log(f"    FAILED (see {out / 'train.log'})")
        return None
    info = json.loads(prep_path.read_text())
    return info["models"][0], info["labels"]


def main():
    parser = argparse.ArgumentParser(description="Sweep model settings with leave-one-room-out and scenarios")
    parser.add_argument("--data", default=str(REPO / "data"), help="training rooms (keep scenarios out of here)")
    parser.add_argument("--models", nargs="+", default=["cnn1d"], choices=["mlp", "lenet", "cnn1d"])
    parser.add_argument("--windows", nargs="+", type=int, default=[50, 100], help="packets per window")
    parser.add_argument("--strides", nargs="+", type=int,
                        help="packets between windows (default: half of each window)")
    parser.add_argument("--filters", nargs="+", default=["16,32"],
                        help="cnn1d filters per layer, one setting each, e.g. 8,16 16,32 16,32,32")
    parser.add_argument("--scenarios", help="also train on every room and score this scenario folder")
    parser.add_argument("--no-folds", action="store_true", help="skip leave-one-room-out (scenarios only)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default=str(REPO / "results" / "sweep"))
    parser.add_argument("--fresh", action="store_true", help="retrain even where results already exist")
    args = parser.parse_args()

    if not TRAIN.exists():
        sys.exit(f"Not found: {TRAIN}")
    if args.scenarios and not EVALUATE.exists():
        sys.exit(f"Not found: {EVALUATE}\nPut evaluate.py in tools/ (needed for --scenarios).")
    rooms = find_rooms(args.data)
    if not args.no_folds and len(rooms) < 2:
        sys.exit(f"Leave-one-room-out needs at least two rooms in '{args.data}', found {rooms}.")

    configs = []
    for model, window in itertools.product(args.models, args.windows):
        strides = args.strides or [window // 2]
        filter_sets = args.filters if model == "cnn1d" else ["16,32"]
        for stride, filt in itertools.product(strides, filter_sets):
            f = tuple(int(x) for x in filt.split(","))
            if model == "cnn1d" and (window - 4) // 2 - 4 * (len(f) - 1) < 1:
                print(f"  skipping cnn1d {filt} with window {window}: too deep for that window")
                continue
            name = f"{model}_w{window}_s{stride}" + (f"_f{filt.replace(',', '-')}" if model == "cnn1d" else "")
            configs.append({"name": name, "model": model, "window": window, "stride": stride,
                            "filters": filt, "macs": macs(model, window, f)})

    folds = [] if args.no_folds else rooms
    runs = len(configs) * (len(folds) + (1 if args.scenarios else 0))
    print(f"Rooms: {', '.join(rooms)}")
    print(f"{len(configs)} settings x ({len(folds)} folds" + (" + 1 all-rooms model" if args.scenarios else "")
          + f") = {runs} training runs. Results go to {args.out}\n")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "sweep.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    rows = []
    started = time.time()
    for n, cfg in enumerate(configs, 1):
        log(f"[{n}/{len(configs)}] {cfg['name']}")
        row = {k: cfg[k] for k in ("name", "model", "window", "stride", "filters", "macs")}
        fold_scores = []
        for room in folds:
            res = train(cfg, args.data, room, out_root / cfg["name"] / f"test_{room}",
                        args.epochs, args.fresh, log)
            if res is None:
                continue
            entry, labels = res
            row["parameters"], row["int8_kb"] = entry["parameters"], round(entry["int8_kb"], 1)
            acc, mf, qo, bal = scores(entry["confusion_int8"], labels)
            fold_scores.append((acc, mf, qo, bal))
            row[f"acc_{room}"] = acc
            row[f"balanced_{room}"] = bal
            log(f"    unseen {room:18} 3-class {acc:.1%}   motion balanced "
                + (f"{bal:.1%}" if bal is not None else "-"))
        for i, key in enumerate(("acc_mean", "moving_found_mean", "quiet_ok_mean", "balanced_mean")):
            vals = [s[i] for s in fold_scores if s[i] is not None]
            row[key] = float(np.mean(vals)) if vals else None
        row["balanced_min"] = min((s[3] for s in fold_scores if s[3] is not None), default=None)

        if args.scenarios:
            out = out_root / cfg["name"] / "all_rooms"
            res = train(cfg, args.data, "none", out, args.epochs, args.fresh, log)
            if res is not None:
                entry, _ = res
                row["parameters"], row["int8_kb"] = entry["parameters"], round(entry["int8_kb"], 1)
                scen_csv = out / "scenarios" / "scenario_results.csv"
                if not scen_csv.exists() or args.fresh:
                    with open(out / "evaluate.log", "w") as f:
                        subprocess.run([sys.executable, str(EVALUATE), "--data", args.scenarios,
                                        "--model", str(out / f"{cfg['model']}_int8.tflite"),
                                        "--prep", str(out / "preprocess.json"),
                                        "--out", str(out / "scenarios")], stdout=f, stderr=subprocess.STDOUT)
                if scen_csv.exists():
                    with open(scen_csv) as f:
                        scen = [r for r in csv.DictReader(f) if r["balanced"]]
                    vals = [float(r["balanced"]) for r in scen]
                    row["scenario_balanced_mean"] = float(np.mean(vals)) if vals else None
                    row["scenario_balanced_min"] = min(vals) if vals else None
                    for r in scen:
                        row[f"scen_{r['scenario']}"] = float(r["balanced"])
                    if vals:
                        log(f"    scenarios: balanced mean {np.mean(vals):.1%}, worst {min(vals):.1%}")
                else:
                    log(f"    scenario scoring failed (see {out / 'evaluate.log'})")
        rows.append(row)
        log(f"    ({(time.time() - started) / 60:.0f} min so far)")

    # ---- Table ----
    key = "balanced_mean" if not args.no_folds else "scenario_balanced_mean"
    rows.sort(key=lambda r: -(r.get(key) or 0))
    fmt = lambda v: "   -  " if v is None else f"{v * 100:5.1f}%"
    header = f"\n{'setting':28}{'params':>8}{'int8 KB':>9}{'MACs':>9}"
    if not args.no_folds:
        header += f"{'3-class':>9}{'motion bal.':>12}{'worst room':>11}"
    if args.scenarios:
        header += f"{'scen. bal.':>11}{'worst scen.':>12}"
    log(header)
    for r in rows:
        line = (f"{r['name'][:27]:28}{r.get('parameters', 0):>8,}{r.get('int8_kb', 0):>9.1f}"
                f"{r['macs'] / 1000:>8.0f}k")
        if not args.no_folds:
            line += f"{fmt(r.get('acc_mean')):>9}{fmt(r.get('balanced_mean')):>12}{fmt(r.get('balanced_min')):>11}"
        if args.scenarios:
            line += f"{fmt(r.get('scenario_balanced_mean')):>11}{fmt(r.get('scenario_balanced_min')):>12}"
        log(line)
    log("\n3-class and motion balanced: mean over the unseen rooms (leave-one-room-out). "
        "Scenario columns: the all-rooms model on the scenario recordings. Chance: 33% and 50%.")

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    table = out_root / "sweep_results.csv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Table saved to {table}")


if __name__ == "__main__":
    main()
