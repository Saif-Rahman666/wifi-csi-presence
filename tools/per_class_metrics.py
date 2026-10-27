import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def prf(table):
    t = np.asarray(table, dtype=float)
    tp = np.diag(t)
    predicted = t.sum(axis=0)
    actual = t.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.full_like(tp, np.nan), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.full_like(tp, np.nan),
                   where=np.nan_to_num(denom) > 0)
    return precision, recall, f1, actual.astype(int)


def motion_table(table, labels):
    t = np.asarray(table)
    m = labels.index("moving")
    q = [i for i in range(len(labels)) if i != m]
    return [[t[m, m], t[m, q].sum()],
            [t[np.ix_(q, [m])].sum(), t[np.ix_(q, q)].sum()]], ["moving", "quiet"]


def load(folder):
    info = json.loads((folder / "preprocess.json").read_text())
    entry = info["models"][0]
    if "confusion_int8" not in entry:
        return None
    return entry["confusion_int8"], info["labels"]


def pct(v):
    return "–" if v is None or np.isnan(v) else f"{v * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Per-class precision, recall and F1 from the sweep")
    parser.add_argument("--sweep", default=str(REPO / "results" / "sweep"))
    parser.add_argument("--setting", default="cnn1d_w100_s50_f8-16", help="setting for the markdown tables")
    args = parser.parse_args()

    sweep = Path(args.sweep)
    rows = []
    summary = {}  
    for setting in sorted(p for p in sweep.iterdir() if p.is_dir()):
        for run in sorted(p for p in setting.iterdir() if p.is_dir() and (p / "preprocess.json").exists()):
            loaded = load(run)
            if loaded is None:
                print(f"  no confusion matrix in {run / 'preprocess.json'} (trained before it was saved)")
                continue
            table, labels = loaded
            fold = run.name.replace("test_", "") if run.name.startswith("test_") else "all_rooms"
            for view, (tab, labs) in (("3-class", (table, labels)), ("motion", motion_table(table, labels))):
                p, r, f1, n = prf(tab)
                summary.setdefault((setting.name, view), {})[fold] = (labs, p, r, f1, n)
                for i, lab in enumerate(labs):
                    rows.append({"setting": setting.name, "fold": fold, "view": view, "class": lab,
                                 "precision": p[i], "recall": r[i], "f1": f1[i], "support": n[i]})

    if not rows:
        raise SystemExit(f"No preprocess.json with confusion matrices found under {sweep}")

    out_csv = sweep / "per_class_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    md = [f"# Per-class metrics: {args.setting}\n"]
    for view in ("3-class", "motion"):
        runs = summary.get((args.setting, view))
        if not runs:
            continue
        folds = sorted(k for k in runs if k != "all_rooms")
        labs = runs[folds[0] if folds else "all_rooms"][0]
        title = "Three classes" if view == "3-class" else "Motion view (quiet = empty + still)"
        if folds:
            md.append(f"## {title}: leave-one-room-out\n")
            md.append("| Held-out room | Class | Precision | Recall | F1 | Windows |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for room in folds:
                _, p, r, f1, n = runs[room]
                for i, lab in enumerate(labs):
                    md.append(f"| {room} | {lab} | {pct(p[i])} | {pct(r[i])} | {pct(f1[i])} | {n[i]} |")
            md.append("")
            md.append("| Class | Precision (mean of rooms) | Recall (mean of rooms) | F1 (mean of rooms) |")
            md.append("| --- | --- | --- | --- |")
            for i, lab in enumerate(labs):
                p = np.nanmean([runs[k][1][i] for k in folds])
                r = np.nanmean([runs[k][2][i] for k in folds])
                f = np.nanmean([runs[k][3][i] for k in folds])
                md.append(f"| {lab} | {pct(p)} | {pct(r)} | {pct(f)} |")
            macro = np.nanmean([np.nanmean(runs[k][3]) for k in folds])
            md.append(f"| **Macro F1** | | | {pct(macro)} |\n")
        if "all_rooms" in runs:
            _, p, r, f1, n = runs["all_rooms"]
            md.append(f"## {title}: all-rooms model, same-session test\n")
            md.append("| Class | Precision | Recall | F1 | Windows |")
            md.append("| --- | --- | --- | --- | --- |")
            for i, lab in enumerate(labs):
                md.append(f"| {lab} | {pct(p[i])} | {pct(r[i])} | {pct(f1[i])} | {n[i]} |")
            md.append("")
    out_md = sweep / "per_class_tables.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_md}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
