
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

LABEL_ORDER = ["empty", "still", "moving"]
REPO = Path(__file__).resolve().parents[2]         



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



class Preprocessor:
    """Turns raw amplitudes into model inputs. The firmware has to repeat these steps."""

    def __init__(self, window, stride, n_bins, detrend):
        self.window, self.stride, self.n_bins, self.detrend = window, stride, n_bins, detrend
        self.active = None     
        self.bins = None    
        self.scale = 1.0      

    def fit_layout(self, amplitudes):
        """Choose the active subcarriers and group them into n_bins columns."""
        mean_amp = np.concatenate(amplitudes).mean(axis=0)
        self.active = np.flatnonzero(mean_amp > 0.1 * np.median(mean_amp[mean_amp > 0]))
        n_bins = min(self.n_bins, self.active.size)
        self.bins = [chunk.tolist() for chunk in np.array_split(self.active, n_bins)]
        self.n_bins = n_bins

    def rows(self, amp):
        """Per packet: average subcarriers into bins, then divide by the packet's mean.

        Dividing by the mean removes the receiver's gain steps. A still room then
        gives values close to 1 in every column.
        """
        binned = np.stack([amp[:, idx].mean(axis=1) for idx in self.bins], axis=1)
        return binned / np.maximum(binned.mean(axis=1, keepdims=True), 1e-6)

    def windows(self, rows):
        """Cut into windows of shape (window, n_bins), centred on zero."""
        starts = range(0, len(rows) - self.window + 1, self.stride)
        if len(rows) < self.window:
            return np.zeros((0, self.window, self.n_bins), dtype=np.float32)
        out = np.stack([rows[s:s + self.window] for s in starts]).astype(np.float32)
        if self.detrend:
            out = out / np.maximum(out.mean(axis=1, keepdims=True), 1e-6) - 1.0
        else:
            out = out - 1.0
        return out

    def to_json(self, labels, input_details=None):
        return {
            "window_packets": self.window,
            "stride_packets": self.stride,
            "input_columns": self.n_bins,
            "subcarrier_bins": self.bins,
            "steps": [
                "amplitude = sqrt(imag^2 + real^2) for each subcarrier",
                "column = mean amplitude of the subcarriers in each bin",
                "divide the packet's columns by their mean",
                "divide each column by its mean over the window, then subtract 1.0" if self.detrend
                else "subtract 1.0",
                "divide by scale",
            ],
            "detrend": self.detrend,
            "scale": self.scale,
            "labels": labels,
        }



def build_model(name, window, n_bins, n_classes):
    """SenseFi-style architectures, shrunk for a microcontroller."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(window, n_bins, 1), name="csi_window")
    if name == "mlp":
        x = layers.Flatten()(inputs)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
    elif name == "lenet":
        x = layers.Conv2D(8, 5, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
    elif name == "cnn1d":
        x = layers.Conv2D(16, (5, n_bins), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Conv2D(32, (5, 1), activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(name)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model, sample_inputs, int8):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        def representative():
            for x in sample_inputs:
                yield [x[None, ...]]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def predict_tflite(model_bytes, inputs):
    """Run a TFLite model on the PC, the same arithmetic the ESP32 will do."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    predictions = []
    for x in inputs:
        x = x[None, ...]
        if inp["dtype"] == np.int8:
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        predictions.append(int(np.argmax(interpreter.get_tensor(out["index"])[0])))
    return np.array(predictions), {"scale": float(scale), "zero_point": int(zero)}


def confusion(true, predicted, n):
    table = np.zeros((n, n), dtype=int)
    for t, p in zip(true, predicted):
        table[t, p] += 1
    return table



def main():
    parser = argparse.ArgumentParser(description="Train SenseFi-style models on ESP32 CSI")
    parser.add_argument("--data", default=str(REPO / "data"), help="folder with the recordings (searched recursively)")
    parser.add_argument("--out", default=str(REPO / "models" / "checkpoints"), help="folder for the trained models")
    parser.add_argument("--model", default="all", choices=["all", "mlp", "lenet", "cnn1d"])
    parser.add_argument("--test-room", help="room held out for testing (never trained on)")
    parser.add_argument("--binary", action="store_true",
                        help="two classes: empty vs person (still and moving merged)")
    parser.add_argument("--window", type=int, default=50, help="packets per window")
    parser.add_argument("--stride", type=int, default=25, help="packets between window starts")
    parser.add_argument("--columns", type=int, default=64,
                        help="subcarriers are averaged down to this many input columns")
    parser.add_argument("--keep-level", action="store_true",
                        help="keep each window's static subcarrier profile instead of removing it "
                             "(more information within a room, but it is a fingerprint of that room)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        sys.exit("TensorFlow is not installed. Run:  python -m pip install tensorflow numpy")
    keras.utils.set_random_seed(args.seed)

    recordings = find_recordings(args.data)
    if not recordings:
        sys.exit(f"No recordings found in '{args.data}'.")
    print(f"Loading {len(recordings)} recordings from '{args.data}'...")
    loaded = []
    for room, label, path in recordings:
        amp = load_recording(path)
        if len(amp) < args.window * 4:
            print(f"  skipping {path.name}: only {len(amp)} packets")
            continue
        loaded.append((room, label, amp, path.name))
    sizes = Counter(r[2].shape[1] for r in loaded)
    usual = sizes.most_common(1)[0][0]
    if len(sizes) > 1:
        print(f"  recordings disagree on subcarrier count {dict(sizes)}; keeping {usual}")
        loaded = [r for r in loaded if r[2].shape[1] == usual]

    label_names = ["empty", "person"] if args.binary else [l for l in LABEL_ORDER
                                                           if any(r[1] == l for r in loaded)]

    def class_of(label):
        if args.binary:
            return 0 if label == "empty" else 1
        return label_names.index(label)

    rooms = sorted({r[0] for r in loaded})
    if args.test_room and args.test_room not in rooms:
        sys.exit(f"Room '{args.test_room}' not found. Rooms in the data: {rooms}")
    test_room = args.test_room or (rooms[0] if len(rooms) > 1 else None)
    if test_room:
        print(f"Rooms: {rooms}.  Testing on unseen room '{test_room}', training on the rest.")
    else:
        print(f"Only one room ({rooms[0]}): testing on the last quarter of each recording.\n"
              "  This is NOT an unseen-room test; expect it to look better than reality.")

    train_parts, val_parts, test_parts = [], [], []   
    groups = {}
    for room, label, amp, fname in loaded:
        c = class_of(label)
        if test_room and room == test_room:
            test_parts.append((amp, c))
            continue
        if not test_room:
            cut = int(len(amp) * 0.75)
            test_parts.append((amp[cut:], c))
            amp = amp[:cut]
        groups.setdefault((room, label), []).append((fname, amp, c))

    whole_sessions = True
    for sessions in groups.values():
        sessions.sort()                              
        if len(sessions) >= 2 and test_room:
            *train_sessions, (_, amp, c) = sessions
            val_parts.append((amp, c))
            train_parts += [(a, cc) for _, a, cc in train_sessions]
        else:
            whole_sessions = False
            for _, amp, c in sessions:
                cut = int(len(amp) * 0.85)
                train_parts.append((amp[:cut], c))
                val_parts.append((amp[cut:], c))
    print("Validation: " + ("the latest session of each room and label."
                            if whole_sessions else "the last 15% of each training recording."))

    prep = Preprocessor(args.window, args.stride, args.columns, detrend=not args.keep_level)
    prep.fit_layout([amp for amp, _ in train_parts])

    def make_set(parts):
        xs, ys = [], []
        for amp, c in parts:
            w = prep.windows(prep.rows(amp))
            xs.append(w)
            ys.append(np.full(len(w), c))
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train = make_set(train_parts)
    x_val, y_val = make_set(val_parts)
    x_test, y_test = make_set(test_parts)
    prep.scale = float(x_train.std()) or 1.0
    x_train, x_val, x_test = (x[..., None] / prep.scale for x in (x_train, x_val, x_test))

    print(f"Input: {args.window} packets x {prep.n_bins} columns "
          f"(from {prep.active.size} active of {usual} subcarriers)")
    print(f"Windows: train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}")
    for name, y in (("train", y_train), ("test", y_test)):
        counts = ", ".join(f"{label_names[i]} {int((y == i).sum())}" for i in range(len(label_names)))
        print(f"  {name}: {counts}")
    if len(x_test) == 0 or len(x_train) == 0:
        sys.exit("Not enough data to train and test.")

    os.makedirs(args.out, exist_ok=True)
    order = np.random.default_rng(args.seed).permutation(len(x_train))
    representative = x_train[order[:200]]
    names = ["mlp", "lenet", "cnn1d"] if args.model == "all" else [args.model]
    results = []

    for name in names:
        print(f"\n=== {name} ===")
        model = build_model(name, args.window, prep.n_bins, len(label_names))
        stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                             restore_best_weights=True)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=args.epochs, batch_size=32, callbacks=[stop], verbose=0)
        print(f"trained {len(history.history['loss'])} epochs, "
              f"best validation accuracy {max(history.history['val_accuracy']):.3f}")

        float_bytes = export_tflite(model, representative, int8=False)
        int8_bytes = export_tflite(model, representative, int8=True)
        float_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        int8_pred, quant = predict_tflite(int8_bytes, x_test)

        model.save(os.path.join(args.out, f"{name}.keras"))
        Path(args.out, f"{name}_float.tflite").write_bytes(float_bytes)
        Path(args.out, f"{name}_int8.tflite").write_bytes(int8_bytes)

        acc_float = float((float_pred == y_test).mean())
        acc_int8 = float((int8_pred == y_test).mean())
        print(f"test accuracy: float {acc_float:.3f}, int8 {acc_int8:.3f}")
        print("int8 confusion matrix (rows = true, columns = predicted):")
        table = confusion(y_test, int8_pred, len(label_names))
        print("          " + "".join(f"{n:>9}" for n in label_names))
        for i, n in enumerate(label_names):
            print(f"  {n:>8}" + "".join(f"{v:>9}" for v in table[i]))
        results.append({"model": name, "parameters": int(model.count_params()),
                        "float_kb": len(float_bytes) / 1024, "int8_kb": len(int8_bytes) / 1024,
                        "test_accuracy_float": acc_float, "test_accuracy_int8": acc_int8,
                        "int8_input": quant})

    info = prep.to_json(label_names)
    info["test_room"] = test_room
    info["models"] = results
    Path(args.out, "preprocess.json").write_text(json.dumps(info, indent=2))

    print("\nSummary" + (f" (tested on unseen room '{test_room}')" if test_room else ""))
    print(f"  {'model':8}{'parameters':>12}{'float KB':>10}{'int8 KB':>9}{'acc float':>11}{'acc int8':>10}")
    for r in results:
        print(f"  {r['model']:8}{r['parameters']:>12,}{r['float_kb']:>10.1f}{r['int8_kb']:>9.1f}"
              f"{r['test_accuracy_float']:>11.3f}{r['test_accuracy_int8']:>10.3f}")
    print(f"\nSaved models and preprocess.json in '{args.out}'.")


if __name__ == "__main__":
    main()
