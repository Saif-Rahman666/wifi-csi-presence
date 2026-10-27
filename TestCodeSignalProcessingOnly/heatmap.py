import argparse
import sys
import threading
import time
from collections import deque

import numpy as np

SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#8a8a85"
GRID = "#e4e3df"
BLUE, NEUTRAL, RED = "#1f5fd0", "#f0efec", "#cf3a2f"


def parse_amplitudes(line):
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


def moving_average(rows, k):
    if k <= 1 or len(rows) < k:
        return rows
    padded = np.vstack([np.repeat(rows[:1], k - 1, axis=0), rows])
    total = np.cumsum(padded, axis=0)
    total[k:] = total[k:] - total[:-k]
    return total[k - 1:] / k


class Viewer:

    def __init__(self, seconds=20.0, scale=15.0, smooth=3):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.seconds, self.scale, self.smooth = seconds, scale, smooth
        self.lock = threading.Lock()
        self.packets = deque()      
        self.n_sub = None

        cmap = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, NEUTRAL, RED])
        self.fig, (self.ax_map, self.ax_line) = plt.subplots(
            2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[3, 1],
            facecolor=SURFACE, layout="constrained")
        self.fig.canvas.manager.set_window_title("CSI heat map")

        self.image = self.ax_map.imshow(
            np.zeros((2, 2)), aspect="auto", origin="lower", cmap=cmap,
            vmin=-scale, vmax=scale, extent=(-seconds, 0, 0, 1), interpolation="nearest")
        self.ax_map.set_title("Waiting for CSI...", loc="left", color=INK, fontsize=12)
        self.ax_map.set_ylabel("Subcarrier", color=INK)
        bar = self.fig.colorbar(self.image, ax=self.ax_map, pad=0.01)
        bar.set_label("Change from typical level (%)", color=INK)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=MUTED, length=0)

        (self.line,) = self.ax_line.plot([], [], color=INK, linewidth=2)
        self.ax_line.set_ylabel("Fluctuation (%)", color=INK)
        self.ax_line.set_xlabel("Seconds (0 = now)", color=INK)
        self.ax_line.set_xlim(-seconds, 0)
        self.ax_line.set_ylim(0, 5)
        self.ax_line.grid(axis="y", color=GRID, linewidth=1)

        for ax in (self.ax_map, self.ax_line):
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=MUTED, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)

    def add_packet(self, when, amp):
        with self.lock:
            if self.n_sub is None:
                self.n_sub = amp.size
            if amp.size != self.n_sub:
                return                        
            self.packets.append((when, amp))
            while self.packets and when - self.packets[0][0] > self.seconds:
                self.packets.popleft()

    def redraw(self, *_):
        with self.lock:
            if len(self.packets) < 20:
                return
            times = np.array([p[0] for p in self.packets])
            data = np.array([p[1] for p in self.packets])
        now = times[-1]
        span = max(now - times[0], 1e-3)
        rate = (len(times) - 1) / span

        mean_amp = data.mean(axis=0)
        mask = mean_amp > 0.1 * np.median(mean_amp[mean_amp > 0])
        data = data[:, mask]
        data = data / data.mean(axis=1, keepdims=True)
        data = moving_average(data, self.smooth)

        change = (data / np.median(data, axis=0) - 1.0) * 100.0

        win = max(5, int(rate))
        if len(change) > win:
            c1 = np.cumsum(np.vstack([np.zeros((1, change.shape[1])), change]), axis=0)
            c2 = np.cumsum(np.vstack([np.zeros((1, change.shape[1])), change ** 2]), axis=0)
            mean = (c1[win:] - c1[:-win]) / win
            var = np.maximum((c2[win:] - c2[:-win]) / win - mean ** 2, 0)
            level = np.sqrt(var).mean(axis=1)
            self.line.set_data(times[win - 1:] - now, level)
            top = max(5.0, float(level.max()) * 1.2)
            if top > self.ax_line.get_ylim()[1] or top < 0.5 * self.ax_line.get_ylim()[1]:
                self.ax_line.set_ylim(0, top)

        self.image.set_data(change.T)
        self.image.set_extent((times[0] - now, 0, 0, change.shape[1]))
        self.ax_map.set_xlim(-self.seconds, 0)
        self.ax_map.set_title(
            f"CSI amplitude change per subcarrier   ·   {rate:.0f} packets/s, "
            f"{change.shape[1]} subcarriers", loc="left", color=INK, fontsize=12)


def read_serial(port, viewer, stop):
    while not stop.is_set():
        try:
            raw = port.readline()
        except Exception:
            break
        if not raw:
            continue
        amp = parse_amplitudes(raw.decode("utf-8", errors="ignore").strip())
        if amp is not None:
            viewer.add_packet(time.time(), amp)


def main():
    parser = argparse.ArgumentParser(description="Live heat map of ESP32 CSI")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM8")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=20, help="how much history to show")
    parser.add_argument("--scale", type=float, default=15,
                        help="colour range in percent (smaller = more sensitive colours)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="packets averaged together to reduce speckle (1 = off)")
    args = parser.parse_args()

    try:
        import serial
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as err:
        sys.exit(f"{err}\nInstall with:  python -m pip install pyserial numpy matplotlib")

    try:
        port = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as err:
        sys.exit(f"Could not open {args.port}: {err}\n"
                 "Close any monitor or other script that is using the port.")

    viewer = Viewer(args.seconds, args.scale, args.smooth)
    stop = threading.Event()
    reader = threading.Thread(target=read_serial, args=(port, viewer, stop), daemon=True)
    reader.start()
    animation = FuncAnimation(viewer.fig, viewer.redraw, interval=150, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
        reader.join(timeout=2)
        port.close()
    del animation


if __name__ == "__main__":
    main()
