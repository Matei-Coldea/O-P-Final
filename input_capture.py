import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev


class FreehandDrawer:
    def __init__(self, xlim=(-5, 10), ylim=(-5, 10)):
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.patch.set_facecolor("#fefae0")
        self.ax.set_facecolor("#faedcd")
        self.ax.set_title("Draw something please!\n(click + drag, then close window)", fontsize=13, color="#6b705c", pad=12)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3, color="#d4a373", linestyle="-", linewidth=0.5)
        self.ax.tick_params(colors="#a98467", labelsize=8)
        self.ax.set_xlabel("x", color="#a98467", fontsize=9)
        self.ax.set_ylabel("y", color="#a98467", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#d4a373")
            spine.set_linewidth(1.5)
        self.is_drawing = False
        self.raw_points = []
        self.line, = self.ax.plot([], [], color="#e07a5f", linewidth=2.5, solid_capstyle="round", solid_joinstyle="round")
        self.line_shadow, = self.ax.plot([], [], color="#bc6c50", linewidth=4, alpha=0.25, solid_capstyle="round")
        self.status_text = self.ax.text(0.5, 0.03, "go ahead, draw something...", transform=self.ax.transAxes, ha="center", fontsize=10, color="#6b705c", style="italic")
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_move = self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.is_drawing = True
        self.raw_points = [[event.xdata, event.ydata]]
        self.status_text.set_text("nice! keep going...")
        self.status_text.set_color("#e07a5f")
        self.fig.canvas.draw_idle()

    def _on_move(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.raw_points.append([event.xdata, event.ydata])
        pts = np.array(self.raw_points)
        self.line.set_data(pts[:, 0], pts[:, 1])
        self.line_shadow.set_data(pts[:, 0], pts[:, 1])
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False
        total = len(self.raw_points)
        self.status_text.set_text(f"got it! ({total} pts) - close window when ready")
        self.status_text.set_color("#81b29a")
        self.fig.canvas.draw_idle()

    def get_target_path(self, num_output_points=100, smoothing=2.0, close_threshold=1.0):
        print("\n  Opening sketch pad...")
        print("  Draw your shape, then close the window when done.")
        plt.show()
        if len(self.raw_points) < 5:
            raise ValueError("get_target_path")
        raw = np.array(self.raw_points)
        print(f"  Processing {len(raw)} points...")
        start = raw[0]
        end = raw[-1]
        dist = np.linalg.norm(start - end)
        closed = dist < close_threshold
        if closed:
            print("  Closed shape!")
        else:
            print("  Open path detected")
        try:
            tck, u = splprep([raw[:, 0], raw[:, 1]], s=smoothing, per=int(closed))
        except Exception:
            print("  Using backup method...")
            return self._fallback_resample(raw, num_output_points)
        u_new = np.linspace(0, 1, num_output_points)
        x_new, y_new = splev(u_new, tck)
        path = np.column_stack((x_new, y_new))
        print(f"  Smoothed to {num_output_points} points")
        print(f"  Range: x=[{x_new.min():.1f}, {x_new.max():.1f}], y=[{y_new.min():.1f}, {y_new.max():.1f}]")
        return path

    def _fallback_resample(self, raw, num_points):
        diffs = np.diff(raw, axis=0)
        arc = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(arc)])
        total = cum[-1]
        if total < 1e-6:
            return np.tile(raw[0], (num_points, 1))
        desired = np.linspace(0, total, num_points)
        x_new = np.interp(desired, cum, raw[:, 0])
        y_new = np.interp(desired, cum, raw[:, 1])
        return np.column_stack((x_new, y_new))
