import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kinematics import FourBarLinkage


class OptimizationVisualizer:
    def __init__(self, target_path, parameter_history, cost_history=None):
        self.target_path = target_path
        self.parameter_history = parameter_history
        self.cost_history = cost_history
        self.n_frames = len(parameter_history)
        self.n_points = len(target_path)
        if self.parameter_history:
            p0 = self.parameter_history[0]
            self.linkage = FourBarLinkage(L1=p0[0], L2=p0[1], L3=p0[2], L4=p0[3], Gx=p0[4], Gy=p0[5], g_angle=p0[6], Cx=p0[7], Cy=p0[8])

    def _get_plot_limits(self):
        ok = self.target_path[~np.isnan(self.target_path[:, 0])]
        if len(ok) == 0:
            return -5.0, 5.0, -5.0, 5.0
        x0, x1 = float(ok[:, 0].min()), float(ok[:, 0].max())
        y0, y1 = float(ok[:, 1].min()), float(ok[:, 1].max())
        dx = x1 - x0
        dy = y1 - y0
        pad = max(0.4 * max(dx, dy), 1.5)
        x0 -= pad
        x1 += pad
        y0 -= pad
        y1 += pad
        span = max(x1 - x0, y1 - y0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return cx - 0.5 * span, cx + 0.5 * span, cy - 0.5 * span, cy + 0.5 * span

    def animate(self, save_path="evolution.gif", fps=5, show_mechanism=True):
        x0, x1, y0, y1 = self._get_plot_limits()
        fig, (ax_mech, ax_conv) = plt.subplots(1, 2, figsize=(14, 6))

        ax_mech.plot(self.target_path[:, 0], self.target_path[:, 1], "b--", linewidth=2.5, label="Target", alpha=0.7)
        trace_line, = ax_mech.plot([], [], "r-", linewidth=2, label="Trace")
        ground_line, = ax_mech.plot([], [], "k-", linewidth=4, label="Ground")
        crank_line, = ax_mech.plot([], [], "b-", linewidth=3, label="Crank")
        rocker_line, = ax_mech.plot([], [], "m-", linewidth=3, label="Rocker")
        coupler_line, = ax_mech.plot([], [], color="#4CAF50", linewidth=4, label="Coupler")
        pen_offset_line, = ax_mech.plot([], [], color="#00C853", linewidth=2, linestyle=":", label="Pen Offset", zorder=4)
        joints_scatter = ax_mech.scatter([], [], s=100, c="black", zorder=5)
        coupler_point_scatter = ax_mech.scatter([], [], s=150, c="red", marker="*", zorder=6)
        base_point_scatter = ax_mech.scatter([], [], s=60, c="#00C853", marker="o", zorder=5)

        ax_mech.set_xlim(x0, x1)
        ax_mech.set_ylim(y0, y1)
        ax_mech.set_aspect("equal")
        ax_mech.grid(True, alpha=0.3)
        ax_mech.set_xlabel("X", fontsize=12)
        ax_mech.set_ylabel("Y", fontsize=12)
        ax_mech.legend(loc="upper right", fontsize=9)
        mech_title = ax_mech.set_title("Iteration 0", fontsize=14)

        conv_dot = None
        if self.cost_history:
            steps = range(len(self.cost_history))
            ax_conv.semilogy(steps, self.cost_history, "b-", linewidth=2, alpha=0.5)
            conv_dot, = ax_conv.semilogy([], [], "ro", markersize=12)
            ax_conv.set_xlabel("Iteration", fontsize=12)
            ax_conv.set_ylabel("Cost (log scale)", fontsize=12)
            ax_conv.set_title("Optimization Convergence", fontsize=14)
            ax_conv.grid(True, alpha=0.3, which="both")
            ax_conv.set_xlim(-1, len(self.cost_history))
            ax_conv.set_ylim(min(self.cost_history) * 0.5, max(self.cost_history) * 2)
        else:
            ax_conv.text(0.5, 0.5, "No cost data", transform=ax_conv.transAxes, ha="center", va="center", fontsize=14)

        def update(idx):
            params = self.parameter_history[idx]
            self.linkage.set_design_vector(params)
            trace, poses = self.linkage.get_trace_with_joints(self.n_points)
            trace_line.set_data(trace[:, 0], trace[:, 1])

            if show_mechanism:
                usable = [p for p in poses if p is not None]
                pick = usable[len(usable) // 8] if usable else None
                if pick is not None:
                    P, A, B = pick
                    O, D = self.linkage.O, self.linkage.D
                    ground_line.set_data([O[0], D[0]], [O[1], D[1]])
                    crank_line.set_data([O[0], A[0]], [O[1], A[1]])
                    rocker_line.set_data([D[0], B[0]], [D[1], B[1]])
                    coupler_line.set_data([A[0], B[0]], [A[1], B[1]])
                    AB = B - A
                    AB_len = np.linalg.norm(AB)
                    if AB_len > 1e-10:
                        unit = AB / AB_len
                        base = A + self.linkage.Cx * self.linkage.L2 * unit
                        if abs(self.linkage.Cy) > 0.01:
                            pen_offset_line.set_data([base[0], P[0]], [base[1], P[1]])
                            base_point_scatter.set_offsets([[base[0], base[1]]])
                        else:
                            pen_offset_line.set_data([], [])
                            base_point_scatter.set_offsets(np.empty((0, 2)))
                    else:
                        pen_offset_line.set_data([], [])
                        base_point_scatter.set_offsets(np.empty((0, 2)))
                    joints_scatter.set_offsets([[O[0], O[1]], [A[0], A[1]], [B[0], B[1]], [D[0], D[1]]])
                    coupler_point_scatter.set_offsets([[P[0], P[1]]])
                else:
                    ground_line.set_data([], [])
                    crank_line.set_data([], [])
                    rocker_line.set_data([], [])
                    coupler_line.set_data([], [])
                    pen_offset_line.set_data([], [])
                    joints_scatter.set_offsets(np.empty((0, 2)))
                    coupler_point_scatter.set_offsets(np.empty((0, 2)))
                    base_point_scatter.set_offsets(np.empty((0, 2)))

            if conv_dot is not None and idx < len(self.cost_history):
                conv_dot.set_data([idx], [self.cost_history[idx]])
            if self.cost_history and idx < len(self.cost_history):
                mech_title.set_text(f"Iteration {idx} | Cost: {self.cost_history[idx]:.4f}")
            return trace_line

        anim = FuncAnimation(fig, update, frames=self.n_frames, interval=1000 // fps, blit=False, repeat=True)
        plt.tight_layout()
        try:
            anim.save(save_path, writer="pillow", fps=fps)
        except Exception:
            print("visualizer.py: OptimizationVisualizer.animate")
        plt.close(fig)
        return anim

    def plot_summary(self, save_path="optimization_summary.png"):
        if not self.parameter_history:
            return
        first = self.parameter_history[0]
        last = self.parameter_history[-1]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        def draw(ax, params, title, color="r"):
            self.linkage.set_design_vector(params)
            ax.plot(self.target_path[:, 0], self.target_path[:, 1], "b--", lw=2, label="Target")
            trace, poses = self.linkage.get_trace_with_joints(self.n_points)
            ax.plot(trace[:, 0], trace[:, 1], f"{color}-", lw=2, label="Trace")
            good = [p for p in poses if p is not None]
            if good:
                pick = good[min(len(good) - 1, 12)]
                P, A, B = pick
                O, D = self.linkage.O, self.linkage.D
                ax.plot([O[0], D[0]], [O[1], D[1]], "k-", lw=4)
                ax.plot([O[0], A[0]], [O[1], A[1]], "b-", lw=3)
                ax.plot([A[0], B[0]], [A[1], B[1]], color="#4CAF50", lw=4)
                ax.plot([D[0], B[0]], [D[1], B[1]], "m-", lw=3)
                AB = B - A
                AB_len = np.linalg.norm(AB)
                if AB_len > 1e-10 and abs(self.linkage.Cy) > 0.01:
                    unit = AB / AB_len
                    base = A + self.linkage.Cx * self.linkage.L2 * unit
                    ax.plot([base[0], P[0]], [base[1], P[1]], color="#00C853", lw=2, linestyle=":", label="Pen Offset")
                    ax.scatter([base[0]], [base[1]], s=60, c="#00C853", zorder=5)
                ax.scatter([O[0], A[0], B[0], D[0]], [O[1], A[1], B[1], D[1]], s=80, c="k", zorder=5)
                ax.scatter([P[0]], [P[1]], s=120, c=color, marker="*", zorder=6)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=14)
            ax.legend()

        draw(axes[0, 0], first, "Initial Configuration", "r")
        draw(axes[0, 1], last, "Final Configuration", "g")

        ax3 = axes[1, 0]
        if self.cost_history:
            ax3.semilogy(self.cost_history, "b-", lw=2)
            ax3.set_title("Convergence History", fontsize=14)
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Cost (log)")
            ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        names = ["L1", "L2", "L3", "L4", "Gx", "Gy", "θ", "Cx", "Cy"]
        idxs = np.arange(len(names))
        w = 0.35
        ax4.bar(idxs - w / 2, first, w, label="Initial", color="r", alpha=0.6)
        ax4.bar(idxs + w / 2, last, w, label="Final", color="g", alpha=0.6)
        ax4.set_xticks(idxs)
        ax4.set_xticklabels(names)
        ax4.legend()
        ax4.set_title("Parameter Changes", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


def _rotate_points_viz(points, angle, center):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (points - center) @ rot.T + center


class OptimizationVisualizerWithFlip(OptimizationVisualizer):
    def plot_summary_with_flip(self, save_path, center, original_target):
        if not self.parameter_history:
            return
        first = self.parameter_history[0]
        last = self.parameter_history[-1]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        def draw(ax, params, title, tgt, color="r"):
            self.linkage.set_design_vector(params)
            ax.plot(tgt[:, 0], tgt[:, 1], "b--", lw=2, label="Target")
            trace, poses = self.linkage.get_trace_with_joints(self.n_points)
            ax.plot(trace[:, 0], trace[:, 1], f"{color}-", lw=2, label="Trace")
            usable = [p for p in poses if p]
            if usable:
                P, A, B = usable[min(len(usable) - 1, 12)]
                O, D = self.linkage.O, self.linkage.D
                ax.plot([O[0], D[0]], [O[1], D[1]], "k-", lw=4)
                ax.plot([O[0], A[0]], [O[1], A[1]], "b-", lw=3)
                ax.plot([A[0], B[0]], [A[1], B[1]], color="#4CAF50", lw=4)
                ax.plot([D[0], B[0]], [D[1], B[1]], "m-", lw=3)
                ax.scatter([O[0], A[0], B[0], D[0]], [O[1], A[1], B[1], D[1]], s=80, c="k", zorder=5)
                ax.scatter([P[0]], [P[1]], s=120, c=color, marker="*", zorder=6)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=14)
            ax.legend()

        draw(axes[0, 0], first, "Initial (Rotated Space)", self.target_path, "r")
        draw(axes[0, 1], last, "Final (Rotated Space)", self.target_path, "g")

        flipped = np.array(last)
        pivot_back = _rotate_points_viz(np.array([[flipped[4], flipped[5]]]), -np.pi / 2, center)[0]
        flipped[4], flipped[5], flipped[6] = pivot_back[0], pivot_back[1], flipped[6] - np.pi / 2
        draw(axes[1, 0], flipped, "↻ Flipped Back to Original", original_target, "g")

        if self.cost_history:
            axes[1, 1].semilogy(self.cost_history, "b-", lw=2)
            axes[1, 1].set_title("Convergence")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Cost (log)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def animate_with_flip(self, save_path, fps, center, original_target, rotation_angle=np.pi / 2):
        intro = 3
        outro = 5
        total = intro + self.n_frames + outro

        def limits(tgt):
            good = tgt[~np.isnan(tgt[:, 0])]
            if len(good) == 0:
                return -5, 5, -5, 5
            pad = max(0.4 * max(np.ptp(good[:, 0]), np.ptp(good[:, 1])), 1.5)
            xc, yc = good[:, 0].mean(), good[:, 1].mean()
            span = max(np.ptp(good[:, 0]), np.ptp(good[:, 1])) + 2 * pad
            return xc - span / 2, xc + span / 2, yc - span / 2, yc + span / 2

        lim_orig = limits(original_target)
        lim_rot = limits(self.target_path)
        fig, (ax_mech, ax_conv) = plt.subplots(1, 2, figsize=(14, 6))

        tgt_line, = ax_mech.plot([], [], "b--", lw=2.5, alpha=0.7)
        trace_line, = ax_mech.plot([], [], "r-", lw=2)
        ground_line, = ax_mech.plot([], [], "k-", lw=4)
        crank_line, = ax_mech.plot([], [], "b-", lw=3)
        coupler_line, = ax_mech.plot([], [], color="#4CAF50", lw=4)
        rocker_line, = ax_mech.plot([], [], "m-", lw=3)
        joints = ax_mech.scatter([], [], s=100, c="black", zorder=5)
        pen = ax_mech.scatter([], [], s=150, c="red", marker="*", zorder=6)

        ax_mech.set_aspect("equal")
        ax_mech.grid(True, alpha=0.3)
        title = ax_mech.set_title("", fontsize=14)

        conv_dot = None
        if self.cost_history:
            ax_conv.semilogy(self.cost_history, "b-", lw=2, alpha=0.5)
            conv_dot, = ax_conv.semilogy([], [], "ro", ms=12)
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Cost (log)")
            ax_conv.set_title("Convergence")
            ax_conv.grid(True, alpha=0.3)

        def draw_link(params):
            self.linkage.set_design_vector(params)
            trace, poses = self.linkage.get_trace_with_joints(self.n_points)
            trace_line.set_data(trace[:, 0], trace[:, 1])
            good = [p for p in poses if p]
            if good:
                P, A, B = good[len(good) // 8]
                O, D = self.linkage.O, self.linkage.D
                ground_line.set_data([O[0], D[0]], [O[1], D[1]])
                crank_line.set_data([O[0], A[0]], [O[1], A[1]])
                coupler_line.set_data([A[0], B[0]], [A[1], B[1]])
                rocker_line.set_data([D[0], B[0]], [D[1], B[1]])
                joints.set_offsets([[O[0], O[1]], [A[0], A[1]], [B[0], B[1]], [D[0], D[1]]])
                pen.set_offsets([[P[0], P[1]]])

        def update(frame):
            if frame < intro:
                tgt_line.set_data(original_target[:, 0], original_target[:, 1])
                ax_mech.set_xlim(lim_orig[0], lim_orig[1])
                ax_mech.set_ylim(lim_orig[2], lim_orig[3])
                start = np.array(self.parameter_history[0])
                pivot = _rotate_points_viz(np.array([[start[4], start[5]]]), -rotation_angle, center)[0]
                start[4], start[5], start[6] = pivot[0], pivot[1], start[6] - rotation_angle
                draw_link(start)
                if frame == intro - 1:
                    title.set_text(f"↻ Rotating {np.degrees(rotation_angle):.0f}° for optimization...")
                else:
                    title.set_text("Original Target")
            elif frame < intro + self.n_frames:
                idx = frame - intro
                tgt_line.set_data(self.target_path[:, 0], self.target_path[:, 1])
                ax_mech.set_xlim(lim_rot[0], lim_rot[1])
                ax_mech.set_ylim(lim_rot[2], lim_rot[3])
                draw_link(self.parameter_history[idx])
                if conv_dot and idx < len(self.cost_history):
                    conv_dot.set_data([idx], [self.cost_history[idx]])
                title.set_text(f"Iteration {idx} (Rotated Space)")
            else:
                tgt_line.set_data(original_target[:, 0], original_target[:, 1])
                ax_mech.set_xlim(lim_orig[0], lim_orig[1])
                ax_mech.set_ylim(lim_orig[2], lim_orig[3])
                final = np.array(self.parameter_history[-1])
                pivot = _rotate_points_viz(np.array([[final[4], final[5]]]), -rotation_angle, center)[0]
                final[4], final[5], final[6] = pivot[0], pivot[1], final[6] - rotation_angle
                draw_link(final)
                title.set_text("↻ Flipped Back to Original!")
            return trace_line,

        anim = FuncAnimation(fig, update, frames=total, interval=1000 // fps, blit=False)
        plt.tight_layout()
        try:
            anim.save(save_path, writer="pillow", fps=fps)
        except Exception:
            print("visualizer.py: OptimizationVisualizerWithFlip.animate_with_flip")
        plt.close(fig)


def create_mechanism_animation(linkage, n_frames=72, save_path="mechanism_motion.gif", fps=12, target_path=None):
    n_trace_points = max(n_frames * 3, 200)
    trace_angles = np.linspace(0, 2 * np.pi, n_trace_points, endpoint=False)
    all_poses = []
    prev_B = None
    for ang in trace_angles:
        P, A, B = linkage.solve_pose(ang, prev_joint_B=prev_B)
        if P is not None and A is not None and B is not None:
            all_poses.append((P, A, B))
            prev_B = B
        else:
            all_poses.append(None)
            prev_B = None

    trace_pts = []
    for pose in all_poses:
        if pose is not None:
            trace_pts.append(pose[0])
        else:
            trace_pts.append([np.nan, np.nan])
    trace = np.array(trace_pts)

    step = n_trace_points // n_frames
    poses = [all_poses[i * step] for i in range(n_frames)]

    cloud = []
    for p in poses:
        if p:
            cloud.extend([linkage.O, linkage.D, p[0], p[1], p[2]])
    if target_path is not None:
        good_t = target_path[~np.isnan(target_path[:, 0])]
        cloud.extend(good_t)

    cloud = np.array(cloud)
    if len(cloud) > 0:
        xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
        ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
        pad = 0.5
    else:
        xmin, xmax, ymin, ymax = -2, 5, -2, 5
        pad = 0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#f5f5f5")
    if target_path is not None:
        ax.plot(target_path[:, 0], target_path[:, 1], "b--", lw=2, alpha=0.5, label="Target")
    ax.plot(trace[:, 0], trace[:, 1], color="lightblue", lw=3, alpha=0.5, label="Trace")

    ground_line, = ax.plot([], [], "k-", lw=6, label="Ground")
    crank_line, = ax.plot([], [], color="#2196F3", lw=5, label="Crank")
    coupler_line, = ax.plot([], [], color="#4CAF50", lw=5, label="Coupler", zorder=4)
    pen_offset_line, = ax.plot([], [], color="#00C853", lw=3, linestyle=":", label="Pen Offset", zorder=4)
    rocker_line, = ax.plot([], [], color="#9C27B0", lw=5, label="Rocker")
    joints_scatter = ax.scatter([], [], s=150, c="k", zorder=5)
    pen_scatter = ax.scatter([], [], s=200, c="r", marker="*", zorder=6)
    base_point_scatter = ax.scatter([], [], s=80, c="#00C853", marker="o", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Four-Bar Linkage", fontsize=16)

    def update(frame):
        pose = poses[frame]
        if pose is not None:
            P, A, B = pose
            O, D = linkage.O, linkage.D
            ground_line.set_data([O[0], D[0]], [O[1], D[1]])
            crank_line.set_data([O[0], A[0]], [O[1], A[1]])
            coupler_line.set_data([A[0], B[0]], [A[1], B[1]])
            rocker_line.set_data([D[0], B[0]], [D[1], B[1]])
            AB = B - A
            AB_len = np.linalg.norm(AB)
            if AB_len > 1e-10 and abs(linkage.Cy) > 0.01:
                unit = AB / AB_len
                base = A + linkage.Cx * linkage.L2 * unit
                pen_offset_line.set_data([base[0], P[0]], [base[1], P[1]])
                base_point_scatter.set_offsets([[base[0], base[1]]])
            else:
                pen_offset_line.set_data([], [])
                base_point_scatter.set_offsets(np.empty((0, 2)))
            joints_scatter.set_offsets([O, A, B, D])
            pen_scatter.set_offsets([P])
        else:
            ground_line.set_data([], [])
            crank_line.set_data([], [])
            coupler_line.set_data([], [])
            pen_offset_line.set_data([], [])
            rocker_line.set_data([], [])
            joints_scatter.set_offsets(np.empty((0, 2)))
            pen_scatter.set_offsets(np.empty((0, 2)))
            base_point_scatter.set_offsets(np.empty((0, 2)))
        return ground_line, coupler_line, pen_offset_line

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    try:
        anim.save(save_path, writer="pillow", fps=fps)
    except Exception:
        print("visualizer.py: create_mechanism_animation")
    plt.close(fig)
