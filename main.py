import numpy as np
from kinematics import FourBarLinkage
from targets import TargetGenerator
from optimizer import LevenbergMarquardtSolver
from visualizer import OptimizationVisualizer, create_mechanism_animation
from input_capture import FreehandDrawer


def print_menu():
    print("\n  [1] Circle         - A perfect circle")
    print("  [2] Ellipse        - An oval shape (recommended)")
    print("  [3] Wide Ellipse   - A wider oval")
    print("  [4] Tall Ellipse   - A taller oval")
    print("  [5] Small Circle   - A smaller circle")
    print("  [6] Large Circle   - A larger circle")
    print()
    print("  [7] DRAW YOUR OWN  - Drawing mode")
    print()
    print("  [0] Exit")


def get_user_choice():
    while True:
        text = input("\nEnter your choice (0-7): ").strip()
        if text == "":
            print("Please enter a number.")
            continue
        try:
            pick = int(text)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if 0 <= pick <= 7:
            return pick
        print("Invalid choice. Please enter a number between 0 and 7.")


def _normalize_drawing(raw_pts, n_points):
    def _resample_closed_path(points, n_out):
        closed = np.vstack([points, points[0]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-9:
            return np.tile(points[0], (n_out, 1))
        u = np.linspace(0.0, total, n_out, endpoint=False)
        x = np.interp(u, cum, closed[:, 0])
        y = np.interp(u, cum, closed[:, 1])
        return np.column_stack([x, y])

    a = raw_pts[:, 0].min()
    b = raw_pts[:, 0].max()
    c = raw_pts[:, 1].min()
    d = raw_pts[:, 1].max()
    cx = (a + b) / 2
    cy = (c + d) / 2
    moved = raw_pts - np.array([cx, cy])
    w = b - a
    h = d - c
    m = max(w, h)
    scale_to = 3.0
    if m > 0.1:
        moved = moved * (scale_to / m)
    resampled = _resample_closed_path(moved, n_points)


    start_idx = int(np.argmax(resampled[:, 0]))
    phased = np.roll(resampled, -start_idx, axis=0)

    final = phased + np.array([2.5, 1.5])
    return final


def get_target_shape(choice, n_points=100):
    maker = TargetGenerator()
    label = "Ellipse"
    info = "Width: 2.0, Height: 1.2, Center: (2.5, 1.5)"

    if choice == 1:
        pts = maker.get_circle(1.0, (2.5, 1.5), n_points)
        label = "Circle"
        info = "Radius: 1.0, Center: (2.5, 1.5)"
    elif choice == 2:
        pts = maker.get_ellipse(2.0, 1.2, (2.5, 1.5), n_points)
        label = "Ellipse"
        info = "Width: 2.0, Height: 1.2, Center: (2.5, 1.5)"
    elif choice == 3:
        pts = maker.get_ellipse(3.0, 1.0, (2.5, 1.5), n_points)
        label = "Wide Ellipse"
        info = "Width: 3.0, Height: 1.0, Center: (2.5, 1.5)"
    elif choice == 4:
        pts = maker.get_ellipse(1.0, 2.0, (2.5, 1.5), n_points)
        label = "Tall Ellipse"
        info = "Width: 1.0, Height: 2.0, Center: (2.5, 1.5)"
    elif choice == 5:
        pts = maker.get_circle(0.6, (2.5, 1.5), n_points)
        label = "Small Circle"
        info = "Radius: 0.6, Center: (2.5, 1.5)"
    elif choice == 6:
        pts = maker.get_circle(1.5, (2.5, 1.5), n_points)
        label = "Large Circle"
        info = "Radius: 1.5, Center: (2.5, 1.5)"
    elif choice == 7:
        pad = FreehandDrawer(xlim=(-2, 8), ylim=(-2, 8))
        raw = pad.get_target_path(num_output_points=n_points * 2, smoothing=2.0, close_threshold=1.0)
        pts = _normalize_drawing(raw, n_points)
        label = "Custom Drawing"
        a = pts[:, 0].min()
        b = pts[:, 0].max()
        c = pts[:, 1].min()
        d = pts[:, 1].max()
        cx = (a + b) / 2
        cy = (c + d) / 2
        info = f"Freehand path (normalized + angular), Center: ({cx:.2f}, {cy:.2f})"
    else:
        pts = maker.get_ellipse(2.0, 1.2, (2.5, 1.5), n_points)

    return pts, label, info


def analyze_target(target):
    xmin = target[:, 0].min()
    xmax = target[:, 0].max()
    ymin = target[:, 1].min()
    ymax = target[:, 1].max()

    clean = target[~np.isnan(target[:, 0])]
    cx = clean[:, 0].mean()
    cy = clean[:, 1].mean()

    w = xmax - xmin
    h = ymax - ymin
    aspect = w / max(h, 0.01)

    shifted = target - np.array([cx, cy])
    cov = np.cov(shifted.T)
    vals, vecs = np.linalg.eigh(cov)
    main_vec = vecs[:, np.argmax(vals)]
    angle = np.arctan2(main_vec[1], main_vec[0])

    size_guess = (w + h) / 2

    return {
        "center_x": cx,
        "center_y": cy,
        "width": w,
        "height": h,
        "aspect_ratio": aspect,
        "orientation": angle,
        "char_size": size_guess,
        "x_min": xmin,
        "x_max": xmax,
        "y_min": ymin,
        "y_max": ymax,
    }


def create_initial_linkage(props, classic_mode):
    basic = props["char_size"] / 2
    l1 = max(0.3, min(basic * 0.5, 3.0))
    l2 = max(l1 * 2.5, props["char_size"] * 1.2)
    l3 = l2 * 0.8
    l4 = l2 * 0.9

    base_angle = props["orientation"] - np.pi / 2

    push_out = props["char_size"] * 0.6 + l1
    side_angle = base_angle + np.pi / 2
    gx = props["center_x"] - push_out * np.cos(side_angle)
    gy = props["center_y"] - push_out * np.sin(side_angle)

    cx = 0.5
    cy = 0.0 if classic_mode else 0.3

    return FourBarLinkage(L1=l1, L2=l2, L3=l3, L4=l4, Gx=gx, Gy=gy, g_angle=base_angle, Cx=cx, Cy=cy)


def run_optimization(target, classic_mode=True, iterations=150, regularization_weight=0.001):
    bits = analyze_target(target)
    mech = create_initial_linkage(bits, classic_mode)
    solver = LevenbergMarquardtSolver(regularization_weight=regularization_weight, classic_mode=classic_mode)
    res = solver.solve(mech, target, iterations=iterations)
    last_cost = res["cost_history"][-1] if res["cost_history"] else float("inf")
    return {"linkage": mech, "result": res, "cost": last_cost, "target": target}


def get_optimization_mode():
    print("\n  [1] CLASSIC     - Pen on coupler bar (Cy=0)")
    print("  [2] NON-CLASSIC - Pen can be offset (better fit)")
    while True:
        txt = input("\nEnter mode (1 or 2) [default=2]: ").strip()
        if txt == "":
            return False
        try:
            num = int(txt)
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")
            continue
        if num == 1:
            return True
        if num == 2:
            return False
        print("Invalid choice. Please enter 1 or 2.")


def run_for_target(choice):
    total_pts = 100
    try:
        tgt, name, desc = get_target_shape(choice, total_pts)
    except ValueError as err:
        print(f"\nDrawing cancelled: {err}")
        return None

    classic = get_optimization_mode()
    mode_txt = "classic" if classic else "nonclassic"
    reg = 0.0 if choice == 7 else 0.001

    print(f"\nOptimizing {name}...")
    outcome = run_optimization(target=tgt, classic_mode=classic, iterations=150, regularization_weight=reg)
    mech = outcome["linkage"]
    data = outcome["result"]

    file_tag = name.lower().replace(" ", "_") + f"_{mode_txt}"
    show = OptimizationVisualizer(target_path=tgt, parameter_history=data["parameter_history"], cost_history=data["cost_history"])

    png_path = f"summary_{file_tag}.png"
    gif_path = f"evolution_{file_tag}.gif"
    mech_path = f"mechanism_{file_tag}.gif"

    print("Generating visualizations...")
    show.plot_summary(png_path)
    show.animate(gif_path, fps=3)
    create_mechanism_animation(linkage=mech, n_frames=72, save_path=mech_path, fps=12, target_path=tgt)

    print(f"\nDone! Files: {png_path}, {gif_path}, {mech_path}")
    return data


def main():
    print("\nThe Mechanical Inventor - Four-Bar Linkage Synthesizer")
    while True:
        print_menu()
        pick = get_user_choice()
        if pick == 0:
            print("\nGoodbye!")
            break
        res = run_for_target(pick)
        if res is None:
            continue
        again = input("\nTry another shape? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
