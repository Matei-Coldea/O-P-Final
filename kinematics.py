import numpy as np


class FourBarLinkage:
    def __init__(self, L1, L2, L3, L4, Gx=0.0, Gy=0.0, g_angle=0.0, Cx=0.5, Cy=0.5):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.Gx = Gx
        self.Gy = Gy
        self.g_angle = g_angle
        self.Cx = Cx
        self.Cy = Cy
        self.O = np.array([self.Gx, self.Gy])
        self.D = np.array([self.Gx + self.L4 * np.cos(self.g_angle), self.Gy + self.L4 * np.sin(self.g_angle)])

    def _get_circle_intersections(self, p1, r1, p2, r2):
        diff = p2 - p1
        dist = float(np.linalg.norm(diff))
        if dist > r1 + r2:
            return []
        if dist < abs(r1 - r2):
            return []
        if dist < 1e-12:
            return []

        a = (dist * dist + r1 * r1 - r2 * r2) / (2.0 * dist)
        h2 = r1 * r1 - a * a
        if h2 < 0.0:
            if h2 > -1e-12:
                h2 = 0.0
            else:
                return []

        h = float(np.sqrt(h2))
        ex = diff / dist
        ey = np.array([-ex[1], ex[0]])
        base = p1 + a * ex
        if h == 0.0:
            return [base]
        return [base + h * ey, base - h * ey]

    def solve_pose(self, theta, prev_joint_B=None):
        crank = theta + self.g_angle
        A = np.array([self.O[0] + self.L1 * np.cos(crank), self.O[1] + self.L1 * np.sin(crank)])
        hits = self._get_circle_intersections(A, self.L2, self.D, self.L3)
        if len(hits) == 0:
            return (None, None, None)
        if len(hits) == 1:
            B = hits[0]
        else:
            j1, j2 = hits[0], hits[1]
            if prev_joint_B is None:
                B = j1 if j1[1] >= j2[1] else j2
            else:
                d1 = float(np.linalg.norm(j1 - prev_joint_B))
                d2 = float(np.linalg.norm(j2 - prev_joint_B))
                B = j1 if d1 <= d2 else j2
        AB = B - A
        ab_len = np.linalg.norm(AB)
        if ab_len < 1e-10:
            return (None, None, None)
        unit = AB / ab_len
        perp = np.array([-unit[1], unit[0]])
        P = A + self.Cx * self.L2 * unit + self.Cy * self.L2 * perp
        return (P, A, B)

    def get_trace(self, n_points=100):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        pts = []
        prev = None
        for ang in angles:
            P, _, B = self.solve_pose(ang, prev_joint_B=prev)
            if P is not None and B is not None:
                pts.append(P)
                prev = B
            else:
                pts.append(np.array([np.nan, np.nan]))
                prev = None
        return np.array(pts)

    def get_trace_with_joints(self, n_points=100):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        trace = np.full((n_points, 2), np.nan)
        poses = []
        prev = None
        for i, ang in enumerate(angles):
            P, A, B = self.solve_pose(ang, prev_joint_B=prev)
            if P is not None and A is not None and B is not None:
                poses.append((P, A, B))
                trace[i] = P
                prev = B
            else:
                poses.append(None)
                trace[i] = np.array([np.nan, np.nan])
                prev = None
        return trace, poses

    def get_all_joints(self, theta, prev_joint_B=None):
        P, A, B = self.solve_pose(theta, prev_joint_B=prev_joint_B)
        if P is None or A is None or B is None:
            return None
        return {"O": self.O.copy(), "A": A, "B": B, "D": self.D.copy(), "P": P}

    def get_design_vector(self):
        return np.array([self.L1, self.L2, self.L3, self.L4, self.Gx, self.Gy, self.g_angle, self.Cx, self.Cy])

    def set_design_vector(self, x):
        if len(x) != 9:
            raise ValueError("set_design_vector")
        self.L1 = x[0]
        self.L2 = x[1]
        self.L3 = x[2]
        self.L4 = x[3]
        self.Gx = x[4]
        self.Gy = x[5]
        self.g_angle = x[6]
        self.Cx = x[7]
        self.Cy = x[8]
        self._update_derived_values()

    def _update_derived_values(self):
        self.O = np.array([self.Gx, self.Gy])
        self.D = np.array([self.Gx + self.L4 * np.cos(self.g_angle), self.Gy + self.L4 * np.sin(self.g_angle)])

