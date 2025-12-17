import numpy as np


class TargetGenerator:
    @staticmethod
    def get_circle(radius=1.0, center=(0.0, 0.0), n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        return np.column_stack([x, y])

    @staticmethod
    def get_lemniscate(width=1.0, center=(0.0, 0.0), n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center[0] + width * np.cos(t)
        y = center[1] + (width / 2) * np.sin(2 * t)
        return np.column_stack([x, y])

    @staticmethod
    def get_letter_d(height=2.0, center=(0.0, 0.0), n_points=100):
        n_straight = n_points // 3
        n_curve = n_points - n_straight
        half = height / 2
        radius = half
        straight_y = np.linspace(-half, half, n_straight)
        straight_x = np.zeros(n_straight)
        curve_t = np.linspace(np.pi / 2, -np.pi / 2, n_curve)
        curve_x = radius * np.cos(curve_t)
        curve_y = radius * np.sin(curve_t)
        x = np.concatenate([straight_x, curve_x]) + center[0]
        y = np.concatenate([straight_y, curve_y]) + center[1]
        return np.column_stack([x, y])

    @staticmethod
    def get_ellipse(width=2.0, height=1.0, center=(0.0, 0.0), n_points=100):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center[0] + (width / 2) * np.cos(t)
        y = center[1] + (height / 2) * np.sin(t)
        return np.column_stack([x, y])

