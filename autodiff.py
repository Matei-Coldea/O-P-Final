import numpy as np
from kinematics import FourBarLinkage


class FiniteDiff:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def compute_jacobian(self, mechanism, n_points):
        base_vec = mechanism.get_design_vector().copy()
        base_trace = mechanism.get_trace(n_points)
        actual = len(base_trace)
        if actual == 0:
            return np.array([]).reshape(0, 9)
        base_flat = base_trace.flatten()
        rows = len(base_flat)
        cols = 9
        jac = np.zeros((rows, cols))
        for j in range(cols):
            tweak = base_vec.copy()
            tweak[j] = tweak[j] + self.epsilon
            mechanism.set_design_vector(tweak)
            new_trace = mechanism.get_trace(n_points)
            if len(new_trace) != actual:
                jac[:, j] = 0.0
            else:
                jac[:, j] = (new_trace.flatten() - base_flat) / self.epsilon
            mechanism.set_design_vector(base_vec)
        return jac

    def compute_residuals(self, mechanism, target):
        n_points = len(target)
        trace = mechanism.get_trace(n_points)
        if len(trace) != n_points:
            use = min(len(trace), n_points)
            trace = trace[:use]
            target = target[:use]
        return (trace - target).flatten()


def compute_jacobian(mechanism, n_points, epsilon=1e-5):
    return FiniteDiff(epsilon=epsilon).compute_jacobian(mechanism, n_points)

