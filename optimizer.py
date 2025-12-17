import numpy as np
from kinematics import FourBarLinkage
from autodiff import FiniteDiff


class LevenbergMarquardtSolver:
    def __init__(self, regularization_weight=1e-6, min_link_length=0.3, max_link_length=20.0, classic_mode=True):
        self.lambda_reg = regularization_weight
        self.min_link_length = min_link_length
        self.max_link_length = max_link_length
        self.classic_mode = classic_mode
        self.finite_diff = FiniteDiff(epsilon=1e-5)

    def _clamp_parameters(self, vec):
        out = vec.copy()
        for i in range(4):
            out[i] = np.clip(out[i], self.min_link_length, self.max_link_length)
        out[4] = np.clip(out[4], -20.0, 20.0)
        out[5] = np.clip(out[5], -20.0, 20.0)
        out[6] = np.clip(out[6], -np.pi, np.pi)
        if self.classic_mode:
            out[7] = np.clip(out[7], 0.0, 1.0)
            out[8] = 0.0
        else:
            out[7] = np.clip(out[7], 0.0, 1.5)
            out[8] = np.clip(out[8], -2.0, 2.0)
        return out

    def compute_cost(self, residuals, parameters):
        fit = 0.5 * np.dot(residuals, residuals)
        if self.lambda_reg <= 0.0:
            return fit
        reg = self.lambda_reg * np.dot(parameters, parameters)
        return fit + reg

    def _print_progress(self, step, total, cost, bar_length=40):
        prog = (step + 1) / total
        filled = int(bar_length * prog)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\r  [{bar}] {step + 1}/{total} | Cost: {cost:.4f}", end="", flush=True)

    def solve(self, mechanism, target_path, iterations=100):
        n_pts = len(target_path)
        target_flat = target_path.flatten()

        mu = 1e-3
        mu_up = 2.0
        mu_down = 3.0
        mu_min, mu_max = 1e-10, 1e10

        start_vec = mechanism.get_design_vector()
        x = self._clamp_parameters(start_vec)
        mechanism.set_design_vector(x)

        n_params = len(x)
        history = [x.copy()]
        costs = []

        trace_now = mechanism.get_trace(n_pts)
        if len(trace_now) != n_pts:
            n_pts = min(len(trace_now), n_pts)
            target_path = target_path[:n_pts]
            target_flat = target_path.flatten()

        res = trace_now.flatten() - target_flat
        current = self.compute_cost(res, x)
        costs.append(current)

        for i in range(iterations):
            self._print_progress(i, iterations, current)
            J = self.finite_diff.compute_jacobian(mechanism, n_pts)
            JTJ = np.dot(J.T, J)
            eye = np.eye(n_params)
            if self.lambda_reg <= 0.0:
                H = JTJ + mu * eye
                grad = np.dot(J.T, res)
            else:
                H = JTJ + (mu + self.lambda_reg) * eye
                grad = np.dot(J.T, res) + self.lambda_reg * x
            try:
                step = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                mu = min(mu * mu_up, mu_max)
                continue

            guess = x + step
            guess = self._clamp_parameters(guess)

            mechanism.set_design_vector(guess)
            trace_new = mechanism.get_trace(n_pts)
            if len(trace_new) != n_pts:
                mechanism.set_design_vector(x)
                mu = min(mu * mu_up, mu_max)
                continue

            res_new = trace_new.flatten() - target_flat
            cost_new = self.compute_cost(res_new, guess)

            if cost_new < current:
                x = guess.copy()
                res = res_new.copy()
                current = cost_new
                mu = max(mu / mu_down, mu_min)
                history.append(x.copy())
                costs.append(current)
            else:
                mechanism.set_design_vector(x)
                mu = min(mu * mu_up, mu_max)

        self._print_progress(iterations - 1, iterations, current)
        print()

        return {"parameter_history": history, "cost_history": costs}


def optimize_linkage(mechanism, target_path, iterations=100, regularization=0.01):
    solver = LevenbergMarquardtSolver(regularization_weight=regularization)
    return solver.solve(mechanism, target_path, iterations)


