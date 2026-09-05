import numpy as np

# Pure math -- unit-testable without a simulator

def ridge_pinv(B, lam):
    B = np.asarray(B, dtype=np.float64)
    n_out = B.shape[0]
    gram = B @ B.T + lam * np.eye(n_out)
    return B.T @ np.linalg.inv(gram)


class RLSEstimator:
    def __init__(self, n_out, n_in, theta0=None, forgetting=0.995, p0=1.0):
        self.n_out = n_out
        self.n_in = n_in                       # includes the bias column
        self.theta = np.zeros((n_out, n_in)) if theta0 is None else np.array(theta0, dtype=np.float64)
        self.P = p0 * np.eye(n_in)
        self.mu = forgetting
        self.n_updates = 0

    def update(self, phi, y):
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        Pphi = self.P @ phi
        denom = self.mu + float(phi @ Pphi)
        if denom < 1e-12:
            return
        K = Pphi / denom                                  # gain, (n_in,)
        residual = y - self.theta @ phi                   # (n_out,)
        self.theta = self.theta + np.outer(residual, K)
        self.P = (self.P - np.outer(K, Pphi)) / self.mu
        # keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1


# The adapter

class ResidualAdapter:
    def __init__(
        self,
        command=(0.5, 0.0, 0.0),
        B0=None,
        n_act=12,
        mode="fixed",               # "none" | "fixed" | "rls"
        eta=0.03,                   # integral gain (swept: 0.02-0.05 all work; 0.10 overshoots)
        lam=0.05,                   # ridge regularization on B^+
        delta_max=0.30,             # bound on |delta| per joint (action units)
        smooth_window=30,           # steps; ~one stride, to reject gait oscillation
        response_lag=3,             # steps between applying delta and seeing its effect
        dither=0.01,                # exploration amplitude for RLS identification
        rls_forgetting=0.995,
        rls_p0=1.0,
        warmup_steps=15,            # steps before delta starts moving
        rng=None,
    ):
        if mode not in ("none", "fixed", "rls"):
            raise ValueError(f"mode must be none|fixed|rls, got {mode!r}")
        self.command = np.asarray(command, dtype=np.float64)
        self.n_out = len(self.command)
        self.n_act = n_act
        self.mode = mode
        self.eta = eta
        self.lam = lam
        self.delta_max = delta_max
        self.smooth_window = smooth_window
        self.response_lag = response_lag
        self.dither = dither if mode == "rls" else 0.0
        self.rls_forgetting = rls_forgetting
        self.rls_p0 = rls_p0
        self.warmup_steps = warmup_steps
        self.rng = np.random.default_rng() if rng is None else rng

        if B0 is None:
            # Not a usable default
            B0 = np.zeros((self.n_out, n_act))
        self.B0 = np.array(B0, dtype=np.float64)
        if self.B0.shape != (self.n_out, n_act):
            raise ValueError(f"B0 must be {(self.n_out, n_act)}, got {self.B0.shape}")

        self.reset()

    # lifecycle 
    def reset(self):
        self.delta = np.zeros(self.n_act)
        self.B = self.B0.copy()
        self._y_hist = []
        self._delta_hist = []
        self.step_count = 0
        self._last_error = np.zeros(self.n_out)

        if self.mode == "rls":
            # theta = [B | bias]
            theta0 = np.hstack([self.B0, self.command.reshape(-1, 1)])
            self.rls = RLSEstimator(self.n_out, self.n_act + 1, theta0=theta0,
                                    forgetting=self.rls_forgetting, p0=self.rls_p0)
        else:
            self.rls = None

    # per-step 
    def apply(self, base_action):
        """Add the current residual (plus dither, in rls mode) to the policy output."""
        base_action = np.asarray(base_action, dtype=np.float64).reshape(-1)
        applied_delta = self.delta.copy()
        if self.dither > 0:
            applied_delta = applied_delta + self.rng.normal(0.0, self.dither, size=self.n_act)
        self._delta_hist.append(applied_delta)
        return np.clip(base_action + applied_delta, -1.0, 1.0)

    def observe(self, info):
        """Feed back the measured body-frame response and update delta (and B)."""
        y = np.array([info["forward_vel"], info["lateral_vel"], info["yaw_rate"]],
                     dtype=np.float64)
        self._y_hist.append(y)
        self.step_count += 1

        if self.mode == "none":
            return

        # Smooth over ~1 stride
        w = min(self.smooth_window, len(self._y_hist))
        y_smooth = np.mean(self._y_hist[-w:], axis=0)
        error = self.command - y_smooth
        self._last_error = error

        # identify B (rls only) 
        # Pair the current response with the delta applied response_lag steps ago
        if self.mode == "rls" and len(self._delta_hist) > self.response_lag + w:
            # The target is y averaged over `w` steps, so the regressor must be the delta averaged over the SAME window (shifted by response_lag)
            lo = len(self._delta_hist) - self.response_lag - w
            hi = len(self._delta_hist) - self.response_lag
            phi_delta = np.mean(self._delta_hist[lo:hi], axis=0)
            phi = np.concatenate([phi_delta, [1.0]])
            self.rls.update(phi, y_smooth)
            self.B = self.rls.theta[:, :self.n_act]

        # integral update of delta 
        if self.step_count <= self.warmup_steps:
            return                              # let the smoothing window fill first
        step = self.eta * (ridge_pinv(self.B, self.lam) @ error)
        self.delta = np.clip(self.delta + step, -self.delta_max, self.delta_max)

    # introspection 
    def diagnostics(self):
        return {
            "mode": self.mode,
            "delta_norm": float(np.linalg.norm(self.delta)),
            "delta_max_abs": float(np.max(np.abs(self.delta))) if self.n_act else 0.0,
            "tracking_error_norm": float(np.linalg.norm(self._last_error)),
            "B_norm": float(np.linalg.norm(self.B)),
            "B_drift_from_nominal": float(np.linalg.norm(self.B - self.B0)),
            "rls_updates": self.rls.n_updates if self.rls else 0,
        }
