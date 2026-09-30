import numpy as np


class AmplitudeAdapter:

    def __init__(
        self,
        base_scale,
        mode="adaptive",          # "none" | "fixed" | "adaptive"
        fixed_scale=0.70,         # used when mode == "fixed"
        s_min=0.60,
        s_max=1.00,
        k_threshold=3.0,          # risk threshold = healthy mean + k * healthy sd
        decrease=0.04,            # multiplicative back-off per step when at risk
        increase=0.0015,          # additive recovery per step when calm
        risk_mode="osc",          # "tilt" | "osc"
        osc_weight=3.0,           # weight on tilt VARIABILITY when risk_mode="osc"
        smooth_window=30,         # ~1 stride, so gait wobble is not read as risk
        calibrate_steps=120,      # 2 s of healthy walking to set the threshold
        min_threshold=0.02,       # floor, in case healthy walking is unusually smooth
    ):
        if mode not in ("none", "fixed", "adaptive"):
            raise ValueError(f"mode must be none|fixed|adaptive, got {mode!r}")
        self.base_scale = float(base_scale)
        self.mode = mode
        self.fixed_scale = float(fixed_scale)
        self.s_min, self.s_max = float(s_min), float(s_max)
        self.k = float(k_threshold)
        self.decrease = float(decrease)
        self.increase = float(increase)
        self.risk_mode = risk_mode
        self.osc_weight = float(osc_weight)
        self.smooth_window = int(smooth_window)
        self.calibrate_steps = int(calibrate_steps)
        self.min_threshold = float(min_threshold)
        self.reset()

    def reset(self):
        self.s = self.fixed_scale if self.mode == "fixed" else self.s_max
        self.step_count = 0
        self._risk_hist = []
        self._calib = []
        self.threshold = None
        self.n_backoff = 0
        self.s_trace = []

    # per step 
    def apply(self, env):
        """Set the environment's joint-space action scale for this step."""
        env._action_scale = self.base_scale * self.s
        self.s_trace.append(self.s)
        return self.s

    def observe(self, info):
        self.step_count += 1
        if self.mode in ("none", "fixed"):
            return

        # Tilt away from upright. 0 when level.
        tilt = 1.0 - float(info.get("upright_alignment", 1.0))
        self._risk_hist.append(tilt)
        w = min(self.smooth_window, len(self._risk_hist))
        recent = self._risk_hist[-w:]

        if self.risk_mode == "tilt":
            risk_s = float(np.mean(recent))
        else:
            # Delay-induced instability appears as OSCILLATION before it appears as lean
            risk_s = float(np.mean(recent)) + self.osc_weight * float(np.std(recent))

        # Calibrate on the robot's own healthy walking
        if self.step_count <= self.calibrate_steps:
            self._calib.append(risk_s)
            return
        if self.threshold is None:
            arr = np.array(self._calib[self.smooth_window:] or self._calib)
            self.threshold = max(float(arr.mean() + self.k * arr.std()),
                                 self.min_threshold)

        if risk_s > self.threshold:
            self.s = max(self.s_min, self.s * (1.0 - self.decrease))
            self.n_backoff += 1
        else:
            self.s = min(self.s_max, self.s + self.increase)

    # reporting 
    def diagnostics(self):
        return {
            "mode": self.mode,
            "final_scale": round(self.s, 4),
            "mean_scale": round(float(np.mean(self.s_trace)), 4) if self.s_trace else None,
            "min_scale": round(float(np.min(self.s_trace)), 4) if self.s_trace else None,
            "threshold": round(self.threshold, 5) if self.threshold is not None else None,
            "risk_mode": self.risk_mode,
            "backoff_steps": self.n_backoff,
            "steps": self.step_count,
        }