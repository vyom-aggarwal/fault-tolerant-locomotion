import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path
from amplitude_adapter import AmplitudeAdapter
from baseline_fault_eval import FAULT_ONSET_MIN, FAULT_ONSET_MAX, POST_FAULT_WINDOW

CONTROL_HZ = 60.0


def trial(model, env, fault, severity, seed, base_scale, k, safe_scale):
    ad = AmplitudeAdapter(base_scale=base_scale, mode="adaptive", k_threshold=k)
    obs, info = env.reset(seed=seed)
    onset = int(np.random.default_rng(seed).integers(FAULT_ONSET_MIN, FAULT_ONSET_MAX))

    for _ in range(onset):
        ad.apply(env)
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        ad.observe(info)
        if term or trunc:
            env._action_scale = base_scale
            return None

    env.trigger_fault(fault, severity=severity)
    t_detect = t_safe = t_fall = None
    prev_backoff = ad.n_backoff

    for i in range(POST_FAULT_WINDOW):
        ad.apply(env)
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        ad.observe(info)
        if t_detect is None and ad.n_backoff > prev_backoff:
            t_detect = i                      # risk first exceeded threshold
        if t_safe is None and ad.s <= safe_scale:
            t_safe = i                        # reached a demonstrably stable scale
        if term:
            t_fall = i
            break
        if trunc:
            break
    env._action_scale = base_scale
    return {"t_detect": t_detect, "t_safe": t_safe, "t_fall": t_fall,
            "s_at_end": ad.s, "fell": t_fall is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--fault", type=str, default="actuation_delay")
    parser.add_argument("--severity", type=float, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--safe_scale", type=float, default=0.70,
                        help="A scale measured to be stable at this severity "
                             "(from the frontier run).")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False)
    base_scale = float(env._action_scale)

    rows = [trial(model, env, args.fault, args.severity, sd, base_scale,
                  args.k, args.safe_scale) for sd in range(args.trials)]
    env.close()
    rows = [r for r in rows if r is not None]
    if not rows:
        print("No usable trials.")
        return

    fell = [r for r in rows if r["fell"]]
    print(f"Timing on {args.fault}({args.severity}), k={args.k}, "
          f"safe scale {args.safe_scale}")
    print(f"  {len(rows)} trials, {len(fell)} fell ({len(fell)/len(rows):.0%})\n")

    def stat(vals, label, unit="steps"):
        v = [x for x in vals if x is not None]
        if not v:
            print(f"  {label:<34}never")
            return None
        med = float(np.median(v))
        print(f"  {label:<34}median {med:>6.0f} {unit}  "
              f"({med/CONTROL_HZ:.2f} s)   n={len(v)}")
        return med

    print("  Relative to fault onset:")
    t_det = stat([r["t_detect"] for r in rows], "detection fires at")
    t_saf = stat([r["t_safe"] for r in rows], "reaches safe scale at")
    t_fal = stat([r["t_fall"] for r in fell], "falls at (fallen trials)")

    print("\n  Budget vs requirement:")
    if fell:
        budgets = [r["t_fall"] - r["t_detect"] for r in fell
                   if r["t_detect"] is not None and r["t_fall"] is not None]
        if budgets:
            bmed = float(np.median(budgets))
            print(f"    time from detection to fall      median {bmed:>6.0f} steps"
                  f"  ({bmed/CONTROL_HZ:.2f} s)")
        else:
            bmed = None
            print("    detection never fired before the fall in any fallen trial")
    else:
        budgets, bmed = [], None
        print("    no falls -- nothing to outrun at this severity")

    if t_det is not None and t_saf is not None:
        need = t_saf - t_det
        print(f"    detection -> safe scale          median {need:>6.0f} steps"
              f"  ({need/CONTROL_HZ:.2f} s)")
    else:
        need = None
        if t_saf is None:
            print("    the controller never reached the safe scale at all")

    print("\n" + "=" * 70)
    if not fell:
        print("  No falls: reaction is not being outrun at this severity. A")
        print("  constant can still win on speed, but not because reaction is")
        print("  impossible.")
    elif bmed is None:
        print("  DETECTION NEVER FIRES BEFORE THE FALL. The risk signal does not")
        print("  lead the failure at all, so no back-off rate could help. The")
        print("  limit is the observable, not the gains.")
    elif need is not None and bmed < need:
        print(f"  REACTION IS TOO SLOW BY CONSTRUCTION: the robot falls {bmed:.0f} steps")
        print(f"  after detection, but needs {need:.0f} steps to reach a stable scale.")
        print("  No choice of gains closes a deficit this large -- the instability")
        print("  outruns the controller. Pre-emptive conservatism is the only")
        print("  option, which is exactly what the frontier shows.")
    else:
        print(f"  Reaction fits in the budget ({bmed:.0f} steps available, "
              f"{need:.0f} needed).")
        print("  Falls are then a tuning failure, not a physical limit: a faster")
        print("  or deeper back-off should help, and is worth sweeping.")
    print("=" * 70)


if __name__ == "__main__":
    main()
