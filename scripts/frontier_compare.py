import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path
from amplitude_adapter import AmplitudeAdapter
from baseline_fault_eval import FAULT_ONSET_MIN, FAULT_ONSET_MAX, POST_FAULT_WINDOW

FIXED_SCALES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
ADAPTIVE_K = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def trial(model, env, adapter, fault, severity, seed, base_scale):
    adapter.reset()
    obs, info = env.reset(seed=seed)
    onset = int(np.random.default_rng(seed).integers(FAULT_ONSET_MIN, FAULT_ONSET_MAX))
    for _ in range(onset):
        adapter.apply(env)
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        adapter.observe(info)
        if term or trunc:
            env._action_scale = base_scale
            return None
    env.trigger_fault(fault, severity=severity)
    vels, fell = [], False
    for i in range(POST_FAULT_WINDOW):
        adapter.apply(env)
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        adapter.observe(info)
        if i >= 150:
            vels.append(info["forward_vel"])
        if term:
            fell = True
            break
        if trunc:
            env._action_scale = base_scale
            return None
    env._action_scale = base_scale
    return {"fell": fell,
            "vel": float(np.mean(vels)) if (vels and not fell) else None,
            "mean_s": adapter.diagnostics()["mean_scale"]}


def measure(model, env, base_scale, fault, severity, seeds, mode, **kw):
    ad = AmplitudeAdapter(base_scale=base_scale, mode=mode, **kw)
    res = [trial(model, env, ad, fault, severity, sd, base_scale) for sd in seeds]
    res = [r for r in res if r is not None]
    if not res:
        return None
    n = len(res)
    vels = [r["vel"] for r in res if r["vel"] is not None]
    return {"n": n,
            "fall_rate": sum(r["fell"] for r in res) / n,
            # survivors only; comparable across points of similar fall rate
            "speed": float(np.mean(vels)) if vels else float("nan"),
            "mean_s": float(np.mean([r["mean_s"] for r in res]))}


def pareto_front(points):
    front = []
    for i, (li, fi, si) in enumerate(points):
        if np.isnan(si):
            continue
        dominated = any(
            (fj <= fi and sj >= si) and (fj < fi or sj > si)
            for j, (lj, fj, sj) in enumerate(points)
            if i != j and not np.isnan(sj))
        if not dominated:
            front.append((li, fi, si))
    return front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--fault", type=str, default="actuation_delay")
    parser.add_argument("--severity", type=float, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--fixed_scales", type=str, default=None,
                        help="Comma-separated fixed scales. A coarse grid lets an "
                             "adaptive point reach the front merely by filling a "
                             "sampling gap; sample finely where adaptive points land.")
    parser.add_argument("--out", type=str, default="logs/frontier.csv")
    args = parser.parse_args()

    global FIXED_SCALES
    if args.fixed_scales:
        FIXED_SCALES = [float(x) for x in args.fixed_scales.split(",")]

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False)
    base_scale = float(env._action_scale)
    seeds = list(range(args.trials))

    print(f"Frontier comparison on {args.fault}({args.severity}), "
          f"{args.trials} trials per point\n")
    rows, points = [], []

    print(f"  {'mechanism':<12}{'param':>8}{'n':>5}{'falls':>8}{'speed':>10}{'mean s':>9}")
    for s_fixed in FIXED_SCALES:
        mode = "none" if s_fixed >= 0.999 else "fixed"
        r = measure(model, env, base_scale, args.fault, args.severity, seeds,
                    mode, fixed_scale=s_fixed)
        if r is None:
            continue
        label = f"fixed {s_fixed:.2f}"
        rows.append({"mechanism": "fixed", "param": s_fixed, **r})
        points.append((label, r["fall_rate"], r["speed"]))
        print(f"  {'fixed':<12}{s_fixed:>8.2f}{r['n']:>5}{r['fall_rate']:>8.0%}"
              f"{r['speed']:>10.4f}{r['mean_s']:>9.3f}")

    for k in ADAPTIVE_K:
        r = measure(model, env, base_scale, args.fault, args.severity, seeds,
                    "adaptive", k_threshold=k)
        if r is None:
            continue
        label = f"adaptive k={k:.1f}"
        rows.append({"mechanism": "adaptive", "param": k, **r})
        points.append((label, r["fall_rate"], r["speed"]))
        print(f"  {'adaptive':<12}{k:>8.1f}{r['n']:>5}{r['fall_rate']:>8.0%}"
              f"{r['speed']:>10.4f}{r['mean_s']:>9.3f}")
    env.close()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    front = pareto_front(points)
    print("\n" + "=" * 72)
    print("PARETO FRONT  (not dominated on both fall rate and speed)")
    print("=" * 72)
    for label, fr, sp in sorted(front, key=lambda x: x[1]):
        print(f"  {label:<18}falls {fr:>5.0%}   speed {sp:>8.4f}")

    n_ad = sum(1 for l, _, _ in front if l.startswith("adaptive"))
    n_fx = sum(1 for l, _, _ in front if l.startswith("fixed"))
    print("\n" + "=" * 72)
    if n_ad and not n_fx:
        print("  ADAPTATION DOMINATES: every non-dominated point is adaptive.")
        print("  At matched fall rates it is faster than any constant scale.")
    elif n_fx and not n_ad:
        print("  FIXED SCALING DOMINATES: no adaptive configuration survives on the")
        print("  front. A constant gait is the better controller for this fault, and")
        print("  that is the result to report -- it is simpler and it wins.")
    else:
        print(f"  MIXED FRONT: {n_ad} adaptive, {n_fx} fixed.")
        print("  Neither mechanism dominates; each wins in part of the range. Report")
        print("  the frontier, not a single pairwise comparison.")
    print("\n  Speed is averaged over survivors, so it is only meaningful between")
    print("  points with similar fall rates. A fast point with many falls is not")
    print("  better than a slow point with none.")
    print(f"\n  Wrote {len(rows)} points to {args.out}")


if __name__ == "__main__":
    main()
