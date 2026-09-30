import argparse
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path
from amplitude_adapter import AmplitudeAdapter
from baseline_fault_eval import FAULT_ONSET_MIN, FAULT_ONSET_MAX, POST_FAULT_WINDOW

GRID = {
    "risk_mode": ["tilt", "osc"],
    "k_threshold": [1.0, 2.0, 3.0],
    "increase": [0.0002, 0.0015],
    "smooth_window": [10, 30],
}


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
    d = adapter.diagnostics()
    return {"fell": fell,
            "vel": float(np.mean(vels)) if (vels and not fell) else None,
            "mean_s": d["mean_scale"], "min_s": d["min_scale"],
            "backoff": d["backoff_steps"], "thr": d["threshold"]}


def run_config(model, env, base_scale, fault, sev, seeds, **kw):
    ad = AmplitudeAdapter(base_scale=base_scale, mode="adaptive", **kw)
    res = [trial(model, env, ad, fault, sev, sd, base_scale) for sd in seeds]
    res = [r for r in res if r is not None]
    if not res:
        return None
    n = len(res)
    fell = sum(r["fell"] for r in res)
    vels = [r["vel"] for r in res if r["vel"] is not None]
    return {"n": n, "fall_rate": fell / n,
            "vel": float(np.mean(vels)) if vels else float("nan"),
            "mean_s": float(np.mean([r["mean_s"] for r in res])),
            "min_s": float(np.mean([r["min_s"] for r in res])),
            "backoff": float(np.mean([r["backoff"] for r in res])),
            "thr": res[0]["thr"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--fault", type=str, default="actuation_delay")
    parser.add_argument("--severity", type=float, default=5)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--fixed_scale", type=float, default=0.70)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False)
    base_scale = float(env._action_scale)
    seeds = list(range(args.trials))

    print(f"Sweeping adapter settings on {args.fault}({args.severity}), "
          f"{args.trials} trials each")
    print("References first, then every configuration.\n")

    refs = {}
    for label, mode, kw in [("none", "none", {}),
                            ("fixed", "fixed", {"fixed_scale": args.fixed_scale})]:
        ad = AmplitudeAdapter(base_scale=base_scale, mode=mode, **kw)
        res = [trial(model, env, ad, args.fault, args.severity, sd, base_scale)
               for sd in seeds]
        res = [r for r in res if r is not None]
        n = len(res)
        fr = sum(r["fell"] for r in res) / n
        vs = [r["vel"] for r in res if r["vel"] is not None]
        refs[label] = (fr, float(np.mean(vs)) if vs else float("nan"))
        print(f"  {label:<10} falls {fr:>5.0%}   speed "
              f"{refs[label][1]:.4f}   (n={n})")

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"\n  {len(combos)} configurations\n")
    print(f"  {'risk':>5}{'k':>5}{'incr':>8}{'win':>5}{'falls':>8}{'speed':>9}"
          f"{'mean s':>8}{'min s':>8}{'backoff':>9}")
    rows = []
    for vals in combos:
        kw = dict(zip(keys, vals))
        r = run_config(model, env, base_scale, args.fault, args.severity, seeds, **kw)
        if r is None:
            continue
        rows.append((kw, r))
        print(f"  {kw['risk_mode']:>5}{kw['k_threshold']:>5.1f}"
              f"{kw['increase']:>8.4f}{kw['smooth_window']:>5}"
              f"{r['fall_rate']:>8.0%}{r['vel']:>9.4f}{r['mean_s']:>8.3f}"
              f"{r['min_s']:>8.3f}{r['backoff']:>9.0f}")
    env.close()

    if not rows:
        print("\nNo configuration produced usable trials.")
        return

    f_fix, v_fix = refs["fixed"]
    f_none, _ = refs["none"]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Matching the constant on safety comes first
    safe = [(kw, r) for kw, r in rows if r["fall_rate"] <= f_fix + 0.05]
    if safe:
        best = max(safe, key=lambda x: (x[1]["vel"] if not np.isnan(x[1]["vel"]) else -1))
        kw, r = best
        print(f"  Best configuration matching fixed on safety:")
        print(f"    {kw}")
        print(f"    falls {r['fall_rate']:.0%} (fixed {f_fix:.0%}, none {f_none:.0%}), "
              f"speed {r['vel']:.4f} (fixed {v_fix:.4f}), mean s {r['mean_s']:.3f}")
        if r["vel"] > v_fix + 0.01:
            print("    -> Adaptation beats a constant conservative gait: same safety,")
            print("       higher speed. Validate on held-out seeds before claiming it.")
        else:
            print("    -> Matches fixed on safety but not on speed. Adaptation is not")
            print("       earning anything here; a constant gait is the simpler answer.")
    else:
        best_safety = min(rows, key=lambda x: x[1]["fall_rate"])
        kw, r = best_safety
        print(f"  NO configuration matches fixed s={args.fixed_scale} on safety.")
        print(f"  Best was {r['fall_rate']:.0%} falls vs fixed {f_fix:.0%}: {kw}")
        print(f"    mean s {r['mean_s']:.3f}, min s {r['min_s']:.3f}, "
              f"{r['backoff']:.0f} back-off steps per trial")
        if r["backoff"] < 20:
            print("  The risk signal is barely firing -- detection, not the back-off")
            print("  rule, is the limitation. The instability likely develops faster")
            print("  than any tilt-derived signal can lead it.")
        else:
            print("  The signal fires but the robot still falls: back-off is too slow")
            print("  or too shallow relative to how fast the instability grows.")
        print("\n  Reportable either way: a fixed conservative gait is the better")
        print("  controller for this fault, and that is a finding about the fault,")
        print("  not a failure of the experiment.")


if __name__ == "__main__":
    main()
