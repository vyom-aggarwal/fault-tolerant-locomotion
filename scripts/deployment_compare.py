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

SEVERITIES = [2, 3, 5, 10, 20]

CANDIDATES = [
    ("none",            dict(mode="none")),
    ("fixed 0.80",      dict(mode="fixed", fixed_scale=0.80)),
    ("fixed 0.70",      dict(mode="fixed", fixed_scale=0.70)),
    ("fixed 0.65",      dict(mode="fixed", fixed_scale=0.65)),
    ("fixed 0.55",      dict(mode="fixed", fixed_scale=0.55)),
    ("adaptive k=1.0",  dict(mode="adaptive", k_threshold=1.0)),
    ("adaptive k=2.0",  dict(mode="adaptive", k_threshold=2.0)),
    ("adaptive k=3.0",  dict(mode="adaptive", k_threshold=3.0)),
]


def trial(model, env, adapter, severity, seed, base_scale):
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
    env.trigger_fault("actuation_delay", severity=severity)
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
    return fell, (float(np.mean(vels)) if (vels and not fell) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--out", type=str, default="logs/deployment.csv")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False)
    base_scale = float(env._action_scale)
    seeds = list(range(args.trials))

    print("One configuration held fixed across an unknown severity")
    print(f"  severities {SEVERITIES}, {args.trials} trials each\n")
    header = "  " + f"{'candidate':<17}" + "".join(f"{'sev'+str(s):>12}" for s in SEVERITIES)
    print(header + f"{'mean':>11}{'worst':>8}{'speed':>9}")

    rows, points = [], []
    for label, kw in CANDIDATES:
        per_sev_falls, per_sev_speed = [], []
        cells = ""
        for sev in SEVERITIES:
            ad = AmplitudeAdapter(base_scale=base_scale, **kw)
            res = [trial(model, env, ad, sev, sd, base_scale) for sd in seeds]
            res = [r for r in res if r is not None]
            if not res:
                cells += f"{'--':>12}"
                continue
            fr = sum(r[0] for r in res) / len(res)
            vs = [r[1] for r in res if r[1] is not None]
            sp = float(np.mean(vs)) if vs else float("nan")
            per_sev_falls.append(fr)
            per_sev_speed.append(sp)
            cells += f"{fr:>7.0%}{sp:>5.2f}"
            rows.append({"candidate": label, "severity": sev,
                         "fall_rate": round(fr, 4),
                         "speed": round(sp, 4) if not np.isnan(sp) else ""})
        if not per_sev_falls:
            continue
        mean_fall = float(np.mean(per_sev_falls))
        worst_fall = float(np.max(per_sev_falls))
        valid = [v for v in per_sev_speed if not np.isnan(v)]
        mean_speed = float(np.mean(valid)) if valid else float("nan")
        points.append((label, mean_fall, mean_speed, worst_fall))
        print(f"  {label:<17}{cells}{mean_fall:>10.0%}{worst_fall:>8.0%}{mean_speed:>9.3f}")
    env.close()

    if args.out and rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Pareto on (mean fall rate, mean speed)
    front = []
    for i, (li, fi, si, wi) in enumerate(points):
        if np.isnan(si):
            continue
        dominated = any((fj <= fi and sj >= si) and (fj < fi or sj > si)
                        for j, (lj, fj, sj, wj) in enumerate(points)
                        if i != j and not np.isnan(sj))
        if not dominated:
            front.append((li, fi, si, wi))

    print("\n" + "=" * 74)
    print("PARETO FRONT over the severity range")
    print("=" * 74)
    for label, fr, sp, wf in sorted(front, key=lambda x: x[1]):
        print(f"  {label:<18}mean falls {fr:>5.0%}   mean speed {sp:>7.3f}   "
              f"worst {wf:>4.0%}")

    n_ad = sum(1 for l, *_ in front if l.startswith("adaptive"))
    n_fx = sum(1 for l, *_ in front if l.startswith("fixed") or l == "none")
    print("\n" + "=" * 74)
    if n_ad and not n_fx:
        print("  ADAPTATION DOMINATES under unknown severity. No single constant")
        print("  matches it across the range, which is the deployment condition:")
        print("  a robot cannot look up its own fault severity.")
    elif n_fx and not n_ad:
        print("  A CONSTANT STILL WINS even without knowing the severity. The")
        print("  adaptive controller buys nothing a single well-chosen scale does")
        print("  not already provide. Report that -- it is the simpler controller.")
    else:
        print(f"  MIXED: {n_ad} adaptive, {n_fx} constant. Each owns part of the")
        print("  range; report the frontier rather than a winner.")
    print("\n  'worst' is the single worst severity for that candidate. A low mean")
    print("  with a catastrophic worst case is not deployable, so read both.")


if __name__ == "__main__":
    main()
