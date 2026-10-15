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


def evaluate_model(model_path, trials, verbose=True):
    model = PPO.load(model_path)
    env = make_env_from_model_path(model_path, render=False)
    base_scale = float(env._action_scale)
    seeds = list(range(trials))
    rows, points = [], []
    if verbose:
        header = "  " + f"{'candidate':<17}" + "".join(f"{'sev'+str(s):>12}" for s in SEVERITIES)
        print(header + f"{'mean':>11}{'worst':>8}{'speed':>9}")
    for label, kw in CANDIDATES:
        per_f, per_s, cells = [], [], ""
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
            per_f.append(fr); per_s.append(sp)
            cells += f"{fr:>7.0%}{sp:>5.2f}"
            rows.append({"model": os.path.basename(model_path), "candidate": label,
                         "severity": sev, "fall_rate": round(fr, 4),
                         "speed": round(sp, 4) if not np.isnan(sp) else ""})
        if not per_f:
            continue
        valid = [v for v in per_s if not np.isnan(v)]
        pt = (label, float(np.mean(per_f)),
              float(np.mean(valid)) if valid else float("nan"),
              float(np.max(per_f)))
        points.append(pt)
        if verbose:
            print(f"  {label:<17}{cells}{pt[1]:>10.0%}{pt[3]:>8.0%}{pt[2]:>9.3f}")
    env.close()
    return rows, points


def pareto(points):
    front = []
    for i, (li, fi, si, wi) in enumerate(points):
        if np.isnan(si):
            continue
        dom = any((fj <= fi and sj >= si) and (fj < fi or sj > si)
                  for j, (lj, fj, sj, wj) in enumerate(points)
                  if i != j and not np.isnan(sj))
        if not dom:
            front.append(li)
    return front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--all", action="store_true",
                        help="Every converged seed. Required before reporting: the "
                             "adapter was tuned on seed 0, so a result from seed 0 "
                             "alone is not evidence.")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated seed numbers to skip (e.g. the tuning "
                             "seed, to report held-out results only).")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--out", type=str, default="logs/deployment.csv")
    args = parser.parse_args()

    if args.all:
        mp = os.path.join(args.results_dir, "manifest.json")
        if not os.path.exists(mp):
            raise SystemExit(f"{mp} not found.")
        import json
        with open(mp) as f:
            manifest = json.load(f)
        skip = {x.strip() for x in args.exclude.split(",") if x.strip()}
        paths = [os.path.join(args.models_dir, f"seed_{sid}")
                 for sid in sorted((s for s, v in manifest.get("seeds", {}).items()
                                    if v.get("gait", {}).get("converged")), key=int)
                 if sid not in skip]
        paths = [p_ for p_ in paths if os.path.exists(p_ + ".zip")]
    else:
        paths = [args.model]

    print("One configuration held fixed across an unknown severity")
    print(f"  severities {SEVERITIES}, {args.trials} trials each, "
          f"{len(paths)} polic{'y' if len(paths)==1 else 'ies'}\n")

    all_rows, per_model = [], {}
    for path in paths:
        name = os.path.basename(path)
        print(f"{name}")
        rows, points = evaluate_model(path, args.trials, verbose=True)
        all_rows += rows
        per_model[name] = (points, pareto(points))
        print(f"  front: {', '.join(per_model[name][1])}\n")

    if args.out and all_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    labels = [c[0] for c in CANDIDATES]
    n = len(per_model)
    print("=" * 78)
    print(f"ACROSS {n} POLIC{'Y' if n==1 else 'IES'}  "
          f"(unit of analysis is the policy, not the trial)")
    print("=" * 78)
    print(f"  {'candidate':<17}{'on front':>10}{'mean falls':>16}{'mean speed':>16}"
          f"{'worst falls':>15}")
    for lab in labels:
        on = sum(lab in fr for _, fr in per_model.values())
        mf = [pt[1] for pts, _ in per_model.values() for pt in pts if pt[0] == lab]
        ms = [pt[2] for pts, _ in per_model.values() for pt in pts
              if pt[0] == lab and not np.isnan(pt[2])]
        wf = [pt[3] for pts, _ in per_model.values() for pt in pts if pt[0] == lab]
        def f(v, pct):
            if not v:
                return "--"
            m = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            return f"{m:.0%} +/- {sd:.0%}" if pct else f"{m:.3f} +/- {sd:.3f}"
        print(f"  {lab:<17}{on:>6}/{n:<3}{f(mf, True):>16}{f(ms, False):>16}"
              f"{f(wf, True):>15}")

    ad_on = sum(any(l.startswith("adaptive") for l in fr) for _, fr in per_model.values())
    print("\n" + "=" * 78)
    if ad_on == 0:
        print(f"  No adaptive configuration reached the front on ANY of {n} policies.")
        print("  The constant wins robustly, not as a property of one seed.")
    elif ad_on == n:
        print(f"  An adaptive configuration reached the front on all {n} policies.")
    else:
        print(f"  Adaptive reached the front on {ad_on} of {n} policies -- mixed. Report")
        print("  the per-policy breakdown rather than a single verdict.")


if __name__ == "__main__":
    main()
    