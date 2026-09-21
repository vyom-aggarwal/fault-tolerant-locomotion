import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401  (DLL load order on Windows)
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

FAULTS = [
    ("torque_limit", 0.2),
    ("joint_lock", 1.0),
    ("actuation_delay", 5),
    ("actuation_delay", 10),
    ("sensor_dropout", 1.0),
    ("sensor_noise", 0.3),
]

FAULT_STEP = 150      # onset
MEASURE_FROM = 300    # settled: 150 steps (2.5 s) after onset
MEASURE_TO = 800      # ends before episode truncation
MAX_STEPS = 1000


def rollout(model, env, seed, fault=None, severity=1.0):
    obs, info = env.reset(seed=seed)
    vels = []
    fell = False
    for t in range(MAX_STEPS):
        if fault is not None and t == FAULT_STEP:
            env.trigger_fault(fault, severity=severity)
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if MEASURE_FROM <= t < MEASURE_TO:
            vels.append(info["forward_vel"])
        if term:
            fell = True
            break
        if trunc:
            break
    complete = (t + 1) >= MEASURE_TO
    if not complete or not vels:
        return None, fell, t + 1
    return float(np.mean(vels)), fell, t + 1


def process(model_path, trials, verbose=True):
    model = PPO.load(model_path)
    env = make_env_from_model_path(model_path, render=False)

    # Healthy reference, one per seed, over the identical window
    healthy = {}
    for s in range(trials):
        v, fell, n = rollout(model, env, seed=s)
        if v is not None and not fell:
            healthy[s] = v

    if not healthy:
        print(f"  {os.path.basename(model_path)}: no healthy trial survived the "
              f"measurement window; cannot form a reference.")
        env.close()
        return []

    h_mean = float(np.mean(list(healthy.values())))
    if verbose:
        print(f"\n  {os.path.basename(model_path)}")
        print(f"    healthy reference: {h_mean:.4f} m/s "
              f"(n={len(healthy)}/{trials} survived the window)")
        print(f"    {'fault':<17}{'sev':>5}{'faulted':>10}{'deficit':>10}"
              f"{'paired n':>10}{'fell':>7}")

    rows = []
    for fname, fsev in FAULTS:
        deficits, n_fell, n_short = [], 0, 0
        faulted_vals = []
        for s in range(trials):
            if s not in healthy:
                continue                      # no paired reference for this seed
            v, fell, n = rollout(model, env, seed=s, fault=fname, severity=fsev)
            if fell:
                n_fell += 1
                continue                      # excluded: window incomplete
            if v is None:
                n_short += 1
                continue
            faulted_vals.append(v)
            deficits.append(healthy[s] - v)   # paired, same seed

        if deficits:
            d_mean = float(np.mean(deficits))
            d_sd = float(np.std(deficits, ddof=1)) if len(deficits) > 1 else 0.0
            f_mean = float(np.mean(faulted_vals))
        else:
            d_mean = d_sd = f_mean = float("nan")

        if verbose:
            print(f"    {fname:<17}{fsev:>5}{f_mean:>10.4f}{d_mean:>+10.4f}"
                  f"{len(deficits):>10}{n_fell:>7}")
        rows.append({"model": os.path.basename(model_path), "fault": fname,
                     "severity": fsev, "healthy": h_mean, "faulted": f_mean,
                     "deficit": d_mean, "deficit_sd": d_sd,
                     "n_paired": len(deficits), "n_fell": n_fell})
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--trials", type=int, default=12,
                        help="Paired trials per fault. Each needs a surviving "
                             "healthy run at the same seed.")
    parser.add_argument("--authority", type=float, default=0.055,
                        help="Best forward correction a DC offset delivers, from "
                             "probe_authority.py. Used only for the comparison.")
    args = parser.parse_args()

    print("Settled post-fault velocity deficit")
    print(f"  measured over steps {MEASURE_FROM}-{MEASURE_TO} "
          f"(fault at {FAULT_STEP}); paired by seed; fallen trials excluded")

    if args.all:
        mp = os.path.join(args.results_dir, "manifest.json")
        if not os.path.exists(mp):
            print(f"ERROR: {mp} not found.")
            return
        with open(mp) as f:
            manifest = json.load(f)
        seeds = sorted((s for s, v in manifest.get("seeds", {}).items()
                        if v.get("gait", {}).get("converged")), key=int)
        all_rows = []
        for sid in seeds:
            path = os.path.join(args.models_dir, f"seed_{sid}")
            if os.path.exists(path + ".zip"):
                all_rows += process(path, args.trials)
    else:
        all_rows = process(args.model, args.trials)

    if not all_rows:
        return

    # across whatever was run 
    print("\n" + "=" * 74)
    print("DEFICIT vs AVAILABLE CORRECTION")
    print("=" * 74)
    print(f"  A DC joint offset delivers at most ~{args.authority:.3f} m/s "
          f"(probe_authority.py).\n")
    print(f"  {'fault':<17}{'sev':>5}{'deficit':>12}{'covered by':>12}   verdict")
    by = {}
    for r in all_rows:
        by.setdefault((r["fault"], r["severity"]), []).append(r["deficit"])
    for (fname, fsev), vals in by.items():
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            print(f"  {fname:<17}{fsev:>5}      (no surviving paired trials)")
            continue
        d = float(np.mean(vals))
        frac = args.authority / d if d > 1e-6 else float("inf")
        if d <= 0:
            verdict = "no deficit to correct"
        elif frac >= 1.0:
            verdict = "CORRECTABLE in principle"
        elif frac >= 0.5:
            verdict = "partially correctable"
        else:
            verdict = "beyond DC-offset authority"
        print(f"  {fname:<17}{fsev:>5}{d:>+12.4f}{min(frac,9.99):>11.0%}   {verdict}")

    print("\n  'Covered by' is the fraction of the settled deficit that the best")
    print("  achievable constant offset can restore. It says nothing about whether")
    print("  a fallen robot can be saved -- fallen trials are excluded here, and")
    print("  for faults with high fall rates the dominant failure is loss of")
    print("  stability, not loss of speed.")


if __name__ == "__main__":
    main()
