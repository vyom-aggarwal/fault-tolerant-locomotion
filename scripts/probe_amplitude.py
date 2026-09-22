import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401  (DLL load order on Windows)
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

SCALES = (0.70, 0.85, 1.00, 1.15, 1.30)
FAULTS = [("torque_limit", 0.2), ("joint_lock", 1.0), ("actuation_delay", 5)]
CHANGE_STEP = 150
MEASURE_FROM = 300
MEASURE_TO = 800
MAX_STEPS = 1000
DC_OFFSET_AUTHORITY = 0.055   # best forward change a constant offset delivered

MEASURED_DEFICIT = {"torque_limit": 0.106, "joint_lock": 0.099, "actuation_delay": 0.197}


def rollout(model, env, seed, base_scale, scale=1.0, mode="joint",
            fault=None, severity=1.0):
    env._action_scale = base_scale            # restore: reset() does not
    obs, info = env.reset(seed=seed)
    vels, sat = [], []
    fell = False
    for t in range(MAX_STEPS):
        if t == CHANGE_STEP:
            if fault is not None:
                env.trigger_fault(fault, severity=severity)
            if mode == "joint":
                env._action_scale = base_scale * scale
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action, dtype=np.float64)
        sat.append(float(np.mean(np.abs(a) > 0.95)))
        if mode == "action" and t >= CHANGE_STEP:
            a = a * scale
        obs, r, term, trunc, info = env.step(a)
        if MEASURE_FROM <= t < MEASURE_TO:
            vels.append(info["forward_vel"])
        if term:
            fell = True
            break
        if trunc:
            break
    env._action_scale = base_scale
    complete = (t + 1) >= MEASURE_TO and not fell
    v = float(np.mean(vels)) if (complete and vels) else None
    return v, fell, float(np.mean(sat)) if sat else 0.0


def sweep(model, env, base_scale, trials, mode, fault=None, severity=1.0):
    out = {}
    for c in SCALES:
        vels, n_fell, sats = {}, 0, []
        for sd in range(trials):
            v, fell, s = rollout(model, env, sd, base_scale, scale=c, mode=mode,
                                 fault=fault, severity=severity)
            sats.append(s)
            if fell:
                n_fell += 1
            elif v is not None:
                vels[sd] = v
        out[c] = {"vels": vels, "fell": n_fell, "sat": float(np.mean(sats))}
    return out


def paired_gain(res, c, ref=1.00):
    common = sorted(set(res[c]["vels"]) & set(res[ref]["vels"]))
    if not common:
        return None, 0
    d = [res[c]["vels"][s] - res[ref]["vels"][s] for s in common]
    return float(np.mean(d)), len(common)


def print_table(res, trials, label):
    print(f"\n  {label}")
    print(f"    {'scale':>6}{'settled v':>11}{'dv vs 1.0':>11}{'paired n':>10}"
          f"{'fell':>8}{'saturated':>11}")
    for c in SCALES:
        r = res[c]
        v = np.mean(list(r["vels"].values())) if r["vels"] else float("nan")
        g, n = paired_gain(res, c)
        gs = f"{g:+.4f}" if g is not None else "   --"
        print(f"    {c:>6.2f}{v:>11.4f}{gs:>11}{n:>10}"
              f"{r['fell']:>5}/{trials:<2}{r['sat']:>10.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--trials", type=int, default=8,
                        help="Seeds per (condition, scale). Paired across scales.")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False, randomize_init=False)
    base_scale = float(env._action_scale)

    print("Amplitude-channel authority probe")
    print(f"  scales {SCALES}, change at step {CHANGE_STEP}, settled over "
          f"{MEASURE_FROM}-{MEASURE_TO}, {args.trials} paired seeds each")
    print(f"  base joint action_scale = {base_scale}")

    # healthy: action-space vs joint-space 
    print("\n" + "=" * 74)
    print("HEALTHY ROBOT")
    print("=" * 74)
    h_joint = sweep(model, env, base_scale, args.trials, "joint")
    h_action = sweep(model, env, base_scale, args.trials, "action")
    print_table(h_joint, args.trials, "joint-space scaling (after the clip)")
    print_table(h_action, args.trials, "action-space scaling (before the clip)")
    sat = h_joint[1.00]["sat"]
    print(f"\n  Policy saturation at scale 1.0: {sat:.1%} of action components "
          f"exceed |0.95|.")
    if sat > 0.20:
        print("  High saturation: action-space scaling is largely clipped away, so")
        print("  joint-space scaling is the fair test of the amplitude channel.")

    # under fault 
    print("\n" + "=" * 74)
    print("UNDER FAULT  (joint-space scaling)")
    print("=" * 74)
    fault_results = {}
    for fname, fsev in FAULTS:
        res = sweep(model, env, base_scale, args.trials, "joint",
                    fault=fname, severity=fsev)
        fault_results[(fname, fsev)] = res
        print_table(res, args.trials, f"{fname} (severity {fsev})")
    env.close()

    # ---- verdict ----
    print("\n" + "=" * 74)
    print("AMPLITUDE vs OFFSET, judged against the measured deficits")
    print("=" * 74)
    print(f"  Constant joint offset delivered at most {DC_OFFSET_AUTHORITY:+.3f} m/s.\n")
    print(f"  {'fault':<20}{'deficit':>9}{'best dv':>10}{'scale':>7}"
          f"{'covers':>9}{'vs offset':>11}")
    speed_rows = []
    for (fname, fsev), res in fault_results.items():
        gains = {c: paired_gain(res, c)[0] for c in SCALES if c != 1.00}
        gains = {c: g for c, g in gains.items() if g is not None}
        deficit = MEASURED_DEFICIT.get(fname)
        label = f"{fname}({fsev})"
        if not gains:
            print(f"  {label:<20}  (no surviving paired trials at any scale)")
            continue
        c_best = max(gains, key=gains.get)
        g_best = gains[c_best]
        cover = g_best / deficit if deficit else float("nan")
        vs_off = g_best / DC_OFFSET_AUTHORITY
        speed_rows.append((fname, cover, g_best))
        print(f"  {label:<20}{deficit:>9.3f}{g_best:>+10.4f}{c_best:>7.2f}"
              f"{cover:>8.0%}{vs_off:>10.1f}x")

    print(f"\n  {'fault':<20}{'falls @1.0':>12}{'fewest falls':>14}{'at scale':>10}")
    stab_rows = []
    for (fname, fsev), res in fault_results.items():
        falls = {c: res[c]["fell"] for c in SCALES}
        label = f"{fname}({fsev})"
        if len(set(falls.values())) == 1:
            print(f"  {label:<20}{falls[1.00]:>8}/{args.trials:<3}"
                  f"{'(no difference across scales)':>24}")
            continue
        fewest = min(falls.values())
        best_scales = [c for c in SCALES if falls[c] == fewest]
        stab_rows.append((fname, falls[1.00], fewest))
        print(f"  {label:<20}{falls[1.00]:>8}/{args.trials:<3}{fewest:>9}/{args.trials:<3}"
              f"{', '.join(f'{c:.2f}' for c in best_scales):>10}")

    print()
    closes = [f for f, cov, _ in speed_rows if cov >= 0.8]
    partial = [f for f, cov, _ in speed_rows if 0.4 <= cov < 0.8]
    if closes:
        print(f"  SPEED: amplitude closes >=80% of the settled deficit for "
              f"{', '.join(closes)}.")
        print("  It reaches a channel the constant offset could not.")
    elif partial:
        print(f"  SPEED: amplitude closes 40-80% of the deficit for {', '.join(partial)}"
              f" -- a real but incomplete channel.")
    else:
        print("  SPEED: amplitude does not close a meaningful share of any deficit.")
        print("  The policy may cancel multiplicative changes as well as additive ones.")

    helped = [f for f, f1, fmin in stab_rows if f1 - fmin >= max(1, round(0.25 * args.trials))]
    if helped:
        print(f"  STABILITY: some scale cuts falls by >=25% of trials for "
              f"{', '.join(helped)}.")
        print("  That addresses the failure that actually dominates these faults,")
        print("  independently of whether speed improves.")
    else:
        print("  STABILITY: no scale materially reduces falls at these settings.")
    print(f"\n  {args.trials}-seed probes on one policy: they indicate which channel is")
    print("  worth building a method around, not effect sizes for the paper.")


if __name__ == "__main__":
    main()
