import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401  (DLL load order on Windows)
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

LABELS = ("forward", "lateral", "yaw")


SETTLED_DEFICIT = {
    "torque_limit": 0.117,
    "joint_lock": 0.137,
    "sensor_dropout": 0.019,
    "sensor_noise": 0.009,
}


# Two-sided 95% Student-t critical values by degrees of freedom, to 5 decimals, each rounded UP so no interval is ever narrower than exact.
_T975 = {1: 12.70621, 2: 4.30266, 3: 3.18245, 4: 2.77645, 5: 2.57059, 6: 2.44692, 7: 2.36463, 8: 2.30601, 9: 2.26216, 10: 2.22814, 11: 2.20099, 12: 2.17882, 13: 2.16037, 14: 2.14479, 15: 2.13145, 16: 2.11991, 17: 2.10982, 18: 2.10093, 19: 2.09303, 20: 2.08597, 21: 2.07962, 22: 2.07388, 23: 2.06866, 24: 2.06390, 25: 2.05954, 26: 2.05553, 27: 2.05184, 28: 2.04841, 29: 2.04523, 30: 2.04228, 40: 2.02108, 60: 2.00030, 120: 1.97994}
MIN_PAIRS = 4   # an under-fault gain from fewer surviving pairs is not compared


def t_interval(mean, sd, n):
    if n < 2 or not np.isfinite(sd):
        return float("nan"), float("nan")
    df = n - 1
    t = _T975[max(k for k in _T975 if k <= df)]
    h = t * sd / np.sqrt(n)
    return mean - h, mean + h


def model_fingerprint(model_path):
    zp = model_path + ".zip"
    if not os.path.exists(zp):
        return None
    st = os.stat(zp)
    return {"size": st.st_size, "mtime": round(st.st_mtime, 3)}


def write_authority_sidecar(model_path, record):
    path = model_path + "_authority.json"
    record = dict(record, model=os.path.basename(model_path),
                  fingerprint=model_fingerprint(model_path))
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def ridge_pinv(Bm, lam=0.05):
    return Bm.T @ np.linalg.inv(Bm @ Bm.T + lam * np.eye(Bm.shape[0]))


def run_episode(model, env, offset, max_steps, seed, settle=40,
                fault=None, fault_step=150, fault_severity=1.0,
                offset_start=0, measure_from=None):
    obs, info = env.reset(seed=seed)
    if measure_from is None:
        measure_from = settle
    fwd, lat, yaw = [], [], []
    fell = False
    for t in range(max_steps):
        if fault is not None and t == fault_step:
            env.trigger_fault(fault, severity=fault_severity)
        base, _ = model.predict(obs, deterministic=True)
        off = offset if t >= offset_start else np.zeros_like(offset)
        obs, r, term, trunc, info = env.step(np.clip(base + off, -1.0, 1.0))
        if t >= measure_from:
            fwd.append(info["forward_vel"])
            lat.append(info["lateral_vel"])
            yaw.append(info["yaw_rate"])
        if term:
            fell = True
            break
        if trunc:
            break
    if not fwd:
        return None, t + 1, fell
    return np.array([np.mean(fwd), np.mean(lat), np.mean(yaw)]), t + 1, fell


def split_half(X, Y):
    mid = len(X) // 2
    if mid < 14:                              # need > n_params rows per half
        return None, None

    def fit(Xs, Ys):
        Xa = np.hstack([Xs, np.ones((len(Xs), 1))])
        th, *_ = np.linalg.lstsq(Xa, Ys, rcond=None)
        return th[:-1].T

    B1, B2 = fit(X[:mid], Y[:mid]), fit(X[mid:], Y[mid:])
    rel = np.linalg.norm(B1 - B2) / max(np.linalg.norm(0.5 * (B1 + B2)), 1e-12)
    cos = []
    for i in range(B1.shape[0]):
        a, b = B1[i], B2[i]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        cos.append(float(a @ b / d) if d > 1e-12 else 0.0)
    return rel, np.array(cos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--episodes", type=int, default=60,
                        help="One constant offset per episode. Needs > 13 per half.")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.15,
                        help="Std of the per-joint constant offset.")
    parser.add_argument("--delta_max", type=float, default=0.30,
                        help="Per-joint bound on the correction, matching ResidualAdapter.")
    parser.add_argument("--direct_trials", type=int, default=6,
                        help="Initial conditions per requested correction in the direct "
                             "test. Was 1 (a single fixed trajectory per level).")
    parser.add_argument("--fault_trials", type=int, default=10,
                        help="Paired trials per fault in the under-fault test. Was 4, "
                             "with randomization off and falls not excluded.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(args.model)
    # Randomization OFF here, deliberately, unlike the tests below
    env = make_env_from_model_path(args.model, render=False, randomize_init=False)
    rng = np.random.default_rng(args.seed)
    n_act = env.action_space.shape[0]

    print(f"Probing control authority: {args.episodes} episodes, one constant "
          f"offset each (std {args.scale})\n")

    # Baseline: no offset, repeated, to measure how much the outputs vary on their own
    base_env = make_env_from_model_path(args.model, render=False, randomize_init=True)
    base_runs = []
    for k in range(8):
        y, n, _ = run_episode(model, base_env, np.zeros(n_act), args.max_steps, seed=1000 + k)
        if y is not None:
            base_runs.append(y)
    base_env.close()
    base_runs = np.array(base_runs)
    base_sd = base_runs.std(axis=0, ddof=1) if len(base_runs) > 1 else np.zeros(3)
    print("Zero-offset repeats (intrinsic episode-to-episode variation):")
    for i, l in enumerate(LABELS):
        print(f"  {l:<8} mean={base_runs[:, i].mean():+.4f}  sd={base_sd[i]:.4f}")

    X, Y, short = [], [], 0
    for e in range(args.episodes):
        off = rng.normal(0.0, args.scale, size=n_act)
        y, n, _ = run_episode(model, env, off, args.max_steps, seed=e)
        if y is None:
            continue
        if n < 0.5 * args.max_steps:
            short += 1
        X.append(off)
        Y.append(y)
    env.close()

    X, Y = np.array(X), np.array(Y)
    if len(X) < 30:
        print(f"\nOnly {len(X)} usable episodes -- too few. Lower --scale "
              f"(the robot may be falling) or raise --episodes.")
        return

    Xa = np.hstack([X, np.ones((len(X), 1))])
    theta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    B, y0 = theta[:-1].T, theta[-1]
    pred = Xa @ theta
    ss_res = np.sum((Y - pred) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    rel, cos = split_half(X, Y)

    print(f"\nEpisode-level fit  (n={len(X)}"
          + (f", {short} episodes ended early" if short else "") + ")")
    print(f"  output   spread(sd)   vs zero-offset sd   R2     cos(split-half)")
    for i, l in enumerate(LABELS):
        spread = Y[:, i].std(ddof=1)
        ratio = spread / base_sd[i] if base_sd[i] > 1e-9 else float("inf")
        c = cos[i] if cos is not None else float("nan")
        print(f"  {l:<8} {spread:>9.4f}   {ratio:>14.1f}x   {r2[i]:>5.3f}   {c:>+6.3f}")
    print(f"\n  ||B|| = {np.linalg.norm(B):.4f}   baseline y0 = {np.round(y0, 4)}")
    if rel is not None:
        print(f"  split-half rel.diff = {rel:.3f}")

    # direct authority test
    print("\n" + "-" * 70)
    print("DIRECT TEST: request a forward correction, measure what is delivered")
    print(f"  (delta bounded at +/-{args.delta_max} per joint, as in ResidualAdapter)")
    # Randomized initial states and several paired trials per level
    env2 = make_env_from_model_path(args.model, render=False, randomize_init=True)
    dseeds = [2000 + k for k in range(args.direct_trials)]
    refs = {}
    for sd in dseeds:
        y, _, fell = run_episode(model, env2, np.zeros(n_act), args.max_steps, seed=sd)
        if y is not None and not fell:
            refs[sd] = y
    print(f"\n  {len(refs)}/{len(dseeds)} reference episodes survived; each requested "
          f"correction is paired against them")
    print(f"\n  {'requested':>10}{'||delta||':>11}{'clipped':>9}{'delivered':>11}"
          f"{'sd':>8}{'n':>4}{'fell':>6}{'fraction':>10}")
    delivered_abs = []
    level_stats = []          # (requested, mean, sd, n) per level
    for target in (0.10, 0.20, 0.30):
        d = ridge_pinv(B) @ np.array([target, 0.0, 0.0])
        norm_raw = np.linalg.norm(d)
        d_cl = np.clip(d, -args.delta_max, args.delta_max)
        was_clipped = not np.allclose(d, d_cl)
        diffs, n_fell = [], 0
        for sd in dseeds:
            if sd not in refs:
                continue
            y_new, _, fell = run_episode(model, env2, d_cl, args.max_steps, seed=sd)
            if fell or y_new is None:
                n_fell += 1
                continue
            diffs.append(y_new[0] - refs[sd][0])
        if diffs:
            delivered = float(np.mean(diffs))
            spread = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        else:
            delivered, spread = float("nan"), float("nan")
        frac = delivered / target if target else float("nan")
        delivered_abs.append(delivered)
        level_stats.append((target, delivered, spread, len(diffs)))
        print(f"  {target:>10.2f}{norm_raw:>11.3f}{str(was_clipped):>9}"
              f"{delivered:>+11.4f}{spread:>8.4f}{len(diffs):>4}{n_fell:>6}{frac:>9.0%}")
    env2.close()

    max_delivered = float(np.nanmax(delivered_abs)) if delivered_abs else 0.0
    env2.close()
    print("\n  Settled deficits to compensate (Baseline A, 10 seeds): "
          f"torque_limit {SETTLED_DEFICIT['torque_limit']:.3f} m/s, "
          f"joint_lock {SETTLED_DEFICIT['joint_lock']:.3f} m/s,")
    print(f"  sensor faults {SETTLED_DEFICIT['sensor_noise']:.3f}-"
          f"{SETTLED_DEFICIT['sensor_dropout']:.3f} m/s. If the delivered correction "
          "falls well short,")
    print("  a DC offset cannot restore commanded speed regardless of how well B")
    print("  is identified.")

    WORST_DROP = max(SETTLED_DEFICIT["torque_limit"], SETTLED_DEFICIT["joint_lock"])

    # authority on a DAMAGED robot 
    print("\n" + "-" * 70)
    print("AUTHORITY UNDER FAULT (paired: same seed, same fault, delta on vs off)")
    env3 = make_env_from_model_path(args.model, render=False, randomize_init=True)
    FSTEP, MEASURE = 150, 260
    faults = [("torque_limit", 0.2), ("joint_lock", 1.0), ("actuation_delay", 5)]
    print(f"\n  {'fault':<17}{'sev':>6}{'v_fault':>10}{'v_+delta':>10}{'gain':>9}"
          f"{'n':>4}{'fell off':>10}{'fell on':>9}{'healthy':>9}")
    fault_rows = {}
    for fname, fsev in faults:
        gains, fell_off, fell_on, attempted = [], 0, 0, 0
        for k in range(args.fault_trials):
            sd = 3000 + k
            y0f, _, f0 = run_episode(model, env3, np.zeros(n_act), args.max_steps,
                                     seed=sd, fault=fname, fault_severity=fsev,
                                     fault_step=FSTEP, offset_start=FSTEP,
                                     measure_from=MEASURE)
            attempted += 1
            fell_off += bool(f0)
            # Request the shortfall the fault created on this trial
            short = (y0[0] - y0f[0]) if (y0f is not None and not f0) else 0.05
            want = float(np.clip(short, 0.05, 0.40))
            d = np.clip(ridge_pinv(B) @ np.array([want, 0.0, 0.0]),
                        -args.delta_max, args.delta_max)
            y1f, _, f1 = run_episode(model, env3, d, args.max_steps,
                                     seed=sd, fault=fname, fault_severity=fsev,
                                     fault_step=FSTEP, offset_start=FSTEP,
                                     measure_from=MEASURE)
            fell_on += bool(f1)
            if f0 or f1 or y0f is None or y1f is None:
                continue
            gains.append((y0f[0], y1f[0], y1f[0] - y0f[0]))
        fault_rows[fname] = {"n_pairs": len(gains), "attempted": attempted,
                             "fell_off": fell_off, "fell_on": fell_on}
        if not gains:
            print(f"  {fname:<17}{fsev:>6}   (no pair survived both runs; "
                  f"fell off {fell_off}, on {fell_on} of {attempted})")
            continue
        g = np.array(gains)
        fault_rows[fname]["gain"] = round(float(g[:, 2].mean()), 5)
        print(f"  {fname:<17}{fsev:>6}{g[:,0].mean():>10.4f}{g[:,1].mean():>10.4f}"
              f"{g[:,2].mean():>+9.4f}{len(gains):>4}{fell_off:>7}/{attempted:<2}"
              f"{fell_on:>6}/{attempted:<2}{max_delivered:>+9.4f}")
    env3.close()
    print("\n  'fell off' / 'fell on': falls without and with the offset. Velocity")
    print("  gain uses only pairs where neither run fell; a difference in fall")
    print("  counts is reported separately rather than folded into speed.")

    # healthy ceiling, with its uncertainty
    valid = [ls for ls in level_stats if ls[3] > 0 and np.isfinite(ls[1])]
    if valid:
        req_c, max_delivered, sd_c, n_c = max(valid, key=lambda ls: ls[1])
        ci_lo, ci_hi = t_interval(max_delivered, sd_c, n_c)
    else:
        req_c, max_delivered, sd_c, n_c = float("nan"), 0.0, float("nan"), 0
        ci_lo = ci_hi = float("nan")
    by_req = {ls[0]: ls[1] for ls in level_stats}
    growth = (by_req.get(0.30, float("nan")) - by_req.get(0.10, float("nan")))

    # under fault: compare against the healthy INTERVAL, not a ratio
    estimable = {k: v for k, v in fault_rows.items()
                 if v.get("n_pairs", 0) >= MIN_PAIRS and "gain" in v}
    best_fault_gain = max((v["gain"] for v in estimable.values()), default=float("nan"))
    print(f"\n  Best under-fault gain with >= {MIN_PAIRS} surviving pairs: "
          + (f"{best_fault_gain:+.4f} m/s" if np.isfinite(best_fault_gain) else "none"))
    print(f"  Healthy ceiling: {max_delivered:+.4f} m/s, "
          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    if not np.isfinite(best_fault_gain):
        fault_vs_healthy = "not estimable"
    elif not np.isfinite(ci_lo):
        fault_vs_healthy = "healthy interval unavailable"
    elif best_fault_gain > ci_hi:
        fault_vs_healthy = "above the healthy interval"
    elif best_fault_gain < ci_lo:
        fault_vs_healthy = "below the healthy interval"
    else:
        fault_vs_healthy = "within the healthy interval"
    print(f"  -> Under-fault gain is {fault_vs_healthy}.")
    print("     Caveat: under fault each trial requests its own shortfall, while the")
    print("     healthy ceiling comes from requests up to 0.30 m/s. The comparison is")
    print("     like-for-like only where delivered correction saturates well below")
    print("     both request sizes.")

    # verdict: descriptive, interval-based, scoped 
    D = WORST_DROP
    print("\n" + "=" * 70)
    print(f"Delivered forward correction (best level, requested {req_c:.2f}): "
          f"{max_delivered:+.4f} m/s")
    print(f"  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] m/s  =  "
          f"[{ci_lo/D:.0%}, {ci_hi/D:.0%}] of the largest settled deficit ({D:.3f} m/s)")
    if not np.isfinite(ci_hi):
        closes = "cannot be assessed (no usable interval)"
    elif ci_hi < D:
        closes = f"does NOT close the settled deficit (upper 95% bound {ci_hi/D:.0%})"
    elif ci_lo >= D:
        closes = "closes the settled deficit"
    else:
        closes = "cannot be determined at this n -- the interval spans the full deficit"
    print(f"  -> {closes}")
    if np.isfinite(ci_lo) and ci_lo <= 0:
        print("  -> and is not distinguishable from zero")
    print(f"  Tripling the request (0.10 -> 0.30) changed the delivered correction "
          f"by {growth:+.4f} m/s, against +0.20 requested.")
    fc = cos[0] if cos is not None else float("nan")
    print(f"\n  Scope: this is the offset a residual method would apply using the")
    print(f"  identified influence matrix B (forward split-half cosine {fc:.2f}). It is")
    print(f"  not an upper bound on every constant offset, and a non-scaling response")
    print(f"  cannot by itself distinguish the policy cancelling the offset from B")
    print(f"  pointing it in a poor direction.")
    print("=" * 70)

    out = write_authority_sidecar(args.model, {
        "offset_authority_mps": round(float(max_delivered), 5),
        "offset_authority_under_fault_mps": (round(float(best_fault_gain), 5)
                                             if np.isfinite(best_fault_gain) else None),
        "largest_settled_deficit_mps": round(float(WORST_DROP), 5),
        "fraction_of_deficit_healthy": round(float(max_delivered / WORST_DROP), 4),
        "fraction_of_deficit_faulted": (round(float(best_fault_gain / WORST_DROP), 4)
                                        if np.isfinite(best_fault_gain) else None),
        "offset_authority_ci95_mps": [round(float(ci_lo), 5), round(float(ci_hi), 5)],
        "delivered_by_request": {f"{ls[0]:.2f}": {"mean": round(float(ls[1]), 5),
                                                  "sd": round(float(ls[2]), 5),
                                                  "n": int(ls[3])} for ls in level_stats},
        "growth_when_request_tripled_mps": round(float(growth), 5),
        "under_fault_vs_healthy": fault_vs_healthy,
        "forward_r2": round(float(r2[0]), 4),
        "forward_split_half_cos": (round(float(cos[0]), 4) if cos is not None else None),
        "episodes": int(args.episodes),
        "direct_trials": int(args.direct_trials),
        "fault_trials": int(args.fault_trials),
        "under_fault": fault_rows,
        "offset_scale": float(args.scale),
    })
    print(f"\nWrote {out}  (read automatically by probe_amplitude.py)")


if __name__ == "__main__":
    main()