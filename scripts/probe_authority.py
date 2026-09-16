import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

LABELS = ("forward", "lateral", "yaw")


def run_episode(model, env, offset, max_steps, seed, settle=40):
    obs, info = env.reset(seed=seed)
    fwd, lat, yaw = [], [], []
    for t in range(max_steps):
        base, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(np.clip(base + offset, -1.0, 1.0))
        if t >= settle:                       # drop the startup transient
            fwd.append(info["forward_vel"])
            lat.append(info["lateral_vel"])
            yaw.append(info["yaw_rate"])
        if term or trunc:
            break
    if not fwd:
        return None, t + 1
    return np.array([np.mean(fwd), np.mean(lat), np.mean(yaw)]), t + 1


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False, randomize_init=False)
    rng = np.random.default_rng(args.seed)
    n_act = env.action_space.shape[0]

    print(f"Probing control authority: {args.episodes} episodes, one constant "
          f"offset each (std {args.scale})\n")
    
    base_env = make_env_from_model_path(args.model, render=False, randomize_init=True)
    base_runs = []
    for k in range(8):
        y, n = run_episode(model, base_env, np.zeros(n_act), args.max_steps, seed=1000 + k)
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
        y, n = run_episode(model, env, off, args.max_steps, seed=e)
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
    def ridge_pinv(Bm, lam=0.05):
        return Bm.T @ np.linalg.inv(Bm @ Bm.T + lam * np.eye(Bm.shape[0]))

    print("\n" + "-" * 70)
    print("DIRECT TEST: request a forward correction, measure what is delivered")
    print(f"  (delta bounded at +/-{args.delta_max} per joint, as in ResidualAdapter)")
    env2 = make_env_from_model_path(args.model, render=False, randomize_init=False)
    y_ref, _ = run_episode(model, env2, np.zeros(n_act), args.max_steps, seed=7)
    print(f"\n  {'requested':>10}{'||delta||':>11}{'clipped':>9}{'delivered':>11}{'fraction':>10}")
    delivered_abs = []
    for target in (0.10, 0.20, 0.30):
        d = ridge_pinv(B) @ np.array([target, 0.0, 0.0])
        norm_raw = np.linalg.norm(d)
        d_cl = np.clip(d, -args.delta_max, args.delta_max)
        was_clipped = not np.allclose(d, d_cl)
        y_new, _ = run_episode(model, env2, d_cl, args.max_steps, seed=7)
        delivered = (y_new[0] - y_ref[0]) if y_new is not None else float("nan")
        frac = delivered / target if target else float("nan")
        delivered_abs.append(delivered)
        print(f"  {target:>10.2f}{norm_raw:>11.3f}{str(was_clipped):>9}"
              f"{delivered:>+11.4f}{frac:>9.0%}")
    max_delivered = float(np.nanmax(delivered_abs)) if delivered_abs else 0.0
    env2.close()
    print("\n  Fault-induced drops to compensate: torque_limit ~0.27 m/s, "
          "joint_lock ~0.22 m/s,")
    print("  sensor faults ~0.12 m/s. If delivered correction falls well short of")
    print("  those, a DC offset cannot restore commanded speed regardless of how")
    print("  well B is identified.")

    # verdict 
    print("\n" + "=" * 70)
    # Largest fault-induced drop the method would have to undo (m/s).
    WORST_DROP = 0.27
    sufficient = max_delivered >= 0.6 * WORST_DROP
    partial = max_delivered >= 0.25 * WORST_DROP

    fwd_ok = (cos is not None and cos[0] > 0.7 and r2[0] > 0.4)
    fwd_weak = (cos is not None and cos[0] > 0.35 and r2[0] > 0.2)
    others_ok = (cos is not None and cos[1] > 0.6 and cos[2] > 0.6)
    print(f"Best delivered forward correction: {max_delivered:+.4f} m/s "
          f"({max_delivered/WORST_DROP:.0%} of the worst fault-induced drop)\n")

    if not sufficient and not partial:
        print("INSUFFICIENT AUTHORITY. Even at the delta bound, a constant joint")
        print("offset moves forward velocity far less than the faults remove it.")
        print("Identification quality is beside the point: no achievable delta")
        print("restores commanded speed. The residual must act on a channel that")
        print("reaches gait timing or amplitude, not joint posture.")
    elif not sufficient:
        print("PARTIAL AUTHORITY. The correction moves forward velocity in the")
        print("right direction but cannot fully undo the larger faults. Expect")
        print("recovery on mild faults and partial recovery on severe ones --")
        print("which is a reportable result, not a failure, provided it is framed")
        print("as a bound on what this parameterisation can do.")
    elif fwd_weak and not fwd_ok:
        print("SUFFICIENT AUTHORITY, WEAKLY IDENTIFIED.")
        print("The achievable correction covers the fault-induced drop, but the")
        print("forward row of B is noisy. Improve identification (more episodes,")
        print("larger dither) before running the evaluation.")
        print("The forward row is positive and explains real variance, so this is")
        print("not a missing control channel -- it is a poorly identified and")
        print("low-gain one. Whether that is sufficient is answered by the direct")
        print("test above, not by these correlations: compare the delivered")
        print("correction against the fault-induced velocity drops.")
    elif fwd_ok:
        print("AUTHORITY CONFIRMED and well identified for forward velocity.")
        print("Averaging over whole episodes resolves the response, so the earlier")
        print("failure was signal-to-noise, not a missing control channel.")
        print("Fix identification: raise --dither and --steps in")
        print("fit_influence_matrix.py, or widen its smoothing window.")
    elif others_ok:
        print("STRUCTURAL LIMIT: lateral and yaw respond to a constant offset;")
        print("forward velocity does not, even averaged over whole episodes.")
        print("A DC joint offset changes posture, and forward speed is set by")
        print("stride frequency and length -- properties of the limit cycle that")
        print("a constant offset does not reach. More data will not fix this.")
        print("The residual parameterisation needs a channel that affects gait")
        print("timing or amplitude, not just posture.")
    else:
        print("NO CLEAR AUTHORITY on any output at this offset scale.")
        print("Try a larger --scale. If the robot falls instead of responding,")
        print("additive action offsets may be the wrong actuation channel entirely.")
    print("=" * 70)


if __name__ == "__main__":
    main()