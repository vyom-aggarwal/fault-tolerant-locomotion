import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

SMOOTH_WINDOW = 30     # must match ResidualAdapter.smooth_window
RESPONSE_LAG = 3       # must match ResidualAdapter.response_lag


def collect(model, env, steps, dither, rng, hold=120, settle=60):
    obs, info = env.reset()
    deltas, ys = [], []
    n_act = env.action_space.shape[0]
    current = rng.normal(0.0, dither, size=n_act)
    for t in range(steps):
        if t % hold == 0:
            current = rng.normal(0.0, dither, size=n_act)
        base, _ = model.predict(obs, deterministic=True)
        d = current
        obs, r, term, trunc, info = env.step(np.clip(base + d, -1.0, 1.0))
        deltas.append(d)
        ys.append([info["forward_vel"], info["lateral_vel"], info["yaw_rate"]])
        if term or trunc:
            obs, info = env.reset()
            # drop the transient after a reset rather than regressing across it
            for _ in range(settle):
                base, _ = model.predict(obs, deterministic=True)
                d2 = current
                obs, r, term2, trunc2, info = env.step(np.clip(base + d2, -1.0, 1.0))
                deltas.append(d2)
                ys.append([info["forward_vel"], info["lateral_vel"], info["yaw_rate"]])
                if term2 or trunc2:
                    obs, info = env.reset()
                    break
    return np.array(deltas), np.array(ys)


def fit(deltas, ys, w=SMOOTH_WINDOW, lag=RESPONSE_LAG):
    X, Y = [], []
    for t in range(w + lag, len(ys)):
        Y.append(np.mean(ys[t - w + 1:t + 1], axis=0))
        lo, hi = t - lag - w + 1, t - lag + 1
        X.append(np.mean(deltas[lo:hi], axis=0))
    X, Y = np.array(X), np.array(Y)

    Xa = np.hstack([X, np.ones((len(X), 1))])          # bias column
    theta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)      # (n_act+1, 3)
    B = theta[:-1].T                                    # (3, n_act)
    y0 = theta[-1]

    pred = Xa @ theta
    ss_res = np.sum((Y - pred) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    return B, y0, r2, len(X)


def process(model_path, steps, dither, seed, hold=120):
    model = PPO.load(model_path)
    env = make_env_from_model_path(model_path, render=False)
    rng = np.random.default_rng(seed)
    deltas, ys = collect(model, env, steps, dither, rng, hold=hold)
    env.close()

    B, y0, r2, n = fit(deltas, ys)
    out = model_path + "_influence.npz"
    np.savez(out, B=B, y0=y0, r2=r2, n_samples=n, dither=dither, steps=steps,
             hold=hold, smooth_window=SMOOTH_WINDOW, response_lag=RESPONSE_LAG)

    labels = ("forward", "lateral", "yaw")
    print(f"  {os.path.basename(model_path)}: n={n}, "
          f"R2 = " + ", ".join(f"{l}={v:.3f}" for l, v in zip(labels, r2)))
    print(f"    baseline y0 = {np.round(y0, 4)}   ||B|| = {np.linalg.norm(B):.4f}")
    if r2[0] < 0.05:
        print(f"    WARNING: forward-velocity R2 is very low. The linear influence "
              f"model explains little variance, so B^+ may point corrections in "
              f"poorly-chosen directions. Try a larger --dither or more --steps.")
    print(f"    saved -> {out}")
    return B, y0, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Fit for every converged seed in results/manifest.json")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--steps", type=int, default=24000,
                        help="Dithered steps to collect. With a 120-step hold this "
                             "is ~200 independent excitation levels.")
    parser.add_argument("--dither", type=float, default=0.10,
                        help="Std of the action dither. Too small gives poor "
                             "signal-to-noise; too large perturbs the gait being measured.")
    parser.add_argument("--hold", type=int, default=120,
                        help="Steps to hold each dither sample. Long holds excite the "
                             "low frequencies the fit depends on; per-step dither does not.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        mpath = os.path.join(args.results_dir, "manifest.json")
        if not os.path.exists(mpath):
            print(f"ERROR: {mpath} not found.")
            return
        with open(mpath) as f:
            manifest = json.load(f)
        seeds = sorted((s for s, v in manifest.get("seeds", {}).items()
                        if v.get("gait", {}).get("converged")), key=int)
        print(f"Fitting influence matrices for {len(seeds)} converged seeds\n")
        r2s = []
        for sid in seeds:
            mp = os.path.join(args.models_dir, f"seed_{sid}")
            if os.path.exists(mp + ".zip"):
                _, _, r2 = process(mp, args.steps, args.dither, args.seed, args.hold)
                r2s.append(r2)
        if r2s:
            r2s = np.array(r2s)
            print(f"\nAcross seeds, R2 mean +/- sd:")
            for i, l in enumerate(("forward", "lateral", "yaw")):
                print(f"  {l:<8}{r2s[:,i].mean():.3f} +/- {r2s[:,i].std(ddof=1):.3f}")
    else:
        if not args.model:
            print("Pass --model or --all")
            return
        process(args.model, args.steps, args.dither, args.seed, args.hold)


if __name__ == "__main__":
    main()
