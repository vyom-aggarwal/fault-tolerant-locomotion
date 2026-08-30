import argparse
import json
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gait_check(model_path, steps=1000, target_velocity=0.5, tol=0.30,
               n_episodes=5, survival_threshold=0.8, init_noise=0.05):
    import pybullet as p
    from stable_baselines3 import PPO
    from envs.quadruped_env import QuadrupedFaultEnv

    model = PPO.load(model_path)
    # Without init noise every "episode" here is identical, making the multi-episode survival rate a single measurement repeated n times.
    env = QuadrupedFaultEnv(render=False, init_noise_scale=init_noise)

    ep_survived, ep_speeds, ep_steps, ep_ranges, ep_contact_var = [], [], [], [], []
    ep_net_speeds, ep_lat, ep_yaw = [], [], []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        start_pos = np.array(p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env._client)[0])
        joint_history, contact_counts = [], []
        fwd_vels, lat_vels, yaw_rates = [], [], []

        for step in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            joint_history.append([
                p.getJointState(env.robot_id, j, physicsClientId=env._client)[0]
                for j in env.joint_ids
            ])
            contact_counts.append(len(p.getContactPoints(
                env.robot_id, env.plane_id, physicsClientId=env._client)))
            fwd_vels.append(info.get("forward_vel", 0.0))
            lat_vels.append(info.get("lateral_vel", 0.0))
            yaw_rates.append(info.get("yaw_rate", 0.0))
            if terminated or truncated:
                break

        end_pos = np.array(p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env._client)[0])

        steps_survived = step + 1
        elapsed = steps_survived * 4 / 240.0      # action_repeat=4, 240 Hz physics
        displacement = float(np.linalg.norm(end_pos - start_pos))

        ep_steps.append(steps_survived)
        ep_survived.append(steps_survived >= 0.9 * steps)
        # Judge tracking on FORWARD velocity in the body frame, not net displacement
        ep_speeds.append(float(np.mean(fwd_vels)) if fwd_vels else 0.0)
        ep_net_speeds.append(displacement / elapsed if elapsed > 0 else 0.0)
        ep_lat.append(float(np.mean(np.abs(lat_vels))) if lat_vels else 0.0)
        ep_yaw.append(float(np.mean(np.abs(yaw_rates))) if yaw_rates else 0.0)
        jh = np.array(joint_history)
        ep_ranges.append(float((jh.max(axis=0) - jh.min(axis=0)).mean()) if len(jh) else 0.0)
        ep_contact_var.append(
            float(np.mean(np.diff(contact_counts) != 0)) if len(contact_counts) > 1 else 0.0)

    env.close()

    survival_rate = sum(ep_survived) / len(ep_survived)

    # Speed is only meaningful on episodes that ran to completion
    full_speeds = [sp for sp, ok in zip(ep_speeds, ep_survived) if ok]
    mean_speed = float(np.mean(full_speeds)) if full_speeds else float(np.mean(ep_speeds))
    speed_sd = float(np.std(full_speeds, ddof=1)) if len(full_speeds) > 1 else 0.0

    tracking_error = abs(mean_speed - target_velocity) / target_velocity if target_velocity else 1.0

    stable = survival_rate >= survival_threshold
    on_target = tracking_error <= tol
    converged = stable and on_target

    if converged:
        failure_mode = ""
    elif not stable and not on_target:
        failure_mode = "unstable+off_target"
    elif not stable:
        failure_mode = "unstable"
    else:
        failure_mode = "off_target"

    return {
        "converged": bool(converged),
        "failure_mode": failure_mode,
        "n_episodes": n_episodes,
        "survival_rate": round(survival_rate, 3),
        "stable": bool(stable),
        "on_target": bool(on_target),
        "tracking_error": round(tracking_error, 4),
        "steps_survived": int(np.mean(ep_steps)),
        "steps_survived_per_episode": [int(x) for x in ep_steps],
        "fell": bool(not all(ep_survived)),
        "mean_speed_mps": round(mean_speed, 4),
        "mean_net_speed_mps": round(float(np.mean(ep_net_speeds)), 4) if ep_net_speeds else 0.0,
        "mean_abs_lateral_vel": round(float(np.mean(ep_lat)), 4) if ep_lat else 0.0,
        "mean_abs_yaw_rate": round(float(np.mean(ep_yaw)), 4) if ep_yaw else 0.0,
        "speed_sd_mps": round(speed_sd, 4),
        "mean_joint_range_rad": round(float(np.mean(ep_ranges)), 4),
        "contact_variation": round(float(np.mean(ep_contact_var)), 4),
    }


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..N-1)")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=100, help="Fault trials per fault type, per seed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--skip_training", action="store_true",
                        help="Evaluate existing policies without retraining")
    parser.add_argument("--gait_steps", type=int, default=1000)
    parser.add_argument("--target_velocity", type=float, default=0.5,
                        help="Commanded FORWARD velocity (vx) the policy tracks.")
    parser.add_argument("--convergence_tol", type=float, default=0.30,
                        help="A seed counts as converged only if it survives the episode AND "
                             "tracks the commanded velocity within this fraction. The old "
                             "criterion (speed >= 0.2 absolute) admitted policies running at "
                             "60%% of target, whose fault responses are not comparable to "
                             "on-target policies.")
    parser.add_argument("--gait_episodes", type=int, default=5,
                        help="Episodes used to judge convergence. 1 makes the reported "
                             "convergence rate an n=1 measurement per seed.")
    parser.add_argument("--survival_threshold", type=float, default=0.8,
                        help="Fraction of gait-check episodes a policy must survive.")
    parser.add_argument("--init_noise", type=float, default=0.05,
                        help="Initial joint-angle perturbation (rad) used during EVALUATION "
                             "so repeated episodes/trials are genuinely independent.")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a model already exists at the target path.")
    parser.add_argument("--log_format", type=str, default="csv",
                        choices=["csv", "tensorboard", "none"],
                        help="Passed through to training. csv is the robust default.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    manifest_path = os.path.join(args.results_dir, "manifest.json")
    manifest = {"config": vars(args), "seeds": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["config"] = vars(args)
        except json.JSONDecodeError:
            pass

    started = time.time()

    for seed in range(args.seeds):
        print("\n" + "=" * 70)
        print(f"SEED {seed} / {args.seeds - 1}")
        print("=" * 70)

        model_path = os.path.join(args.models_dir, f"seed_{seed}")
        seed_results_dir = os.path.join(args.results_dir, f"seed_{seed}")
        os.makedirs(seed_results_dir, exist_ok=True)

        # train 
        if not args.skip_training:
            existing_cfg_path = model_path + "_trainconfig.json"
            stale = False
            if os.path.exists(model_path + ".zip") and not args.force_retrain:
                if os.path.exists(existing_cfg_path):
                    with open(existing_cfg_path) as f:
                        prev = json.load(f)
                    mismatches = {k: (prev.get(k), v) for k, v in
                                  (("timesteps", args.timesteps),
                                   ("ent_coef", args.ent_coef),
                                   ("n_envs", args.n_envs))
                                  if prev.get(k) != v}
                    if mismatches:
                        stale = True
                        print(f"[seed {seed}] EXISTING MODEL WAS TRAINED WITH DIFFERENT SETTINGS:")
                        for k, (was, now) in mismatches.items():
                            print(f"    {k}: existing={was}  requested={now}")
                        print(f"  Mixing training budgets across seeds confounds the experiment.")
                        print(f"  Retraining this seed with the requested settings.")
                else:
                    stale = True
                    print(f"[seed {seed}] existing model has no _trainconfig.json -- provenance "
                          f"unknown, so it cannot be trusted to match this batch. Retraining.")

            if os.path.exists(model_path + ".zip") and not stale and not args.force_retrain:
                print(f"[seed {seed}] {model_path}.zip already exists with matching settings "
                      f"-- skipping training.")
            else:
                ok = run([sys.executable, "scripts/train_base_policy.py",
                          "--seed", str(seed),
                          "--timesteps", str(args.timesteps),
                          "--n_envs", str(args.n_envs),
                          "--ent_coef", str(args.ent_coef),
                          "--save_path", model_path,
                          "--log_dir", os.path.join("logs", f"seed_{seed}"),
                          "--log_format", args.log_format])
                if not ok:
                    print(f"[seed {seed}] TRAINING FAILED -- skipping this seed")
                    manifest["seeds"][str(seed)] = {"status": "train_failed"}
                    continue

        if not os.path.exists(model_path + ".zip"):
            print(f"[seed {seed}] no model at {model_path}.zip -- skipping")
            manifest["seeds"][str(seed)] = {"status": "missing_model"}
            continue

        # gait check 
        print(f"\n[seed {seed}] checking whether the policy walks...")
        try:
            gait = gait_check(model_path, steps=args.gait_steps,
                              target_velocity=args.target_velocity,
                              tol=args.convergence_tol,
                              n_episodes=args.gait_episodes,
                              survival_threshold=args.survival_threshold,
                              init_noise=args.init_noise)
        except Exception as exc:
            print(f"[seed {seed}] gait check errored: {exc}")
            manifest["seeds"][str(seed)] = {"status": "gait_check_failed", "error": str(exc)}
            continue

        status = "converged" if gait["converged"] else "NOT converged"
        detail = f" [{gait['failure_mode']}]" if gait['failure_mode'] else ""
        print(f"[seed {seed}] {status}{detail}: fwd={gait['mean_speed_mps']} +/- "
              f"{gait['speed_sd_mps']} m/s (err {gait['tracking_error']:.0%} vs "
              f"{args.target_velocity}), net={gait.get('mean_net_speed_mps')} m/s, "
              f"lat={gait.get('mean_abs_lateral_vel')} m/s, "
              f"yaw={gait.get('mean_abs_yaw_rate')} rad/s, "
              f"survived {gait['survival_rate']:.0%} of {gait['n_episodes']} eps")

        manifest["seeds"][str(seed)] = {"status": "ok", "gait": gait,
                                        "model_path": model_path + ".zip"}

        # A non-converged seed is a REPORTABLE RESULT, not a failure of the experiment. It is still useful to know how the policy responds to faults, but it is not comparable to a policy that actually learned to walk.
        if not gait["converged"]:
            print(f"[seed {seed}] skipping fault evaluation -- policy did not learn to walk. "
                  f"This counts toward the reported convergence rate.")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            continue

        # fault evaluation 
        out_csv = os.path.join(seed_results_dir, "baseline_fault_results.csv")
        ok = run([sys.executable, "scripts/baseline_fault_eval.py",
                  "--model", model_path,
                  "--trials", str(args.trials),
                  "--init_noise", str(args.init_noise),
                  "--out", out_csv])
        manifest["seeds"][str(seed)]["baseline_csv"] = out_csv if ok else None

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # summary 
    elapsed_min = (time.time() - started) / 60.0
    ok_seeds = [s for s, v in manifest["seeds"].items() if v.get("status") == "ok"]
    converged = [s for s in ok_seeds if manifest["seeds"][s]["gait"]["converged"]]

    print("\n" + "=" * 70)
    print(f"DONE in {elapsed_min:.1f} min")
    print(f"Seeds run: {len(ok_seeds)} | converged to a walking gait: "
          f"{len(converged)}/{len(ok_seeds)}")
    if ok_seeds:
        speeds = [manifest["seeds"][s]["gait"]["mean_speed_mps"] for s in converged]
        if speeds:
            print(f"Converged-seed speed: {np.mean(speeds):.3f} +/- "
                  f"{np.std(speeds, ddof=1) if len(speeds) > 1 else 0.0:.3f} m/s")
    print(f"Manifest: {manifest_path}")
    print("\nNext: python scripts/aggregate_seeds.py")


if __name__ == "__main__":
    main()