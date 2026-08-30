import argparse
import csv
import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path

# Experiment configuration

# Default: one severity per fault type (matches the original experiment).
FAULT_CONFIGS = [
    {"type": "torque_limit", "severity": 0.2},
    {"type": "joint_lock", "severity": 1.0},
    {"type": "actuation_delay", "severity": 10},
    {"type": "sensor_dropout", "severity": 1.0},
    {"type": "sensor_noise", "severity": 0.3},
]

# --sweep: vary severity to get a dose-response curve. Far more informative than a single point
FAULT_SWEEP = [
    {"type": "torque_limit", "severity": s} for s in (0.5, 0.3, 0.2, 0.1, 0.05)
] + [
    {"type": "actuation_delay", "severity": s} for s in (5, 10, 20, 40)
] + [
    {"type": "sensor_noise", "severity": s} for s in (0.1, 0.3, 0.6, 1.0)
] + [
    # joint_lock and sensor_dropout have no continuous severity axis at one joint
    {"type": "joint_lock", "severity": 1.0},
    {"type": "sensor_dropout", "severity": 1.0},
]

# Fault onset is DRAWN PER TRIAL, not fixed
FAULT_ONSET_MIN = 150
FAULT_ONSET_MAX = 250
POST_FAULT_WINDOW = 300         # observe ~5 s afterwards
BASELINE_WINDOW = 50            # steps averaged to define pre-fault speed
SMOOTHING_WINDOW = 30           # ~0.5 s rolling mean -- spans a full stride
SUSTAINED_STEPS = 30            # recovery must hold this long (~0.5 s)
RECOVERY_TOLERANCE = 0.15       # within 15% of pre-fault speed
CONTROL_HZ = 60.0               # 240 Hz physics / action_repeat 4


# Metric computation (pure -- no simulator, so it is directly testable)

def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def analyze_trace(pre_fault_vels, post_fault_vels, fell,
                  smoothing_window=SMOOTHING_WINDOW,
                  sustained_steps=SUSTAINED_STEPS,
                  tolerance=RECOVERY_TOLERANCE,
                  control_hz=CONTROL_HZ):

    base_chunk = pre_fault_vels[-BASELINE_WINDOW:] if pre_fault_vels else [0.0]
    baseline = statistics.mean(base_chunk)
    baseline_sd = statistics.stdev(base_chunk) if len(base_chunk) > 1 else 0.0

    result = {
        "baseline_vel": baseline,
        "baseline_vel_sd": baseline_sd,
        "velocity_drop_frac": 0.0,
        "degraded": False,
        "recovery_status": "no_degradation",
        "recovery_time_s": None,
    }

    if abs(baseline) < 1e-6 or not post_fault_vels:
        result["recovery_status"] = "fell" if fell else "no_degradation"
        return result

    smoothed = rolling_mean(post_fault_vels, smoothing_window)
    lower = baseline * (1.0 - tolerance)

    # How far did the smoothed speed fall below the pre-fault level?
    min_smoothed = min(smoothed)
    result["velocity_drop_frac"] = (baseline - min_smoothed) / abs(baseline)

    # First point where degradation is actually visible.
    deg_idx = next((i for i, v in enumerate(smoothed) if v < lower), None)
    result["degraded"] = deg_idx is not None

    if fell:
        # A fall is never a recovery, whatever the velocity did beforehand.
        result["recovery_status"] = "fell"
        return result

    if deg_idx is None:
        # The fault never measurably slowed the robot. This is a real and interesting outcome
        result["recovery_status"] = "no_degradation"
        return result

    # Search from the point of degradation onward for a SUSTAINED return.
    for i in range(deg_idx, len(smoothed) - sustained_steps + 1):
        window = smoothed[i:i + sustained_steps]
        if all(abs(v - baseline) <= tolerance * abs(baseline) for v in window):
            result["recovery_status"] = "recovered"
            result["recovery_time_s"] = i / control_hz
            return result

    result["recovery_status"] = "no_recovery"
    return result


# Trial execution

def run_trial(model, env, fault_config, seed, n_joints=1):
    obs, info = env.reset(seed=seed)

    trial_rng = np.random.default_rng(seed)
    onset_step = int(trial_rng.integers(FAULT_ONSET_MIN, FAULT_ONSET_MAX))

    pre_fault_vels = []
    for step in range(onset_step):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        pre_fault_vels.append(info["forward_vel"])
        if terminated or truncated:
            return None  # fell before the fault -- says nothing about fault response

    # Inject. n_joints>1 calls trigger_fault repeatedly, which is the severity axis for the otherwise-binary joint_lock and sensor_dropout faults.
    for _ in range(max(1, n_joints)):
        env.trigger_fault(fault_config["type"], severity=fault_config["severity"])

    start_pos = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])

    post_fault_vels = []
    fell = False
    steps_after = 0

    for step in range(POST_FAULT_WINDOW):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        post_fault_vels.append(info["forward_vel"])
        steps_after = step + 1
        if terminated:
            fell = True
            break
        if truncated:
            break

    end_pos = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])
    distance = float(np.linalg.norm(end_pos - start_pos))

    metrics = analyze_trace(pre_fault_vels, post_fault_vels, fell)

    return {
        "fault_type": fault_config["type"],
        "severity": fault_config["severity"],
        "n_joints": n_joints,
        "seed": seed,
        "onset_step": onset_step,
        "baseline_vel": round(metrics["baseline_vel"], 5),
        "baseline_vel_sd": round(metrics["baseline_vel_sd"], 5),
        "velocity_drop_frac": round(metrics["velocity_drop_frac"], 5),
        "degraded": metrics["degraded"],
        "recovery_status": metrics["recovery_status"],
        "recovery_time_s": round(metrics["recovery_time_s"], 4)
                           if metrics["recovery_time_s"] is not None else "",
        "post_fault_distance_m": round(distance, 5),
        "post_fault_steps_survived": steps_after,
        "fell": fell,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/base_policy")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--out", type=str, default="logs/baseline_fault_results.csv")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep fault severity instead of using one value per type. "
                             "Produces a dose-response curve.")
    # Initial-state randomization now lives in the environment (randomize_init), and is loaded from the training config, so it always matches training.
    parser.add_argument("--n_joints", type=int, default=1,
                        help="How many joints each fault affects. >1 is the severity "
                             "axis for joint_lock and sensor_dropout.")
    args = parser.parse_args()

    configs = FAULT_SWEEP if args.sweep else FAULT_CONFIGS

    model = PPO.load(args.model)
    env = make_env_from_model_path(args.model, render=False)


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    discarded = 0

    for cfg in configs:
        label = f"{cfg['type']} (severity={cfg['severity']}, n_joints={args.n_joints})"
        print(f"\n=== {label} ===")
        status_counts = {}
        for trial in range(args.trials):
            r = run_trial(model, env, cfg, seed=trial, n_joints=args.n_joints)
            if r is None:
                discarded += 1
                continue
            results.append(r)
            status_counts[r["recovery_status"]] = status_counts.get(r["recovery_status"], 0) + 1
        total = sum(status_counts.values())
        if total:
            summary = "  ".join(f"{k}={v}({v/total:.0%})" for k, v in sorted(status_counts.items()))
            print(f"  {summary}")
            drops = [r["velocity_drop_frac"] for r in results if r["fault_type"] == cfg["type"]
                     and r["severity"] == cfg["severity"]]
            if drops:
                print(f"  mean velocity drop: {statistics.mean(drops):.1%}")

    env.close()

    if results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} trials to {args.out}")
        if discarded:
            print(f"Discarded {discarded} trials (robot fell before fault onset)")
    else:
        print("\nNo valid trials -- check that the policy survives "
              f"{FAULT_ONSET_MAX} steps before any fault is injected.")


if __name__ == "__main__":
    main()