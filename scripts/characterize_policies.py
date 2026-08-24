import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs.quadruped_env import QuadrupedFaultEnv

GRAVITY = 9.81


def total_mass(env):
    m = 0.0
    for link in range(-1, p.getNumJoints(env.robot_id, physicsClientId=env._client)):
        m += p.getDynamicsInfo(env.robot_id, link, physicsClientId=env._client)[0]
    return m


def foot_link_indices(env):
    feet = []
    for j in env.joint_ids:
        name = p.getJointInfo(env.robot_id, j, physicsClientId=env._client)[1].decode()
        if "lower_leg" in name:
            feet.append(j)
    return feet


def duty_and_stride(contact_bool, control_hz):
    if not contact_bool:
        return 0.0, 0.0, []
    duty = sum(contact_bool) / len(contact_bool)
    onsets = [i for i in range(1, len(contact_bool))
              if contact_bool[i] and not contact_bool[i - 1]]
    if len(onsets) < 2:
        return duty, 0.0, onsets
    periods = [(onsets[i + 1] - onsets[i]) / control_hz for i in range(len(onsets) - 1)]
    mean_period = statistics.mean(periods)
    freq = 1.0 / mean_period if mean_period > 0 else 0.0
    return duty, freq, onsets


def run_episode(model, env, steps, mass, feet):
    obs, info = env.reset(seed=None)
    control_dt = env.action_repeat / 240.0

    start = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])

    energy = 0.0
    powers, torques = [], []
    fwd_vels, lat_vels = [], []
    heights, uprights = [], []
    contacts = {f: [] for f in feet}
    prev_action = None
    action_deltas = []

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        if prev_action is not None:
            action_deltas.append(float(np.mean(np.abs(action - prev_action))))
        prev_action = np.array(action)

        obs, reward, terminated, truncated, info = env.step(action)

        # mechanical power this control step 
        step_power = 0.0
        step_torque = 0.0
        for j in env.joint_ids:
            _, vel, _, tau = p.getJointState(env.robot_id, j, physicsClientId=env._client)
            step_power += abs(tau * vel)
            step_torque += abs(tau)
        energy += step_power * control_dt
        powers.append(step_power)
        torques.append(step_torque / len(env.joint_ids))

        # velocities in the robot's own frame 
        _, orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)
        pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)
        lin_vel, _ = p.getBaseVelocity(env.robot_id, physicsClientId=env._client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        fwd_vels.append(float(np.dot(lin_vel, rot[:, 2])))   # local Z = forward
        lat_vels.append(float(np.dot(lin_vel, rot[:, 0])))   # local X = lateral
        heights.append(pos[2])
        uprights.append(float(rot[2, 1]))                     # local Y = up

        # per-foot ground contact 
        cps = p.getContactPoints(env.robot_id, env.plane_id, physicsClientId=env._client)
        touching = {cp[3] for cp in cps}          # index 3 = linkIndexA
        for f in feet:
            contacts[f].append(1 if f in touching else 0)

        if terminated or truncated:
            break

    end = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])

    n = step + 1
    elapsed = n * control_dt
    distance = float(np.linalg.norm(end - start))
    control_hz = 1.0 / control_dt

    cot = energy / (mass * GRAVITY * distance) if distance > 1e-6 else float("nan")

    duties, freqs, onset_sets = [], [], []
    for f in feet:
        d, fr, on = duty_and_stride(contacts[f], control_hz)
        duties.append(d)
        if fr > 0:
            freqs.append(fr)
        onset_sets.append(on)

    # phase of each foot relative to the first, in stride fractions
    phases = []
    if freqs and onset_sets and onset_sets[0]:
        period_steps = control_hz / statistics.mean(freqs)
        ref = onset_sets[0][0]
        for on in onset_sets:
            if on:
                phases.append(((on[0] - ref) % period_steps) / period_steps)
            else:
                phases.append(float("nan"))

    return {
        "steps": n,
        "completed": not (n < steps),
        "distance_m": distance,
        "net_speed_mps": distance / elapsed if elapsed > 0 else 0.0,
        "mean_forward_vel_mps": float(np.mean(fwd_vels)),
        "mean_abs_lateral_vel_mps": float(np.mean(np.abs(lat_vels))),
        "cost_of_transport": cot,
        "mean_power_W": float(np.mean(powers)),
        "peak_power_W": float(np.max(powers)),
        "mean_torque_Nm": float(np.mean(torques)),
        "duty_factor": float(np.mean(duties)) if duties else float("nan"),
        "stride_freq_hz": float(np.mean(freqs)) if freqs else float("nan"),
        "phase_offsets": phases,
        "action_delta": float(np.mean(action_deltas)) if action_deltas else float("nan"),
        "height_mean_m": float(np.mean(heights)),
        "height_sd_m": float(np.std(heights)),
        "upright_sd": float(np.std(uprights)),
    }


def summarize(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(m, sd, dec=3):
    if m is None:
        return "n/a"
    return f"{m:.{dec}f}" if sd is None else f"{m:.{dec}f} ± {sd:.{dec}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="logs/policy_characterization.csv")
    args = parser.parse_args()

    manifest_path = os.path.join(args.results_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found. Run run_multiseed.py first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    converged = sorted(
        (s for s, v in manifest.get("seeds", {}).items()
         if v.get("gait", {}).get("converged")),
        key=int)
    if not converged:
        print("ERROR: no converged seeds in the manifest.")
        return

    print(f"Characterizing {len(converged)} converged seeds: "
          f"{', '.join('seed_' + s for s in converged)}")
    print(f"{args.episodes} episodes x {args.steps} steps each\n")

    per_seed = []
    for sid in converged:
        model_path = os.path.join(args.models_dir, f"seed_{sid}")
        if not os.path.exists(model_path + ".zip"):
            print(f"  seed_{sid}: model missing, skipping")
            continue

        model = PPO.load(model_path)
        env = QuadrupedFaultEnv(render=False)
        mass = total_mass(env)
        feet = foot_link_indices(env)

        eps = [run_episode(model, env, args.steps, mass, feet)
               for _ in range(args.episodes)]
        env.close()

        agg = {"seed": sid, "mass_kg": round(mass, 2)}
        for key in ("net_speed_mps", "mean_forward_vel_mps", "mean_abs_lateral_vel_mps",
                    "cost_of_transport", "mean_power_W", "peak_power_W", "mean_torque_Nm",
                    "duty_factor", "stride_freq_hz", "action_delta",
                    "height_mean_m", "height_sd_m", "upright_sd"):
            m, _ = summarize([e[key] for e in eps])
            agg[key] = round(m, 5) if m is not None else ""
        per_seed.append(agg)

        print(f"  seed_{sid}: CoT={agg['cost_of_transport']}  "
              f"fwd={agg['mean_forward_vel_mps']} m/s  "
              f"net={agg['net_speed_mps']} m/s  "
              f"lat={agg['mean_abs_lateral_vel_mps']} m/s  "
              f"duty={agg['duty_factor']}  stride={agg['stride_freq_hz']} Hz")

    if not per_seed:
        print("No seeds characterized.")
        return

    # across-seed summary 
    print("\n" + "=" * 78)
    print(f"BASE POLICY CHARACTERIZATION -- across {len(per_seed)} converged seeds")
    print("=" * 78)

    def col(key):
        return summarize([r[key] for r in per_seed if r[key] != ""])

    print("\nEFFICIENCY")
    m, sd = col("cost_of_transport")
    print(f"  Cost of Transport      : {fmt(m, sd)}   (dimensionless; "
          f"animals ~0.2-0.5, legged robots ~0.5-2)")
    m, sd = col("mean_power_W");   print(f"  Mean mech. power       : {fmt(m, sd, 1)} W")
    m, sd = col("peak_power_W");   print(f"  Peak mech. power       : {fmt(m, sd, 1)} W")
    m, sd = col("mean_torque_Nm"); print(f"  Mean |torque| / joint  : {fmt(m, sd, 2)} Nm "
                                          f"(limit 20 Nm)")

    print("\nGAIT STRUCTURE")
    m, sd = col("duty_factor");    print(f"  Duty factor            : {fmt(m, sd)}   "
                                          f"(~0.5 trot-like; higher = more support time)")
    m, sd = col("stride_freq_hz"); print(f"  Stride frequency       : {fmt(m, sd, 2)} Hz")
    m, sd = col("action_delta");   print(f"  Action smoothness      : {fmt(m, sd, 4)} "
                                          f"(mean |a_t - a_(t-1)|; lower = smoother)")

    print("\nDRIFT  (do reported speed and tracked velocity agree?)")
    net_m, net_sd = col("net_speed_mps")
    fwd_m, fwd_sd = col("mean_forward_vel_mps")
    lat_m, lat_sd = col("mean_abs_lateral_vel_mps")
    print(f"  Net displacement speed : {fmt(net_m, net_sd)} m/s   <- gait_check reports this")
    print(f"  Mean FORWARD velocity  : {fmt(fwd_m, fwd_sd)} m/s   <- the reward tracks this")
    print(f"  Mean |lateral| velocity: {fmt(lat_m, lat_sd)} m/s")
    if fwd_m and abs(fwd_m) > 1e-6:
        ratio = lat_m / abs(fwd_m)
        gap = abs(net_m - fwd_m)
        print(f"  lateral/forward ratio  : {ratio:.0%}")
        if ratio > 0.15:
            print("  -> DRIFTING. Report forward velocity, not net displacement, as the")
            print("     tracked quantity -- they are different measurements here.")
        elif gap > 0.05:
            print(f"  -> {gap:.3f} m/s gap between the two; worth a sentence in the paper.")
        else:
            print("  -> walks essentially straight; the two measures agree.")

    print("\nSTABILITY")
    m, sd = col("height_mean_m"); print(f"  Base height            : {fmt(m, sd)} m")
    m, sd = col("height_sd_m");   print(f"  Height variation (sd)  : {fmt(m, sd, 4)} m")
    m, sd = col("upright_sd");    print(f"  Upright-alignment sd   : {fmt(m, sd, 4)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed)
    print(f"\nWrote per-seed characterization to {args.out}")
    print("\nThese pre-fault numbers are the reference for the 'energy cost of")
    print("adaptation' metric: re-measure power and CoT during the adaptation")
    print("window and compare against these.")


if __name__ == "__main__":
    main()
