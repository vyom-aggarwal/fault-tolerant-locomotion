import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from envs.quadruped_env import QuadrupedFaultEnv, save_env_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000,
                        help="1.5M is the budget that produced a reliable gait. 500k "
                             "undertrains: it yields slow, off-target policies and a "
                             "lower convergence rate.")
    parser.add_argument("--n_envs", type=int, default=4, help="Parallel envs; keep small on a laptop CPU")
    parser.add_argument("--save_path", type=str, default="models/base_policy")
    parser.add_argument("--log_dir", type=str, default="logs/base_policy")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient; higher values encourage more exploration")
    parser.add_argument("--log_format", type=str, default="csv",
                        choices=["csv", "tensorboard", "none"],
                        help="csv (default) writes progress.csv -- robust, and directly "
                             "plottable for learning-curve figures. tensorboard uses an "
                             "async writer thread that has proven fragile on Windows "
                             "(it can die mid-run and take training down with it).")
    parser.add_argument("--target_velocity", type=float, default=0.5,
                        help="Commanded forward velocity vx. The full command is "
                             "(vx, 0, 0): move forward at this speed, no lateral "
                             "drift, no turning.")
    parser.add_argument("--max_torque", type=float, default=20.0,
                        help="Per-joint torque limit (Nm). The URDF allows up to 100.")
    parser.add_argument("--action_scale", type=float, default=0.5,
                        help="Max per-joint deviation (rad) from the standing pose.")
    parser.add_argument("--no_randomize_init", action="store_true",
                        help="Disable initial-state randomization. Debugging only -- "
                             "without it, repeated trials are not independent samples.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed. Set explicitly for every run -- results are not "
                             "comparable across runs without it.")
    args = parser.parse_args()

    # Seed everything reachable: python random, numpy, torch (CPU+CUDA).
    set_random_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Guard against silently destroying a good policy by retraining over it.
    if os.path.exists(args.save_path + ".zip"):
        print(f"WARNING: {args.save_path}.zip already exists and will be OVERWRITTEN "
              f"when this run finishes. Ctrl+C now and pass a different --save_path "
              f"if you meant to keep it.")

    print(f"Training with seed={args.seed}, timesteps={args.timesteps}, "
          f"n_envs={args.n_envs}, ent_coef={args.ent_coef}")

    # seed= here gives each parallel env a distinct but deterministic sub-seed
    env_kwargs = dict(
        command_vel=(args.target_velocity, 0.0, 0.0),
        max_torque=args.max_torque,
        action_scale=args.action_scale,
        randomize_init=not args.no_randomize_init,
    )

    # Written before training so it exists even if the run is interrupted and only checkpoints survive
    cfg_path = save_env_config(os.path.dirname(args.save_path) or ".",
                               {**env_kwargs, "max_episode_steps": 1000, "action_repeat": 4})
    print(f"Saved env config to {cfg_path}")

    def make_env():
        return QuadrupedFaultEnv(render=False, **env_kwargs)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        seed=args.seed,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    # Configure logging AFTER construction so the format is explicit and the tensorboard writer thread is never created unless asked for.
    if args.log_format == "csv":
        model.set_logger(configure(args.log_dir, ["stdout", "csv"]))
        print(f"Logging to {os.path.join(args.log_dir, 'progress.csv')}")
    elif args.log_format == "tensorboard":
        model.set_logger(configure(args.log_dir, ["stdout", "tensorboard"]))
    # "none" -> keep SB3's default stdout-only logger

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix=os.path.basename(args.save_path) + "_ckpt",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
    model.save(args.save_path)

    # Sidecar recording HOW this policy was trained
    with open(args.save_path + "_trainconfig.json", "w") as f:
        json.dump({
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "ent_coef": args.ent_coef,
            "seed": args.seed,
        }, f, indent=2)
    print(f"Saved base policy to {args.save_path}.zip (seed={args.seed})")


if __name__ == "__main__":
    main()  