import os
import json
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


# Frame helpers -- pure functions of a quaternion, so they are unit-testable without a running simulator.

def _frame_from_quat(orn):
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    lateral = rot[:, 0]
    up = rot[:, 1]
    forward = rot[:, 2]
    return up, forward, lateral, rot


def _to_body_frame(world_vec, rot):
    return rot.T @ np.asarray(world_vec, dtype=np.float64)


def _projected_gravity(rot):
    return rot.T @ np.array([0.0, 0.0, -1.0])


class QuadrupedFaultEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render: bool = False,
        command_vel=(0.5, 0.0, 0.0),      # (vx, vy, yaw_rate) in the body frame
        max_episode_steps: int = 1000,
        action_repeat: int = 4,
        max_torque: float = 20.0,
        action_scale: float = 0.5,
        # initial-state randomization 
        randomize_init: bool = True,
        init_joint_noise: float = 0.10,   # rad, uniform around the standing pose
        init_height_noise: float = 0.03,  # m
        init_rp_noise: float = 0.05,      # rad, roll/pitch perturbation
        randomize_yaw: bool = True,
        init_lin_vel_noise: float = 0.10, # m/s
        init_ang_vel_noise: float = 0.20, # rad/s
        # reward weights 
        w_lin_vel: float = 1.5,
        w_yaw_rate: float = 0.5,
        w_energy: float = 0.03,
        w_stability: float = 1.0,
        alive_bonus: float = 1.5,
        fall_penalty: float = 10.0,
        err_clip: float = 1.0,        # bound on tracking-error penalties
    ):
        super().__init__()
        self.render_mode = "human" if render else None
        self.command_vel = np.array(command_vel, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
        self.max_torque = max_torque

        self.randomize_init = randomize_init
        self.init_joint_noise = init_joint_noise
        self.init_height_noise = init_height_noise
        self.init_rp_noise = init_rp_noise
        self.randomize_yaw = randomize_yaw
        self.init_lin_vel_noise = init_lin_vel_noise
        self.init_ang_vel_noise = init_ang_vel_noise

        self.w_lin_vel = w_lin_vel
        self.w_yaw_rate = w_yaw_rate
        self.w_energy = w_energy
        self.w_stability = w_stability
        self.alive_bonus = alive_bonus
        self.fall_penalty = fall_penalty
        self.err_clip = err_clip

        self._client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        # Per leg [hip, thigh, calf], applied to all four legs
        self._standing_pose = np.array([0.0, -0.7, 0.7] * 4, dtype=np.float64)
        self._action_scale = action_scale

        # Base spawn orientation. NOT identity: see the axis-convention note.
        self._base_orn = [0, 0.5, 0.5, 0]

        self.robot_id = None
        self.plane_id = None
        self.joint_ids = []
        self.num_joints = 0
        self.step_count = 0
        self._episode_pose = self._standing_pose.copy()

        # fault state
        self.active_fault = None
        self._torque_scale = None
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias = None
        self._sensor_dropout_mask = None
        self._actuation_delay_steps = 0
        self._action_buffer = []

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 3 + 3 + 3 + 3   # = 36; see the layout note at the top
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # Core Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        # randomized spawn pose 
        height = 0.48
        orn = self._base_orn
        if self.randomize_init:
            height += float(self.np_random.uniform(-self.init_height_noise,
                                                   self.init_height_noise))
            roll = float(self.np_random.uniform(-self.init_rp_noise, self.init_rp_noise))
            pitch = float(self.np_random.uniform(-self.init_rp_noise, self.init_rp_noise))
            yaw = float(self.np_random.uniform(-np.pi, np.pi)) if self.randomize_yaw else 0.0
            pert = p.getQuaternionFromEuler([roll, pitch, yaw])
            # Apply the world-frame perturbation on top of the base orientation.
            _, orn = p.multiplyTransforms([0, 0, 0], pert, [0, 0, 0], self._base_orn)

        self.robot_id = p.loadURDF(
            "laikago/laikago.urdf",
            [0, 0, height],
            orn,
            physicsClientId=self._client,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self.joint_ids = [
            j for j in range(p.getNumJoints(self.robot_id, physicsClientId=self._client))
            if p.getJointInfo(self.robot_id, j, physicsClientId=self._client)[2] == p.JOINT_REVOLUTE
        ]
        self.num_joints = len(self.joint_ids)

        # randomized joint angles around the standing pose 
        pose = self._standing_pose.copy()
        if self.randomize_init:
            pose = pose + self.np_random.uniform(
                -self.init_joint_noise, self.init_joint_noise, size=pose.shape)
        self._episode_pose = pose

        for idx, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot_id, joint_id, float(pose[idx]),
                              physicsClientId=self._client)

        # clear faults (an episode always starts healthy) 
        self.active_fault = None
        self._torque_scale = np.ones(self.num_joints, dtype=np.float32)
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias = np.zeros(self.num_joints, dtype=np.float32)
        self._sensor_dropout_mask = np.zeros(self.num_joints, dtype=bool)
        self._actuation_delay_steps = 0
        self._action_buffer = []

        # settle onto the (randomized) pose under active position control 
        for _ in range(60):
            for idx, joint_id in enumerate(self.joint_ids):
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.POSITION_CONTROL,
                    targetPosition=float(pose[idx]), force=self.max_torque,
                    physicsClientId=self._client,
                )
            p.stepSimulation(physicsClientId=self._client)

        # ---- initial base velocity, applied AFTER settling so it is not damped out ----
        if self.randomize_init and (self.init_lin_vel_noise > 0 or self.init_ang_vel_noise > 0):
            lin = self.np_random.uniform(-self.init_lin_vel_noise,
                                         self.init_lin_vel_noise, size=3)
            ang = self.np_random.uniform(-self.init_ang_vel_noise,
                                         self.init_ang_vel_noise, size=3)
            p.resetBaseVelocity(self.robot_id, linearVelocity=lin.tolist(),
                                angularVelocity=ang.tolist(),
                                physicsClientId=self._client)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # actuation-delay fault: FIFO queue, inert when delay == 0
        self._action_buffer.append(action)
        delay = self._actuation_delay_steps
        if len(self._action_buffer) > delay:
            applied_action = self._action_buffer.pop(0)
        else:
            applied_action = np.zeros_like(action)

        applied_action = self._apply_fault_to_action(applied_action)

        for _ in range(self.action_repeat):
            for idx, joint_id in enumerate(self.joint_ids):
                if joint_id in self._joint_lock_angle:
                    target = self._joint_lock_angle[joint_id]        # joint-lock fault
                else:
                    # Action is a DELTA around the standing pose, so a fresh policy (outputs ~0) starts near a valid stance.
                    target = self._standing_pose[idx] + float(applied_action[idx]) * self._action_scale

                torque_limit = self.max_torque * self._torque_scale[idx]   # torque-limit fault
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.POSITION_CONTROL,
                    targetPosition=target, force=torque_limit,
                    physicsClientId=self._client,
                )
            p.stepSimulation(physicsClientId=self._client)

        self.step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = self._check_fallen()
        if terminated:
            reward -= self.fall_penalty
        truncated = self.step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected(physicsClientId=self._client):
            p.disconnect(physicsClientId=self._client)

    # Fault injection
    def trigger_fault(self, fault_type: str, joint: int = None, severity: float = 1.0):
        """
        "torque_limit"    - cap a joint's torque to `severity` x nominal
        "joint_lock"      - freeze a joint at its current angle
        "actuation_delay" - lag all commands by `severity` steps
        "sensor_dropout"  - a joint's angle reading returns 0
        "sensor_noise"    - gaussian noise (std=severity) on all angle readings
        """
        if joint is None and fault_type in ("torque_limit", "joint_lock", "sensor_dropout"):
            joint = self.np_random.integers(0, self.num_joints)

        if fault_type == "torque_limit":
            self._torque_scale[joint] = severity
        elif fault_type == "joint_lock":
            angle = p.getJointState(self.robot_id, self.joint_ids[joint],
                                    physicsClientId=self._client)[0]
            self._joint_lock_angle[self.joint_ids[joint]] = angle
        elif fault_type == "actuation_delay":
            self._actuation_delay_steps = int(severity)
        elif fault_type == "sensor_dropout":
            self._sensor_dropout_mask[joint] = True
        elif fault_type == "sensor_noise":
            self._sensor_noise_std = severity
        else:
            raise ValueError(f"Unknown fault_type: {fault_type}")

        self.active_fault = {"type": fault_type, "joint": joint, "severity": severity}

    def clear_faults(self):
        self._torque_scale[:] = 1.0
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias[:] = 0.0
        self._sensor_dropout_mask[:] = False
        self._actuation_delay_steps = 0
        self.active_fault = None

    def _apply_fault_to_action(self, action):
        return action   # extension point

    # Observation / reward / termination
    def _get_obs(self):
        joint_angles = np.zeros(self.num_joints, dtype=np.float32)
        joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        for idx, joint_id in enumerate(self.joint_ids):
            angle, vel, _, _ = p.getJointState(self.robot_id, joint_id,
                                               physicsClientId=self._client)
            joint_angles[idx] = angle
            joint_velocities[idx] = vel

        # Sensor faults corrupt the observation only
        # ground truth is untouched.
        if self._sensor_noise_std > 0:
            joint_angles = joint_angles + self.np_random.normal(
                0, self._sensor_noise_std, size=joint_angles.shape)
        joint_angles = joint_angles + self._sensor_bias
        joint_angles[self._sensor_dropout_mask] = 0.0

        _, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        _, _, _, rot = _frame_from_quat(orn)

        obs = np.concatenate([
            joint_angles,
            joint_velocities,
            _projected_gravity(rot),
            _to_body_frame(lin_vel, rot),
            _to_body_frame(ang_vel, rot),
            self.command_vel,
        ]).astype(np.float32)
        return obs

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        up, forward, lateral, rot = _frame_from_quat(orn)

        body_lin = _to_body_frame(lin_vel, rot)
        body_ang = _to_body_frame(ang_vel, rot)

        # Body axes: X lateral, Y up, Z forward.
        forward_vel = float(body_lin[2])   # local Z = forward
        lateral_vel = float(body_lin[0])   # local X = lateral
        yaw_rate = float(body_ang[1])      # local Y = up, so this is yaw rate

        cmd_vx, cmd_vy, cmd_yaw = (float(c) for c in self.command_vel)

        # Tracks the full command, not just forward speed
        lin_err = min(float(np.hypot(forward_vel - cmd_vx, lateral_vel - cmd_vy)),
                      self.err_clip)
        yaw_err = min(abs(yaw_rate - cmd_yaw), self.err_clip)
        vel_reward = -self.w_lin_vel * lin_err
        yaw_reward = -self.w_yaw_rate * yaw_err

        # Normalized to each joint's own budget
        torque_penalty = 0.0
        for joint_id in self.joint_ids:
            _, _, _, tau = p.getJointState(self.robot_id, joint_id,
                                           physicsClientId=self._client)
            nt = tau / self.max_torque if self.max_torque > 0 else 0.0
            torque_penalty += nt ** 2
        energy_penalty = -self.w_energy * torque_penalty

        upright_alignment = float(up[2])          # 1 = upright, 0 = 90 deg, -1 = inverted
        stability_penalty = -self.w_stability * (1.0 - upright_alignment) ** 2
        height_penalty = -1.0 * max(0.0, 0.50 - pos[2])

        # Keeps every reasonable behavior net-positive. 
        reward = (vel_reward + yaw_reward + energy_penalty
                  + stability_penalty + height_penalty + self.alive_bonus)

        info = {
            "forward_vel": forward_vel,
            "lateral_vel": lateral_vel,
            "yaw_rate": yaw_rate,
            "lin_vel_error": float(lin_err),
            "vel_reward": vel_reward,
            "yaw_reward": yaw_reward,
            "energy_penalty": energy_penalty,
            "stability_penalty": stability_penalty,
            "height": pos[2],
            "upright_alignment": upright_alignment,
        }
        return reward, info

    def _check_fallen(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        up, _, _, _ = _frame_from_quat(orn)
        return pos[2] < 0.35 or up[2] < 0.5


# Config sidecars: keep evaluation environments matched to training.

_CONFIG_KEYS = ["command_vel", "max_episode_steps", "action_repeat", "max_torque",
                "action_scale", "randomize_init", "w_lin_vel", "w_yaw_rate"]


def save_env_config(models_dir, config: dict):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "env_config.json")
    out = {}
    for k in _CONFIG_KEYS:
        if k in config:
            v = config[k]
            out[k] = list(v) if isinstance(v, (tuple, np.ndarray)) else v
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_env_config(models_dir):
    path = os.path.join(models_dir, "env_config.json")
    if not os.path.exists(path):
        print(f"WARNING: no env_config.json in {models_dir} -- using defaults. If this "
              f"policy was trained with non-default settings, evaluating it now will "
              f"silently use the WRONG ones.")
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if "command_vel" in cfg:
        cfg["command_vel"] = tuple(cfg["command_vel"])
    return cfg


def make_env_from_model_path(model_path, render=False, **overrides):
    models_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    cfg = load_env_config(models_dir)
    if cfg:
        print(f"Loaded env config from {models_dir}/env_config.json: {cfg}")
    cfg.update(overrides)
    return QuadrupedFaultEnv(render=render, **cfg)