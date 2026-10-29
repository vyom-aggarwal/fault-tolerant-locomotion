# Fault Recovery in RL Locomotion Through Online Residual Adaptation

Research project at the **UC Santa Cruz AIEA Lab**, investigating whether a small residual
correction module that updates *during execution* can help a quadruped recover from unexpected
actuator and sensor faults, without retraining its base locomotion policy.

Everything runs in simulation (PyBullet), on CPU, on a laptop.

---

## Project status

| Component | State |
|---|---|
| Fault-injectable quadruped environment | **Built** — [`envs/quadruped_env.py`](envs/quadruped_env.py): 36-dim body-frame observation that includes the velocity command, command-tracking reward, randomized initial state |
| PPO base policies, 10 seeds @ 1.5M steps | **Built + run** — all 10 pass the gait check; `models/seed_0..9.zip`, training curves in `logs/seed_N/progress.csv` |
| Baseline A (faults, *no* adaptation), 700-step window | **Built + run** — 4,890 trials, see [Results](#results-baseline-a); trial data in `results/seed_N/baseline_fault_results.csv` |
| Across-seed aggregation & headroom analysis | **Built + run** — [`logs/across_seed_summary.csv`](logs/across_seed_summary.csv) |
| Pre-fault policy characterization (cost of transport, gait structure, drift) | **Built, output not committed** — [`scripts/characterize_policies.py`](scripts/characterize_policies.py) |
| Influence matrices (action → body velocity) | **Fitted for all 10 seeds**, but weakly determined — see [Residual correction](#1-residual-correction-adaptationpy) |
| Constant-offset residual correction (`ResidualAdapter`, integral / RLS) | **Library built; no script runs it closed-loop yet.** Offset authority probed on seed 0 only (partial coverage); replication on seeds 1, 4, 7 pending |
| Adaptive gait-amplitude controller (AIMD on torso tilt) | **Built, tested, negative result** — a constant amplitude reduction beats it on held-out policies. ⚠ The committed `amplitude_adapter.py` currently raises `NameError` on construction ([Known issues](#known-issues-at-this-commit)) |
| Paired statistical analysis of that comparison | **Built + run** — [`scripts/analyze_deployment.py`](scripts/analyze_deployment.py) on `logs/deployment.csv` (9 held-out policies) |
| Baseline B (full policy retraining) | **Not built yet** |
| Held-out fault split for H3 | **Not yet fixed** |

**Where the project stands.** Baseline A leaves real headroom: unadapted recovery ranges from 0%
for `actuation_delay` to 65% for `sensor_noise`. The first adaptation mechanism tried, a reactive
amplitude controller, does not beat simply walking with a constant, smaller gait amplitude: on
held-out policies 1–9, `fixed 0.80` is faster than every adaptive configuration (Holm-adjusted
p = 0.023–0.031) and no difference in falls was detected. The open question is whether any
*adaptive* method (direct delay estimation, a fault-conditioned policy, or a better residual) can
beat a constant. The residual module that H1–H3 describe is still to be built.

> **Heads-up: three scripts are broken at the current commit** (`7ecd6de`; details in
> [Known issues](#known-issues-at-this-commit)). `amplitude_adapter.py` raises `NameError` on
> construction, which takes down every script that builds an adapter; `run_multiseed.py`'s gait
> check uses an environment it never creates; and `diagnose_gait.py` crashes on its last line.
> The committed policies, results, and logs are unaffected. Commands below that depend on a broken
> script are marked ⚠.

---

## Contents

- [Abstract](#abstract)
- [Hypotheses](#hypotheses)
- [Results (Baseline A)](#results-baseline-a)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The environment](#the-environment)
- [The fault model](#the-fault-model)
- [Baseline A protocol](#baseline-a-protocol)
- [Adaptation work so far](#adaptation-work-so-far)
- [Full reproduction pipeline](#full-reproduction-pipeline)
- [How the code fits together](#how-the-code-fits-together)
- [Known issues and limitations](#known-issues-and-limitations)
- [Roadmap](#roadmap)

---

## Abstract

Most reinforcement learning controllers for legged robots are trained assuming the robot is
working perfectly. In the real world that assumption doesn't hold: motors weaken, joints seize,
and sensors start reporting inaccurate values. A policy trained only under ideal conditions
often struggles when that happens, because it has never learned how to respond to failure.

This project asks whether a small residual correction module, updated online, can recover
locomotion performance after an unexpected hardware fault without retraining the original
policy. A quadruped locomotion policy is trained in PyBullet with PPO (Stable-Baselines3), then
actuator and sensor faults are injected mid-rollout. Online residual adaptation is compared
against two baselines: **no adaptation** (Baseline A) and **full policy retraining**
(Baseline B). Staying entirely in simulation makes it possible to test failure modes
systematically and repeatably, including ones too risky or impractical to induce on hardware.

## Hypotheses

- **H1** — Residual adaptation recovers most of the lost locomotion performance within a small,
  bounded number of post-fault timesteps.
- **H2** — It does so without retraining the base policy, landing between Baseline A
  (no adaptation) and Baseline B (full retraining).
- **H3** — Recovery generalizes to fault types the residual module never trained on.

A subset of fault categories is held out exclusively for H3. **The split must be fixed before the
adaptation module is written**: choosing a test set after observing how a method performs on it
invalidates the generalization claim. (This is a split over *fault types*. The "held-out policies"
mentioned below are a different thing: seeds 1–9, kept away from the tuning seed 0.)

---

## Results (Baseline A)

Base policy, **no adaptation**, fault injected mid-episode. Everything in this section can be
re-derived from the committed per-seed CSVs with `python scripts/aggregate_seeds.py` (see
[Quickstart](#4-reproduce-the-tables-without-a-simulator)). Numbers are from the 700-step (11.7 s)
post-fault window and the settled-deficit metric described in
[Baseline A protocol](#baseline-a-protocol).

### Training reliability

| | |
|---|---|
| Seeds attempted | 10 (seeds 0–9), 1.5M timesteps each (`n_envs=4`, `ent_coef=0.01`; identical for every seed, per `models/seed_N_trainconfig.json`) |
| Converged to a usable gait | **10 / 10** (`results/manifest.json`). Seeds 7 and 8 survived 4 of 5 gait-check episodes, exactly the 80% threshold; the other eight survived all five |
| Forward velocity (body frame) | **0.544 ± 0.039 m/s** against a 0.5 m/s command (+8.8%); range 0.456 (seed 7) to 0.590 (seed 2) |
| Net-displacement speed | 0.531 ± 0.044 m/s |
| Mean absolute lateral velocity | 0.087 ± 0.007 m/s |
| Mean absolute yaw rate | 0.247 ± 0.015 rad/s |
| Last logged training episode length | 859 ± 41 of 1000 steps (stochastic policy) |
| Training wall-clock | 0.63–1.93 h per seed (median 0.64 h, 8.4 h in total), from `logs/seed_N/progress.csv`; hardware is not recorded |
| Influence matrices | `models/seed_N_influence.npz`, fitted for all 10 seeds |

All ten seeds pass the gait check, so every analysis below uses all ten. Non-converged seeds, if
they ever occur, are excluded by `aggregate_seeds.py` via the manifest, with the failure recorded
as `failure_mode` (`off_target`, `unstable`, or `unstable+off_target`).

Each seed runs about 100 trials per fault (89–100 valid trials: a trial is discarded if the robot
falls before the fault is injected, which costs seed 8 eleven trials and seed 7 six). Trial `i`
uses seed `i`, so the same trials are discarded for every fault type.

### Fault response

**The unit of analysis is the seed.** Each of the 10 seeds contributes one number per metric
from its ~100 trials; the ± is the spread *across seeds*, not across trials.

| Fault (severity) | Degraded | Velocity drop | **Recovery** | Fall rate | Settled deficit (m/s) | Recovery time (s) |
|---|---|---|---|---|---|---|
| `actuation_delay` (10 steps) | 99.9% | 100.2% | **0.0% ± 0.0%** | 97.7% ± 2.7% | 0.321 ± 0.048 † | — |
| `torque_limit` (0.2×, one joint) | 84.2% | 64.0% | **20.6% ± 6.2%** | 40.8% ± 8.6% | 0.117 ± 0.048 | 6.64 ± 0.47 |
| `joint_lock` (one joint) | 85.2% | 57.5% | **29.6% ± 4.8%** | 8.1% ± 12.3% | 0.137 ± 0.035 | 7.37 ± 0.48 |
| `sensor_dropout` (one joint) | 60.3% | 40.8% | **56.8% ± 10.0%** | 5.8% ± 9.3% | 0.019 ± 0.014 | 6.84 ± 0.47 |
| `sensor_noise` (σ = 0.3, all joints) | 59.4% | 38.7% | **65.0% ± 13.1%** | 4.1% ± 9.0% | 0.009 ± 0.014 | 6.57 ± 0.59 |

*Velocity drop is the worst point of a 30-step smoothed trace (a transient, descriptive only).
Recovery rate is **conditional on degradation**: of the trials where the fault measurably slowed
the robot, the fraction that returned to pre-fault speed and held it. Settled deficit is
pre-fault speed minus the speed over the final 2 s of the window.*

*† Settled deficit exists only for trials that did not fall, so it understates the harm of faults
with high fall rates. For `actuation_delay` it rests on 23 of 978 trials from 6 of the 10 seeds;
for `torque_limit`, on 580 of 978. Treat the `actuation_delay` cell as indicative only.*

Per-seed recovery rates (seeds 0–9):

```
actuation_delay   0%    0%    0%    0%    0%    0%    0%    0%    0%    0%
torque_limit     27%   21%   24%   17%   16%   17%   26%   16%   10%   30%
joint_lock       37%   25%   39%   30%   30%   27%   29%   26%   24%   30%
sensor_dropout   53%   49%   63%   67%   62%   42%   66%   56%   42%   68%
sensor_noise     79%   56%   69%   59%   63%   62%   77%   73%   36%   78%
```

### What these numbers say

1. **Actuation-side faults are harder than sensor-side faults.** Ordered by recoverability:
   `actuation_delay` (0%) < `torque_limit` (20.6%) < `joint_lock` (29.6%) < `sensor_dropout`
   (56.8%) < `sensor_noise` (65.0%). The physics/sensor split is an empirical property of the
   system, which makes it a defensible axis for the H3 split. Sensor faults leave settled deficits
   of only 0.009–0.019 m/s, so there is little for any method to recover there.

2. **The longer window changed the picture.** Requiring recovery to *hold* through the final 2 s
   of a 700-step window lowers every recovery rate relative to the earlier 300-step window (that
   summary is in git history at commit `aa776b2`: 82.3% for `sensor_noise`, 53.7% for
   `joint_lock`), because early returns to speed often do not hold, and it exposes falls that take
   several seconds to develop.

3. **`torque_limit` is the most dangerous non-delay fault.** It has a 40.8% fall rate and a
   0.117 m/s settled deficit with the lowest recovery of the non-delay faults (20.6%).
   `joint_lock` has the largest settled deficit among them (0.137 m/s) but rarely topples the
   robot (8.1%). Between-seed spread is modest for both; `sensor_noise` varies most (36–79%, and
   `aggregate_seeds.py` flags it as high-variance).

4. **`actuation_delay` is saturated at severity 10.** It topples the robot in 97.7% of trials and
   slows the survivors to near zero. The dose-response sweep below and the deployment severities
   (2–20) are what characterize it; see [Adaptation work](#adaptation-work-so-far).

### Dose-response: `actuation_delay` (seed 0)

From `logs/ad_sweep.csv`: 30 trials per severity on the seed-0 policy, 700-step protocol.

| Delay (steps) | Degraded | Fell | Recovered (of degraded) | Settled deficit, non-fallen (m/s) |
|---|---|---|---|---|
| 1 | 83% | 20% | 40% | 0.023 |
| 2 | 97% | 7% | 41% | 0.081 |
| 3 | 97% | 37% | 14% | 0.106 |
| 5 | 100% | 43% | 10% | 0.151 |
| 10 | 100% | 80% | 0% | 0.283 |
| 20 | 100% | 100% | 0% | — (no survivors) |

At n = 30 a difference of a few trials is noise (20% falls at 1 step vs 7% at 2). The trend is what
matters: delay degrades gradually through severity 5 and saturates by 10. `logs/sweep_seed0.csv`
holds sweeps of the other faults, but it predates the 700-step protocol (300-step window, no
settled-deficit columns) and is superseded.

---

## Quickstart

Run every command from the repository root: default paths (`models/`, `logs/`, `results/`) are
relative.

### 1. Clone

```bash
git clone https://github.com/vyom-aggarwal/fault-tolerant-locomotion
cd fault-tolerant-locomotion
```

### 2. Install

**On Windows, use conda.** `pip install pybullet` compiles a C++ core from source and fails with
`Microsoft Visual C++ 14.0 or greater is required` unless MSVC build tools are present.
conda-forge ships a prebuilt binary and avoids the compiler entirely.

```bash
conda create -n locomotion python=3.11 -y
conda activate locomotion
conda install -c conda-forge pybullet -y
pip install "stable-baselines3[extra]" gymnasium
```

The `[extra]` is required, not optional: the base install omits `tensorboard`, `tqdm`, and
`rich`, and each surfaces as a separate `ImportError` partway into a training run.

On macOS/Linux, `pip install pybullet "stable-baselines3[extra]" gymnasium` works directly.

The committed policies were trained with Python 3.11, Stable-Baselines3 2.9.0, PyTorch 2.13.0
(CPU), NumPy 2.4.6, and Gymnasium 1.3.0 (recorded inside each `models/seed_N.zip`). Other versions
may work, but those are the ones the policies were produced with. There is no `requirements.txt`
yet (see [Roadmap](#roadmap)).

Everything is CPU-only by design: the networks are small (two hidden layers of 128) and PyBullet
is the bottleneck, so a GPU build of torch buys nothing. The robot model (`laikago.urdf`) and
ground plane ship inside `pybullet_data`; there are no assets to download separately.

### 3. Verify the install

```bash
python scripts/smoke_test.py
```

Expected: observation shape `(36,)` matching `env.observation_space`, action shape `(12,)`, a joint
count of 12, and confirmation that a `torque_limit` injection on joint 0 at step 50 doesn't crash
the simulation. It also prints the final base height (the script's hint: above 0.25 m if still
standing).

### 4. Reproduce the tables without a simulator

The across-seed and paired analyses are pure standard-library Python and read the committed CSVs,
so they run without PyBullet or Stable-Baselines3:

```bash
python scripts/aggregate_seeds.py
python scripts/analyze_deployment.py --csv logs/deployment.csv --confirm "fixed 0.80"
```

The first reprints the [fault-response table](#fault-response) and regenerates
`logs/across_seed_summary.csv` (identical to the committed file; pass `--out` to write somewhere
else). The second reprints the [held-out comparison](#held-out-result).
Add `--exclude_severities 20` to check that the result does not depend on the saturated severity.

### 5. Re-run Baseline A on the committed policies

The trained policies are committed, so the evaluation half of the pipeline needs no training.
`run_multiseed.py` would be the single-command route, but its gait check is currently broken
(⚠ [Known issues](#known-issues-at-this-commit)), so run the per-seed evaluation directly. Write to
a separate directory so the committed CSVs are not overwritten, and copy the committed manifest so
`aggregate_seeds.py` knows which seeds converged:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/baseline_fault_eval.py --model models/seed_$s --trials 100 \
      --out results_rerun/seed_$s/baseline_fault_results.csv
done
cp results/manifest.json results_rerun/manifest.json
python scripts/aggregate_seeds.py --results_dir results_rerun --out logs/rerun_summary.csv
```

(PowerShell: `0..9 | ForEach-Object { python scripts/baseline_fault_eval.py --model "models/seed_$_" --trials 100 --out "results_rerun/seed_$_/baseline_fault_results.csv" }`,
then `Copy-Item results/manifest.json results_rerun/manifest.json`.)

**The committed table used `actuation_delay` severity 10, but the code default is now 5**
(`FAULT_CONFIGS` in `baseline_fault_eval.py`). To regenerate the committed table exactly, change
that entry to 10 first; otherwise the `actuation_delay` row will differ.

---

## Repository layout

```
envs/
  quadruped_env.py          the ONLY file with real robot logic:
                            physics, observations, reward, fault injection,
                            and the env_config.json load/save helpers

scripts/
  smoke_test.py             does the env run at all?
  check_reset_pose.py       is the starting stance sane?  [opens a GUI, waits on Enter]
  train_base_policy.py      PPO training for one seed
  evaluate_policy.py        roll out a policy, optionally with GUI and a fault
  diagnose_gait.py          is it REALLY walking, numerically?   (⚠ crashes on its last line)
  check_drift.py            does it walk straight, or crab sideways?
  characterize_policies.py  pre-fault cost of transport, duty factor, stride, drift, stability
  baseline_fault_eval.py    Baseline A: inject faults, log trials
  analyze_baseline.py       summarize ONE seed's trial CSV                (stdlib only)
  aggregate_seeds.py        summarize ACROSS seeds + headroom analysis    (stdlib only)
  run_multiseed.py          orchestrates train -> gait check -> fault eval (⚠ gait check broken)

  -- adaptation research (after Baseline A) --
  adaptation.py             ResidualAdapter + RLSEstimator: bounded joint-space
                            correction from an influence matrix           (library, numpy only)
  fit_influence_matrix.py   identify the action -> body-velocity influence matrix per policy
  probe_authority.py        how much can a constant joint-space offset change body velocity?
  measure_fault_deficit.py  settled post-fault velocity deficit vs. offset authority
  probe_amplitude.py        does scaling gait amplitude recover velocity / stability?
  amplitude_adapter.py      AmplitudeAdapter: none | fixed | adaptive action scale
                            (library, numpy only; ⚠ fails on construction)
  eval_amplitude.py         none vs fixed vs adaptive per fault/severity    (⚠ needs the adapter)
  sweep_adapter.py          grid over adapter hyperparameters on one fault  (⚠ needs the adapter)
  frontier_compare.py       fall-rate vs speed Pareto front, fixed scales vs adaptive (⚠)
  measure_latency.py        detection / safe-scale / fall timing after fault onset    (⚠)
  deployment_compare.py     one committed config across severities 2-20, per policy   (⚠)
  analyze_deployment.py     paired Wilcoxon + Holm + bootstrap on deployment.csv (stdlib only)

models/
  env_config.json                    ONE environment config shared by every policy in this folder
  seed_0.zip .. seed_9.zip           trained base policies
  seed_N_trainconfig.json            provenance: timesteps, ent_coef, n_envs, seed
  seed_N_influence.npz               fitted action -> velocity influence matrix (+ diagnostics)

logs/
  across_seed_summary.csv   across-seed Baseline A results
  seed_N/progress.csv       per-seed PPO training curves
  ad_sweep.csv              actuation_delay severity sweep, seed 0, 700-step protocol
  sweep_seed0.csv           older sweep of all faults, seed 0, 300-step window (superseded)
  amplitude_eval.csv        none / fixed / adaptive on seed 0, actuation_delay severities 2-20
  frontier.csv              one fall-rate-vs-speed frontier run (fault and severity not recorded)
  deployment.csv            8 candidates x 5 severities x 9 held-out policies

results/
  manifest.json                       per-seed status, gait-check metrics, failure_mode
  seed_N/baseline_fault_results.csv   trial-level Baseline A data (4,890 trials)
```

**What is and isn't committed.** The trained policies, influence files, training curves, the
manifest, every per-seed trial CSV, and the `logs/*.csv` outputs above are all in the repo.
*Not* committed: periodic training checkpoints (`seed_N_ckpt_*_steps.zip`, generated by
`train_base_policy.py`), the CSV written by `characterize_policies.py`
(`logs/policy_characterization.csv`), and the console output of the probe scripts
(`probe_authority.py`, `probe_amplitude.py`, `measure_fault_deficit.py`, `measure_latency.py`).
Numbers quoted below from those scripts come from the author's runs. There is no `.gitignore`,
`requirements.txt`, or license file yet.

**Environment config sidecar.** `train_base_policy.py` writes `models/env_config.json` before
training starts (command, episode length, action repeat, torque ceiling, action scale, whether
initial-state randomization is on, and the two tracking-reward weights).
`make_env_from_model_path()` loads it so evaluation environments match training. The file is shared
by every policy in the folder, so policies trained with different settings need separate folders.
`smoke_test.py`, `check_reset_pose.py`, `check_drift.py`, `diagnose_gait.py`, and
`evaluate_policy.py` instead construct `QuadrupedFaultEnv` with its constructor defaults, which
currently equal the saved config.

**Provenance sidecars.** Every training run writes `models/seed_N_trainconfig.json`.
`run_multiseed.py` reads it before reusing an existing model and retrains if `timesteps`,
`ent_coef`, or `n_envs` differ from the requested batch. (It does not compare the seed or any
environment setting.) This exists because mixing policies trained under different budgets into one
experiment confounds the across-seed variance, and nothing else would catch it.

---

## The environment

`QuadrupedFaultEnv` is a standard Gymnasium environment. One instance owns one PyBullet client.

| Property | Value |
|---|---|
| Robot | `laikago/laikago.urdf` from `pybullet_data`, 12 revolute joints |
| Physics rate | 240 Hz (`setTimeStep = 1/240`) |
| Action repeat | 4 → **control rate 60 Hz** |
| Episode length | 1000 control steps ≈ 16.7 s of sim time |
| Observation | 36-dim `Box(-inf, inf)` |
| Action | 12-dim `Box(-1, 1)` |
| Command | `command_vel = (vx, vy, yaw_rate)` in the body frame, default `(0.5, 0, 0)`: walk forward at 0.5 m/s, no lateral motion, no turning |
| Terrain | flat plane |

### Observation (36-dim, body frame)

| Slice | Contents |
|---|---|
| `0:12` | joint angles (rad) — **the only channel faults corrupt** |
| `12:24` | joint velocities (rad/s) |
| `24:27` | projected gravity (unit gravity vector in the body frame) |
| `27:30` | base linear velocity, body frame |
| `30:33` | base angular velocity, body frame |
| `33:36` | velocity command `(vx, vy, yaw_rate)` |

### Action

Actions are **offsets around a fixed standing pose**, not raw torques:

```
target_angle[i] = standing_pose[i] + action[i] * action_scale      # standing_pose = [0.0, -0.7, 0.7] * 4, action_scale = 0.5
```

This matters: a freshly initialized network outputs values near zero, so an untrained policy
begins by roughly standing still rather than commanding a collapsed pose it can never escape.

Each joint is then driven by PyBullet `POSITION_CONTROL` with a force ceiling of `max_torque`
(20 N·m). This matters for interpreting the `torque_limit` fault: it caps the position controller's
force budget, which is not identical to saturating a true torque command.

### Reward

```
r =  -1.5  * min( || (v_fwd, v_lat) - (cmd_vx, cmd_vy) ||, 1 )     linear-velocity tracking
     -0.5  * min( | yaw_rate - cmd_yaw | , 1 )                     yaw-rate tracking
     -0.03 * sum_i (tau_i / max_torque)^2                           energy, normalized to the torque budget
     -1.0  * (1 - up_z)^2                                           stay upright
     -1.0  * max(0, 0.50 - height)                                  don't crouch
     +1.5                                                           alive bonus
     -10.0                                                          one-time penalty on falling
```

All velocities are in the robot's own frame. Laikago's URDF frame is **Y-up / Z-forward**, not the
usual Z-up, so `_frame_from_quat` takes lateral = local X, up = local Y, forward = local Z, and
`up_z` is the *world* z-component of the local up axis. Reading world-frame roll/pitch directly
gives wrong answers for this robot: a perfectly upright Laikago reports roll ≈ 90°.

Note that the reward tracks the whole command, lateral velocity and yaw rate included, so there is
a direct penalty on drift. Two terms exist for specific reasons worth knowing before changing them:

- **The alive bonus is load-bearing.** Every other term is ≤ 0, so without it the fastest way to
  maximize total episode reward is to end the episode immediately.
- **The energy penalty is normalized** by `max_torque` rather than using raw N·m². Raw units would
  rescale the term by the square of any change to the torque ceiling, silently altering the
  reward balance whenever that ceiling is swept.

### Reset and termination

- **Reset** rebuilds the simulation, spawns the robot, and runs 60 physics steps (0.25 s) of
  position control onto the (randomized) standing pose before handing back the first observation.
  Spawn quaternion `[0, 0.5, 0.5, 0]` (**not** identity: identity stands the robot on its tail),
  nominal start height 0.48 m.
- **Initial-state randomization** (`randomize_init=True`, the default): joint angles ±0.10 rad,
  height ±0.03 m, roll and pitch ±0.05 rad, yaw uniform over ±π, and an initial base velocity of
  ±0.10 m/s linear and ±0.20 rad/s angular applied *after* settling so it isn't damped out. Trials
  of the same fault are only independent samples because of this; `--no_randomize_init` exists for
  debugging only.
- **Terminated** when base height < 0.35 m **or** `up_z` < 0.5 (roughly 60° from vertical).
  **Truncated** at 1000 steps.
- **Reset clears all faults.** Faults must be triggered *after* `reset()`, mid-episode.

---

## The fault model

Call `env.trigger_fault(...)` at any point during an episode.

| `fault_type` | Scope | `severity` means | Mechanism |
|---|---|---|---|
| `torque_limit` | one joint | fraction of nominal (e.g. `0.2`) | scales that joint's 20 N·m force ceiling |
| `joint_lock` | one joint | *ignored* | holds the joint's position target at the angle it had at injection |
| `actuation_delay` | all joints | integer control steps of lag (e.g. `10` = 167 ms) | actions queue through a FIFO buffer |
| `sensor_dropout` | one joint | *ignored* | that joint's **angle** reading is forced to 0 |
| `sensor_noise` | all joints | Gaussian σ (e.g. `0.3`) | noise added to all **angle** readings |

```python
env.trigger_fault("torque_limit", joint=3, severity=0.2)   # specific joint
env.trigger_fault("torque_limit", severity=0.2)            # random joint, drawn from env.np_random
env.trigger_fault("sensor_noise", severity=0.3)            # global fault, joint ignored
env.clear_faults()                                          # revert to healthy
```

The first three faults corrupt **physics** (what the robot can do); the last two corrupt
**observations** (what the robot knows). Sensor faults never touch ground truth: the actual joint
angles are unaffected, only the policy's perception of them. That is the structural distinction the
Baseline A results validate empirically.

Details worth keeping in mind when designing experiments:

- `joint=None` on a per-joint fault draws a uniform random joint from `env.np_random`, so the
  choice is reproducible given the episode seed.
- **Actuation delay fills a buffer from empty.** For the first `delay` control steps after the
  fault, the robot is commanded a *zero* action (the standing pose), and only then do the delayed
  policy actions arrive.
- Sensor faults corrupt **joint angles only**. Joint velocities, projected gravity, base
  velocities, and the command pass through clean, so the policy always retains an uncorrupted view
  of its own body motion. That is a generous assumption and a reasonable knob to tighten later.
- Faults **compose**. Calling `trigger_fault` twice leaves both active; `active_fault` only
  records the most recent one. `baseline_fault_eval.py --n_joints N` uses this deliberately to
  affect multiple joints, the severity axis for `joint_lock` and `sensor_dropout` (neither has a
  continuous one at a single joint). Each call draws its joint independently, so with replacement:
  `N` calls can hit fewer than `N` distinct joints.

### Extension points

Two hooks exist in `envs/quadruped_env.py` but are currently dormant, and both are natural
attachment points for the residual work:

- `_apply_fault_to_action(action)` — an identity pass-through today, called once per control step
  after the delay buffer. This is where an action-space fault, or the residual correction itself,
  would be applied. `ResidualAdapter` is not wired into it.
- `_sensor_bias` — allocated and added to joint angles every step, but no fault type currently
  sets it. Wiring up a `sensor_bias` / drift fault is a few lines.

---

## Baseline A protocol

Implemented in [`scripts/baseline_fault_eval.py`](scripts/baseline_fault_eval.py). One trial:

| Phase | Steps | What happens |
|---|---|---|
| Pre-fault | 0 – onset (random, 150–249) | policy walks; trial is **discarded** if it falls here |
| Baseline measurement | last 120 pre-fault steps | `baseline_vel` = mean forward velocity |
| Injection | onset step | `trigger_fault(type, severity)`, random joint |
| Post-fault window | 700 steps (11.7 s) | degradation, recovery, settled deficit, and falls are scored |

The window is 700 steps so that "recovered" must *hold* and falls that take several seconds to
develop (notably under `actuation_delay`) are observed. Trial `i` calls `env.reset(seed=i)` and
draws its onset from `default_rng(i)`, so every candidate evaluated with the same trial index sees
the same initial state and the same onset, and fault types are compared on matched conditions. The
script refuses to run if onset + window would exceed the 1000-step episode.

### How recovery is scored

Every trial receives an explicit `recovery_status`, so nothing is ambiguous:

| Status | Meaning |
|---|---|
| `recovered` | degraded, then ended the window back within 15% of pre-fault speed |
| `no_recovery` | degraded and still outside that band at the end of the window |
| `no_degradation` | the fault never measurably slowed the robot |
| `fell` | the robot fell after the fault |

The procedure (`analyze_trace`; every threshold is a constant at the top of the file):

1. `baseline_vel` is the mean forward velocity over the last **120 pre-fault steps** (2 s, about six
   strides).
2. **Degradation is measured first.** A trial is `degraded` if the mean forward velocity of *any*
   120-step (2 s) window after the fault drops below `baseline_vel × 0.85`. If it never does, the
   outcome is `no_degradation`: a real finding (the policy was already robust to this fault), but
   not recovery. Windows of 2 s are used so that normal gait oscillation cannot trigger detection.
3. **A trial that falls is never scored as recovered**, whatever its velocity did beforehand.
4. For trials that survive the window, `settled_vel` is the mean over the **final 120 steps** and
   `settled_deficit = baseline_vel − settled_vel`. This is the lasting loss a correction would have
   to restore.
5. The trial is `recovered` if `settled_vel` is within 15% of `baseline_vel`, otherwise
   `no_recovery`. `recovery_time_s` is the start of the first 2 s window after which *every*
   later window stays inside the band (and not earlier than the first degraded window),
   in seconds (`step / 60`).
6. The separate **velocity drop** is the worst point of a 30-step causal rolling mean, as a
   fraction of baseline. It is a transient and descriptive only.

Aggregation (`aggregate_seeds.py`): per seed, recovery rate = `recovered / degraded` (so it is
conditional on degradation, and fallen trials count as degraded if their velocity dropped);
degradation rate = `degraded / trials`; fall rate = `fell / trials`. Then mean ± sd across seeds.

> **This replaces an earlier, invalid metric.** The original criterion checked *instantaneous*
> velocity against the baseline every step and latched on the first crossing, with the recovery
> check running before the fall check. Because a 25 kg robot's velocity barely changes in the first
> few steps after a fault, and because gait velocity oscillates around its own mean twice per
> stride, that criterion fired within a few control steps essentially always and produced
> near-100% recovery on every fault type, including the logically impossible combination of high
> recovery with a high fall rate. Any recovery number predating the sustained-window criterion
> should be discarded.

### Severities

```python
# FAULT_CONFIGS defaults in baseline_fault_eval.py
torque_limit 0.2 · joint_lock 1.0 · actuation_delay 5 · sensor_dropout 1.0 · sensor_noise 0.3
# The COMMITTED Baseline A data used actuation_delay = 10 (see the note in Quickstart step 5).
```

`--sweep` replaces these with a dose-response grid:

```python
torque_limit     0.5, 0.3, 0.2, 0.1, 0.05
actuation_delay  1, 2, 3, 5, 10, 20
sensor_noise     0.1, 0.3, 0.6, 1.0
joint_lock       (single point; use --n_joints for severity)
sensor_dropout   (single point; use --n_joints for severity)
```

`--faults a,b` restricts a run to some fault types, and `--n_joints N` sets how many joints each
fault touches.

---

## Adaptation work so far

Baseline A showed the faults leave headroom; the work since then asks which adaptation mechanism
can use it. Everything sits on the frozen base policy; nothing is retrained. The short version:
**a constant, reduced gait amplitude is a strong baseline that the adaptive controller does not
beat.**

### 1. Residual correction (`adaptation.py`)

`ResidualAdapter` adds a bounded joint-space offset `delta` (|delta| ≤ `delta_max` = 0.30 per
joint, in action units) to the policy's action. An integral controller (`eta = 0.03`) drives
the 30-step-smoothed body velocity `(forward, lateral, yaw)` toward the command through a ridge
pseudo-inverse of an influence matrix `B` (3 × 12: how each joint action moves each velocity).
`mode="fixed"` uses an offline `B` from `fit_influence_matrix.py` (archived per seed as
`models/seed_N_influence.npz`); `mode="rls"` refines it online with a forgetting-factor
`RLSEstimator` and a small exploratory dither. **No script runs `ResidualAdapter` closed-loop yet.**

**The influence matrices are weakly determined.** `fit_influence_matrix.py` regresses smoothed body
velocity on a dithered action offset (24,000 steps per seed, dither σ = 0.10, each sample held for
120 steps). Across the ten seeds the regression R² is 0.053 ± 0.019 for forward velocity, 0.089 ±
0.014 for lateral, and 0.112 ± 0.030 for yaw. Only `seed_0`'s file carries the split-half agreement
check (seeds 1–9 were fitted by an earlier version of the script); on seed 0 the two independent
halves agree on the lateral and yaw rows (cosine 0.77 and 0.75) but not on the forward row
(cosine 0.29), which is the row that matters. By the script's own criteria that is a "halves
disagree" result.

`probe_authority.py` and `measure_fault_deficit.py` ask how much speed a bounded constant offset
can buy back. On seed 0 the best delivered forward correction is **0.055 m/s**, against the
largest settled deficit from Baseline A (0.137 m/s, `joint_lock`): about **40%, partial rather
than insufficient**. A constant offset therefore cannot restore commanded speed, but it is not
useless. Both probes originally compared against a 0.27 m/s worst *transient* dip, which overstated
the target. Commit `6f218ae` switched them to the settled deficits from the 10-seed Baseline A,
which moves the seed-0 verdict from 20% ("insufficient") to 40% ("partial"); the same commit
reports that `measure_fault_deficit.py` independently found `torque_limit` and `joint_lock`
52–56% correctable. The probe output is not committed, and the verdict is seed-0 only. Replication on held-out seeds 1, 4, and 7 (chosen in
advance) is pending:

```bash
python scripts/probe_authority.py --model models/seed_N
python scripts/probe_amplitude.py --model models/seed_N --trials 12 --offset_authority <value from probe_authority>
```

### 2. Gait-amplitude scaling (`amplitude_adapter.py`)

Scale the policy's joint-space action amplitude (`env._action_scale * s`, applied after the action
clip), trading speed for stability. `probe_amplitude.py` measures fixed scales per fault, comparing
scaling before and after the clip because a saturated policy cancels the former. `AmplitudeAdapter`
has three modes: `none` (`s = 1`), `fixed` (constant `s`), and `adaptive` (`s ∈ [0.5, 1.0]`).

The adaptive rule needs no fault detector. It calibrates on the robot's own first 2 s of walking
and sets a risk threshold at `mean + k·sd` of healthy risk (floor 0.02). Risk is the smoothed torso
tilt (`1 − upright_alignment`) plus `osc_weight` × its short-window standard deviation, since
delay-induced instability shows up as oscillation before it shows up as lean. Above the threshold
the controller backs off, in proportion to how far risk exceeds the threshold
(`s ← s · (1 − prop_gain · excess)`, `excess = risk/threshold − 1`); otherwise it recovers additively by
`increase` per step (0.0015 in the evaluated configurations). This is additive-increase /
multiplicative-decrease (AIMD). ⚠ In the committed file, `increase` and `excess` are undefined; see
[Known issues](#known-issues-at-this-commit).

### 3. Is adaptation worth it? Four comparisons, one answer

| Script | Question | Finding |
|---|---|---|
| `frontier_compare.py` | At a *fixed* severity, does adaptive reach a better fall-rate/speed Pareto front than constant scales? | The earlier approach of comparing one adaptive point to one constant was invalid, because each mechanism has its own trade-off curve. In the committed `logs/frontier.csv` (seed 0, 20 trials per point) a single adaptive point (k = 1.0) is on the front, between `fixed 0.65` and `fixed 0.75`. Fall rates are not monotonic in the fixed scale (75% at 0.70, 45% at 0.75, 60% at 0.80), so at n = 20 the front is partly sampling noise. |
| `measure_latency.py` | Is the controller simply too slow to react? | **No** (author's runs, output not committed). Detection takes 0.6–1.1 s and reaching a safe scale about 1.3 s, against a budget of 2.5–6.5 s before the robot falls: roughly 10× the time needed. |
| `deployment_compare.py` | With one configuration committed *before* a fault of unknown severity (2–20), does adaptive win? | Constants win. On seed 0, `fixed 0.80` had equal or lower falls and higher speed than every adaptive configuration, and every adaptive controller fell ~100% at severity 20 (author's seed-0 run; `logs/deployment.csv` now holds the held-out policies only). |
| `analyze_deployment.py` | Does that hold on policies it was not tuned on, under paired statistics? | Yes on seeds 1–9 (below). |

**Why the controller loses.** It detects quickly and backs off correctly, then climbs back
(`increase = 0.0015`) to a scale that falls a large share of the time. AIMD assumes conditions
improve; these faults are permanent, so the safe amplitude is permanently lower and any recovery is
wrong by construction. Back-off-and-hold (`increase = 0`) also fails: the fault transient alone
drives the scale to its floor at every severity, which is a fixed minimum with extra steps.
`increase` trades conservatism for speed exactly as a fixed scale does. Three successive
explanations for the controller's behaviour (sweep defaults, detection latency, back-off-and-hold)
were each wrong and each caught by measurement.

#### Held-out result

Policies 1–9, `actuation_delay`, severities 2–20. Up to 15 trials per cell (`deployment_compare.py --all --exclude 0 --trials 15`; trials that fall
before the fault are discarded). **The policy is the unit of analysis**: tests are exact two-sided
Wilcoxon signed-rank with Holm correction, and bootstrap 95% intervals resample policies. The
confirmatory claim (`fixed 0.80` dominates every adaptive configuration) was fixed from the seed-0
tuning run *before* the held-out run; the family is 2 metrics × 3 configs = 6 tests.

| Adaptive config | Speed gain of `fixed 0.80` | 95% CI | Holm adj. p | Fall difference (fixed − adaptive) | 95% CI | adj. p |
|---|---|---|---|---|---|---|
| k = 1.0 | +0.091 m/s | [+0.063, +0.117] | 0.023 | +0.4 pt | [−2.5, +2.9] | 1.00 |
| k = 2.0 | +0.075 m/s | [+0.050, +0.102] | 0.023 | −1.0 pt | [−4.1, +2.1] | 1.00 |
| k = 3.0 | +0.072 m/s | [+0.042, +0.097] | 0.031 | −1.7 pt | [−3.6, +0.6] | 0.63 |

All three are dominated: `fixed 0.80` is faster, and no difference in falls was detected. The
claim is that the constant is not shown to be safer, only not detectably riskier. The result is
robust to excluding severity 20, which is saturated for every candidate (speed gains are
unchanged, fall differences remain undetected). Exploratory only: `fixed 0.70` shows a 6–8 point
fall advantage over every adaptive configuration with intervals excluding zero, which does not
survive correction across the full family of 24 comparisons (the smallest achievable exact p at
n = 9, 0.0039, exceeds the strictest corrected threshold, 0.0021).

The same data by severity (mean over the 9 held-out policies; each cell is fall rate / mean speed
of the trials that did not fall, in m/s):

| Candidate | Sev 2 | Sev 3 | Sev 5 | Sev 10 | Sev 20 |
|---|---|---|---|---|---|
| `none` | 16% / 0.504 | 34% / 0.447 | 68% / 0.353 | 98% / 0.236 ‡ | 100% / — |
| `fixed 0.80` | 2% / 0.431 | 3% / 0.371 | 31% / 0.275 | 94% / 0.145 ‡ | 100% / — |
| `fixed 0.70` | 0% / 0.351 | 1% / 0.307 | 15% / 0.216 | 85% / 0.079 | 99% / 0.110 ‡ |
| `fixed 0.65` | 0% / 0.305 | 1% / 0.262 | 7% / 0.168 | 67% / 0.036 | 99% / 0.009 ‡ |
| `fixed 0.55` | 0% / 0.152 | 0% / 0.111 | 3% / 0.040 | 52% / −0.035 | 98% / −0.039 ‡ |
| adaptive k = 1.0 | 8% / 0.390 | 11% / 0.320 | 27% / 0.197 | 83% / 0.046 | 100% / — |
| adaptive k = 2.0 | 10% / 0.402 | 14% / 0.327 | 30% / 0.209 | 81% / 0.052 | 100% / — |
| adaptive k = 3.0 | 11% / 0.421 | 14% / 0.353 | 33% / 0.204 | 81% / 0.052 | 100% / — |

*‡ Speed averaged over fewer than 6 of the 9 policies, because most trials fell.*

**Scope.** This establishes that AIMD on a tilt-oscillation signal does not beat a constant. It
does not show that no adaptive method could. Directly estimating the delay, or a fault-conditioned
policy, remain untested. The remaining argument for adaptation is the cost of a constant chosen
for the worst case: `fixed 0.55`, the safest constant here, manages only 0.152 m/s at severity 2
against 0.504 m/s with no adaptation (30%), and at severities 10 and 20 even it is slightly
negative, meaning the survivors are moving backward, which is not restored locomotion.

```bash
python scripts/frontier_compare.py --model models/seed_0 --fault actuation_delay --severity 5 \
    --fixed_scales 0.5,0.55,0.6,0.65,0.7,0.75,0.8                         # ⚠
python scripts/measure_latency.py --model models/seed_0 --fault actuation_delay --severity 5   # ⚠
python scripts/deployment_compare.py --model models/seed_0 --trials 15                         # ⚠
python scripts/deployment_compare.py --all --exclude 0 --trials 15     # ⚠ held-out policies
python scripts/analyze_deployment.py --csv logs/deployment.csv --confirm "fixed 0.80"
python scripts/eval_amplitude.py --model models/seed_0 --trials 30     # ⚠ none / fixed / adaptive
python scripts/sweep_adapter.py --model models/seed_0                  # ⚠ hyperparameter grid
```

Outputs: `logs/frontier.csv`, `logs/deployment.csv`, `logs/amplitude_eval.csv`. (`sweep_adapter.py`
only prints.) `deployment_compare.py` runs 8 candidates (`none`, fixed 0.80/0.70/0.65/0.55,
adaptive k = 1.0/2.0/3.0), each held unchanged across severities 2, 3, 5, 10, 20, and
hard-codes `actuation_delay`. `analyze_deployment.py` implements the exact Wilcoxon, Holm
correction, and bootstrap itself with no external dependencies.

---

## Full reproduction pipeline

### Cheap sanity checks first

These take seconds and catch mistakes that otherwise cost hours of training. Don't skip them.

```bash
python scripts/smoke_test.py
python scripts/check_reset_pose.py
```

`check_reset_pose.py` opens a PyBullet GUI window and waits on Enter before closing.

### Train one seed

```bash
python scripts/train_base_policy.py --seed 0 --timesteps 1500000 --save_path models/seed_0 --log_dir logs/seed_0
```

| Flag | Default | Notes |
|---|---|---|
| `--seed` | `0` | Set explicitly on every run. Seeds Python, NumPy, PyTorch, the vec-envs, and PPO. |
| `--timesteps` | `1_500_000` | 500k undertrains: slower, off-target policies and a lower convergence rate. |
| `--n_envs` | `4` | Parallel envs; keep small on a laptop CPU. |
| `--ent_coef` | `0.01` | Entropy bonus. SB3's PPO default is `0.0`, which lets action noise collapse as soon as the policy finds anything locally stable, typically standing still. |
| `--target_velocity` | `0.5` | Commanded forward velocity; the command is `(vx, 0, 0)`. |
| `--max_torque` | `20.0` | Per-joint torque ceiling (N·m). The URDF allows up to 100. |
| `--action_scale` | `0.5` | Max per-joint deviation (rad) from the standing pose. |
| `--no_randomize_init` | off | Debugging only: without randomization, repeated trials are not independent samples. |
| `--log_format` | `csv` | Writes a directly plottable `progress.csv`. `tensorboard` uses an async writer thread that has proven fragile on Windows: it can die mid-run and take training down with it. `none` for stdout only. |
| `--save_path`, `--log_dir` | `models/base_policy`, `logs/base_policy` | Pass explicitly. |

PPO hyperparameters: `n_steps=2048`, `batch_size=256`, `n_epochs=10`, `lr=3e-4`, `γ=0.99`,
`gae_lambda=0.95`, `clip_range=0.2`, `net_arch=[128, 128]`. Checkpoints every ~50k timesteps.
Training prints a warning before overwriting an existing `<save_path>.zip`.

On the logged runs a seed took 0.63–1.93 h (median 0.64 h) for 1.5M timesteps with `--n_envs 4`.
Throughput *drops* as policies improve, since a surviving robot generates more contact computation
per episode, so a progress bar's early ETA is optimistic.

### Train and evaluate all seeds

```bash
python scripts/run_multiseed.py --seeds 10 --timesteps 1500000 --n_envs 4 --trials 100     # ⚠ gait check
```

For each seed this runs training → a gait check → fault evaluation, writing
`results/manifest.json` after each seed finishes fault evaluation (or fails the gait screen) so a
crash mid-run doesn't lose completed seeds. Pass arguments explicitly rather than relying on defaults (`--seeds` defaults
to 5): it puts the exact command that produced your results in your shell history, which is what
the methods section needs.

The **gait check** between training and evaluation is the quality gate. It runs **5 episodes with
different environment seeds** (0–4), deterministic policy, and requires both:

- **stability** — surviving ≥ 90% of a 1000-step rollout in ≥ 80% of episodes, and
- **tracking** — mean *forward* body-frame velocity (over the surviving episodes) within 30% of
  the commanded 0.5 m/s.

Both are needed, and they fail independently: a policy can walk stably but far too slowly
(`off_target`), or track the right speed and then fall (`unstable`). Seeds that fail either test
are excluded from fault evaluation and counted against the reported convergence rate. The manifest
also records net-displacement speed, mean |lateral| velocity, mean |yaw rate|, joint range, and
contact variation per seed. Tune with `--gait_episodes`, `--survival_threshold`, and
`--convergence_tol`, but don't drop below a handful of episodes: judging convergence from a single
episode makes the reported convergence rate an n = 1 measurement per seed.

### Aggregate

```bash
python scripts/aggregate_seeds.py
```

Prints training reliability, per-fault across-seed statistics, a high-variance warning when the
per-seed recovery spread exceeds 40 points, and a headroom ranking classifying each fault as
`CEILING` (≥ 95% recovery) / `moderate headroom` / `large headroom` (≤ 30%). Writes
[`logs/across_seed_summary.csv`](logs/across_seed_summary.csv).

It reads `results/manifest.json` to determine which seeds converged and **excludes the rest**,
printing what it dropped and why. Result directories persist across runs, so a seed that passed
under an older, looser criterion will leave a stale CSV on disk; globbing alone would silently read
it back in and contaminate the summary.

### Characterize the base policies

```bash
python scripts/characterize_policies.py
```

For every converged seed, runs `--episodes` (3) rollouts and reports cost of transport, mechanical
power, mean |torque|, duty factor, stride frequency, action smoothness, base-height variation, and
a drift check (net-displacement speed vs forward velocity vs |lateral| velocity). Writes
`logs/policy_characterization.csv`, which is not committed. These pre-fault numbers are the
reference for an "energy cost of adaptation" metric.

### Inspect a single policy

```bash
python scripts/evaluate_policy.py --model models/seed_0 --render --episodes 3
python scripts/evaluate_policy.py --model models/seed_0 --fault torque_limit --fault_step 150 --fault_severity 0.2
python scripts/diagnose_gait.py --model models/seed_0 --steps 1000      # ⚠ crashes on its last line
python scripts/check_drift.py --model models/seed_0 --episodes 2
python scripts/analyze_baseline.py --csv results/seed_0/baseline_fault_results.csv
```

`diagnose_gait.py` is the numerical counterpart to watching the GUI: per-joint angle ranges (a real
gait needs tens of degrees of swing, not a few), ground-contact variation (a constant count means
all feet stay planted and the robot is sliding), and net displacement versus target speed.
`check_drift.py` answers a narrower question: does the robot travel along its own heading, or crab
sideways?

> **Gotchas.**
> - `baseline_fault_eval.py`, `evaluate_policy.py`, and `diagnose_gait.py` default `--model` to
>   `models/base_policy`, which does not exist (policies are `models/seed_N`). Always pass
>   `--model` explicitly.
> - `analyze_baseline.py` reports recovery rate **unconditionally** (recovered ÷ all trials),
>   while `aggregate_seeds.py` reports it **conditional on degradation** (recovered ÷ degraded).
>   The two disagree on the same data (seed 0, `sensor_dropout`: 27% vs 53%); the conditional
>   figure is the one quoted in [Results](#results-baseline-a). `analyze_baseline.py` also
>   defaults to `logs/baseline_fault_results.csv`, so point it at
>   `results/seed_N/baseline_fault_results.csv`.

---

## How the code fits together

Exactly one file contains real quadruped logic. Everything in `scripts/` is a thin driver that
imports it and does one job. Dependencies point one way only: scripts import the environment;
the environment imports nothing from scripts.

```
                           envs/quadruped_env.py
     physics · observations · reward · fault injection · models/env_config.json helpers
                                     │
    ┌───────────┬────────────┬───────┴─────┬──────────────┬──────────────┬───────────────┐
    │           │            │             │              │              │               │
smoke_test  train_base_  evaluate_    diagnose_gait   baseline_      characterize_   fit_influence_
check_reset  policy       policy      check_drift     fault_eval     policies        matrix
 _pose
    │           │            │             │              │              │               │
"does it    "learn to    "watch it"   "is it REALLY   "how does it   "how efficient    "action ->
 run?"       walk"                     walking?"       fail?"         is the gait?"     velocity?"


run_multiseed.py    ── orchestrates ─▶  train → gait check → baseline_fault_eval, per seed
analyze_baseline.py ── reduces ─────▶   one seed's trials  → console + summary CSV
aggregate_seeds.py  ── reduces ─────▶   all seeds' trials  → logs/across_seed_summary.csv
```

The adaptation scripts follow the same rule. `adaptation.py` (`ResidualAdapter`, `RLSEstimator`)
and `amplitude_adapter.py` (`AmplitudeAdapter`) are libraries holding the controller logic and
import only numpy. `eval_amplitude.py`, `sweep_adapter.py`, `frontier_compare.py`,
`measure_latency.py`, and `deployment_compare.py` are drivers that import the environment, an
`AmplitudeAdapter`, and `baseline_fault_eval`'s onset constants (`eval_amplitude.py` also reuses
its `analyze_trace` scoring). `probe_authority.py`, `measure_fault_deficit.py`, and
`fit_influence_matrix.py` import only the environment; `probe_amplitude.py` also imports the
adapter and the onset constants but only uses `env._action_scale`. `analyze_deployment.py` reads
`logs/deployment.csv` and nothing else.

---

## Known issues and limitations

Stated plainly because they determine what the next round of experiments has to address.

### Known issues at this commit

Found by reading the code at commit `7ecd6de` and confirmed where it could be executed; each
should be a small fix.

1. **`scripts/amplitude_adapter.py` cannot be constructed.** `__init__` executes
   `self.increase = float(increase)`, but `increase` is not in the signature, and the proportional
   back-off uses an `excess` that is never computed. Every mode raises
   `NameError: name 'increase' is not defined`. This takes down `eval_amplitude.py`,
   `sweep_adapter.py`, `frontier_compare.py`, `measure_latency.py`, and `deployment_compare.py`.
   The last working version (commit `0a0c425`) had a parameter `increase=0.0015` and
   `excess = (risk_s / self.threshold) - 1.0`; both were lost in `a5a1e32`. The committed
   `logs/*.csv` are static outputs and are unaffected, but cannot be regenerated until this is
   fixed. `probe_amplitude.py` imports the class but never instantiates it, so it still runs.
2. **`scripts/run_multiseed.py`'s gait check has no environment.** `gait_check()` loads the policy
   but never builds the environment (`make_env_from_model_path` is imported and never called; the
   `env = ...` line was dropped in `74e0533`), so the first use of `env` raises `NameError`. The
   exception is caught and the seed is recorded as `gait_check_failed`, so no seed reaches fault
   evaluation, including with `--skip_training`. Use `baseline_fault_eval.py` directly until it is
   fixed (see [Quickstart step 5](#5-re-run-baseline-a-on-the-committed-policies)).
3. **`scripts/diagnose_gait.py` ends with an `AttributeError`.** Its last diagnostic prints
   `env.target_velocity`, which the environment no longer defines (the command is `env.command_vel`).
   Everything before that line prints normally.
4. **Committed Baseline A and the code disagree on `actuation_delay`.** The committed data used
   severity 10; `FAULT_CONFIGS` now says 5.
5. **Influence matrices came from different script versions.** Only `seed_0`'s file carries the
   split-half diagnostics and the `randomize_init` metadata; seeds 1–9 were fitted by an earlier
   version.
6. **`logs/frontier.csv` does not record the fault or severity** it was run at, and
   `logs/sweep_seed0.csv` predates the 700-step protocol.
7. **Several scripts bypass `models/env_config.json`** (see [Repository layout](#repository-layout)),
   and three scripts default `--model` to the nonexistent `models/base_policy`.
8. **`check_reset_pose.py` prints "expected ~0.30" for the reset height.** That cannot be a
   standing height: the environment terminates below 0.35 m and the reset nominally starts at
   0.48 m (now randomized). The printed measurement is right; the parenthetical hint is stale.

### Experimental limitations

1. **`actuation_delay` is saturated at severity 10.** It is a 97.7% fall rate on Baseline A and
   98–100% for every candidate at severity 20 on held-out policies, so it discriminates poorly
   between methods there. Severities 2–5 are where methods can differ.

2. **Systematic velocity overshoot.** Seeds average 0.544 m/s forward velocity against a 0.5 m/s
   command, about 9% high (range 0.456–0.590). The reward tracks the full command, and
   net-displacement speed (0.531 m/s) is close to forward velocity, so this does not look like
   a lateral drift problem. But mean |lateral| velocity is 0.087 m/s, about 16% of forward velocity
   on these averages, which is above the 15% threshold `check_drift.py` uses to flag drifting
   (its output is not committed). The gait check's 30% tolerance is loose enough to admit it.

3. **n = 10 seeds.** Enough for paired exact tests, but the smallest achievable exact Wilcoxon p
   at n = 9 held-out policies is 0.0039, so large exploratory families cannot reach significance.
   Fix the family before looking at results. Baseline A tables still report mean ± sd across seeds
   rather than bootstrap intervals.

4. **Random joint selection adds uncontrolled variance.** Per-joint faults draw a random joint each
   trial, so 100 trials spread across 12 joints is ~8 per joint. Whether a hip or a calf is
   affected is currently invisible in the results. A systematic per-joint sweep would both remove
   the variance and answer "which joints are critical?"

5. **Sensor faults touch joint angles only.** Velocities, projected gravity, base velocities, and
   the command are never corrupted, so the policy always keeps a clean view of its body motion.

6. **`torque_limit` caps a position controller's force budget**, which is related to but not the
   same as true torque saturation on a torque-controlled actuator.

7. **Single robot, flat terrain, simulation only.** Only the *initial state* is randomized (pose,
   orientation, yaw, joint angles, velocities). There is no terrain variation, no dynamics
   randomization (mass, friction, motor strength), and no sim-to-real claim is being made.

8. **Seed 0 is the tuning seed.** Adapter defaults came from sweeps on seed 0. Report held-out
   results on seeds 1–9 (`--exclude 0`); the seed-0 numbers are development results.

9. **The adaptation evidence is mostly `actuation_delay`.** The deployment comparison hard-codes
   that fault, and the frontier, latency, and amplitude-evaluation runs used it. Constant
   amplitude reduction is shown to beat AIMD there; it has not been tested on `torque_limit` or
   `joint_lock` under the same protocol. Note also that the candidate H3 hold-out set (see
   [Roadmap](#roadmap)) contains `actuation_delay`, which the amplitude work has already been
   tuned and evaluated on; if it stays held out, that exposure should be documented.

10. **Amplitude scaling is not residual adaptation.** It changes gait amplitude, not a learned
    correction. It is a baseline for the residual module, which is still to be built;
    `ResidualAdapter` has not been run closed-loop.

11. **Offset-probe verdicts are seed-0 only,** and the fitted influence matrices are weakly
    determined (see [Residual correction](#1-residual-correction-adaptationpy)). The 40%
    partial-coverage figure needs replication on seeds 1, 4, and 7.

---

## Roadmap

- [x] Redefine the recovery criterion (sustained window; degradation required; falls disqualify).
- [x] Seed training reproducibly and record provenance alongside every model.
- [x] Screen seeds on velocity tracking, over multiple episodes.
- [x] Exclude non-converged seeds from aggregation via the manifest.
- [x] Retrain on a command-conditioned environment with a full-command tracking reward and
      randomized initial state; 10 / 10 seeds converge.
- [x] Extend the Baseline A window to 700 steps and report settled deficits.
- [x] Characterize the base policies (cost of transport, gait structure, drift): script written
      (output not committed).
- [x] Prototype a residual (integral / RLS) correction and probe its authority (seed 0).
- [x] Prototype an adaptive gait-amplitude controller; add a fixed-scale control, tuning sweep,
      and Pareto-front comparison.
- [x] Compare adaptive vs constant amplitude on frontiers, then under deployment conditions.
- [x] Replicate the comparison on held-out policies 1–9 with paired, corrected statistics.
- [ ] **Fix the three broken scripts** ([Known issues](#known-issues-at-this-commit)) so the
      committed experiments can be regenerated.
- [ ] Re-fit the influence matrices for seeds 1–9 with the current script (split-half check), and
      commit the console output of the probe scripts.
- [ ] Replicate `probe_authority.py` / `probe_amplitude.py` on seeds 1, 4, 7.
- [ ] **Fix the held-out fault split for H3 and write down the reasoning — before the adaptation
      module exists.** Current candidate: train on `torque_limit`, `joint_lock`, `sensor_noise`;
      hold out `actuation_delay` and `sensor_dropout`. Both categories are represented in the
      held-out set, and the most structurally distinct fault (temporal, not magnitude) is held
      out. Caveat: the amplitude work was tuned on `actuation_delay`.
- [ ] Test an adaptive method that is not AIMD: estimate the delay directly, or condition the
      policy on the fault, and compare against constant amplitude on the same protocol.
- [ ] Extend the constant-vs-adaptive comparison beyond `actuation_delay`.
- [ ] Implement the residual module on the `_apply_fault_to_action` hook, run it closed-loop, and
      compare it to the constant-amplitude baseline.
- [ ] Implement Baseline B (full policy retraining post-fault) for the upper-bound comparison.
- [ ] Add a domain-randomization baseline: reviewers will ask why randomized training isn't
      sufficient on its own.
- [ ] Report Baseline A with paired, across-seed statistics and bootstrap CIs rather than
      mean ± sd at this sample size (see Agarwal et al., 2021).
- [ ] Add a `sensor_bias` / drift fault using the existing `_sensor_bias` state.
- [ ] Pin dependency versions in `requirements.txt`; add a `.gitignore` for generated output
      (`results_rerun/`, checkpoints).

---

## Notes

No license file is currently specified; please get in touch before reusing this work.
Conducted through the UC Santa Cruz AIEA Lab.