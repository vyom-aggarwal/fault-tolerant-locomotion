# Fault Recovery in RL Locomotion Through Online Residual Adaptation

Research project at the **UC Santa Cruz AIEA Lab**, investigating whether a small residual
correction module that updates *during execution* can help a quadruped recover from unexpected
actuator and sensor faults — without retraining its base locomotion policy.

Everything runs in simulation (PyBullet), on CPU, on a laptop.

---

## Project status

| Component | State |
|---|---|
| Fault-injectable quadruped environment | **Built** — [`envs/quadruped_env.py`](envs/quadruped_env.py) |
| PPO base policy, trained across 10 seeds @ 1.5M steps | **Built + run** — `models/seed_0..9.zip` |
| Baseline A (fault, *no* adaptation) evaluation | **Built + run** — see [Results](#results-baseline-a) |
| Across-seed aggregation & headroom analysis | **Built + run** — [`logs/across_seed_summary.csv`](logs/across_seed_summary.csv) |
| Fault severity sweep | Built, **not yet run across seeds** |
| Residual adaptation module | **Not built yet** ← *the actual contribution* |
| Baseline B (full policy retraining) | **Not built yet** |
| Held-out fault split for H3 | **Not yet fixed** — must be chosen before the adaptation module exists |

Baseline A is complete and the faults separate cleanly: unadapted recovery ranges from **0%**
(`actuation_delay`) to **86.7%** (`sensor_noise`), leaving real headroom for an adaptation method
to demonstrate an effect. All three actuation-side faults are less recoverable than both
sensor-side faults, with no overlap — an empirical basis for the physics/sensor split used in H3.

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
- [Full reproduction pipeline](#full-reproduction-pipeline)
- [How the code fits together](#how-the-code-fits-together)
- [Known limitations & open issues](#known-limitations--open-issues)
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
adaptation module is written** — choosing a test set after observing how a method performs on it
invalidates the generalization claim.

---

## Results (Baseline A)

Base policy, **no adaptation**, fault injected mid-episode.

### Training reliability

| | |
|---|---|
| Seeds attempted | 10 (seeds 0–9), 1.5M timesteps each |
| Converged to a usable gait | **6 / 10 (60%)** — seeds 0, 2, 4, 6, 7, 8 |
| Excluded | seeds 1, 3, 5, 9 |
| Forward speed, converged seeds | **0.585 ± 0.053 m/s** against a 0.5 m/s command |
| Wall-clock, full pipeline | ~18.6 h (train + evaluate, 10 seeds, laptop CPU) |

A 60% convergence rate is a reported result, not a defect to hide: it characterizes how reliably
this training setup produces a usable policy. The two failure modes are distinct and recorded
separately in `results/manifest.json` as `failure_mode`:

- `off_target` — walks stably for the full episode but does not track the commanded velocity
  (e.g. seed 1 at 0.308 m/s, 38% tracking error).
- `unstable` — tracks velocity acceptably but falls before the episode ends.

### Fault response

**The unit of analysis is the seed.** Each of the 6 converged seeds contributes one number per
metric from 100 trials; the ± is the spread *across seeds*, not across trials. Non-converged
seeds are excluded by `aggregate_seeds.py`, which reads `results/manifest.json` rather than
trusting whatever CSVs happen to be on disk.

| Fault (severity) | Degraded | Velocity drop | **Recovery** | Fall rate | Recovery time (s) | Post-fault distance (m) |
|---|---|---|---|---|---|---|
| `actuation_delay` (10 steps) | 100.0% ± 0.0% | 85.9% ± 12.7% | **0.0% ± 0.0%** | 0.0% ± 0.0% | — | 1.565 ± 0.220 |
| `torque_limit` (0.2×) | 95.2% ± 5.3% | 55.1% ± 7.7% | **41.2% ± 3.0%** | **33.2% ± 10.3%** | 1.864 ± 0.618 | 1.880 ± 0.211 |
| `joint_lock` | 91.2% ± 2.6% | 44.9% ± 4.2% | **54.9% ± 18.6%** | 0.0% ± 0.0% | 2.222 ± 0.628 | 2.432 ± 0.185 |
| `sensor_dropout` | 95.7% ± 5.0% | 28.3% ± 3.2% | **81.4% ± 14.3%** | 0.0% ± 0.0% | 2.077 ± 0.940 | 3.041 ± 0.178 |
| `sensor_noise` (σ = 0.3) | 83.2% ± 14.9% | 23.7% ± 3.6% | **86.7% ± 5.6%** | 0.0% ± 0.0% | 2.227 ± 0.451 | 3.154 ± 0.138 |

*Recovery rate is **conditional on degradation** — of the trials where the fault measurably
slowed the robot, the fraction that returned to pre-fault speed and held it. Trials the fault
never affected are reported under "Degraded" instead of being scored as recoveries.*

Per-seed recovery rates (seeds 0, 2, 4, 6, 7, 8):

```
actuation_delay    0%   0%   0%   0%   0%   0%
torque_limit      39%  39%  39%  45%  45%  41%
joint_lock        49%  64%  76%  48%  68%  24%     <- high spread
sensor_dropout    93%  91%  91%  55%  81%  77%
sensor_noise      88%  85%  78%  95%  90%  84%
```

### What these numbers say

1. **Actuation-side faults are strictly harder than sensor-side faults.** Ordered by
   recoverability: `actuation_delay` (0%) < `torque_limit` (41.2%) < `joint_lock` (54.9%) <
   `sensor_dropout` (81.4%) < `sensor_noise` (86.7%). All three physics faults fall below both
   sensor faults with no overlap. The physics/sensor distinction is therefore an empirical
   property of the system, not just a labelling convention — which is what makes it a
   defensible axis for the H3 generalization split.

2. **`torque_limit` is the highest-power test case.** 41.2% ± 3.0 across six seeds — moderate
   headroom combined with very low between-seed variance. It is also the only fault that causes
   falls (33.2%); every other fault degrades the gait without toppling the robot.

3. **`actuation_delay` may be saturated.** 0% recovery on every seed and every trial, with an
   85.9% velocity drop and no falls: the robot slows to near-zero and never returns within the
   5 s window, but stays upright. This is either the most dramatic H1 demonstration available or
   a fault so severe that no method could discriminate. **The severity sweep decides which**, and
   has not been run yet.

4. **`joint_lock` varies genuinely across seeds** (24–76%). This survived the exclusion of
   non-converged seeds, so it reflects real policy-to-policy variation — some learned gaits
   tolerate a frozen joint, others do not. `aggregate_seeds.py` flags it automatically when the
   per-seed spread exceeds 40 points.

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/vyom-aggarwal/fault-recovery-quadruped-rl
```

### 2. Install

**On Windows, use conda.** `pip install pybullet` compiles a C++ core from source and fails with
`Microsoft Visual C++ 14.0 or greater is required` unless MSVC build tools are present.
conda-forge ships a prebuilt binary and avoids the compiler entirely.

```bash
conda create -n locomotion python=3.11 -y
```

```bash
conda activate locomotion
```

```bash
conda install -c conda-forge pybullet -y
```

```bash
pip install "stable-baselines3[extra]" gymnasium
```

The `[extra]` is required, not optional: the base install omits `tensorboard`, `tqdm`, and
`rich`, and each surfaces as a separate `ImportError` partway into a training run.

On macOS/Linux, `pip install pybullet "stable-baselines3[extra]" gymnasium` works directly.

Everything is CPU-only by design — the networks are small (two hidden layers of 128) and PyBullet
is the bottleneck, so a GPU build of torch buys nothing. The robot model (`laikago.urdf`) and
ground plane ship inside `pybullet_data`; there are no assets to download separately.

### 3. Verify the install

```bash
python scripts/smoke_test.py
```

Expected: observation/action shapes that match (34 and 12), a joint count of 12, and confirmation
that a mid-episode fault injection doesn't crash the sim.

### 4. Reproduce the Baseline A results without training anything

The trained policies are committed, so the evaluation half of the pipeline runs from a clean
clone in minutes:

```bash
python scripts/run_multiseed.py --seeds 10 --trials 100 --skip_training
```

```bash
python scripts/aggregate_seeds.py
```

This regenerates `results/seed_*/baseline_fault_results.csv`, `results/manifest.json`, and
`logs/across_seed_summary.csv`.

---

## Repository layout

```
envs/
  quadruped_env.py          the ONLY file with real robot logic:
                            physics, observations, reward, fault injection

scripts/
  smoke_test.py             does the env run at all?                    (seconds)
  check_reset_pose.py       is the starting stance sane?  [GUI]         (seconds)
  train_base_policy.py      PPO training for one seed                   (~1.5-2 h @ 1.5M)
  evaluate_policy.py        roll out a policy, optionally with GUI      (seconds)
  diagnose_gait.py          is it REALLY walking, numerically?          (seconds)
  check_drift.py            does it walk straight, or crab sideways?    (seconds)
  baseline_fault_eval.py    Baseline A: inject faults, log trials       (~20-40 min/seed)
  analyze_baseline.py       summarize ONE seed's trial CSV
  aggregate_seeds.py        summarize ACROSS seeds + headroom analysis
  run_multiseed.py          orchestrates the whole per-seed pipeline

models/
  seed_0.zip .. seed_9.zip           trained base policies
  seed_N_trainconfig.json            provenance: timesteps, ent_coef, n_envs, seed
  seed_N_ckpt_*_steps.zip            periodic checkpoints (generated)

logs/
  across_seed_summary.csv   committed across-seed results
  seed_*/progress.csv       per-seed training curves (generated, not committed)

results/                    generated, not committed
  manifest.json             per-seed status, gait-check metrics, failure_mode
  seed_*/baseline_fault_results.csv   trial-level Baseline A data
```

**Provenance sidecars.** Every training run writes `models/seed_N_trainconfig.json` recording the
timesteps, entropy coefficient, parallel-env count, and seed used. `run_multiseed.py` reads it
before reusing an existing model and retrains if the settings don't match the requested batch.
This exists because mixing policies trained under different budgets into one experiment
confounds the across-seed variance, and nothing else would catch it.

**What is and isn't committed:** the trained policies and the final across-seed summary are in the
repo. Trial-level CSVs, training curves, and `manifest.json` are regenerated by the commands
above. There is no `.gitignore` yet, so generated files show up in `git status` — don't commit
them by accident.

---

## The environment

`QuadrupedFaultEnv` is a standard Gymnasium environment. One instance owns one PyBullet client.

| Property | Value |
|---|---|
| Robot | `laikago.urdf` from `pybullet_data`, 12 revolute joints |
| Physics rate | 240 Hz (`setTimeStep = 1/240`) |
| Action repeat | 4 → **control rate 60 Hz** |
| Episode length | 1000 control steps ≈ 16.7 s of sim time |
| Observation | 34-dim `Box(-inf, inf)` |
| Action | 12-dim `Box(-1, 1)` |
| Target velocity | 0.5 m/s forward, flat terrain |
| Settled standing height | ~0.558 m |

### Observation (34-dim)

| Slice | Contents |
|---|---|
| `0:12` | joint angles (rad) — **the only channel faults corrupt** |
| `12:24` | joint velocities (rad/s) |
| `24:28` | base orientation quaternion |
| `28:31` | base linear velocity |
| `31:34` | base angular velocity |

### Action

Actions are **offsets around a fixed standing pose**, not raw torques:

```
target_angle[i] = standing_pose[i] + action[i] * 0.5        # standing_pose = [0.0, -0.7, 0.7] * 4
```

This matters: a freshly initialized network outputs values near zero, so an untrained policy
begins by roughly standing still rather than commanding a collapsed pose it can never escape.

Each joint is then driven by PyBullet `POSITION_CONTROL` with a force ceiling of 20 N·m. This
matters for interpreting the `torque_limit` fault: it caps the position controller's force
budget, which is not identical to saturating a true torque command.

### Reward

```
r = -|v_forward - 0.5|                  velocity tracking
    - 0.03 * Σ (τ_i / 20)²              energy penalty, normalized to the torque budget
    - (1 - up_z)²                       stay upright
    - max(0, 0.50 - height)             don't crouch
    + 1.0                               alive bonus
    - 10.0                              one-time penalty on falling
```

`v_forward` is the base linear velocity projected onto the robot's forward axis, and `up_z` is
the world-frame z-component of its up axis — both derived from the rotation matrix in
`_get_orientation_frame`, because Laikago's URDF frame is **Y-up / Z-forward**, not the usual
Z-up. Reading world-frame roll/pitch directly gives wrong answers for this robot: a perfectly
upright Laikago reports roll ≈ 90°.

Two terms exist for specific reasons worth knowing before changing them:

- **The alive bonus is load-bearing.** Every other term is ≤ 0, so without it the fastest way to
  maximize total episode reward is to end the episode immediately. Standing still scores
  **+0.50/step** with the bonus and −0.50/step without it.
- **The energy penalty is normalized** by the 20 N·m budget rather than using raw N·m². Raw
  units would rescale the term by the square of any change to the torque ceiling, silently
  altering the reward balance whenever that ceiling is swept.

### Reset and termination

- **Reset** snaps every joint directly to the standing pose (rather than starting at zero and
  letting the policy fight its way up), then runs 60 settle steps under position control before
  handing back the first observation. Start height 0.48 m, spawn quaternion `[0, 0.5, 0.5, 0]`
  (**not** identity — identity stands the robot on its tail).
- **Terminated** when base height < 0.35 m **or** `up_z` < 0.5 (roughly 60° from vertical).
- **Reset clears all faults.** Faults must be triggered *after* `reset()`, mid-episode.

---

## The fault model

Call `env.trigger_fault(...)` at any point during an episode.

| `fault_type` | Scope | `severity` means | Mechanism |
|---|---|---|---|
| `torque_limit` | one joint | fraction of nominal (e.g. `0.2`) | scales that joint's 20 N·m force ceiling |
| `joint_lock` | one joint | *ignored* | freezes the joint at its angle at injection time |
| `actuation_delay` | all joints | integer steps of lag (e.g. `10`) | actions queue through a FIFO buffer |
| `sensor_dropout` | one joint | *ignored* | that joint's **angle** reading is forced to 0 |
| `sensor_noise` | all joints | Gaussian σ (e.g. `0.3`) | noise added to all **angle** readings |

```python
env.trigger_fault("torque_limit", joint=3, severity=0.2)   # specific joint
env.trigger_fault("torque_limit", severity=0.2)            # random joint, drawn from env.np_random
env.trigger_fault("sensor_noise", severity=0.3)            # global fault, joint ignored
env.clear_faults()                                          # revert to healthy
```

The first three faults corrupt **physics** (what the robot can do); the last two corrupt
**observations** (what the robot knows). Sensor faults never touch ground truth — the actual
joint angles are unaffected, only the policy's perception of them. That is the structural
distinction the Baseline A results validate empirically.

`joint=None` on a per-joint fault draws a uniform random joint from `env.np_random`, so the
choice is reproducible given the episode seed.

Two properties worth keeping in mind when designing experiments:

- Sensor faults corrupt **joint angles only**. Joint velocities, the base quaternion, and the
  base velocities pass through clean, so the policy always retains an uncorrupted view of its
  own body pose. That is a generous assumption and a reasonable knob to tighten later.
- Faults **compose**. Calling `trigger_fault` twice with different types leaves both active;
  `active_fault` only records the most recent one. `baseline_fault_eval.py --n_joints N` uses
  this deliberately to affect multiple joints, which is the severity axis for `joint_lock` and
  `sensor_dropout` (neither has a continuous one at a single joint).

### Extension points

Two hooks exist in `envs/quadruped_env.py` but are currently dormant, and both are natural
attachment points for the residual work:

- `_apply_fault_to_action(action)` — an identity pass-through today. This is where an
  action-space fault (or the residual correction itself) would be applied.
- `_sensor_bias` — allocated and added to joint angles every step, but no fault type currently
  sets it. Wiring up a `sensor_bias` / drift fault is a few lines.

---

## Baseline A protocol

Implemented in [`scripts/baseline_fault_eval.py`](scripts/baseline_fault_eval.py). One trial:

| Phase | Steps | What happens |
|---|---|---|
| Pre-fault | 0 – 199 (3.3 s) | policy walks; trial is **discarded** if it falls here |
| Baseline measurement | last 50 pre-fault steps | `baseline_vel` = mean forward velocity |
| Injection | step 200 | `trigger_fault(type, severity)`, random joint |
| Post-fault window | 300 steps (5 s) | degradation, recovery, and falls are scored |

### How recovery is scored

Every trial receives an explicit `recovery_status`, so nothing is ambiguous:

| Status | Meaning |
|---|---|
| `recovered` | degraded, then returned to pre-fault speed **and held it** |
| `no_recovery` | degraded and never returned within the window |
| `no_degradation` | the fault never measurably slowed the robot |
| `fell` | the robot fell after the fault |

The procedure:

1. Post-fault forward velocity is smoothed with a **30-step causal rolling mean** (~0.5 s, about
   one stride), so normal gait oscillation cannot trigger detection.
2. **Degradation is measured first.** If the smoothed velocity never drops below
   `baseline_vel × 0.85`, the outcome is `no_degradation` — a real finding (the policy was
   already robust to this fault) but *not* recovery.
3. Recovery is the first step, *at or after* the point of degradation, from which the smoothed
   velocity stays within 15% of `baseline_vel` for **30 consecutive steps**. Recorded in seconds
   (`step / 60`).
4. **A trial that falls is never scored as recovered**, whatever its velocity did beforehand.

> **This replaces an earlier, invalid metric.** The original criterion checked *instantaneous*
> velocity against the baseline every step and latched on the first crossing, with the recovery
> check running before the fall check. Because a 25 kg robot's velocity barely changes in the
> first few steps after a fault, and because gait velocity oscillates around its own mean twice
> per stride, that criterion fired within 2–10 control steps essentially always. It produced
> ~100% recovery on every fault type — including the logically impossible combination of 100%
> recovery with a 36.5% fall rate. Any recovery number predating this change should be discarded.

### Severities

```python
torque_limit 0.2 · joint_lock 1.0 · actuation_delay 10 · sensor_dropout 1.0 · sensor_noise 0.3
```

`--sweep` replaces these with a dose-response grid:

```python
torque_limit     0.5, 0.3, 0.2, 0.1, 0.05
actuation_delay  5, 10, 20, 40
sensor_noise     0.1, 0.3, 0.6, 1.0
joint_lock       (single point; use --n_joints for severity)
sensor_dropout   (single point; use --n_joints for severity)
```

Trial `i` uses seed `i`, and the same seed set is reused across every fault type, so fault types
are compared on matched initial conditions.

---

## Full reproduction pipeline

### Cheap sanity checks first

These take seconds and catch mistakes that otherwise cost hours of training. Don't skip them.

```bash
python scripts/smoke_test.py
```

```bash
python scripts/check_reset_pose.py
```

`check_reset_pose.py` opens a PyBullet GUI window and waits on Enter before closing.

### Train and evaluate all seeds

```bash
python scripts/run_multiseed.py --seeds 10 --timesteps 1500000 --n_envs 4 --trials 100
```

For each seed this runs training → a gait check → fault evaluation, writing
`results/manifest.json` incrementally so a crash mid-run doesn't lose completed seeds.

**Budget ~18–19 hours for all ten seeds** at 1.5M timesteps with `--n_envs 4` on a laptop CPU.
Throughput *drops* as policies improve — a surviving robot generates more contact computation per
episode — so the progress bar's early ETA is optimistic.

Pass arguments explicitly rather than relying on defaults. It puts the exact command that
produced your results in your shell history, which is what the methods section needs.

The **gait check** between training and evaluation is the quality gate. It runs **5 episodes with
different environment seeds** and requires both:

- **stability** — surviving ≥ 90% of a 1000-step rollout in ≥ 80% of episodes, and
- **tracking** — mean speed within 30% of the commanded 0.5 m/s.

Both are needed, and they fail independently: a policy can walk stably but far too slowly
(`off_target`), or track the right speed and then fall (`unstable`). Seeds that fail either test
are excluded from fault evaluation and counted against the reported convergence rate.

> Judging convergence from a single episode makes the reported convergence rate an n=1
> measurement per seed. Tune with `--gait_episodes` and `--survival_threshold` if needed, but
> don't drop below a handful of episodes.

### Aggregate

```bash
python scripts/aggregate_seeds.py
```

Prints training reliability, per-fault across-seed statistics, a high-variance warning when the
per-seed recovery spread exceeds 40 points, and a headroom ranking classifying each fault as
`CEILING` / `moderate headroom` / `large headroom`. Writes
[`logs/across_seed_summary.csv`](logs/across_seed_summary.csv).

It reads `results/manifest.json` to determine which seeds converged and **excludes the rest**,
printing what it dropped and why. Result directories persist across runs, so a seed that passed
under an older, looser criterion will leave a stale CSV on disk; globbing alone would silently
read it back in and contaminate the summary.

### Inspect a single policy

```bash
python scripts/evaluate_policy.py --model models/seed_0 --render --episodes 3
```

```bash
python scripts/evaluate_policy.py --model models/seed_0 --fault torque_limit --fault_step 150 --fault_severity 0.2
```

```bash
python scripts/diagnose_gait.py --model models/seed_0 --steps 1000
```

```bash
python scripts/check_drift.py --model models/seed_0 --episodes 2
```

`diagnose_gait.py` is the numerical counterpart to watching the GUI: per-joint angle ranges (a
real gait needs tens of degrees of swing, not a few), ground-contact variation (a constant count
means all feet stay planted and the robot is sliding), and net displacement versus target speed.

`check_drift.py` answers a narrower question: does the robot travel along its own heading, or
crab sideways? The reward tracks velocity projected onto the forward axis while the gait check
reports net displacement — quantities that agree only if the path is straight, and nothing in the
reward penalizes lateral velocity or yaw.

> **Gotchas.**
> - `baseline_fault_eval.py` and `evaluate_policy.py` default to `--model models/base_policy`,
>   which no longer exists — the multi-seed refactor replaced it with `models/seed_N`. Always
>   pass `--model` explicitly.
> - `analyze_baseline.py` reports recovery rate **unconditionally** (recovered ÷ all trials),
>   while `aggregate_seeds.py` reports it **conditional on degradation** (recovered ÷ degraded).
>   The two will disagree on the same data; the conditional figure is the one quoted in
>   [Results](#results-baseline-a). Point `analyze_baseline.py` at
>   `results/seed_N/baseline_fault_results.csv`, not its default path.

### Training options worth knowing

```bash
python scripts/train_base_policy.py --seed 0 --timesteps 1500000 --save_path models/seed_0 --log_dir logs/seed_0
```

| Flag | Default | Notes |
|---|---|---|
| `--seed` | `0` | Set explicitly on every run. Seeds Python, NumPy, PyTorch, the vec-envs, and PPO. |
| `--timesteps` | `1_500_000` | 500k undertrains: slower, off-target policies and a lower convergence rate. |
| `--n_envs` | `4` | Parallel envs; keep small on a laptop CPU. |
| `--ent_coef` | `0.01` | Entropy bonus. SB3's PPO default is `0.0`, which lets action noise collapse as soon as the policy finds anything locally stable — standing still, typically. |
| `--log_format` | `csv` | Writes a directly plottable `progress.csv`. `tensorboard` uses an async writer thread that has proven fragile on Windows: it can die mid-run and take training down with it. `none` for stdout only. |

PPO hyperparameters: `n_steps=2048`, `batch_size=256`, `n_epochs=10`, `lr=3e-4`, `γ=0.99`,
`gae_lambda=0.95`, `clip_range=0.2`, `net_arch=[128, 128]`. Checkpoints every ~50k timesteps.

**Seeding is verified.** Re-running a seed with identical settings reproduces the policy
bit-identically — the same speed, survival, and joint-range figures to four decimal places. That
is a reproducibility property worth stating in the paper, and it is also what makes accidental
configuration drift detectable.

---

## How the code fits together

Exactly one file contains real quadruped logic. Everything in `scripts/` is a thin driver that
imports it and does one job. Dependencies point one way only: scripts import the environment;
the environment imports nothing from scripts.

```
                          envs/quadruped_env.py
             physics · observations · reward · fault injection
                                    │
      ┌──────────┬───────────┬──────┴─────┬─────────────┬──────────────┐
      │          │           │            │             │              │
 smoke_test  check_reset  train_base  evaluate_    diagnose_    baseline_fault
             _pose        _policy     policy       gait         _eval
      │          │           │            │        check_drift        │
 "does it    "is the      "learn to   "watch it"   "is it REALLY   "how does it
  run?"       pose sane?"  walk"                    walking?"       fail?"


 run_multiseed.py    ── orchestrates ──▶  train → gait check → fault eval, per seed
 analyze_baseline.py ── reduces ──▶       one seed's trials  → console + summary CSV
 aggregate_seeds.py  ── reduces ──▶       all seeds' trials  → logs/across_seed_summary.csv
```

---

## Known limitations & open issues

Stated plainly because they determine what the next round of experiments has to address.

1. **`actuation_delay` may be a floor effect.** 0% recovery on every seed and trial gives an
   adaptation method maximum headroom but may not discriminate between methods either. Running
   `--sweep` on one seed settles whether a milder lag is partially recoverable.

2. **Systematic velocity overshoot.** Converged seeds average 0.585 m/s against a 0.5 m/s
   command — 17% high, consistently, across all six. Because the gait check measures net
   displacement while the reward tracks forward-axis velocity, these diverge if the robot drifts
   laterally, and the reward contains no lateral or yaw penalty. `check_drift.py` resolves it;
   the fix would require retraining.

3. **Weak velocity-tracking gradient.** Walking at target scores ≈ +0.97/step, undershooting by
   38% scores ≈ +0.80/step — an ~18% incentive gap, against a fall cost of −10 plus every
   forfeited alive bonus. The policy is strongly pushed not to fall and only weakly pushed to
   track speed, which plausibly explains both the `off_target` failures and the speed spread. An
   exponential tracking reward (Rudin et al., 2022) would sharpen it while staying positive.

4. **n = 6 converged seeds.** Workable, thin for a method comparison. The adaptation comparison
   is paired (same base policies with and without adaptation), which recovers considerable
   power, but more seeds would be better. Expect ~40% attrition when planning.

5. **Random joint selection adds uncontrolled variance.** Per-joint faults draw a random joint
   each trial, so 100 trials spread across 12 joints is ~8 per joint. Whether a hip or a calf is
   affected is currently invisible in the results. A systematic per-joint sweep would both remove
   the variance and answer "which joints are critical?"

6. **Sensor faults touch joint angles only.** Velocities, base orientation, and base velocities
   are never corrupted, so the policy always keeps a clean view of its body state.

7. **`torque_limit` caps a position controller's force budget**, which is related to but not the
   same as true torque saturation on a torque-controlled actuator.

8. **`max_torque` is hardcoded in three places** in `envs/quadruped_env.py` — the reset settle
   loop, `step()`, and the `getattr` fallback in `_compute_reward()`. They currently agree at
   20 N·m. Note the asymmetry: the reward fallback would pick up a `self.max_torque` attribute
   automatically while `step()` would not, so adding one without updating all three would make
   the reward normalize against a different value than physics uses.

9. **Single robot, flat terrain, simulation only.** No terrain variation, no domain
   randomization, no sim-to-real claim is being made.

10. **`check_reset_pose.py` prints "expected ~0.30" for the reset height.** The correct settled
    standing height is ~0.558 m. The printed measurement is right; the parenthetical hint is
    stale.

---

## Roadmap

- [x] Redefine the recovery criterion (sustained window; degradation required; falls disqualify).
- [x] Seed training reproducibly and record provenance alongside every model.
- [x] Screen seeds on velocity tracking, over multiple episodes.
- [x] Exclude non-converged seeds from aggregation via the manifest.
- [ ] Run `--sweep` on one seed to determine whether `actuation_delay` is saturated, then fix
      final severities.
- [ ] Run `check_drift.py` to establish whether the 0.585 m/s figure is forward velocity or
      includes lateral drift.
- [ ] **Fix the held-out fault split for H3 and write down the reasoning — before the adaptation
      module exists.** Current candidate: train on `torque_limit`, `joint_lock`, `sensor_noise`;
      hold out `actuation_delay` and `sensor_dropout`. Both categories are represented in the
      held-out set, and the most structurally distinct fault (temporal, not magnitude) is held
      out.
- [ ] Implement the residual adaptation module on the `_apply_fault_to_action` hook.
- [ ] Implement Baseline B (full policy retraining post-fault) for the upper-bound comparison.
- [ ] Add a domain-randomization baseline — reviewers will ask why randomized training isn't
      sufficient on its own.
- [ ] Add statistical tests over the per-seed CSVs (paired, across seeds; bootstrap CIs rather
      than mean ± sd at this sample size — see Agarwal et al., 2021).
- [ ] Add a `sensor_bias` / drift fault using the existing `_sensor_bias` state.
- [ ] Pin dependency versions in `requirements.txt`; add a `.gitignore` for `results/` and
      `logs/seed_*/`.

---

## Notes

No license file is currently specified — please get in touch before reusing this work.
Conducted through the UC Santa Cruz AIEA Lab.