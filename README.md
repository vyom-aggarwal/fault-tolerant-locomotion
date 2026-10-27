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
| PPO base policy, 10 seeds @ 1.5M steps | **Built + run** — all 10 converge, `models/seed_0..9.zip` |
| Baseline A (fault, *no* adaptation), 700-step window | **Built + run** — see [Results](#results-baseline-a) |
| Across-seed aggregation & headroom analysis | **Built + run** — [`logs/across_seed_summary.csv`](logs/across_seed_summary.csv) |
| Constant-offset residual correction (integral / RLS) | **Built + probed on seed 0** — covers a *partial* share of the deficit; replication on seeds 1, 4, 7 pending |
| Adaptive gait-amplitude controller (AIMD on tilt) | **Built, tested, negative result** — a constant amplitude reduction beats it on held-out seeds |
| Paired statistical analysis of the comparison | **Built + run** — [`scripts/analyze_deployment.py`](scripts/analyze_deployment.py) |
| Baseline B (full policy retraining) | **Not built yet** |
| Held-out fault split for H3 | **Not yet fixed** |

**Where the project stands.** Baseline A leaves real headroom (unadapted recovery from 0% for
`actuation_delay` to 65% for `sensor_noise`). The first adaptation mechanism tried, a reactive
amplitude controller, does not beat simply walking with a constant, smaller gait amplitude: on
held-out seeds 1–9, `fixed 0.80` is faster than every adaptive configuration with intervals
excluding zero, and no difference in falls was detected. The open question is whether any
*adaptive* method (direct delay estimation, a fault-conditioned policy, or a better residual) can
beat a constant. The residual module that H1–H3 describe is still to be built.

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

Base policy, **no adaptation**, fault injected mid-episode. Numbers below are from the corrected
protocol: a 700-step (11.7 s) post-fault window and a settled-deficit metric (see
[Baseline A protocol](#baseline-a-protocol)). Earlier numbers from a 300-step window are superseded.

### Training reliability

| | |
|---|---|
| Seeds attempted | 10 (seeds 0–9), 1.5M timesteps each |
| Converged to a usable gait | **10 / 10** (`results/manifest.json`) |
| Forward speed | **0.544 ± 0.039 m/s** against a 0.5 m/s command |
| Influence matrices | `models/seed_N_influence.npz`, fitted for all 10 seeds |

All ten seeds pass the gait check, so every analysis below uses all ten. Non-converged seeds, if
they ever occur again, are still excluded by `aggregate_seeds.py` via the manifest, with the
failure recorded as `failure_mode` (`off_target` or `unstable`).

### Fault response

**The unit of analysis is the seed.** Each of the 10 seeds contributes one number per metric
from about 100 trials; the ± is the spread *across seeds*, not across trials.

| Fault (severity) | Degraded | Velocity drop | **Recovery** | Fall rate | Settled deficit (m/s) | Recovery time (s) |
|---|---|---|---|---|---|---|
| `actuation_delay` (10 steps) | 99.9% | 100.2% | **0.0% ± 0.0%** | 97.7% ± 2.7% | 0.321 ± 0.048 | — |
| `torque_limit` (0.2×) | 84.2% | 64.0% | **20.6% ± 6.2%** | 40.8% ± 8.6% | 0.117 ± 0.048 | 6.64 ± 0.47 |
| `joint_lock` | 85.2% | 57.5% | **29.6% ± 4.8%** | 8.1% ± 12.3% | 0.137 ± 0.035 | 7.37 ± 0.48 |
| `sensor_dropout` | 60.3% | 40.8% | **56.9% ± 10.0%** | 5.8% ± 9.3% | 0.019 ± 0.014 | 6.84 ± 0.47 |
| `sensor_noise` (σ = 0.3) | 59.4% | 38.7% | **65.0% ± 13.1%** | 4.1% ± 9.0% | 0.009 ± 0.014 | 6.57 ± 0.59 |

*Recovery rate is **conditional on degradation**: of the trials where the fault measurably slowed
the robot, the fraction that returned to pre-fault speed and held it. Settled deficit is
pre-fault speed minus the speed in the settled part of the window.*

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
   (56.9%) < `sensor_noise` (65.0%). The physics/sensor split is an empirical property of the
   system, which makes it a defensible axis for the H3 split. Sensor faults leave settled deficits
   of only 0.009–0.020 m/s, so there is little for any method to recover there.

2. **The longer window changed the picture.** With the 700-step window, recovery rates fell
   sharply (e.g. `sensor_noise` 86.7% → 65.0%, `joint_lock` 54.9% → 29.6%) because early returns
   to speed often do not hold. `actuation_delay` is now also a fall problem: **97.7%** of trials
   end in a fall, where the 300-step window showed 0%.

3. **`torque_limit` is the most dangerous non-delay fault.** It has a 40.8% fall rate and a
   0.117 m/s settled deficit with the lowest recovery of the non-delay faults (20.6%).
   `joint_lock` has the largest settled deficit among them (0.137 m/s) but rarely topples the
   robot (8.1%). Between-seed spread is now modest for both; `sensor_noise` varies most (36–79%).

4. **`actuation_delay` is saturated at severity 10.** It slows the robot to near zero or
   topples it on every seed. The severity sweep and `eval_amplitude`/deployment severities
   (2–20) are what characterize it; see [Adaptation work](#adaptation-work-so-far).

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

  -- adaptation research (after Baseline A) --
  adaptation.py             ResidualAdapter + RLSEstimator: bounded joint-space
                            correction from an influence matrix (library)
  fit_influence_matrix.py   identify the action -> body-velocity influence matrix
                            per policy, with a split-half agreement check
  probe_authority.py        how much can a joint-space offset change body velocity?
  measure_fault_deficit.py  settled post-fault velocity deficit vs. offset authority
  probe_amplitude.py        does scaling gait amplitude recover velocity / stability?
  amplitude_adapter.py      AmplitudeAdapter: none | fixed | adaptive action scale (library)
  eval_amplitude.py         none vs fixed vs adaptive, per fault/severity, one or all seeds
  sweep_adapter.py          grid over adapter hyperparameters on one fault
  frontier_compare.py       fall-rate vs speed Pareto front, fixed scales vs adaptive
  measure_latency.py        detection / safe-scale / fall timing after fault onset
  deployment_compare.py     one committed config across severities 2-20, per policy
  analyze_deployment.py     paired Wilcoxon + Holm + bootstrap on deployment.csv

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

Adaptation scripts write to `logs/` by default (`logs/amplitude_eval.csv`, `logs/frontier.csv`,
`logs/deployment.csv`); `logs/ad_sweep.csv` and `logs/sweep_seed0.csv` hold severity-sweep output.
`models/seed_N_influence.npz` holds each policy's fitted influence matrix.

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
| Pre-fault | 0 – onset (random in 150–250) | policy walks; trial is **discarded** if it falls here |
| Baseline measurement | last 120 pre-fault steps | `baseline_vel` = mean forward velocity |
| Injection | onset step | `trigger_fault(type, severity)`, random joint |
| Post-fault window | 700 steps (11.7 s) | degradation, recovery, settled deficit, and falls are scored |

The window was extended from 300 to 700 steps so that "recovered" must *hold* and falls that take
several seconds to develop (notably under `actuation_delay`) are observed. The onset is drawn per
trial from the trial seed, so every candidate evaluated with the same seed sees the same onset.

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
actuation_delay  1, 2, 3, 5, 10, 20
sensor_noise     0.1, 0.3, 0.6, 1.0
joint_lock       (single point; use --n_joints for severity)
sensor_dropout   (single point; use --n_joints for severity)
```

Trial `i` uses seed `i`, and the same seed set is reused across every fault type, so fault types
are compared on matched initial conditions.

---

## Adaptation work so far

Baseline A showed the faults leave headroom; the work since then asks which adaptation mechanism
can use it. Everything sits on the frozen base policy; nothing is retrained. The short version:
**a constant, reduced gait amplitude is a strong baseline that the adaptive controller does not
beat.**

### 1. Residual correction (`adaptation.py`)

`ResidualAdapter` adds a bounded joint-space offset `delta` (|delta| ≤ `delta_max` per joint) to
the policy's action. An integral controller drives smoothed body velocity toward the command
through a ridge pseudo-inverse of an influence matrix `B` (action → forward/lateral/yaw velocity).
`mode="fixed"` uses an offline `B` from `fit_influence_matrix.py` (archived per seed as
`models/seed_N_influence.npz`); `mode="rls"` refines it online with a forgetting-factor
`RLSEstimator`.

`probe_authority.py` and `measure_fault_deficit.py` ask how much speed a bounded constant offset can
buy back. On seed 0 the best delivered forward correction is **0.055 m/s**, against the largest
settled deficit from Baseline A (0.137 m/s, `joint_lock`): about **40% — partial, not
insufficient**. `measure_fault_deficit.py` independently finds `torque_limit` and `joint_lock`
52–56% correctable. A constant offset therefore cannot restore commanded speed, but it is not
useless. Both probes originally compared against a 0.27 m/s worst *transient* dip, which overstated
the target about 2.5×; they now use settled deficits from the 10-seed Baseline A. Replication on
held-out seeds 1, 4, and 7 (chosen in advance) is pending:

```bash
python scripts/probe_authority.py --model models/seed_N
python scripts/probe_amplitude.py --model models/seed_N --trials 12 --offset_authority <value from probe_authority>
```

### 2. Gait-amplitude scaling (`amplitude_adapter.py`)

Scale the policy's joint-action amplitude (`env._action_scale * s`), trading speed for stability.
`probe_amplitude.py` measures fixed scales per fault. `AmplitudeAdapter` has three modes:
`none` (`s = 1`), `fixed` (constant `s`), and `adaptive` (`s ∈ [0.5, 1.0]`).

The adaptive rule needs no fault detector. It calibrates on the robot's own first 2 s of walking,
sets a risk threshold at `mean + k·sd` of healthy risk, and treats smoothed torso tilt
(`1 − upright_alignment`) plus `osc_weight` × its short-window standard deviation as risk. Above
the threshold it backs off (proportionally to the excess by default); otherwise it recovers
additively by `increase` per step. This is additive-increase / multiplicative-decrease (AIMD).

### 3. Is adaptation worth it? Four comparisons, one answer

| Script | Question | Finding |
|---|---|---|
| `frontier_compare.py` | At a *fixed* severity, does adaptive reach a better fall-rate/speed Pareto front than constant scales? | Severity 5: no adaptive point on the front. Severity 10: adaptive appears on the front only in a sampling gap between fixed scales; at n = 20 fixed 0.70–1.00 are statistically indistinguishable. Comparing one adaptive point to one constant (the earlier approach) was invalid because each mechanism has its own trade-off curve. |
| `measure_latency.py` | Is the controller simply too slow to react? | **No.** Detection takes 0.6–1.1 s and reaching a safe scale about 1.3 s, with a budget of 2.5–6.5 s before the robot falls: roughly 10× the time needed. |
| `deployment_compare.py` | With one configuration committed *before* a fault of unknown severity (2–20), does adaptive win? | Constants win. On seed 0 `fixed 0.80` has equal or lower falls and higher speed than every adaptive configuration, and every adaptive controller falls 100% at severity 20. |
| `analyze_deployment.py` | Does that hold on policies it was not tuned on, under paired statistics? | Yes on seeds 1–9 (below). |

**Why the controller loses.** It detects quickly and backs off correctly, then climbs back
(`increase = 0.0015`) to a scale that falls about 60% of the time. AIMD assumes conditions improve;
these faults are permanent, so the safe amplitude is permanently lower and any recovery is wrong
by construction. Back-off-and-hold (`increase = 0`) also fails: the fault transient alone drives
the scale to its floor at every severity, which is a fixed minimum with extra steps. `increase`
trades conservatism for speed exactly as a fixed scale does. Three successive explanations for
the controller's behaviour (sweep defaults, detection latency, back-off-and-hold) were each wrong
and each caught by measurement.

**Held-out result (seeds 1–9, `actuation_delay`, severities 2–20, 15 trials per cell).**
The policy is the unit of analysis; tests are exact two-sided Wilcoxon signed-rank with Holm
correction, and bootstrap 95% intervals resample policies. The confirmatory claim (`fixed 0.80`
dominates every adaptive configuration) was fixed from the seed-0 tuning run *before* the
held-out run; the family is 2 metrics × 3 configs = 6 tests.

| Adaptive config | Speed gain of `fixed 0.80` | 95% CI | Holm adj. p | Fall difference | adj. p |
|---|---|---|---|---|---|
| k = 1.0 | +0.091 m/s | [+0.063, +0.117] | 0.023 | +0.4 pt | 1.00 |
| k = 2.0 | +0.075 m/s | [+0.050, +0.102] | 0.023 | −0.8 pt | 1.00 |
| k = 3.0 | +0.072 m/s | [+0.042, +0.097] | 0.031 | −1.6 pt | 0.56 |

All three are dominated: `fixed 0.80` is faster, and no difference in falls was detected. The
claim is that the constant is not shown to be safer, only not detectably riskier. The result is
robust to excluding severity 20, which is saturated for every candidate on held-out policies.
Exploratory only: `fixed 0.70` shows a 6–9 point fall advantage over every adaptive configuration
with intervals excluding zero, which does not survive correction across the full family of 24
comparisons (the smallest achievable exact p at n = 9, 0.0039, exceeds the corrected threshold).

**Scope.** This establishes that AIMD on a tilt-oscillation signal does not beat a constant. It
does not show that no adaptive method could. Directly estimating the delay, or a fault-conditioned
policy, remain untested. A constant chosen to survive severity 10 is nearly immobile at severity 2
(`fixed 0.55` manages about −0.04 m/s on seed 0), which is the remaining argument for adaptation.
At severities 10 and 20 settled velocity is negative, meaning upright but drifting backward, which
is not restored locomotion.

```bash
python scripts/frontier_compare.py --model models/seed_0 --fault actuation_delay --severity 5 \
    --fixed_scales 0.5,0.55,0.6,0.65,0.7,0.75,0.8
python scripts/measure_latency.py --model models/seed_0 --fault actuation_delay --severity 5
python scripts/deployment_compare.py --model models/seed_0 --trials 15
python scripts/deployment_compare.py --all --exclude 0 --trials 15     # held-out seeds
python scripts/analyze_deployment.py --csv logs/deployment.csv --confirm "fixed 0.80"
python scripts/eval_amplitude.py --model models/seed_0 --trials 30     # none / fixed / adaptive
python scripts/sweep_adapter.py --model models/seed_0                  # hyperparameter grid
```

Outputs: `logs/frontier.csv`, `logs/deployment.csv`, `logs/amplitude_eval.csv`, `logs/sweep_seed0.csv`.
`deployment_compare.py` runs 8 candidates (`none`, fixed 0.80/0.70/0.65/0.55, adaptive
k = 1.0/2.0/3.0), each held unchanged across severities. `analyze_deployment.py` validates its
own statistics against scipy and statsmodels reference implementations.

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

The adaptation scripts follow the same rule: `adaptation.py` and `amplitude_adapter.py` are
libraries holding the controller logic; `eval_amplitude.py`, `sweep_adapter.py`,
`frontier_compare.py`, and the `probe_*` / `fit_*` / `measure_*` scripts are drivers that import
the environment, one adapter, and (for the amplitude scripts) `baseline_fault_eval`'s onset and
recovery-scoring helpers.

---

## Known limitations & open issues

Stated plainly because they determine what the next round of experiments has to address.

1. **`actuation_delay` is saturated at severity 10.** It is a 97.7% fall rate on Baseline A and
   100% for every candidate at severity 20, so it discriminates poorly between methods there.
   Severities 2–5 are where methods can differ.

2. **Systematic velocity overshoot.** Seeds average 0.544 m/s against a 0.5 m/s command, about 9%
   high. Because the gait check measures net displacement while the reward tracks forward-axis
   velocity, these diverge if the robot drifts laterally, and the reward contains no lateral or
   yaw penalty. `check_drift.py` resolves it; the fix would require retraining.

3. **Weak velocity-tracking gradient.** Walking at target scores ≈ +0.97/step, undershooting by
   38% scores ≈ +0.80/step — an ~18% incentive gap, against a fall cost of −10 plus every
   forfeited alive bonus. The policy is strongly pushed not to fall and only weakly pushed to
   track speed, which plausibly explains both the `off_target` failures and the speed spread. An
   exponential tracking reward (Rudin et al., 2022) would sharpen it while staying positive.

4. **n = 10 seeds.** Enough for paired exact tests, but the smallest achievable exact Wilcoxon p
   at n = 9 held-out policies is 0.0039, so large exploratory families cannot reach significance.
   Fix the family before looking at results.

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

10. **Seed 0 is the tuning seed.** Adapter defaults came from sweeps on seed 0. Report held-out
    results on seeds 1–9 (`--exclude 0`); the seed-0 numbers are development results.

11. **The adaptation evidence is mostly `actuation_delay`.** The deployment comparison and
    frontier use one fault type. Constant amplitude reduction is shown to beat AIMD there; it has
    not been tested on `torque_limit` or `joint_lock` under the same protocol.

12. **Amplitude scaling is not residual adaptation.** It changes gait amplitude, not a learned
    correction. It is a baseline for the residual module, which is still to be built.

13. **Offset-probe verdicts are seed-0 only.** The 40% partial-coverage figure needs replication
    on seeds 1, 4, and 7.

14. **`check_reset_pose.py` prints "expected ~0.30" for the reset height.** The correct settled
    standing height is ~0.558 m. The printed measurement is right; the parenthetical hint is
    stale.

---

## Roadmap

- [x] Redefine the recovery criterion (sustained window; degradation required; falls disqualify).
- [x] Seed training reproducibly and record provenance alongside every model.
- [x] Screen seeds on velocity tracking, over multiple episodes.
- [x] Exclude non-converged seeds from aggregation via the manifest.
- [x] Extend the Baseline A window to 700 steps and report settled deficits.
- [x] Compare adaptive vs constant amplitude on frontiers, then under deployment conditions.
- [x] Replicate the comparison on held-out seeds 1–9 with paired, corrected statistics.
- [ ] Replicate `probe_authority.py` / `probe_amplitude.py` on seeds 1, 4, 7.
- [ ] Run `check_drift.py` to establish whether the 0.544 m/s figure is forward velocity or
      includes lateral drift.
- [ ] **Fix the held-out fault split for H3 and write down the reasoning — before the adaptation
      module exists.** Current candidate: train on `torque_limit`, `joint_lock`, `sensor_noise`;
      hold out `actuation_delay` and `sensor_dropout`. Both categories are represented in the
      held-out set, and the most structurally distinct fault (temporal, not magnitude) is held
      out.
- [x] Prototype a residual (integral / RLS) correction and probe its authority.
- [x] Prototype an adaptive gait-amplitude controller; add a fixed-scale control, tuning sweep,
      and Pareto-front comparison.
- [ ] Test an adaptive method that is not AIMD: estimate the delay directly, or condition the
      policy on the fault, and compare against constant amplitude on the same protocol.
- [ ] Extend the constant-vs-adaptive comparison beyond `actuation_delay`.
- [ ] Implement the residual module on the `_apply_fault_to_action` hook and compare it to the
      constant-amplitude baseline.
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