# ISAL-Humanoid

Independent Isaac Lab workspace for interaction-supervised affordance learning on humanoid rough-terrain locomotion.

The Stage 1 task is a behavior-preserving copy of RoboLab's `RPO-Rough` baseline. Stage 2 adds a separate ordinary-PPO task whose Actor also receives a root-relative RayCaster height scan. Environment, MDP, terrain, and agent implementation live in this package. Robot configuration and URDF/data assets are loaded read-only from `robolab.assets` (with a monorepo namespace fallback).

Registered tasks:

- `ISAL-Humanoid-Rough-v0`: frozen Stage 1 proprioceptive Actor baseline.
- `ISAL-Humanoid-Rough-HeightScan-v0`: Stage 2 Actor height-scan baseline.
- `ISAL-Humanoid-Rough-Interaction-v0`: Stage 3 ordinary-PPO task with interaction label collection.

## Install

```powershell
conda run -n env_isaaclab python -m pip install -e .\isal --no-deps
```

## Validate without training

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless `
  --num_envs 1 `
  --validate-only
```

`--validate-only` constructs the environment, RSL-RL wrapper, policy, algorithm, and runner, resets the environment, performs one fixed zero-action step, and exits before `runner.learn()`.

The Stage 2 observation contract is:

```text
policy       (num_envs, 780)   # 10-frame proprioception history
height_scan  (num_envs, 187)   # current 17 x 11 canonical grid, flattened for standard ActorCritic
critic       (num_envs, 3260)  # 10-frame privileged history with clean scan
actor input  (num_envs, 967)   # policy + height_scan
```

These are the default dimensions. The shared scan preprocessing is now
`clamp(terrain_z - root_z, -1.5, 0.4) / 0.5`, with output in `[-3.0, 0.8]`.
It preserves descending terrain heights that previously saturated at `-0.8 m`
relative to the root. Actor, Critic and interaction snapshots use this same
definition. Non-finite rays still receive the lower-bound fill value, but are
counted separately from finite lower saturation.

### Change scan geometry or observation history

Edit `env_cfg.terrain_perception.size`, `resolution`, and `offset_x` (the
authoritative geometry fields), and `env_cfg.robot.actor_obs_history_length` /
`critic_obs_history_length`. Set native ordering on
`env_cfg.scene.height_scanner.pattern_cfg.ordering` (`xy` or `yx`). Final overrides
are resolved again at environment construction, preserving other scene settings.
Actual Isaac Lab pattern output determines the grid and Critic frame dimension;
no independent rounding formula is used.

For example, this non-training check uses a 19 x 11 grid and different histories:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx
```

The environment exposes `perceptive_observation_layout` with the resolved grid,
ordering, frame dimensions, and history lengths. Actor state remains 78 per
frame; Critic state before scan remains 139 per frame. The new-task mirror
restores each history frame, mirrors state, converts native scan to canonical
`(x,y)`, flips y, then restores native ordering. State dimensions and input
lengths are checked explicitly. Stage 1 retains its legacy augmentation path.

Default network dimensions are unchanged, but the scan distribution has changed.
To reproduce an old experiment, explicitly set `env.terrain_perception.min_height=-0.8`.
Existing checkpoints are not converted; changing ray count also changes network
input dimensions and requires a compatible checkpoint or a new external run.

## Inspect the RayCaster grid without training

Run without `--headless` to view the height-scan markers. The robot receives zero actions and no runner is created:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\debug_height_scan.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --num_envs 1 `
  --steps 1000
```

## Training (external training machine only)

Do not run training on the local development machine.

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\rsl_rl\train.py `
  --task ISAL-Humanoid-Rough-HeightScan-v0 `
  --headless `
  --num_envs 4096
```

## Stage 3: collect interaction labels without training

The new task inherits the Stage 2 observation, action, reward, termination,
terrain, command, randomization, symmetry, and PPO settings. It adds no model,
optimizer, auxiliary loss, or policy update. All sampler settings are in
`isal/interaction/config.py` (`SelfSupervisedCfg`), exposed through
`env_cfg.self_supervised`. Setting `enabled=False` creates no tracker and
returns the Stage 2 observation/extras interface.

Run from the repository root. This script creates no runner and applies only
zero actions; omit `--headless` for a viewer:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal\scripts\debug_interactions.py `
  --headless --num_envs 1 --steps 200 `
  --max_samples 1000 --output outputs\stage3_debug
```

`--num_envs` is restricted to 1–4. The default output is a timestamped folder
under `outputs/`; an explicit output directory is reused. Exports are
`summary.json` (all samples and event counters), `samples.csv` (retained labels
and query coordinates), and `samples.pt` (retained snapshots, contexts and
diagnostics). Retention is capped at 1000 examples; summary statistics count
every emitted sample. No complete gait samples is a valid zero-action result,
not evidence of walking or learning. Synthetic test feedback is never mixed
into the debug export.

### Scan and touchdown-neighborhood diagnostics

The height-scan debug script and `--validate-only` print current-scan point counts
and fractions. Repeated observation reads do not accumulate statistics.
Interaction packets and debug exports additionally contain:

```text
scan_diagnostics_available                  (N,2,P) bool
{scan,query}_{total,finite,lower,upper,invalid}_count
                                           (N,2,P) int64
{scan,query}_{lower,upper,invalid}_fraction  (N,2,P) float32
```

`scan` means the full liftoff snapshot; `query` means the closest grid point to
the touchdown query plus its one-ring neighbors (at most 3 x 3). Borders are
clipped without repeating points. Exact distance ties select the smaller grid
coordinate. Query diagnostics use the liftoff coordinate system and the saved
raw finite/lower/upper masks, never the current touchdown scan. Only compact
counts remain in pending records after touchdown.

Finite heights at or below/above the clipping bounds count as lower/upper
saturation, including equality. Saturation fractions divide by finite points;
invalid fraction divides by total points. Zero finite points gives zero
saturation fractions and an explicit zero finite count. These diagnostics do
not affect observations, labels, validity, rewards or learning.

`summary.json.scan_diagnostics` reports available/missing sample counts and
point-weighted totals/fractions for both scopes; it does not average per-sample
ratios. Missing diagnostics in old packets remain supported: PT uses an explicit
availability flag, CSV leaves unavailable diagnostic cells blank, and summaries
use `available=false`, zero counts and null fractions when no diagnostic samples
exist. The same representation is used for empty runs; no NaN is emitted.

### Sampling contract

The noiseless root-relative Actor scan is copied at liftoff. Each foot can
continue a new swing while older contacts await their survival outcome. The
environment samples once per control step before automatic reset, and publishes
the packet after reset. Reading observations never advances the tracker.

At the default 50 Hz, Python `round` gives 12 outcome frames and 25 survival
steps. Touchdown is outcome frame one / survival age zero; the positive survival
label becomes available 25 control steps later. There are 14 pending slots per
foot by default. Slot capacity is derived from the survival window; explicitly
smaller capacities report overflow rather than replacing existing records.

`extras["auxiliary"]` uses this fixed tensor interface on the environment device:

```text
valid                         (N, 2, P)             bool
height_scan                   (N, 2, P, 1, H, W)    float32
query_xy, foot_side            (N, 2, P, 2)          float32
command, base_ang_vel,
projected_gravity              (N, 2, P, 3)          float32
target                        (N, 2, P, 1)          float32
observed_frames               (N, 2, P)             int64
partial_window                (N, 2, P)             bool
slip_mean, tilt_change,
persistence, survival,
slip_score, tilt_score,
peak_force                    (N, 2, P)             float32
```

The foot axis is left then right; `foot_side` is a corresponding one-hot.
Query xy is in metres in the liftoff root's yaw frame, not relative to the scan
centre. Command and angular velocity are physical unscaled values, angular
velocity/gravity are body-frame values. Height scan alone uses Stage 2 physical
clip/scale preprocessing. Invalid slots are zero. Returned packets own their
data and survive subsequent steps/resets. A future rollout buffer should select
`packet[key][packet["valid"]]`, flattening all three leading axes. The tracker
contains unfinished physical events, not replay data used for learning.

`extras["interaction_stats"]` contains cumulative GPU scalar event/drop counts,
current pending count, and peak pending count per foot. Explicit reset clears
the selected environment's live state and old output, while keeping cumulative
statistics. Automatic reset preserves just-finalized samples for that step.

### Label interpretation

- Contact uses positive world-z force greater than 20 N, a support-force proxy.
  Link position and velocity both refer to the named ankle-roll link frame;
  the query is not an estimated centre of pressure or a sole contact point.
- Slip is mean horizontal speed over contacted outcome frames. Tilt is the
  maximum wrapped roll/pitch change relative to touchdown. Contact persistence
  divides contacted frames by the full outcome window.
- Target is `0.40*exp(-slip/0.15) + 0.25*exp(-tilt/0.20) +
  0.20*persistence + 0.15*survival`, clamped to `[0,1]`.
- A fall immediately finalizes valid contacts, including early partial windows.
  Missing contact frames count as zero persistence; slip/tilt use only observed
  frames. Such samples carry `partial_window=True`. Timeout is not a fall;
  incomplete timeout/manual-reset samples are discarded. A failure before a
  valid touchdown never invents a query or a label.
- Terrain metadata is never used for labels. Peak force is diagnostic only.
  An early-fall target need not equal zero because it combines four scores.

### Stage 3 verification

All tests remain non-training:

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal\tests -q --tb=short -p no:cacheprovider
```

The simulator and pytest need access to their normal USD/Kit and temporary
cache directories. Each Stage 3 simulation case runs in a fresh subprocess,
also explicitly through `conda run -n env_isaaclab python`. This avoids local
Isaac Sim stalls on repeated scene creation and config validation traversing
the previous RayCaster mesh cache into Warp runtime cycles. No upstream source
or global runtime class is patched. See `STAGE3_ACCEPTANCE.md` for results and
remaining gates.

The 2026-09-07 perception improvements and their verification are recorded in
`PERCEPTION_IMPROVEMENTS_ACCEPTANCE.md`. Earlier stage acceptance records remain
historical records of the earlier implementation and preprocessing values.

## Stage 4A: CNN policy and training-only query head

Two additional tasks use the same interaction environment and ordinary PPO:

| Task | Tracker | Query head | Auxiliary optimization |
|---|---|---|---|
| `ISAL-Humanoid-Rough-CNN-v0` | Off | Absent | Off |
| `ISAL-Humanoid-Rough-CNN-Aux-v0` | On | Present | Off |

**Stage 4A does not train the head.** Its effective auxiliary coefficient is zero
because ordinary PPO's auxiliary loss is disabled. The Aux task only exposes
interaction packets and a callable predictor. Current-rollout storage, loss
integration, and schedules belong to Stage 5. Predicted affordance does not enter
the Actor; this is not the Stage 4B affordance-observation policy.

Actor input remains proprio history plus a separate canonical flattened scan.
The model restores the actual grid for its CNN. The Critic retains every history
frame, converts each native-order clean scan to the canonical grid, and encodes
each frame with a Critic-only CNN. Actor/Critic parameters and state normalizers
are independent. Only the query head shares the Actor terrain encoder.

Scans use only `clamp(terrain_z-root_z, -1.5, 0.4)/0.5`. Running normalization is
explicitly enabled for proprio/critic non-scan history only. Query context uses
the Stage 3 physical-unit snapshot, not a slice of normalized Actor history.
`predict_affordance(batch)` accepts valid samples flattened across `[N,2,P]`;
it never consumes targets as inputs or updates normalization/distribution state.

### Parameters and model construction

- `isal/learning/config.py`: convolution channels/kernels/strides/padding,
  terrain FC/latent dimensions, proprio/Critic-state/head hidden dimensions.
- `tasks/direct/humanoid_rough/agents/affordance_agent_cfg.py`: body hidden
  dimensions, explicit normalization flags, action distribution, head switch,
  and ordinary PPO inheritance. Network overrides use `agent.policy.network.*`.
- `tasks/direct/humanoid_rough/terrain_perception_cfg.py`: scan geometry and
  physical preprocessing; `interaction/config.py`: unchanged sampling/labels.

The project training entrypoint calls
`bind_perceptive_model_config(env, agent_cfg)` after constructing the actual
environment and before the runner. Custom callers must do the same. This binds
the final grid, native ordering, histories and preprocessing after all overrides;
the model refuses an absent/mismatched layout. Other task configurations are not
changed by this bridge. RSL-RL resolves the model by its fully qualified class
name; no upstream registry or source patch is required.

Safe local validation (one inferred action and one **zero-action** environment
step; no learning):

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-v0 `
  --headless --num_envs 1 --validate-only

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-Aux-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx
```

### Checkpoint and actor-only export

Model state contains layout, preprocessing, network and variant metadata plus
normalizer buffers. The current runner saves/restores this with optimizer state
and iteration. Old MLP checkpoints, a changed grid/ordering/preprocessing, or a
different head variant are rejected; there is no implicit migration.

Export without launching Isaac Sim or updating weights:

```powershell
conda run -n env_isaaclab python isal/scripts/export_actor.py `
  --checkpoint outputs/stage4a_acceptance/baseline_default/untrained_checkpoint.pt `
  --output outputs/stage4a_acceptance/cli_export
```

The output contains `actor.pt` (TorchScript), `actor.onnx` (opset 17), and
`metadata.json`. Both formats take `policy[B,78*history]` and
`height_scan[B,H*W]` in **already clipped/scaled canonical x/y order**, returning
`action_mean[B,23]`. Feature dimensions are fixed per checkpoint; batch is dynamic.
The exported graph includes state normalization and both Actor encoders, but no
Critic, query head, or stochastic distribution. Action scaling/PD remain outside
the graph. These acceptance checkpoints are **untrained**, not walking policies.

### Stage 4A verification

```powershell
conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests/stage4a -q --tb=short -p no:cacheprovider
```

CPU/CUDA tests use synthetic forward/backward passes only. Simulator cases run
in independent processes, patch learning/update/optimizer entrypoints to fail
if called, and use at most two environments and 40 fixed zero-action steps each.
See `STAGE4A_ACCEPTANCE.md` for actual results, artifacts and remaining gates.

## Stage 4B: full-grid affordance as an Actor input

| Task | Control features | Tracker/head | Auxiliary optimization |
|---|---|---|---|
| `ISAL-Humanoid-Rough-AffordanceObs-v0` | Detached current full-grid scores | On | Off |
| `ISAL-Humanoid-Rough-AffordanceZero-v0` | Permanent zeros | On | Off |

Both tasks keep Stage 4A's environment, Actor/Critic body, ordinary PPO and
symmetry. The zero-input control has identical parameter shapes; this alone does
not eliminate differences in effective network capacity. Existing 4A tasks retain
their original behavior. **There is no auxiliary optimizer, buffer or schedule in
Stage 4B.** The head is untrained until a later stage supplies auxiliary learning.

Each Actor call encodes its current scan once, then queries both feet at every
actual ray origin (including the existing sensor offset) in current root yaw
coordinates. Scores have shape `[B,2,H,W]`, flattened left foot first. Dense
queries run in chunks of 64 without gradients and reuse the terrain latent.
Context comes from angular velocity, gravity and command in the latest raw,
noisy Actor history frame, divided by the bound observation scales. It never
uses running-normalized history, Critic observations or pending tracker samples.

A bias-free `Linear(2*H*W,256)` projects
`input_gate * (2 * detached_scores - 1)` into the original Actor's first linear
output, before ELU. PPO can update this projection and the direct Actor terrain
path; it cannot update the head. Auxiliary prediction retains the Stage 4A API
and only shares the terrain encoder. Critic is unchanged and independent.

`input_gate` is a checkpointed scalar buffer, **default 0**. Set it explicitly with
`model.set_affordance_input_gate(value)` for values in `[0,1]`; no automatic
schedule exists. Predicted mode still computes the full grid at gate 0. Zero
mode always uses zeros and skips control prediction, regardless of gate; its
`predict_affordance_grid(obs)` diagnostic still returns detached predictions.
Mirroring transforms the original observations and then recomputes predictions.
Do not mirror the recomputed scores a second time.

### Configuration and local validation

`isal/learning/config.py::AffordanceObservationCfg` defines the input mode, gate
and chunk size. Agent configuration overrides are:

- `agent.policy.affordance_observation.input_gate=1.0`
- `agent.policy.affordance_observation.query_chunk_size=64`
- `agent.policy.affordance_observation.input_mode=predicted` (or `zero`)

The existing binding function additionally supplies actual ray coordinates and
observation scales for 4B only. Do not hand-enter inferred grid coordinates.
Common network and perception parameters remain in their Stage 4A locations.

These commands validate full inference but **step the environment with zeros**;
they do not execute a random-head policy in closed loop:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-v0 `
  --headless --num_envs 1 --validate-only `
  agent.policy.affordance_observation.input_gate=1.0

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceZero-v0 `
  --headless --num_envs 1 --validate-only

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  env.scene.height_scanner.pattern_cfg.ordering=yx `
  env.normalization.obs_scales.ang_vel=2.0 `
  env.normalization.obs_scales.projected_gravity=3.0 `
  env.normalization.obs_scales.commands=4.0 `
  agent.policy.affordance_observation.input_gate=1.0
```

### Checkpoint, export and inference overhead

4B checkpoints require the same metadata version, input mode, query coordinates,
scales, network and scan definition. They restore the actual gate, weights and
normalizers. 4A-to-4B migration and cross-mode loading are intentionally rejected.
The export CLI dispatches by metadata and still supports 4A:

```powershell
conda run -n env_isaaclab python isal/scripts/export_actor.py `
  --checkpoint outputs/stage4b_acceptance/predicted_default/untrained_checkpoint.pt `
  --output outputs/stage4b_acceptance/cli_export

conda run -n env_isaaclab python isal/scripts/benchmark_affordance.py `
  --checkpoint outputs/stage4b_acceptance/predicted_default/untrained_checkpoint.pt `
  --output outputs/stage4b_acceptance/benchmark.json

conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider
```

TorchScript and ONNX opset 17 retain the same two input tensors as 4A and support
dynamic batch. Predicted mode includes head, context adapter, queries, projection
and the **saved gate**, even at gate 0. Zero-mode export omits the permanently
ineffective head/projection. Neither includes Critic. Export does not change gate.
The benchmark explicitly uses gate 1 in memory, measures the full Actor path
with batch 1 on CPU/CUDA, and does not rewrite the checkpoint or update weights.

Acceptance checkpoints are untrained engineering artifacts. In particular,
current-phase inference versus liftoff-only supervision, and noisy control context
versus clean tracker context, require external training experiments. Random-head
mirror errors and inference overhead do not establish learning quality. See
`STAGE4B_ACCEPTANCE.md` for measured results and artifact locations.

## Stage 5: current-rollout auxiliary learning

Four new tasks use `isal.learning.affordance_runner:AffordanceRunner` and
`isal.learning.ppo_affordance:PPOWithAffordance`. Earlier tasks remain unchanged.

| Task suffix (after `ISAL-Humanoid-Rough-`) | Tracker/head | Auxiliary loss | Predicted control input |
|---|---|---|---|
| `CNN-Train-v0` | Off | Off | Absent |
| `CNN-Aux-Train-v0` | On | On after warmup | Absent |
| `AffordanceObs-Train-v0` | On | On after warmup | Scheduled gate |
| `AffordanceZero-Train-v0` | On | On after warmup | Permanent zeros |

Stage 5 supports **one CPU/CUDA device** and rejects distributed training, RND,
recurrent policies and the unrelated upstream auxiliary hook. All four tasks
share the existing environment, PPO and symmetry settings. No reward, label,
terrain or curriculum change accompanies auxiliary learning.

### Buffer and optimizer behavior

Every physical step forwards newly finalized valid `[N,2,P]` contacts to a local
buffer. Liftoff may predate the current rollout; tracker pending events survive
learning boundaries. Sample timing diagnostics record this age. Only this rollout's
newly finalized samples enter its buffer, which clears after the joint update.
The buffer stores independent, detached tensors usable outside inference mode.

At capacity, independent random priorities retain a uniform subset of contacts.
An independent generator also samples auxiliary minibatches with replacement;
auxiliary sampling never consumes the global PPO RNG. Each PPO minibatch computes
the current full policy and adds `lambda_aux * SmoothL1(pred,target,beta=0.1)` to
the existing loss before a single backward, clipping and Adam step. There is no
separate auxiliary optimizer. Symmetry transforms original observations only.

Auxiliary gradients reach Actor terrain encoder/head only; control predictions
remain detached, so PPO cannot directly update the head. The head can nevertheless
change the next full policy prediction as auxiliary learning changes its weights.
Post-update probes measure this change; PPO clipping is not a hard bound on the
combined auxiliary-induced policy change.

### Parameters and schedules

Defaults live in `isal/learning/training_config.py`; agent overrides live in
`tasks/direct/humanoid_rough/agents/training_agent_cfg.py`.

| Hydra field | Default |
|---|---:|
| `agent.algorithm.auxiliary_learning.enabled` | true except Baseline |
| `agent.algorithm.auxiliary_learning.start_iteration` | 100 |
| `agent.algorithm.auxiliary_learning.ramp_iterations` | 200 |
| `agent.algorithm.auxiliary_learning.loss_coef` | 0.05 |
| `agent.algorithm.auxiliary_learning.aux_batch_size` | 2048 |
| `agent.algorithm.auxiliary_learning.min_samples_per_update` | 256 |
| `agent.algorithm.auxiliary_learning.max_samples_per_rollout` | 16384 |
| `agent.algorithm.auxiliary_learning.smooth_l1_beta` | 0.1 |
| `agent.algorithm.auxiliary_learning.sampling_seed` | initial agent seed |
| `agent.algorithm.affordance_gate_schedule.start_iteration` | 300 |
| `agent.algorithm.affordance_gate_schedule.ramp_iterations` | 100 |
| `agent.algorithm.affordance_gate_schedule.final` | 1.0 |
| `agent.algorithm.diagnostics.enabled` | true |
| `agent.algorithm.diagnostics.probe_size` | 256 |

For `k = completed_updates`, each schedule is `final*clamp((k-start)/ramp,0,1)`.
A zero-length ramp switches at `k >= start`. The runner applies both schedules
before rollout collection and holds them constant through its update epochs.
After an update, completed_updates increases; the next rollout applies the next
values. Zero-input control always uses gate 0. Stage 5 owns the control gate;
change its schedule rather than the Stage 4 model's initial gate setting.

Aux disabled creates no buffer and no auxiliary backward graph. Enabled with
coefficient 0 still collects statistics but skips auxiliary loss. Fewer than the
minimum samples also skips that loss and logs a reason. Tracker enablement remains
a separate environment setting; changing a coefficient does not switch it off.
Sampling seed is explicit: when changing experiment seeds through CLI, override
`agent.algorithm.auxiliary_learning.sampling_seed` too if a different auxiliary
sample stream is desired.

### Safe local commands

Run these from the repository root. They only infer and perform **one zero-action
step**; no learning/update/optimizer step is executed:

```powershell
conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-CNN-Train-v0 `
  --headless --num_envs 1 --validate-only

conda run --no-capture-output -n env_isaaclab python -u `
  isal/scripts/rsl_rl/train.py --task ISAL-Humanoid-Rough-AffordanceObs-Train-v0 `
  --headless --num_envs 1 --validate-only `
  'env.terrain_perception.size=[1.8,1.0]' `
  env.scene.height_scanner.pattern_cfg.ordering=yx `
  env.robot.actor_obs_history_length=3 env.robot.critic_obs_history_length=4 `
  agent.algorithm.auxiliary_learning.loss_coef=0.01 `
  agent.algorithm.auxiliary_learning.start_iteration=0 `
  agent.algorithm.auxiliary_learning.ramp_iterations=0 `
  agent.algorithm.affordance_gate_schedule.start_iteration=0 `
  agent.algorithm.affordance_gate_schedule.ramp_iterations=0

conda run --no-capture-output -n env_isaaclab python -u -m pytest `
  isal/tests -q --tb=short -p no:cacheprovider
```

### Diagnostics, checkpoint and export

The runner writes `stage5_metrics.jsonl`, console records and scalar writer tags.
Records include completed updates and actual environment steps. All retained
targets have distribution/count summaries; bounded probes report prediction
histograms, MAE, valid/invalid correlation, subgroup metrics, action/prediction
drift and post-update KL. Constant targets return `correlation=null` with
`correlation_valid=false`. Auxiliary encoder/head gradients are measured on the
first active minibatch, before coefficient weighting; combined norms are measured
before clipping. Probe diagnostics never update normalization or distributions.

Training checkpoints require the same Stage 5 model, algorithm, environment and
input configuration. They save completed-update count, actual environment steps,
timing, both schedules/current values, optimizer/adaptive learning rate and RNGs.
Save is allowed only at complete update boundaries. Resume uses a fresh env reset
and discards pending contacts and the learning buffer; it does not reproduce the
old physical trajectory. A resumed checkpoint retains its actual saved gate until
the next rollout boundary applies the next schedule value.

The existing metadata-based export CLI accepts Stage 5 checkpoint model state:

```powershell
conda run --no-capture-output -n env_isaaclab python -u isal/scripts/export_actor.py `
  --checkpoint outputs/stage5_acceptance/AffordanceObs/untrained_checkpoint.pt `
  --output outputs/stage5_acceptance/cli_export
```

These acceptance artifacts are **untrained**. Synthetic checkpoint iteration
fields test recovery and do not represent learning. Local tests exercise the
production loss helper and backward, but deliberately do not call the actual
optimization loop. External short training must establish locomotion learning,
sample availability and stability through the fully open gate interval. See
`STAGE5_ACCEPTANCE.md` for actual engineering results and limits.
