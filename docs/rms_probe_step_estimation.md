# RMS probe step estimation

To log RMS during an ordinary training run without changing its stopping point, set:

```toml
total_rms_check_every_n_steps = 20
```

The value is logged as `strength/total_rms` to configured trackers and printed as `total_rms`. The default `0` disables periodic RMS logging.

RMS probe step estimation runs at least two independent trainings:

1. A local probe starts from the base model and stops after `rms_probe_steps` optimizer steps.
2. The trainer measures the effective total scaled LoRA RMS and saves the probe's final weights and full training state.
3. For the calibrated piecewise policy, one adjusted probe may run when the candidate squeeze schedule enters the probe window.
4. Probe weights and optimizer state are discarded from memory.
5. Production training starts from the base model with a fresh optimizer and scheduler.

The default `linear` policy estimates the production step count as:

```text
adjusted_steps = round(max_train_steps * rms_probe_target / observed_probe_rms)
```

Example TOML configuration:

```toml
max_train_steps = 5000
rms_probe_steps = 500
rms_probe_target = 0.0001
rms_probe_scaling_policy = "linear"
rms_probe_adjusted_steps_divisible_by = 5
```

`rms_probe_target` is the reference model's effective total scaled RMS measured at the same probe step. The probe keeps the original `max_train_steps` as its scheduler and LoRA-Squeeze horizon, so its first steps match the unadjusted production configuration.

`rms_probe_adjusted_steps_divisible_by` optionally rounds the estimated production step count to the nearest multiple of the configured value. Ties round upward. This is useful for distributing a LoRA-Squeeze run evenly across its training segments. The option affects only RMS-probe results; ordinary training step counts are unchanged.

Probe artifacts are written beneath the configured output directory in a directory named like `character-rms-probe-step500`. This includes the final LoRA, resumable training state, and `rms_probe_result.json`. Probe artifacts are not uploaded, and probe tracking, sampling, periodic checkpoints, and validation are disabled.

This is a linear approximation. RMS growth may be nonlinear, particularly with non-constant learning-rate schedulers, adaptive optimizers, regularization, or LoRA-Squeeze. Constant learning rates make the estimate easier to interpret.

## Calibrated hybrid energy/RMS policy

`piecewise_energy_v1` keeps the calibrated linear squared-RMS model for the initial rank-36 segment. The policy identifier is retained for existing TOML compatibility; result metadata uses `trajectory_model_version = 2.5` for long Probe 1 segmentation and multi-probe evidence combination. Compressed stages grow linearly in RMS normalized by the step-500 rank-36 reference. At each squeeze, the current RMS is multiplied by the square root of the calibrated energy-retention factor before the next stage's RMS gain is added:

```text
rank-36 energy = reference_rms^2 * (1 + early_energy_slope * delta_steps / 1000)

for each compressed stage:
    rms *= sqrt(energy_retention)
    rms += reference_rms * later_rms_velocity * stage_factor * stage_steps / 1000
```

This is the generalized energy equation `dE/dt proportional to E^gamma` with `gamma = 0.5`. The probe records an RMS curve and splits it at the exact configured squeeze boundaries. Rank-36 energy is fitted only from pre-squeeze samples and projected to the calibration's step-500 reference. Normally the fit starts at step 100; when squeeze one occurs inside the probe, it uses the periodic pre-squeeze samples from the first curve interval so a short initial segment retains as much clean evidence as possible. Compressed-stage RMS velocity is fitted separately inside every observed post-squeeze segment, never across a squeeze discontinuity. The solver then uses both kinds of evidence to select a requested final RMS. When production gradient accumulation differs from the probe, the projected step-500 RMS, normalized rank-36 energy slope, and normalized compressed-stage RMS velocity are mapped to the production accumulation before solving. The compressed-stage velocity regression uses log dataset batches per epoch, log production-reference RMS, log production gradient accumulation, and the normalized early probe energy slope. Log batches per epoch replaced the former inverse-batches feature after the initial two completed 22-batch Marcia runs showed that the inverse form greatly over-extrapolated compressed growth below the previous 93-batch calibration boundary.

Example for the calibrated Anima configuration:

```toml
max_train_steps = 4000
gradient_accumulation_steps = 6

rms_probe_steps = 500
rms_probe_curve_every_n_steps = 20
rms_probe_target = 0.00003878279312630184
rms_probe_final_target = 0.00008425384599385171
rms_probe_scaling_policy = "piecewise_energy_v1"
rms_probe_adjusted_steps_divisible_by = 100
rms_probe_force_adjusted_probe = false

# Optional: adjust gradient accumulation so that actual consumed microbatches
# remain near the 4000-step, accumulation-6 reference budget.
rms_probe_gradient_accumulation_target_microbatches = 24000
rms_probe_gradient_accumulation_rounding_bias = 0.6
rms_probe_min_gradient_accumulation_steps = 3

lora_squeeze_start_dim = 36
network_dim = 9
lora_squeeze_num_squeezes = 4
lora_squeeze_train_after_final_squeeze = true
lora_squeeze_step_schedule = "equal"
lora_squeeze_rank_schedule = "geometric"
```

`rms_probe_final_target` is the desired RMS at the end of production training. Unlike the probe-step reference RMS, it makes the final objective explicit. `rms_probe_target` remains recorded for reference and compatibility.

The piecewise model uses the exact integer segment boundaries produced by LoRA-Squeeze, including `lora_squeeze_first_segment_ratio` and `lora_squeeze_final_segment_ratio`. Probe 1 may end before or after one or more squeezes. It must contain at least two clean pre-squeeze rank-36 samples; extremely short first segments are rejected when the configured curve interval cannot provide them. The default ratios of `1.0` preserve the calibrated equal-fifths schedule.

### Schedule-aware adjusted probe

Probe 1 starts from the base model with the configured production gradient accumulation and the original production squeeze/scheduler horizon. It stops at `rms_probe_steps`. Pre-squeeze samples fit the rank-36 RMS-squared anchor. If Probe 1 reaches squeeze one or later squeezes, samples strictly inside each compressed segment fit normalized RMS velocity, the usable spans weight the stage estimates, and every observed squeeze contributes retained-energy telemetry. All of that evidence enters the provisional complete-trajectory solve. Probe 1's weights and optimizer state are never continued into Probe 2 or production.

If Probe 1 has no usable compressed-stage velocity and the first estimate places squeeze one at or before the configured probe endpoint, the trainer automatically runs one adjusted probe. A long Probe 1 that already measured a compressed segment does not trigger a redundant Probe 2. When needed, Probe 2 uses the provisional production horizon and adjusted gradient accumulation. Its compute-matched horizon preserves the number of microbatches actually consumed by Probe 1, accounting for epoch-boundary remainder updates, and is rounded upward to the RMS curve interval. To retain useful phase evidence, the horizon is extended when possible to the known first-squeeze step plus five RMS intervals, also rounded upward. It is capped at the configured Probe 1 horizon and the provisional production horizon. The result JSON records both the nominal `steps * accumulation` values and these epoch-aware actual microbatch counts.

Set `rms_probe_force_adjusted_probe = true` to run Probe 2 even when Probe 1 already supplied compressed-stage evidence or the provisional schedule's first squeeze is after the configured probe window. The forced probe uses the same provisional production horizon, adjusted gradient accumulation, compute-matched horizon, and phase-coverage rules as an automatically triggered Probe 2. When squeeze one remains after the probe window, the phase-coverage cap makes Probe 2 run for the full configured Probe 1 horizon. Its overlap with Probe 1 then directly measures the change caused by the provisional accumulation pattern.

For example, Probe 1 at 500 steps and accumulation 6 has a nominal budget of 3,000 microbatches. If the provisional Beastgirl schedule is 700 steps with accumulation 20, the raw compute match is 150 steps and the interval-rounded value is 160. Squeeze one is at step 140, so the phase-coverage rule selects step 240. This is 4,800 nominal microbatches instead of the former 10,000-microbatch Probe 2.

Probe 2 starts fresh from the same base model, but uses the provisional production squeeze horizon and provisional production accumulation. Its curve is split at the known squeeze boundaries; measurements are never fitted across a squeeze discontinuity. Before squeeze one, the trainer compares Probe 1 and Probe 2 at exact shared steps that are pre-squeeze under both probes' schedules. In RMS-squared space it fits a relative energy scale through the origin, and it compares the two window slopes after removing that energy scale. The relative RMS and normalized-slope changes are applied to Probe 1's step-500 anchor. Probe 2's short absolute pre-squeeze line is retained only as diagnostic metadata and never becomes the trajectory anchor.

After squeeze one, RMS velocity is fitted only inside individual compressed-rank segments. Each raw `dRMS/dstep` is divided by that probe's compatible reference RMS and its relative stage factor. If both probes measured compressed growth, Probe 1's velocity is first mapped to Probe 2's accumulation/anchor condition and the two estimates are combined by usable within-stage span. Observed retention is combined by matching squeeze stage across probes before the geometric observed-to-calibrated scale is computed; a later stage is never mistaken for an earlier one. The trainer re-solves a complete fresh-production trajectory and reapplies the step/accumulation fixed point. If the final accumulation differs from the observation condition, the measured anchor and later velocity are mapped again to the selected production condition. Retention remains capped at an energy retention of one per stage. The runtime additionally records aggregate energy-weighted retention for future calibration, but the scale uses the historical module-mean statistic so its units match the archived constants. Probe 2's weights and optimizer state are discarded. At most two probes run.

#### Previous behavior versus current behavior

| Part | Previous Marcia behavior | Current behavior |
| --- | --- | --- |
| Probe 1 | Supplied a good unsqueezed curve and projected step-500 anchor, but any samples after squeeze one were ignored by the provisional solve. | Supplies the clean pre-squeeze anchor and directly uses every sufficiently sampled compressed segment plus observed squeeze retention. |
| Accumulation selection | The original implementation treated compute mostly as `steps * accumulation` and capped accumulation at batches per epoch. The interim fix allowed only epoch divisors. | Every integer accumulation at or below one epoch is eligible. Each candidate is evaluated by cycling its real epoch update groups and counting the actual microbatches consumed for the proposed optimizer-step count. Explicit settings above one epoch remain honored when no automatic target is active. |
| Probe 2 rank-36 fit | Probe 2's few pre-squeeze samples were extrapolated as an independent absolute line to step 500 and replaced Probe 1's much better-observed anchor. | The shared pre-squeeze samples measure only Probe 2-to-Probe 1 relative RMS and slope changes. Those changes transfer Probe 1's reliable anchor to Probe 2's exact accumulation pattern. |
| Probe 2 compressed growth | The later raw rate was represented as a constant change in normalized RMS-squared and was normalized by Probe 2's short absolute step-500 extrapolation. The first interim fix retained that value while switching the anchor back to Probe 1, which mixed two different scales. | Each compressed segment is fitted as normalized RMS velocity against a compatible anchor. If Probe 1 also measured compressed growth, both fresh measurements are condition-mapped and span-weighted instead of discarding either curve. |
| Final solve | A large change in the second estimate could select another accumulation without direct accounting for its real epoch microbatch pattern. | The full trajectory and accumulation selection are iterated again. Actual epoch microbatch counts choose the accumulation, and measured quantities are mapped from Probe 2's accumulation if the final choice changes. |

For the failed Marcia run, Probe 1 projected a step-500 RMS of `7.5626712e-5` with normalized early energy slope `2.2633446`. Its provisional 350-step result came from an extreme out-of-range compressed-stage regression for only 22 batches per epoch; this is why the existing schedule rule ran Probe 2. Probe 2's 350-step schedule squeezed at steps 70 and 140, leaving only steps 20, 40, and 60 for rank-36 comparison. Its independent three-sample extrapolation was `4.9249983e-5`. The older anchor implementation used that low value as the absolute reference and ultimately produced the logged 4,850-step, accumulation-5 production run.

The current implementation does not use `4.9249983e-5` as an anchor. The three shared samples measure an energy scale of `1.0046655`, or an RMS scale of `1.0023300`, relative to Probe 1. This transfers the anchor to `7.5802925e-5` and the early energy slope to `2.2534072`. Against that anchor, Probe 2's two short compressed-stage spans produce a normalized RMS velocity of `0.6672414` per 1,000 steps with the current stage factors. Normalizing the same raw RMS velocity by the rejected short anchor produces a different normalized value, but their products with their respective reference RMS values are equal. The former energy-slope value `0.4108008` is not reused because it has different units and changes with entry RMS.

With the existing 24,000-microbatch target, the current replay estimates 1,185 steps, rounds to 1,200, and selects accumulation 22 using the unchanged epoch-aware compute logic. If production accumulation is explicitly held at 5 instead, the same measurements are mapped to accumulation 5 and estimates 1,850 steps, also rounding to 1,850. The difference is the existing accumulation policy, not an added probe rule. A longer Probe 1 can now supply compressed-stage evidence directly; Probe 2 remains available when that evidence is missing or when explicitly forced.

The completed 2,400-step GA-5 Marcia run supplies the more relevant small-dataset check. Replaying its prefix as if Probe 1 had stopped at 250, 400, or 500 steps gives progressively more direct trajectory evidence. The completed adapter reached `8.77655e-5` against the `8.42538e-5` target. A prefix ending before squeeze one relies on the shared compressed-stage regression; a prefix extending past squeeze one uses its measured compressed RMS velocity and retention. The latter no longer runs Probe 2 merely because its provisional schedule squeezes inside the window.

The completed 15-image Beastgirl trajectory adds the first family below Marcia's 22 batches per epoch. At GA 4 and 2,200 steps it reached exact final RMS `8.1586568e-5`. Its schedule-aware rank-36 reference is `5.0190000e-5`, its normalized early energy slope is `2.1833028`, and its four compressed stages have linear-RMS R-squared values from `0.99897` through `0.99963`. Before adding the family, the model predicted `8.4363078e-5` at 2,200 steps (`+3.40%`). With the current calibration it predicts `8.4073344e-5` (`+3.05%`) and solves the standard final target at 2,206 steps.

The third completed Marcia trajectory supplies a same-dataset high-accumulation check. Probe 1 ran for 300 steps at GA 11, projected a step-500 reference RMS of `6.5454940e-5` with normalized energy slope `2.3062346`, and solved 1,680 steps, rounded to 1,700. The then-current model predicted `8.5112022e-5`; the saved adapter reached `9.2126092e-5`, which is `9.34%` above the target and `7.61%` above the model prediction. Its four compressed stages remain very nearly linear in RMS (`R-squared = 0.99967` through `0.99996`) but grow faster than the GA-5 Marcia trajectories. The run is therefore retained as one of Marcia's four rows. The current fitted 1,700-step prediction is `9.0918260e-5` (`-1.31%` versus the adapter), and the standard target solves at 1,550 steps. The actual logged curve crosses the target near step 1,500, so this remains an in-family check rather than an exact dataset-specific override.

The fourth Marcia run uses the revised 27-image dataset at GA 9. Its 200-step isolated probe projects a step-500 reference RMS of `5.4083583e-5` with normalized early energy slope `2.1719512`; production ran for 2,000 steps and reached exact final RMS `9.1941074e-5`. Its four compressed-stage RMS fits have R-squared values from `0.99945` through `0.99973`. Model 2.3 predicted `8.4550521e-5` (`-8.04%`); model 2.4 predicts `8.6421609e-5` (`-6.00%`) and solves the standard target at 1,938 steps. The residual is retained in the shared Marcia family rather than introducing a dataset-specific correction.

The third Beastgirl run uses a revised 22-image dataset at GA 11. Its 200-step isolated probe projects a step-500 reference RMS of `6.3704703e-5` with normalized early energy slope `2.2177240`; production ran for 1,600 steps and reached exact final RMS `8.7174099e-5`. Its compressed-stage RMS fits have R-squared values from `0.99912` through `0.99985`. Model 2.3 predicted `8.4317642e-5` (`-3.28%`); model 2.4 predicts `8.6690027e-5` (`-0.56%`) and solves the standard target at 1,545 steps.

The reason for changing the state variable is visible in the logged portions of Marcia's two trajectories. Rank-25 raw RMS velocity differs by about 6% between the 1,550-step and 4,850-step schedules, and rank-18 velocity differs by about 19%; their normalized energy slopes differ by roughly 2 to 3 times because `d(RMS^2)/dt = 2 * RMS * dRMS/dt`. Anchoring the shorter run at the longer run's first post-squeeze sample, the RMS-velocity model predicts the rank-25 endpoint within `+0.5%` and the later rank-18 endpoint within `-2.4%`. Transferring the old constant-energy rate misses those endpoints by `-10.3%` and `-19.9%`.

The default Accelerate dataloader synchronization ends an accumulation window at every epoch boundary. If the configured accumulation does not divide the dataset's batches per epoch, the final optimizer update in each epoch contains only the remainder microbatches. Accelerate still divides every loss by the configured accumulation value, so this is an under-weighted remainder update, not a second update using a smaller effective accumulation setting. For example, 22 batches per epoch with accumulation 18 produces update groups of 18 and 4 microbatches, with the 4 losses each still divided by 18.

When a microbatch target is configured, every integer accumulation candidate is evaluated using the actual repeating update groups for the dataset and the proposed optimizer-step count. The target is bracketed by the neighboring attainable actual-microbatch totals. `rms_probe_gradient_accumulation_rounding_bias=0.5` uses the midpoint between those totals; the default `0.6` modestly favors the higher-compute choice, while `1.0` always chooses it. Production accumulation changes the rescaled probe RMS, the rank-36 slope, and the compressed-stage velocity prediction, so the trainer alternates the complete trajectory solve and accumulation selection until the pair is stable. Automatic selection does not choose accumulation above one epoch because it cannot collect a full accumulation window before synchronization; if the configured minimum is higher, that minimum remains the sole automatic candidate. If no microbatch target is configured, the trainer preserves the explicitly configured `gradient_accumulation_steps`, including non-divisible values and values above one epoch. The result metadata records update-group sizes, remainder, nominal compute, and actual compute, and the trainer warns when production contains an under-filled update.

The compressed-stage coefficients are fitted over twenty-nine trajectories across nine families: M'rissi, Izutsumi, Neeko, Wilykit, Mutio, Rosine, Crossbreed Priscilla, Marcia, and Beastgirl. Rosine remains the documented exception to equal family weighting: because its trajectories intentionally span unusually substantial dataset edits, it receives twice the total weight of each other family. Its family weight is `2/10`, divided equally across ten rows; every other family receives `1/10`, divided equally across its repeated trajectories. Crossbreed Priscilla contributes one full independent-family share split equally between two completed trajectories, Marcia splits one share between four trajectories, and Beastgirl splits one share between two trajectories. The ridge penalty remains `0.001`.

`tools/refit_rms_probe_calibration.py` contains the archived operational inputs, endpoints, family weights, and exact ridge calculation used to reproduce the implemented constants and validation metrics.

The regression target is normalized RMS velocity per 1,000 steps. Its relative four-stage factors are `0.94667132`, `0.98067753`, `1.03334647`, and `1.03930469`. They are indexed by squeeze-stage position rather than the literal compressed ranks. The family-weighted retained-energy values are `0.93482974`, `0.92453692`, `0.90759665`, and `0.87506596`; these use all twenty-eight archived trajectories with retention telemetry. M'rissi predates that telemetry. The currently validated configuration remains 36-to-9; other rank layouts are still rejected until separately calibrated, but the state equation itself does not require absolute-rank lookup keys.

With both new rows included and the log-batches feature retained, weighted regression-target MAPE is `5.052%` in-sample and `7.798%` under leave-one-family-out validation. End-to-end endpoint MAPE is `2.436%` in-sample and `3.749%` leave-one-family-out. The refit predicts the two Crossbreed endpoints at `-0.95%` and `-0.29%`, the four Marcia endpoints at `+0.72%`, `+0.60%`, `-1.31%`, and `-6.00%`, and the two Beastgirl endpoints at `+3.05%` and `-0.56%`. Leaving the entire Marcia family out gives 2.54% mean absolute endpoint error; the new 27-image row is `-7.16%`. The completed runs continue to support fixed `gamma = 0.5`; the calibration changes the velocity estimate, stage factors, and retention means without adding a dataset-specific runtime branch.

The separate pre-500 accumulation mapping is a through-origin within-pair log-log fit. It contains exact-revision pairs from Izutsumi, Neeko, Wilykit, Mutio, Rosine_first, and Rosine_second, plus the GA 7 versus GA 2 Rosine near-pair whose datasets differ by one image. Same-accumulation reruns are geometrically averaged first, including the isolated-probe and production GA 7 Rosine measurements. The calibration retains the runs' actual configured accumulation values and their natural non-divisible epoch patterns; it does not convert them to divisors. It applies `production_rms = probe_rms * (production_ga / probe_ga) ** 0.13407367061621528` and the analogous exponent `0.028683585047468874` to the normalized energy slope. Across all seven fitting comparisons, step-500 RMS MAE falls from 4.93% with no correction to 0.77%, while slope MAE falls from 1.96% to 1.37%. The 110-image GA 6 Rosine probe was retained as external validation because its dataset differs by six images from the nearest GA 7 revision: the existing map predicts it within 0.74% RMS and 0.41% slope. The 117-image GA 2 trajectory is another non-matched validation row: two reasonable GA 6 priors underpredict its step-500 RMS by 2.75% to 2.99% and overpredict its normalized slope by 2.08% to 2.65%, so it is not used to refit the pre-500 exponents. The 112-image GA 9 and 119-image GA 6 runs each have same-accumulation isolated and production measurements; their step-500 RMS differences are 0.81% and -0.42%, respectively, and do not change the cross-accumulation exponents. The measured accumulation range is 2 through 9; the GA 2 fitting endpoint remains a near-match rather than an identical-dataset comparison.

The combined calibration uses learning rate `6e-5`, constant AdamW, rank/alpha 9/3, start rank 36, four equal geometric squeezes, batches per epoch from 15 through 262, step-500 reference RMS from approximately `3.394e-5` through `6.545e-5`, normalized early energy slope from approximately `1.946` through `2.306`, and production accumulation from 2 through 11. The trainer enforces the rank and squeeze layout. The compressed-stage regression and pre-500 power laws are also used outside their observed feature ranges; this is extrapolation, so predictions farther from the calibration data should be treated with correspondingly more caution.

The result JSON records every probe and its settings, each probe's actual training-step limit, the adjusted-probe nominal and epoch-aware microbatch calculations, the phase-coverage calculation, complete curves, the Probe 1 rank-36 anchor, any Probe 2 shared-window energy/RMS/slope transfer, diagnostic short absolute fits, schedule-separated and combined compressed-stage RMS velocities, observed and combined module-mean retention, aggregate retained energy, detected batches per epoch, predicted final RMS, original and adjusted gradient accumulation, per-epoch update-group sizes, epoch remainder and compatibility, nominal and actual production microbatches, and the selected policy. Model-version-2.5 results include `trajectory_model_version = 2.5`, `later_energy_exponent = 0.5`, `later_mean_rms_velocity_per_1000_steps`, and `squeeze_retention_scale`, so they are distinguishable from older result files that discarded Probe 1's post-squeeze evidence. When the production step count is rounded to a configured multiple, `model.predicted_final_rms` is recomputed for that rounded production schedule; `model.estimated_steps_predicted_final_rms` retains the prediction at the solver's unrounded step count.

Limitations:

- Use `max_train_steps`; `max_train_epochs` is not supported.
- The feature cannot be combined with `resume`, `initial_step`, `initial_epoch`, or DeepSpeed.
- Both probe settings must be provided together.
- `piecewise_energy_v1` requires enough pre-squeeze curve samples to fit rank-36 energy, a positive final RMS target, and the calibrated 36-to-9 four-squeeze schedule. Probe 1 may cross squeeze boundaries; each post-squeeze segment still needs at least two usable interior samples to replace the regression with an observed velocity. Probe horizons other than 500 extrapolate their fitted rank-36 energy line to the step-500 calibration reference.
