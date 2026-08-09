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

## Calibrated piecewise RMS-squared policy

`piecewise_energy_v1` models squared RMS as energy. Energy grows linearly inside each LoRA-Squeeze segment and is multiplied by a calibrated retention factor at each squeeze. The probe records an RMS curve, fits rank-36 energy from the samples at and after step 100, and projects that fit to the calibration's step-500 reference when the configured probe is shorter or longer than 500 steps. It then numerically solves for a requested final RMS. When production gradient accumulation differs from the probe, the projected step-500 RMS and normalized rank-36 energy slope are mapped to the production accumulation before solving. Its later-stage slope regression uses inverse dataset batches per epoch, log production-reference RMS, log production gradient accumulation, and the normalized early probe energy slope.

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

The piecewise model uses the exact integer segment boundaries produced by LoRA-Squeeze, including `lora_squeeze_first_segment_ratio` and `lora_squeeze_final_segment_ratio`. Ratio-adjusted schedules are accepted when the original first squeeze remains after the configured Probe 1 endpoint. Probe 1 must contain at least two rank-36 samples at or after step 100. Extremely short first segments are rejected when an adjusted probe cannot provide two shared pre-squeeze samples. The default ratios of `1.0` preserve the calibrated equal-fifths schedule.

### Schedule-aware adjusted probe

Probe 1 starts from the base model with the configured production gradient accumulation and the original production squeeze/scheduler horizon. It stops at `rms_probe_steps`, which must still be before squeeze one. The trainer fits rank-36 RMS-squared growth from the observed curve and expresses that fit at the calibration's step-500 reference. It then solves a provisional complete squeeze trajectory. Probe 1 is the long, unsqueezed trajectory anchor; its weights and optimizer state are never continued into Probe 2 or production.

If the first estimate places squeeze one at or before the configured Probe 1 endpoint, the trainer automatically runs one adjusted probe. Probe 2 uses the provisional production horizon and adjusted gradient accumulation. Its compute-matched horizon preserves the number of microbatches actually consumed by Probe 1, accounting for epoch-boundary remainder updates, and is rounded upward to the RMS curve interval. To retain useful phase evidence, the horizon is extended when possible to the known first-squeeze step plus five RMS intervals, also rounded upward. It is capped at the configured Probe 1 horizon and the provisional production horizon. The result JSON records both the nominal `steps * accumulation` values and these epoch-aware actual microbatch counts.

Set `rms_probe_force_adjusted_probe = true` to run Probe 2 after Probe 1 even when the provisional schedule's first squeeze is after the configured probe window. The forced probe uses the same provisional production horizon, adjusted gradient accumulation, compute-matched horizon, and phase-coverage rules as an automatically triggered Probe 2. When squeeze one remains after the probe window, the phase-coverage cap makes Probe 2 run for the full configured Probe 1 horizon. Its overlap with Probe 1 then directly measures the change caused by the provisional accumulation pattern.

For example, Probe 1 at 500 steps and accumulation 6 has a nominal budget of 3,000 microbatches. If the provisional Beastgirl schedule is 700 steps with accumulation 20, the raw compute match is 150 steps and the interval-rounded value is 160. Squeeze one is at step 140, so the phase-coverage rule selects step 240. This is 4,800 nominal microbatches instead of the former 10,000-microbatch Probe 2.

Probe 2 starts fresh from the same base model, but uses the provisional production squeeze horizon and provisional production accumulation. Its curve is split at the known squeeze boundaries; measurements are never fitted across a squeeze discontinuity. Before squeeze one, the trainer compares Probe 1 and Probe 2 at the exact same recorded steps. In RMS-squared space it fits a relative energy scale through the origin, and it compares the two window slopes after removing that energy scale. The relative RMS and normalized-slope changes are applied to Probe 1's long step-500 anchor. Probe 2's short absolute pre-squeeze line is retained only as diagnostic metadata and never becomes the trajectory anchor.

After squeeze one, energy growth is fitted only inside individual compressed-rank segments. Those raw rates are normalized by the transferred Probe 1 anchor, so the early anchor and later slope use the same energy scale. The trainer then re-solves a complete fresh-production trajectory and reapplies the step/accumulation fixed point. If the final accumulation differs from Probe 2's accumulation, the measured anchor and later slope are mapped from the observed Probe 2 condition to the selected production condition. Probe 2's weights and optimizer state are discarded. At most two probes run.

#### Previous behavior versus current behavior

| Part | Previous Marcia behavior | Current behavior |
| --- | --- | --- |
| Probe 1 | Supplied a good 200-step unsqueezed curve and projected step-500 anchor, but its provisional solve used an unobserved later-stage regression far outside the calibrated dataset-size range. | Still supplies the long unsqueezed anchor. Its provisional result is used only to configure Probe 2 when the schedule enters the probe window. |
| Accumulation selection | The original implementation treated compute mostly as `steps * accumulation` and capped accumulation at batches per epoch. The interim fix allowed only epoch divisors. | Every integer accumulation at or below one epoch is eligible. Each candidate is evaluated by cycling its real epoch update groups and counting the actual microbatches consumed for the proposed optimizer-step count. Explicit settings above one epoch remain honored when no automatic target is active. |
| Probe 2 rank-36 fit | Probe 2's few pre-squeeze samples were extrapolated as an independent absolute line to step 500 and replaced Probe 1's much better-observed anchor. | The shared pre-squeeze samples measure only Probe 2-to-Probe 1 relative RMS and slope changes. Those changes transfer Probe 1's reliable anchor to Probe 2's exact accumulation pattern. |
| Probe 2 later slope | The later raw slope was normalized by Probe 2's short absolute step-500 extrapolation. The first interim fix retained that normalized value while switching the anchor back to Probe 1, which mixed two different energy scales. | The later raw slope is normalized by the transferred Probe 1 anchor. The anchor and later rate are therefore dimensionally consistent. |
| Final solve | A large change in the second estimate could select another accumulation without direct accounting for its real epoch microbatch pattern. | The full trajectory and accumulation selection are iterated again. Actual epoch microbatch counts choose the accumulation, and measured quantities are mapped from Probe 2's accumulation if the final choice changes. |

For the failed Marcia run, Probe 1 projected a step-500 RMS of `7.5626712e-5` with normalized early slope `2.2633446`. Its provisional 350-step result came from an augmented later-stage slope of approximately `5.3643`, an extreme extrapolation for only 22 batches per epoch; this is why Probe 2 was necessary. Probe 2's 350-step schedule squeezed at steps 70 and 140, leaving only steps 20, 40, and 60 for rank-36 comparison. Its independent three-sample extrapolation was `4.9249983e-5`. The old implementation used that low value as the absolute anchor: at fixed accumulation 22 it produced approximately 3,150 steps, and the 24,000 nominal-microbatch reselection then produced the logged 4,850 steps at accumulation 5.

The current implementation does not use `4.9249983e-5` as an anchor. The three shared samples measure an energy scale of `1.0046655`, or an RMS scale of `1.0023300`, relative to Probe 1. This transfers the anchor to `7.5802925e-5` and the early slope to `2.2534072`. Probe 2's observed later rate, normalized on that same transferred anchor, is `0.4108008`; the former `0.9731751` value was normalized by the rejected low anchor and cannot be combined with Probe 1's RMS. With the existing 24,000-microbatch target, the corrected replay estimates 2,744 steps, rounds to 2,750, and selects accumulation 11 using actual epoch compute. If accumulation is held at 22 instead, the same corrected measurements estimate approximately 2,275 steps and round to 2,300. These are model predictions rather than knob recalibration; the important implementation correction is that no short absolute extrapolation or mismatched normalization can drive the final solve.

The default Accelerate dataloader synchronization ends an accumulation window at every epoch boundary. If the configured accumulation does not divide the dataset's batches per epoch, the final optimizer update in each epoch contains only the remainder microbatches. Accelerate still divides every loss by the configured accumulation value, so this is an under-weighted remainder update, not a second update using a smaller effective accumulation setting. For example, 22 batches per epoch with accumulation 18 produces update groups of 18 and 4 microbatches, with the 4 losses each still divided by 18.

When a microbatch target is configured, every integer accumulation candidate is evaluated using the actual repeating update groups for the dataset and the proposed optimizer-step count. The target is bracketed by the neighboring attainable actual-microbatch totals. `rms_probe_gradient_accumulation_rounding_bias=0.5` uses the midpoint between those totals; the default `0.6` modestly favors the higher-compute choice, while `1.0` always chooses it. Production accumulation changes the rescaled probe RMS, the rank-36 slope, and the augmented later-stage prediction, so the trainer alternates the complete trajectory solve and accumulation selection until the pair is stable. Automatic selection does not choose accumulation above one epoch because it cannot collect a full accumulation window before synchronization; if the configured minimum is higher, that minimum remains the sole automatic candidate. If no microbatch target is configured, the trainer preserves the explicitly configured `gradient_accumulation_steps`, including non-divisible values and values above one epoch. The result metadata records update-group sizes, remainder, nominal compute, and actual compute, and the trainer warns when production contains an under-filled update.

The augmented coefficients were fitted over twenty-one trajectories across six families: M'rissi, Izutsumi, Neeko, Wilykit, Mutio, and Rosine. Rosine is a documented exception to equal family weighting: because its trajectories intentionally span unusually substantial dataset edits, it receives twice the total weight of each other family. Its family weight is `2/7`, divided equally across ten rows as `1/35` each; every other family receives `1/7`, divided equally across its repeated trajectories. The Rosine rows use related dataset revisions with 99, 95, 104, 103, 110, 117, 112, 119, 111, and 103 batches per epoch; the last two additions are the GA7 `Rosine_seventh` run and the newest GA10 run. The earlier 103-image low-accumulation test is a very slight edit of the 104-image revision, while the 117-image low-accumulation test expands the 110-image revision by seven images. Under this weighting, including the early probe slope reduces Rosine-double-weighted leave-one-family-out MAPE from 19.062% to 17.962% and weighted in-sample MAPE from 8.549% to 7.704%, so it remains selected. The ridge penalty remains `0.001`.

The separate pre-500 accumulation mapping is a through-origin within-pair log-log fit. It contains exact-revision pairs from Izutsumi, Neeko, Wilykit, Mutio, Rosine_first, and Rosine_second, plus the GA 7 versus GA 2 Rosine near-pair whose datasets differ by one image. Same-accumulation reruns are geometrically averaged first, including the isolated-probe and production GA 7 Rosine measurements. The calibration retains the runs' actual configured accumulation values and their natural non-divisible epoch patterns; it does not convert them to divisors. It applies `production_rms = probe_rms * (production_ga / probe_ga) ** 0.13407367061621528` and the analogous exponent `0.028683585047468874` to the normalized energy slope. Across all seven fitting comparisons, step-500 RMS MAE falls from 4.93% with no correction to 0.77%, while slope MAE falls from 1.96% to 1.37%. The 110-image GA 6 Rosine probe was retained as external validation because its dataset differs by six images from the nearest GA 7 revision: the existing map predicts it within 0.74% RMS and 0.41% slope. The 117-image GA 2 trajectory is another non-matched validation row: two reasonable GA 6 priors underpredict its step-500 RMS by 2.75% to 2.99% and overpredict its normalized slope by 2.08% to 2.65%, so it is not used to refit the pre-500 exponents. The 112-image GA 9 and 119-image GA 6 runs each have same-accumulation isolated and production measurements; their step-500 RMS differences are 0.81% and -0.42%, respectively, and do not change the cross-accumulation exponents. The measured accumulation range is 2 through 9; the GA 2 fitting endpoint remains a near-match rather than an identical-dataset comparison.

The combined calibration uses learning rate `6e-5`, constant AdamW, rank/alpha 9/3, start rank 36, four equal geometric squeezes, batches per epoch from 95 through 262, probe RMS from approximately `3.394e-5` through `4.676e-5`, normalized early energy slope from approximately `1.946` through `2.217`, and production accumulation from 2 through 10. The trainer enforces the rank and squeeze layout. The augmented regression and pre-500 power laws are also used outside their observed feature ranges; this is extrapolation, so predictions farther from the calibration data should be treated with correspondingly more caution.

The result JSON records every probe and its settings, each probe's actual training-step limit, the adjusted-probe nominal and epoch-aware microbatch calculations, the phase-coverage calculation, complete curves, the Probe 1 rank-36 anchor, Probe 2's shared-window energy/RMS/slope transfer, Probe 2's diagnostic short absolute fit, schedule-separated later energy slopes, detected batches per epoch, predicted final RMS, original and adjusted gradient accumulation, per-epoch update-group sizes, epoch remainder and compatibility, nominal and actual production microbatches, and the selected policy. When the production step count is rounded to a configured multiple, `model.predicted_final_rms` is recomputed for that rounded production schedule; `model.estimated_steps_predicted_final_rms` retains the prediction at the solver's unrounded step count.

Limitations:

- Use `max_train_steps`; `max_train_epochs` is not supported.
- The feature cannot be combined with `resume`, `initial_step`, `initial_epoch`, or DeepSpeed.
- Both probe settings must be provided together.
- `piecewise_energy_v1` requires enough curve samples to fit rank-36 energy, a positive final RMS target, and the calibrated 36-to-9 four-squeeze schedule. The original probe horizon must place squeeze one after the configured probe endpoint; an automatically adjusted Probe 2 may squeeze earlier. Probe horizons other than 500 extrapolate their fitted energy line to the step-500 calibration reference.
