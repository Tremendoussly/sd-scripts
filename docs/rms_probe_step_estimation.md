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

`piecewise_energy_v1` models squared RMS as energy. Energy grows linearly inside each LoRA-Squeeze segment and is multiplied by a calibrated retention factor at each squeeze. The probe records an RMS curve, fits its normalized energy slope between steps 100 and 500, and numerically solves for a requested final RMS. Its later-stage slope regression uses inverse dataset batches per epoch, log probe RMS, and log production gradient accumulation.

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

# Optional: adjust gradient accumulation so that the adjusted step count times
# accumulation remains near the 4000-step, accumulation-6 reference budget.
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

### Schedule-aware adjusted probe

The initial probe keeps the original schedule and must remain at rank 36 through step 500. Its fitted rank-36 energy line can nevertheless evaluate a shorter candidate production schedule by interpolating the energy at that schedule's known first-squeeze boundary, even when the boundary is before step 500.

If the first estimate places squeeze one at or before step 500, the trainer automatically runs one adjusted probe. Probe 2 uses the provisional production horizon and adjusted gradient accumulation, but it no longer always trains for 500 optimizer steps. Its nominal compute-matched horizon is `ceil(Probe 1 steps * Probe 1 accumulation / Probe 2 accumulation)`, rounded upward to the RMS curve interval. To retain useful phase evidence, the horizon is extended when necessary to the known first-squeeze step plus five RMS intervals, also rounded upward. It is capped at the original 500-step probe horizon.

For example, Probe 1 at 500 steps and accumulation 6 has a nominal budget of 3,000 microbatches. If the provisional Beastgirl schedule is 700 steps with accumulation 20, the raw compute match is 150 steps and the interval-rounded value is 160. Squeeze one is at step 140, so the phase-coverage rule selects step 240. This is 4,800 nominal microbatches instead of the former 10,000-microbatch Probe 2.

Probe 2's curve is split at the already-known squeeze boundaries: rank-36 energy is fitted only before squeeze one, and later-stage slopes are fitted only within individual compressed-rank segments. Measurements are never fitted across a squeeze discontinuity. The rank-36 fit is still projected to the calibrated step-500 reference for the final trajectory solve; shortening Probe 2 does not substitute a linear end-to-end estimate.

Probe 2 is the final probe. Its measured per-rank rates are used to re-solve a complete fresh-production trajectory; its weights and optimizer state are discarded. Production keeps Probe 2's gradient accumulation even if the final step estimate changes, so the measured update behavior still matches production. At most two probes run.

Gradient accumulation is capped at the dataset's batches per epoch because the default dataloader synchronization ends an accumulation window at the epoch boundary. For very small datasets, the requested microbatch budget may therefore be unattainable at the estimated RMS target.

When a microbatch target is configured, the trainer divides it by the rounded production step count and rounds the resulting ideal accumulation. `rms_probe_gradient_accumulation_rounding_bias=0.5` is ordinary nearest rounding; the default `0.6` modestly favors more microbatches, while `1.0` always rounds upward. Because production accumulation is an input to the augmented later-stage model, the trainer alternates step prediction and accumulation selection until the pair is stable. The result is not capped by the accumulation used for Probe 1, so a sufficiently short production run may select accumulation 7, 8, or higher. It is capped by the dataset's batches per epoch as described above. Small differences do not force an increase: a 24,000 target with 3,999 steps selects accumulation 6 at the default bias. If no microbatch target is configured, production retains the configured `gradient_accumulation_steps` unchanged unless that value exceeds the dataset batch cap.

The augmented coefficients were fitted with equal total weight per dataset across five datasets: M'rissi, Izutsumi, Neeko, Wilykit, and Mutio. Repeated trajectories divide their character's 20% weight equally. The selected ridge model intentionally omits the early probe slope because it did not improve held-out results. The calibration uses learning rate `6e-5`, constant AdamW, rank/alpha 9/3, start rank 36, four equal geometric squeezes, batches per epoch from 132 through 262, probe RMS from approximately `3.394e-5` through `3.878e-5`, and production accumulation from 4 through 6. The trainer enforces the rank and squeeze layout. If any augmented feature is outside that audited range, the policy explicitly falls back to the previous BPE-only later-slope estimate rather than extrapolating the logarithmic regression.

The result JSON records every probe and its settings, each probe's actual training-step limit, the adjusted-probe microbatch and phase-coverage calculation, the selected probe, complete curves, schedule-separated fitted energy slopes, detected batches per epoch, predicted final RMS, original and adjusted gradient accumulation, and the selected policy.

Limitations:

- Use `max_train_steps`; `max_train_epochs` is not supported.
- The feature cannot be combined with `resume`, `initial_step`, `initial_epoch`, or DeepSpeed.
- Both probe settings must be provided together.
- `piecewise_energy_v1` requires a 500-step probe, a positive final RMS target, and the calibrated 36-to-9 four-squeeze schedule. The original probe horizon must place squeeze one after step 500; an automatically adjusted Probe 2 may squeeze earlier.
