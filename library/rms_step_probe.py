import argparse
import copy
import math
import os
from typing import Dict, Sequence, Tuple


LINEAR_POLICY = "linear"
PIECEWISE_ENERGY_POLICY = "piecewise_energy_v1"
SCALING_POLICIES = (LINEAR_POLICY, PIECEWISE_ENERGY_POLICY)

# Calibrated with equal total weight per dataset from eleven Anima 36->9,
# four-squeeze runs across M'rissi, Izutsumi, Neeko, Wilykit, and Mutio.
# Repeated runs divide their dataset's weight. The later-stage regression
# uses standardized inverse batches per epoch, log probe RMS, and log
# production gradient accumulation. Energy means RMS**2, normalized by the
# energy measured at the 500-step probe.
ENERGY_LATER_SLOPE_FEATURE_MEANS = (
    0.0057603334678737496,
    -10.210902113051919,
    1.6448587169540985,
)
ENERGY_LATER_SLOPE_FEATURE_SCALES = (
    0.0012183354008231285,
    0.041425018893322195,
    0.11937769995213532,
)
ENERGY_LATER_SLOPE_FEATURE_MINS = (
    0.0038167938931297708,
    -10.290870225480576,
    1.3862943611198906,
)
ENERGY_LATER_SLOPE_FEATURE_MAXES = (
    0.007575757575757576,
    -10.157533885825908,
    1.791759469228055,
)
ENERGY_LATER_SLOPE_INTERCEPT = 1.0845692457131164
ENERGY_LATER_SLOPE_COEFFICIENTS = (
    0.4158446719803483,
    -0.33845440477495098,
    0.055874827448662519,
)
ENERGY_LEGACY_DATASET_INTERCEPT = 0.53091727
ENERGY_LEGACY_DATASET_COEFFICIENT = 91.56570293
ENERGY_STAGE_FACTORS = (0.84333973, 0.95492410, 1.08152779, 1.12020838)
ENERGY_SQUEEZE_RETENTION = (0.93776704, 0.91973453, 0.89532405, 0.86626541)
ADJUSTED_PROBE_MIN_POST_SQUEEZE_SAMPLES = 5


def validate_rms_probe_configuration(args: argparse.Namespace) -> bool:
    target = args.rms_probe_target
    steps = args.rms_probe_steps
    step_multiple = args.rms_probe_adjusted_steps_divisible_by
    policy = getattr(args, "rms_probe_scaling_policy", LINEAR_POLICY)
    final_target = getattr(args, "rms_probe_final_target", None)
    curve_interval = getattr(args, "rms_probe_curve_every_n_steps", 20)
    microbatch_target = getattr(args, "rms_probe_gradient_accumulation_target_microbatches", None)
    minimum_gradient_accumulation = getattr(args, "rms_probe_min_gradient_accumulation_steps", 1)
    gradient_accumulation_rounding_bias = getattr(
        args, "rms_probe_gradient_accumulation_rounding_bias", 0.6
    )

    if policy not in SCALING_POLICIES:
        raise ValueError("--rms_probe_scaling_policy must be one of: " + ", ".join(SCALING_POLICIES))

    if target is None and steps is None:
        if step_multiple is not None:
            raise ValueError(
                "--rms_probe_adjusted_steps_divisible_by requires --rms_probe_target and --rms_probe_steps"
            )
        if policy != LINEAR_POLICY or final_target is not None or microbatch_target is not None:
            raise ValueError("RMS probe policy, final-target, and compute-budget options require probe settings")
        return False
    if target is None or steps is None:
        raise ValueError("--rms_probe_target and --rms_probe_steps must be specified together")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("--rms_probe_target must be a finite value greater than 0")
    if steps <= 0:
        raise ValueError("--rms_probe_steps must be greater than 0")
    if args.max_train_epochs is not None:
        raise ValueError("RMS probe step estimation requires --max_train_steps, not --max_train_epochs")
    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be greater than 0 when RMS probe step estimation is enabled")
    if steps > args.max_train_steps:
        raise ValueError("--rms_probe_steps cannot exceed --max_train_steps")
    if args.resume:
        raise ValueError("RMS probe step estimation starts both runs from scratch and cannot be used with --resume")
    if args.initial_step is not None or args.initial_epoch is not None:
        raise ValueError("RMS probe step estimation cannot be used with --initial_step or --initial_epoch")
    if args.deepspeed:
        raise ValueError("RMS probe step estimation does not support --deepspeed")
    if step_multiple is not None and step_multiple <= 0:
        raise ValueError("--rms_probe_adjusted_steps_divisible_by must be greater than 0")
    if microbatch_target is not None and microbatch_target <= 0:
        raise ValueError("--rms_probe_gradient_accumulation_target_microbatches must be greater than 0")
    if minimum_gradient_accumulation <= 0:
        raise ValueError("--rms_probe_min_gradient_accumulation_steps must be greater than 0")
    if (
        not math.isfinite(gradient_accumulation_rounding_bias)
        or gradient_accumulation_rounding_bias < 0
        or gradient_accumulation_rounding_bias > 1
    ):
        raise ValueError("--rms_probe_gradient_accumulation_rounding_bias must be between 0 and 1")

    if policy == PIECEWISE_ENERGY_POLICY:
        if curve_interval <= 0:
            raise ValueError("--rms_probe_curve_every_n_steps must be greater than 0")
        if steps % curve_interval != 0:
            raise ValueError("--rms_probe_steps must be divisible by --rms_probe_curve_every_n_steps")
        earliest_adjusted_squeeze = math.floor(steps / 5.0)
        if 2 * curve_interval >= earliest_adjusted_squeeze:
            raise ValueError(
                "--rms_probe_curve_every_n_steps is too large to fit two pre-squeeze "
                "samples in the shortest adjusted probe schedule"
            )
        if final_target is None or not math.isfinite(final_target) or final_target <= 0:
            raise ValueError(
                "--rms_probe_final_target must be a finite value greater than 0 for piecewise_energy_v1"
            )
        if steps != 500:
            raise ValueError("piecewise_energy_v1 is calibrated for --rms_probe_steps=500")
        if equal_squeeze_steps(args.max_train_steps)[0] <= steps:
            raise ValueError("piecewise_energy_v1 requires the first squeeze to occur after the probe")
        expected = {
            "lora_squeeze_start_dim": 36,
            "network_dim": 9,
            "lora_squeeze_num_squeezes": 4,
            "lora_squeeze_train_after_final_squeeze": True,
            "lora_squeeze_step_schedule": "equal",
            "lora_squeeze_rank_schedule": "geometric",
        }
        mismatches = [
            f"{name}={getattr(args, name, None)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(args, name, None) != value
        ]
        if mismatches:
            raise ValueError(
                "piecewise_energy_v1 is calibrated for the Anima 36->9 four-squeeze schedule: "
                + ", ".join(mismatches)
            )
    elif final_target is not None:
        raise ValueError("--rms_probe_final_target requires --rms_probe_scaling_policy=piecewise_energy_v1")

    return True


def estimate_rms_adjusted_steps(original_steps: int, target_rms: float, observed_rms: float) -> Tuple[int, float]:
    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be a finite value greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("the RMS probe produced a non-finite or zero RMS, so training steps cannot be estimated")

    step_multiplier = target_rms / observed_rms
    adjusted_steps = max(1, math.floor(original_steps * step_multiplier + 0.5))
    return adjusted_steps, step_multiplier


def fit_probe_energy_slope(
    rms_curve: Sequence[Tuple[int, float]], probe_steps: int = 500, fit_start_step: int = 100
) -> float:
    """Fit normalized RMS-squared growth per 1,000 optimizer steps."""

    samples = [(int(step), float(rms)) for step, rms in rms_curve if fit_start_step <= step <= probe_steps]
    if len(samples) < 2:
        raise ValueError("RMS probe curve needs at least two samples between the fit start and probe step")
    if samples[-1][0] != probe_steps:
        raise ValueError("RMS probe curve is missing its final probe-step sample")

    probe_rms = samples[-1][1]
    if not math.isfinite(probe_rms) or probe_rms <= 0:
        raise ValueError("RMS probe curve has a non-finite or zero final RMS")
    if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
        raise ValueError("RMS probe curve contains a non-finite or zero RMS")

    xs = [float(step) for step, _ in samples]
    ys = [(rms / probe_rms) ** 2 for _, rms in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("RMS probe curve samples must use distinct optimizer steps")
    slope_per_step = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    slope_per_1000_steps = slope_per_step * 1000.0
    if not math.isfinite(slope_per_1000_steps) or slope_per_1000_steps <= 0:
        raise ValueError("RMS probe energy slope must be finite and greater than 0")
    return slope_per_1000_steps


def equal_squeeze_steps(total_steps: int) -> Tuple[int, int, int, int]:
    """Return the exact four boundaries used by an equal five-segment schedule."""

    if total_steps < 5:
        raise ValueError("total_steps must be at least 5 for a five-segment schedule")
    return tuple(math.floor(total_steps * index / 5.0) for index in range(1, 5))


def probe_schedule_needs_adjusted_probe(total_steps: int, probe_steps: int) -> bool:
    """Whether the proposed production schedule squeezes during the probe window."""

    if probe_steps <= 0:
        raise ValueError("probe_steps must be greater than 0")
    return equal_squeeze_steps(total_steps)[0] <= probe_steps


def choose_adjusted_probe_training_steps(
    reference_probe_steps: int,
    original_gradient_accumulation_steps: int,
    adjusted_gradient_accumulation_steps: int,
    production_steps: int,
    curve_interval: int,
    minimum_post_squeeze_samples: int = ADJUSTED_PROBE_MIN_POST_SQUEEZE_SAMPLES,
) -> Tuple[int, Dict[str, float]]:
    """Choose Probe 2's observation horizon from compute and phase coverage."""

    if reference_probe_steps <= 0:
        raise ValueError("reference_probe_steps must be greater than 0")
    if original_gradient_accumulation_steps <= 0:
        raise ValueError("original gradient accumulation must be greater than 0")
    if adjusted_gradient_accumulation_steps <= 0:
        raise ValueError("adjusted gradient accumulation must be greater than 0")
    if production_steps < 5:
        raise ValueError("production_steps must be at least 5")
    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    if minimum_post_squeeze_samples <= 0:
        raise ValueError("minimum_post_squeeze_samples must be greater than 0")

    nominal_microbatch_budget = (
        reference_probe_steps * original_gradient_accumulation_steps
    )
    budget_based_steps = math.ceil(
        nominal_microbatch_budget / adjusted_gradient_accumulation_steps
    )
    rounded_budget_steps = (
        math.ceil(budget_based_steps / curve_interval) * curve_interval
    )

    first_squeeze_step = equal_squeeze_steps(production_steps)[0]
    phase_coverage_steps = (
        first_squeeze_step + minimum_post_squeeze_samples * curve_interval
    )
    rounded_phase_coverage_steps = (
        math.ceil(phase_coverage_steps / curve_interval) * curve_interval
    )

    selected_steps = min(
        reference_probe_steps,
        max(rounded_budget_steps, rounded_phase_coverage_steps),
    )
    details = {
        "adjusted_probe_reference_steps": float(reference_probe_steps),
        "adjusted_probe_nominal_microbatch_budget": float(nominal_microbatch_budget),
        "adjusted_probe_budget_based_steps": float(budget_based_steps),
        "adjusted_probe_rounded_budget_steps": float(rounded_budget_steps),
        "adjusted_probe_first_squeeze_step": float(first_squeeze_step),
        "adjusted_probe_min_post_squeeze_samples": float(minimum_post_squeeze_samples),
        "adjusted_probe_phase_coverage_steps": float(phase_coverage_steps),
        "adjusted_probe_rounded_phase_coverage_steps": float(
            rounded_phase_coverage_steps
        ),
        "adjusted_probe_training_steps": float(selected_steps),
        "adjusted_probe_nominal_training_microbatches": float(
            selected_steps * adjusted_gradient_accumulation_steps
        ),
    }
    return selected_steps, details


def fit_schedule_aware_probe_energy_model(
    rms_curve: Sequence[Tuple[int, float]],
    total_steps: int,
    probe_steps: int = 500,
    curve_interval: int = 20,
) -> Tuple[float, float, Dict[str, float]]:
    """Fit rank-36 energy and express it at step 500 for any known schedule."""

    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    first_squeeze_step = equal_squeeze_steps(total_steps)[0]
    samples = [
        (int(step), float(rms))
        for step, rms in rms_curve
        if curve_interval <= step <= probe_steps and step < first_squeeze_step
    ]
    if first_squeeze_step > probe_steps:
        reference_rms = next(
            (float(rms) for step, rms in reversed(rms_curve) if int(step) == probe_steps),
            None,
        )
        if reference_rms is None:
            raise ValueError("RMS probe curve is missing its final probe-step sample")
        slope = fit_probe_energy_slope(rms_curve, probe_steps)
        return reference_rms, slope, {
            "first_squeeze_step": float(first_squeeze_step),
            "rank36_fit_samples": float(
                sum(1 for step, _ in rms_curve if 100 <= int(step) <= probe_steps)
            ),
            "reference_rms_is_extrapolated": 0.0,
        }

    if len(samples) < 2:
        raise ValueError("adjusted RMS probe needs at least two rank-36 samples before squeeze one")
    if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
        raise ValueError("RMS probe curve contains a non-finite or zero RMS")
    xs = [float(step) for step, _ in samples]
    ys = [rms**2 for _, rms in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("RMS probe curve samples must use distinct optimizer steps")
    energy_slope_per_step = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator
    reference_energy = y_mean + energy_slope_per_step * (probe_steps - x_mean)
    if (
        not math.isfinite(energy_slope_per_step)
        or energy_slope_per_step <= 0
        or not math.isfinite(reference_energy)
        or reference_energy <= 0
    ):
        raise ValueError("adjusted RMS probe produced an invalid rank-36 energy fit")
    reference_rms = math.sqrt(reference_energy)
    normalized_slope = energy_slope_per_step / reference_energy * 1000.0
    return reference_rms, normalized_slope, {
        "first_squeeze_step": float(first_squeeze_step),
        "rank36_fit_samples": float(len(samples)),
        "reference_rms_is_extrapolated": 1.0,
    }


def fit_observed_later_mean_energy_slope(
    rms_curve: Sequence[Tuple[int, float]],
    total_steps: int,
    probe_steps: int,
    reference_rms: float,
    curve_interval: int = 20,
) -> Tuple[float | None, Dict[str, float]]:
    """Estimate the calibrated later-stage base slope without crossing squeezes."""

    if not math.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError("reference_rms must be a finite value greater than 0")
    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    squeeze_steps = equal_squeeze_steps(total_steps)
    boundaries = (0, *squeeze_steps, total_steps)
    weighted_slopes = []
    details: Dict[str, float] = {}
    for segment_index in range(1, 5):
        segment_start = boundaries[segment_index]
        segment_end = boundaries[segment_index + 1]
        observed_end = min(segment_end, probe_steps)
        fit_start = segment_start + curve_interval
        fit_end = observed_end
        if segment_end <= probe_steps:
            fit_end = segment_end - curve_interval
        samples = [
            (int(step), float(rms))
            for step, rms in rms_curve
            if fit_start <= step <= fit_end
        ]
        if len(samples) < 2:
            continue
        if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
            raise ValueError("RMS probe curve contains a non-finite or zero RMS")
        xs = [float(step) for step, _ in samples]
        ys = [(rms / reference_rms) ** 2 for _, rms in samples]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 0:
            continue
        raw_slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            / denominator
            * 1000.0
        )
        if not math.isfinite(raw_slope) or raw_slope <= 0:
            continue
        base_slope = raw_slope / ENERGY_STAGE_FACTORS[segment_index - 1]
        observed_span = xs[-1] - xs[0]
        if observed_span <= 0:
            continue
        weighted_slopes.append((base_slope, observed_span))
        details[f"observed_stage_{segment_index}_base_slope"] = base_slope
        details[f"observed_stage_{segment_index}_span"] = observed_span

    if not weighted_slopes:
        details["observed_later_stage_count"] = 0.0
        return None, details
    total_weight = sum(weight for _, weight in weighted_slopes)
    later_mean_slope = sum(slope * weight for slope, weight in weighted_slopes) / total_weight
    details["observed_later_stage_count"] = float(len(weighted_slopes))
    details["observed_later_stage_span"] = total_weight
    details["observed_later_mean_energy_slope"] = later_mean_slope
    return later_mean_slope, details


def _augmented_later_slope_features_are_calibrated(
    dataset_batches_per_epoch: int,
    observed_rms: float,
    gradient_accumulation_steps: int,
) -> bool:
    features = (
        1.0 / dataset_batches_per_epoch,
        math.log(observed_rms),
        math.log(gradient_accumulation_steps),
    )
    return all(
        minimum <= feature <= maximum
        for feature, minimum, maximum in zip(
            features,
            ENERGY_LATER_SLOPE_FEATURE_MINS,
            ENERGY_LATER_SLOPE_FEATURE_MAXES,
        )
    )


def estimate_augmented_later_mean_energy_slope(
    dataset_batches_per_epoch: int,
    observed_rms: float,
    gradient_accumulation_steps: int,
) -> float:
    """Estimate later growth, retaining the legacy model outside the calibrated range."""

    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be finite and greater than 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    features = (
        1.0 / dataset_batches_per_epoch,
        math.log(observed_rms),
        math.log(gradient_accumulation_steps),
    )
    if not _augmented_later_slope_features_are_calibrated(
        dataset_batches_per_epoch,
        observed_rms,
        gradient_accumulation_steps,
    ):
        return (
            ENERGY_LEGACY_DATASET_INTERCEPT
            + ENERGY_LEGACY_DATASET_COEFFICIENT / dataset_batches_per_epoch
        )
    standardized_features = tuple(
        (feature - mean) / scale
        for feature, mean, scale in zip(
            features,
            ENERGY_LATER_SLOPE_FEATURE_MEANS,
            ENERGY_LATER_SLOPE_FEATURE_SCALES,
        )
    )
    later_mean_slope = ENERGY_LATER_SLOPE_INTERCEPT + sum(
        coefficient * feature
        for coefficient, feature in zip(
            ENERGY_LATER_SLOPE_COEFFICIENTS,
            standardized_features,
        )
    )
    if not math.isfinite(later_mean_slope) or later_mean_slope <= 0:
        raise ValueError(
            "augmented later-stage energy slope is non-positive; "
            "the probe configuration is outside the calibrated feature domain"
        )
    return later_mean_slope


def predict_piecewise_energy_final_rms(
    total_steps: int,
    probe_steps: int,
    observed_rms: float,
    dataset_batches_per_epoch: int,
    probe_energy_slope: float,
    gradient_accumulation_steps: int,
    later_mean_energy_slope: float | None = None,
) -> float:
    """Predict final RMS for the calibrated equal five-segment squeeze schedule."""

    if total_steps < probe_steps:
        raise ValueError("piecewise energy prediction requires total_steps >= probe_steps")
    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be a finite value greater than 0")
    if not math.isfinite(probe_energy_slope) or probe_energy_slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    probe_energy = observed_rms**2
    segment_steps = total_steps / 5.0
    energy = probe_energy * (1.0 + probe_energy_slope * (segment_steps - probe_steps) / 1000.0)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("piecewise energy prediction produced non-positive first-segment energy")
    later_mean_slope = later_mean_energy_slope
    if later_mean_slope is None:
        later_mean_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch,
            observed_rms,
            gradient_accumulation_steps,
        )
    if not math.isfinite(later_mean_slope) or later_mean_slope <= 0:
        raise ValueError("later_mean_energy_slope must be a finite value greater than 0")
    for stage_factor, retention in zip(ENERGY_STAGE_FACTORS, ENERGY_SQUEEZE_RETENTION):
        energy *= retention
        energy += probe_energy * later_mean_slope * stage_factor * segment_steps / 1000.0
    return math.sqrt(max(0.0, energy))


def estimate_piecewise_energy_adjusted_steps(
    original_steps: int,
    final_target_rms: float,
    observed_rms: float,
    probe_steps: int,
    dataset_batches_per_epoch: int,
    rms_curve: Sequence[Tuple[int, float]],
    gradient_accumulation_steps: int,
    probe_energy_slope: float | None = None,
    later_mean_energy_slope: float | None = None,
) -> Tuple[int, float, Dict[str, float]]:
    """Solve the calibrated RMS-squared trajectory for the requested final RMS."""

    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if not math.isfinite(final_target_rms) or final_target_rms <= 0:
        raise ValueError("final_target_rms must be a finite value greater than 0")
    slope = probe_energy_slope
    if slope is None:
        slope = fit_probe_energy_slope(rms_curve, probe_steps)
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")
    minimum_positive_segment = probe_steps - 1000.0 / slope
    minimum_steps = max(probe_steps, math.floor(5.0 * minimum_positive_segment) + 1)
    low = minimum_steps
    minimum_prediction = predict_piecewise_energy_final_rms(
        low,
        probe_steps,
        observed_rms,
        dataset_batches_per_epoch,
        slope,
        gradient_accumulation_steps,
        later_mean_energy_slope,
    )
    if minimum_prediction > final_target_rms:
        raise ValueError(
            "requested final RMS is below the minimum schedule compatible with a 500-step probe"
        )
    high = max(original_steps, minimum_steps)
    while predict_piecewise_energy_final_rms(
        high,
        probe_steps,
        observed_rms,
        dataset_batches_per_epoch,
        slope,
        gradient_accumulation_steps,
        later_mean_energy_slope,
    ) < final_target_rms:
        high *= 2
        if high > 100_000_000:
            raise ValueError("piecewise energy model could not bracket the requested final RMS")

    while low < high:
        midpoint = (low + high) // 2
        predicted = predict_piecewise_energy_final_rms(
            midpoint,
            probe_steps,
            observed_rms,
            dataset_batches_per_epoch,
            slope,
            gradient_accumulation_steps,
            later_mean_energy_slope,
        )
        if predicted < final_target_rms:
            low = midpoint + 1
        else:
            high = midpoint

    adjusted_steps = low
    augmented_features_are_calibrated = (
        _augmented_later_slope_features_are_calibrated(
            dataset_batches_per_epoch,
            observed_rms,
            gradient_accumulation_steps,
        )
    )
    details = {
        "probe_energy_slope_per_1000_steps": slope,
        "dataset_batches_per_epoch": float(dataset_batches_per_epoch),
        "later_mean_energy_slope_gradient_accumulation_steps": float(
            gradient_accumulation_steps
        ),
        "later_mean_energy_slope_is_observed": float(
            later_mean_energy_slope is not None
        ),
        "later_mean_energy_slope_uses_augmented_regression": float(
            later_mean_energy_slope is None
            and augmented_features_are_calibrated
        ),
        "later_mean_energy_slope_uses_legacy_fallback": float(
            later_mean_energy_slope is None
            and not augmented_features_are_calibrated
        ),
        "later_mean_energy_slope_per_1000_steps": (
            later_mean_energy_slope
            if later_mean_energy_slope is not None
            else estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch,
                observed_rms,
                gradient_accumulation_steps,
            )
        ),
        "predicted_final_rms": predict_piecewise_energy_final_rms(
            adjusted_steps,
            probe_steps,
            observed_rms,
            dataset_batches_per_epoch,
            slope,
            gradient_accumulation_steps,
            later_mean_energy_slope,
        ),
    }
    return adjusted_steps, adjusted_steps / original_steps, details


def choose_gradient_accumulation_steps(
    total_steps: int,
    target_microbatches: int | None,
    current_gradient_accumulation_steps: int,
    minimum_gradient_accumulation_steps: int = 1,
    rounding_bias: float = 0.6,
) -> int:
    """Choose accumulation near a compute budget with configurable upward bias."""

    if total_steps <= 0 or current_gradient_accumulation_steps <= 0:
        raise ValueError("training steps and gradient accumulation must be greater than 0")
    if minimum_gradient_accumulation_steps <= 0:
        raise ValueError("minimum gradient accumulation must be greater than 0")
    if not math.isfinite(rounding_bias) or rounding_bias < 0 or rounding_bias > 1:
        raise ValueError("rounding_bias must be between 0 and 1")
    if target_microbatches is None:
        return current_gradient_accumulation_steps
    if target_microbatches <= 0:
        raise ValueError("target_microbatches must be greater than 0")

    ideal_accumulation = target_microbatches / total_steps
    rounded = (
        math.ceil(ideal_accumulation)
        if rounding_bias == 1
        else math.floor(ideal_accumulation + rounding_bias)
    )
    return max(minimum_gradient_accumulation_steps, rounded)


def round_steps_to_nearest_multiple(steps: int, multiple: int | None) -> int:
    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if multiple is None:
        return steps
    if multiple <= 0:
        raise ValueError("multiple must be greater than 0")

    lower = (steps // multiple) * multiple
    upper = lower + multiple
    if lower == 0:
        return upper
    return lower if steps - lower < upper - steps else upper


def build_rms_probe_args(
    args: argparse.Namespace,
    probe_index: int = 1,
    production_steps: int | None = None,
    gradient_accumulation_steps: int | None = None,
    training_step_limit: int | None = None,
) -> argparse.Namespace:
    probe_args = copy.deepcopy(args)
    output_name = probe_args.output_name or "last"
    if probe_index not in (1, 2):
        raise ValueError("probe_index must be 1 or 2")
    if probe_index == 1:
        if (
            production_steps is not None
            or gradient_accumulation_steps is not None
            or training_step_limit is not None
        ):
            raise ValueError("adjusted production settings are only valid for probe 2")
        training_step_limit = probe_args.rms_probe_steps
        probe_dir_name = f"{output_name}-rms-probe-step{probe_args.rms_probe_steps}"
    else:
        if production_steps is None or production_steps < probe_args.rms_probe_steps:
            raise ValueError("probe 2 production_steps must be at least rms_probe_steps")
        if gradient_accumulation_steps is None or gradient_accumulation_steps <= 0:
            raise ValueError("probe 2 gradient accumulation must be greater than 0")
        if training_step_limit is None:
            training_step_limit = probe_args.rms_probe_steps
        if training_step_limit <= 0 or training_step_limit > probe_args.rms_probe_steps:
            raise ValueError("probe 2 training_step_limit must be between 1 and rms_probe_steps")
        if training_step_limit > production_steps:
            raise ValueError("probe 2 training_step_limit cannot exceed production_steps")
        probe_args.max_train_steps = production_steps
        probe_args.gradient_accumulation_steps = gradient_accumulation_steps
        probe_dir_name = (
            f"{output_name}-rms-probe2-step{training_step_limit}"
            f"-steps{production_steps}-ga{gradient_accumulation_steps}"
        )

    probe_args.output_dir = os.path.join(probe_args.output_dir or ".", probe_dir_name)
    probe_args.output_name = probe_dir_name
    probe_args._training_step_limit = training_step_limit
    probe_args._is_rms_probe_run = True
    probe_args._rms_probe_index = probe_index

    # Keep the final probe weights and resumable state, but avoid periodic
    # outputs, external uploads, samples, validation, and tracker runs.
    probe_args.save_every_n_steps = None
    probe_args.save_every_n_epochs = None
    probe_args.save_n_epoch_ratio = None
    probe_args.save_state = False
    probe_args.save_state_on_train_end = True
    probe_args.save_state_to_huggingface = False
    probe_args.huggingface_repo_id = None
    probe_args.sample_every_n_steps = None
    probe_args.sample_every_n_epochs = None
    probe_args.sample_at_first = False
    probe_args.max_validation_steps = 0
    probe_args.logging_dir = None
    probe_args.log_with = None
    probe_args.total_rms_check_every_n_steps = (
        probe_args.rms_probe_curve_every_n_steps
        if probe_args.rms_probe_scaling_policy == PIECEWISE_ENERGY_POLICY
        else 0
    )

    return probe_args
