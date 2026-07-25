import os
import sys
import math
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_step_probe import (
    build_rms_probe_args,
    choose_adjusted_probe_training_steps,
    choose_gradient_accumulation_steps,
    equal_squeeze_steps,
    estimate_augmented_later_mean_energy_slope,
    estimate_piecewise_energy_adjusted_steps,
    estimate_rms_adjusted_steps,
    fit_observed_later_mean_energy_slope,
    fit_probe_energy_slope,
    fit_schedule_aware_probe_energy_model,
    probe_schedule_needs_adjusted_probe,
    predict_piecewise_energy_final_rms,
    round_steps_to_nearest_multiple,
    validate_rms_probe_configuration,
)


def make_args(**overrides):
    values = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "rms_probe_scaling_policy": "linear",
        "rms_probe_final_target": None,
        "rms_probe_curve_every_n_steps": 20,
        "rms_probe_adjusted_steps_divisible_by": None,
        "rms_probe_gradient_accumulation_target_microbatches": None,
        "rms_probe_gradient_accumulation_rounding_bias": 0.6,
        "rms_probe_min_gradient_accumulation_steps": 1,
        "max_train_steps": 5000,
        "max_train_epochs": None,
        "resume": None,
        "initial_step": None,
        "initial_epoch": None,
        "deepspeed": False,
        "output_dir": "output",
        "output_name": "character",
        "save_every_n_steps": 100,
        "save_every_n_epochs": 1,
        "save_n_epoch_ratio": 4,
        "save_state": True,
        "save_state_on_train_end": False,
        "save_state_to_huggingface": True,
        "huggingface_repo_id": "owner/repo",
        "sample_every_n_steps": 100,
        "sample_every_n_epochs": 1,
        "sample_at_first": True,
        "max_validation_steps": None,
        "logging_dir": "logs",
        "log_with": "wandb",
        "total_rms_check_every_n_steps": 25,
        "gradient_accumulation_steps": 6,
        "lora_squeeze_start_dim": 36,
        "network_dim": 9,
        "lora_squeeze_num_squeezes": 4,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_step_schedule": "equal",
        "lora_squeeze_rank_schedule": "geometric",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RMSStepProbeTest(unittest.TestCase):
    def test_estimate_scales_original_production_steps(self):
        adjusted_steps, multiplier = estimate_rms_adjusted_steps(5000, 0.0001, 0.00008)

        self.assertEqual(adjusted_steps, 6250)
        self.assertAlmostEqual(multiplier, 1.25)

    def test_estimate_rounds_to_nearest_step_and_never_returns_zero(self):
        self.assertEqual(estimate_rms_adjusted_steps(5, 1.0, 2.0)[0], 3)
        self.assertEqual(estimate_rms_adjusted_steps(5, 1.0, 100.0)[0], 1)

    def test_zero_observed_rms_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero RMS"):
            estimate_rms_adjusted_steps(5000, 0.0001, 0.0)

    def test_adjusted_steps_can_be_rounded_to_nearest_multiple(self):
        self.assertEqual(round_steps_to_nearest_multiple(4398, 5), 4400)
        self.assertEqual(round_steps_to_nearest_multiple(4397, 5), 4395)
        self.assertEqual(round_steps_to_nearest_multiple(10, 4), 12)
        self.assertEqual(round_steps_to_nearest_multiple(1, 5), 5)
        self.assertEqual(round_steps_to_nearest_multiple(4398, None), 4398)

    def test_probe_energy_slope_uses_squared_rms(self):
        observed = 4e-5
        curve = [(step, observed * math.sqrt(1.0 + 0.002 * (step - 500))) for step in range(100, 501, 20)]

        self.assertAlmostEqual(fit_probe_energy_slope(curve), 2.0)

    def test_piecewise_energy_solver_hits_final_target(self):
        observed = 3.527385845165899e-5
        curve = [(step, observed * math.sqrt(1.0 + 0.002024 * (step - 500))) for step in range(100, 501, 20)]

        steps, multiplier, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=250,
            rms_curve=curve,
            gradient_accumulation_steps=6,
        )

        self.assertEqual(steps, 6517)
        self.assertAlmostEqual(multiplier, steps / 4000)
        self.assertEqual(
            details["later_mean_energy_slope_gradient_accumulation_steps"], 6.0
        )
        self.assertEqual(
            details["later_mean_energy_slope_uses_augmented_regression"], 1.0
        )
        self.assertEqual(details["later_mean_energy_slope_uses_legacy_fallback"], 0.0)
        self.assertGreaterEqual(details["predicted_final_rms"], 8.425384599385171e-5)
        self.assertLess(
            predict_piecewise_energy_final_rms(steps - 1, 500, observed, 250, 2.024, 6),
            8.425384599385171e-5,
        )

    def test_piecewise_energy_solver_supports_a_squeeze_before_probe_end(self):
        observed = 5.3288192e-5
        curve = [
            (step, observed * math.sqrt(1.0 + 0.002 * (step - 500)))
            for step in range(100, 501, 20)
        ]

        steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=20,
            rms_curve=curve,
            gradient_accumulation_steps=6,
        )

        self.assertEqual(steps, 669)
        self.assertTrue(probe_schedule_needs_adjusted_probe(steps, 500))

    def test_finished_beastgirl_probe_selects_adjusted_probe_settings(self):
        curve = [
            (20, 7.7112147e-6),
            (40, 1.0898868e-5),
            (60, 1.3565062e-5),
            (80, 1.6082649e-5),
            (100, 1.87522e-5),
            (120, 2.0835095e-5),
            (140, 2.2580525e-5),
            (160, 2.4658147e-5),
            (180, 2.6919618e-5),
            (200, 2.8844781e-5),
            (220, 3.0423066e-5),
            (240, 3.245155e-5),
            (260, 3.4024333e-5),
            (280, 3.5367862e-5),
            (300, 3.7224761e-5),
            (320, 3.8830702e-5),
            (340, 4.0604795e-5),
            (360, 4.2479001e-5),
            (380, 4.4004631e-5),
            (400, 4.5833497e-5),
            (420, 4.7086027e-5),
            (440, 4.8636081e-5),
            (460, 5.0261792e-5),
            (480, 5.1694753e-5),
            (500, 5.3288192e-5),
        ]

        steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=curve[-1][1],
            probe_steps=500,
            dataset_batches_per_epoch=20,
            rms_curve=curve,
            gradient_accumulation_steps=20,
        )
        rounded_steps = round_steps_to_nearest_multiple(steps, 100)
        uncapped_accumulation = choose_gradient_accumulation_steps(
            rounded_steps, 24000, 6, rounding_bias=0.7
        )

        self.assertEqual(steps, 682)
        self.assertAlmostEqual(details["probe_energy_slope_per_1000_steps"], 2.2019441770164145)
        self.assertEqual(
            details["later_mean_energy_slope_uses_augmented_regression"], 0.0
        )
        self.assertEqual(details["later_mean_energy_slope_uses_legacy_fallback"], 1.0)
        self.assertEqual(rounded_steps, 700)
        self.assertEqual(uncapped_accumulation, 34)
        self.assertEqual(min(uncapped_accumulation, 20), 20)
        self.assertEqual(equal_squeeze_steps(rounded_steps), (140, 280, 420, 560))

    def test_beastgirl_adjusted_probe_uses_rounded_budget_and_phase_coverage(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=700,
            curve_interval=20,
        )

        self.assertEqual(probe_steps, 240)
        self.assertEqual(details["adjusted_probe_nominal_microbatch_budget"], 3000.0)
        self.assertEqual(details["adjusted_probe_budget_based_steps"], 150.0)
        self.assertEqual(details["adjusted_probe_rounded_budget_steps"], 160.0)
        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 140.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 240.0)
        self.assertEqual(details["adjusted_probe_rounded_phase_coverage_steps"], 240.0)
        self.assertEqual(details["adjusted_probe_nominal_training_microbatches"], 4800.0)

    def test_adjusted_probe_rounds_unaligned_phase_coverage_upward(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=100,
            production_steps=706,
            curve_interval=20,
        )

        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 141.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 241.0)
        self.assertEqual(details["adjusted_probe_rounded_phase_coverage_steps"], 260.0)
        self.assertEqual(probe_steps, 260)

    def test_shortened_adjusted_probe_can_fit_pre_and_post_squeeze_growth(self):
        production_steps = 700
        observation_steps = 240
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        probe_slope = 2.0
        later_base_slope = 1.0
        first_squeeze = equal_squeeze_steps(production_steps)[0]
        energy_at_squeeze = reference_energy * (
            1.0 + probe_slope * (first_squeeze - 500) / 1000.0
        )

        curve = []
        for step in range(20, observation_steps + 1, 20):
            if step < first_squeeze:
                energy = reference_energy * (
                    1.0 + probe_slope * (step - 500) / 1000.0
                )
            else:
                energy = energy_at_squeeze * 0.93776704
                energy += (
                    reference_energy
                    * later_base_slope
                    * 0.84333973
                    * (step - first_squeeze)
                    / 1000.0
                )
            curve.append((step, math.sqrt(energy)))

        fitted_rms, fitted_probe_slope, _ = fit_schedule_aware_probe_energy_model(
            curve, production_steps, 500, 20
        )
        fitted_later_slope, details = fit_observed_later_mean_energy_slope(
            curve, production_steps, observation_steps, fitted_rms, 20
        )

        self.assertAlmostEqual(fitted_rms, reference_rms)
        self.assertAlmostEqual(fitted_probe_slope, probe_slope)
        self.assertAlmostEqual(fitted_later_slope, later_base_slope)
        self.assertEqual(details["observed_later_stage_count"], 1.0)

    def test_schedule_aware_probe_fits_each_observed_rank_segment_separately(self):
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        later_base_slope = 1.0
        squeeze_steps = equal_squeeze_steps(1000)

        def energy_at(step):
            energy = reference_energy * (1.0 + 2.0 * (squeeze_steps[0] - 500) / 1000.0)
            segment_start = squeeze_steps[0]
            for stage_index, segment_end in enumerate((*squeeze_steps[1:], 1000)):
                energy *= (0.93776704, 0.91973453, 0.89532405, 0.86626541)[stage_index]
                observed_end = min(step, segment_end)
                if observed_end > segment_start:
                    energy += (
                        reference_energy
                        * later_base_slope
                        * (0.84333973, 0.95492410, 1.08152779, 1.12020838)[stage_index]
                        * (observed_end - segment_start)
                        / 1000.0
                    )
                if step <= segment_end:
                    return energy
                segment_start = segment_end
            return energy

        curve = []
        for step in range(20, 501, 20):
            if step < squeeze_steps[0]:
                energy = reference_energy * (1.0 + 2.0 * (step - 500) / 1000.0)
            else:
                energy = energy_at(step)
            curve.append((step, math.sqrt(energy)))

        fitted_rms, fitted_probe_slope, _ = fit_schedule_aware_probe_energy_model(
            curve, 1000, 500, 20
        )
        fitted_later_slope, details = fit_observed_later_mean_energy_slope(
            curve, 1000, 500, fitted_rms, 20
        )

        self.assertAlmostEqual(fitted_rms, reference_rms)
        self.assertAlmostEqual(fitted_probe_slope, 2.0)
        self.assertAlmostEqual(fitted_later_slope, later_base_slope)
        self.assertEqual(details["observed_later_stage_count"], 2.0)

    def test_adjusted_probe_trigger_uses_exact_floor_boundaries(self):
        self.assertTrue(probe_schedule_needs_adjusted_probe(2504, 500))
        self.assertFalse(probe_schedule_needs_adjusted_probe(2505, 500))

    def test_augmented_later_slope_uses_probe_rms_bpe_and_production_accumulation(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=132,
                observed_rms=3.8782791e-5,
                gradient_accumulation_steps=6,
            ),
            1.3369370128008717,
        )
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=5,
            ),
            0.9972399143882617,
        )
        self.assertGreater(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=6,
            ),
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=4,
            ),
        )

    def test_augmented_later_slope_preserves_legacy_out_of_range_behavior(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=20,
                observed_rms=5.3288192e-5,
                gradient_accumulation_steps=20,
            ),
            5.1092024165,
        )

    def test_piecewise_energy_calibration_reproduces_augmented_predictions(self):
        mrissi_predicted = predict_piecewise_energy_final_rms(
            total_steps=4000,
            probe_steps=500,
            observed_rms=3.8782791e-5,
            dataset_batches_per_epoch=132,
            probe_energy_slope=2.103,
            gradient_accumulation_steps=6,
        )
        wilykit_predicted = predict_piecewise_energy_final_rms(
            total_steps=4600,
            probe_steps=500,
            observed_rms=3.804286006720449e-5,
            dataset_batches_per_epoch=157,
            probe_energy_slope=2.063401097961805,
            gradient_accumulation_steps=5,
        )
        wilykit_second_predicted = predict_piecewise_energy_final_rms(
            total_steps=5200,
            probe_steps=500,
            observed_rms=3.804286006720449e-5,
            dataset_batches_per_epoch=157,
            probe_energy_slope=2.063401097961805,
            gradient_accumulation_steps=5,
        )

        self.assertAlmostEqual(mrissi_predicted, 8.431765131805465e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_predicted, 7.950321171961089e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_second_predicted, 8.45530059000021e-5, delta=1e-11)

    def test_gradient_accumulation_uses_biased_nearest_microbatch_budget(self):
        self.assertEqual(choose_gradient_accumulation_steps(3000, 24000, 6), 8)
        self.assertEqual(choose_gradient_accumulation_steps(3500, 24000, 6), 7)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(4500, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5000, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, None, 6), 6)

    def test_gradient_accumulation_rounding_bias_is_configurable(self):
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6, rounding_bias=0.5), 4)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6, rounding_bias=0.7), 5)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6, rounding_bias=0.6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(4000, 24000, 6, rounding_bias=1.0), 6)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6, rounding_bias=1.0), 7)

    def test_gradient_accumulation_respects_configured_minimum(self):
        self.assertEqual(choose_gradient_accumulation_steps(12000, 24000, 6, 3), 3)

    def test_adjusted_step_multiple_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            validate_rms_probe_configuration(make_args(rms_probe_adjusted_steps_divisible_by=0))

    def test_adjusted_step_multiple_requires_probe(self):
        with self.assertRaisesRegex(ValueError, "requires --rms_probe_target"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_target=None,
                    rms_probe_steps=None,
                    rms_probe_adjusted_steps_divisible_by=5,
                )
            )

    def test_probe_parameters_must_be_given_together(self):
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=None))

    def test_probe_rejects_resume_and_initial_position(self):
        with self.assertRaisesRegex(ValueError, "cannot be used with --resume"):
            validate_rms_probe_configuration(make_args(resume="state"))
        with self.assertRaisesRegex(ValueError, "cannot be used with --initial_step"):
            validate_rms_probe_configuration(make_args(initial_step=100))

    def test_probe_cannot_exceed_original_training_horizon(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed --max_train_steps"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=5001))

    def test_piecewise_probe_requires_first_original_squeeze_after_probe(self):
        with self.assertRaisesRegex(ValueError, "first squeeze"):
            validate_rms_probe_configuration(
                make_args(
                    max_train_steps=2504,
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                )
            )

    def test_probe_rejects_deepspeed(self):
        with self.assertRaisesRegex(ValueError, "does not support --deepspeed"):
            validate_rms_probe_configuration(make_args(deepspeed=True))

    def test_piecewise_policy_requires_final_target_and_calibrated_schedule(self):
        with self.assertRaisesRegex(ValueError, "rms_probe_final_target"):
            validate_rms_probe_configuration(make_args(rms_probe_scaling_policy="piecewise_energy_v1"))
        with self.assertRaisesRegex(ValueError, "36->9"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    network_dim=16,
                )
            )

    def test_piecewise_policy_accepts_calibrated_configuration(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5)
            )
        )

    def test_piecewise_policy_requires_enough_pre_squeeze_curve_samples(self):
        with self.assertRaisesRegex(ValueError, "two pre-squeeze samples"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_curve_every_n_steps=50,
                )
            )

    def test_gradient_accumulation_rounding_bias_must_be_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_rms_probe_configuration(
                make_args(rms_probe_gradient_accumulation_rounding_bias=1.1)
            )

    def test_probe_keeps_scheduler_horizon_but_stops_at_probe_step(self):
        probe_args = build_rms_probe_args(make_args())

        self.assertEqual(probe_args.max_train_steps, 5000)
        self.assertEqual(probe_args._training_step_limit, 500)
        self.assertEqual(probe_args.output_dir, os.path.join("output", "character-rms-probe-step500"))
        self.assertTrue(probe_args.save_state_on_train_end)
        self.assertIsNone(probe_args.save_n_epoch_ratio)
        self.assertIsNone(probe_args.huggingface_repo_id)
        self.assertIsNone(probe_args.logging_dir)
        self.assertEqual(probe_args.total_rms_check_every_n_steps, 0)

    def test_piecewise_probe_collects_the_configured_curve(self):
        probe_args = build_rms_probe_args(
            make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5)
        )

        self.assertEqual(probe_args.total_rms_check_every_n_steps, 20)

    def test_second_probe_uses_provisional_production_settings(self):
        probe_args = build_rms_probe_args(
            make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5),
            probe_index=2,
            production_steps=700,
            gradient_accumulation_steps=20,
            training_step_limit=240,
        )

        self.assertEqual(probe_args.max_train_steps, 700)
        self.assertEqual(probe_args.gradient_accumulation_steps, 20)
        self.assertEqual(probe_args._training_step_limit, 240)
        self.assertEqual(
            probe_args.output_dir,
            os.path.join("output", "character-rms-probe2-step240-steps700-ga20"),
        )

if __name__ == "__main__":
    unittest.main()
