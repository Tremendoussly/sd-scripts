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
    count_optimizer_step_microbatches,
    equal_squeeze_steps,
    epoch_accumulation_group_sizes,
    estimate_augmented_later_mean_energy_slope,
    estimate_piecewise_energy_adjusted_steps,
    estimate_piecewise_training_plan,
    estimate_rms_adjusted_steps,
    fit_adjusted_probe_rank36_transfer,
    fit_observed_later_mean_energy_slope,
    fit_probe_energy_slope,
    fit_schedule_aware_probe_energy_model,
    optimizer_steps_for_microbatch_budget,
    probe_schedule_needs_adjusted_probe,
    predict_piecewise_energy_final_rms,
    rescale_probe_measurements_for_gradient_accumulation,
    round_steps_to_nearest_multiple,
    should_run_adjusted_probe,
    squeeze_schedule_steps,
    squeeze_segment_steps,
    update_piecewise_production_prediction,
    validate_rms_probe_configuration,
)
from library.lora_squeeze_schedule import LoRASqueezeSchedule


def make_args(**overrides):
    values = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "rms_probe_scaling_policy": "linear",
        "rms_probe_final_target": None,
        "rms_probe_curve_every_n_steps": 20,
        "rms_probe_adjusted_steps_divisible_by": None,
        "rms_probe_force_adjusted_probe": False,
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
        "network_alpha": 3,
        "lora_squeeze_num_squeezes": 4,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_step_schedule": "equal",
        "lora_squeeze_rank_schedule": "geometric",
        "lora_squeeze_first_segment_ratio": 1.0,
        "lora_squeeze_final_segment_ratio": 1.0,
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

    def test_short_probe_projects_rank36_energy_to_step_500_reference(self):
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        reference_slope = 2.0
        curve = []
        for step in (*range(20, 141, 20), 150):
            energy = reference_energy * (
                1.0 + reference_slope * (step - 500) / 1000.0
            )
            curve.append((step, math.sqrt(energy)))

        projected_rms, projected_slope, details = (
            fit_schedule_aware_probe_energy_model(
                curve,
                total_steps=4000,
                probe_steps=500,
                curve_interval=20,
            )
        )

        self.assertAlmostEqual(projected_rms, reference_rms)
        self.assertAlmostEqual(projected_slope, reference_slope)
        self.assertEqual(details["probe_observation_steps"], 150.0)
        self.assertEqual(details["calibration_reference_steps"], 500.0)
        self.assertEqual(details["reference_rms_is_extrapolated"], 1.0)

    def test_probe_measurements_rescale_with_calibrated_accumulation_elasticity(self):
        adjusted_rms, adjusted_slope, details = (
            rescale_probe_measurements_for_gradient_accumulation(
                observed_rms=4.19757912778963e-5,
                probe_energy_slope=2.1832932595622987,
                probe_gradient_accumulation_steps=6,
                production_gradient_accumulation_steps=9,
            )
        )

        self.assertAlmostEqual(adjusted_rms, 4.43208512565915e-5)
        self.assertAlmostEqual(adjusted_slope, 2.208833613089547)
        self.assertEqual(details["probe_gradient_accumulation_ratio"], 1.5)

    def test_probe_measurement_rescaling_is_reversible(self):
        observed_rms = 3.804286006720449e-5
        observed_slope = 2.0634011093724816
        adjusted_rms, adjusted_slope, _ = (
            rescale_probe_measurements_for_gradient_accumulation(
                observed_rms,
                observed_slope,
                6,
                5,
            )
        )
        restored_rms, restored_slope, _ = (
            rescale_probe_measurements_for_gradient_accumulation(
                adjusted_rms,
                adjusted_slope,
                5,
                6,
            )
        )

        self.assertAlmostEqual(restored_rms, observed_rms)
        self.assertAlmostEqual(restored_slope, observed_slope)

    def test_probe_measurement_rescaling_improves_calibration_pairs(self):
        # Same-GA reruns are represented by their geometric means, matching the
        # calibration fit. The final wide-range Rosine pair differs by one image.
        pairs = (
            (6, 3.54396179318428e-5, 2.038203269953988, 5, 3.469298826530576e-5, 2.006614663143924),
            (6, 3.634618288860564e-5, 2.059958669045801, 5, 3.633632877608761e-5, 2.071582634208073),
            (6, 3.804286006720449e-5, 2.0634011093724816, 5, 3.7372468152551915e-5, 2.0134947818470312),
            (6, 3.6507251192282416e-5, 1.996520566790334, 5, 3.585959711926989e-5, 1.966833287803121),
            (6, 4.31827932468486e-5, 2.1156507756813077, 8, 4.517141292410292e-5, 2.1815015111656755),
            (6, 4.19757912778963e-5, 2.1832932595622987, 9, 4.448729100949966e-5, 2.232738325480896),
            (7, 4.329497199592824e-5, 2.134595960542286, 2, 3.648383790277876e-5, 2.085742066127855),
        )
        raw_rms_errors = []
        adjusted_rms_errors = []
        raw_slope_errors = []
        adjusted_slope_errors = []
        for probe_ga, probe_rms, probe_slope, production_ga, actual_rms, actual_slope in pairs:
            adjusted_rms, adjusted_slope, _ = (
                rescale_probe_measurements_for_gradient_accumulation(
                    probe_rms,
                    probe_slope,
                    probe_ga,
                    production_ga,
                )
            )
            raw_rms_errors.append(abs(probe_rms / actual_rms - 1.0))
            adjusted_rms_errors.append(abs(adjusted_rms / actual_rms - 1.0))
            raw_slope_errors.append(abs(probe_slope / actual_slope - 1.0))
            adjusted_slope_errors.append(abs(adjusted_slope / actual_slope - 1.0))

        mean = lambda values: sum(values) / len(values)
        self.assertAlmostEqual(mean(raw_rms_errors), 0.049279778527764516)
        self.assertAlmostEqual(mean(adjusted_rms_errors), 0.007654149229480108)
        self.assertAlmostEqual(mean(raw_slope_errors), 0.01956964287055209)
        self.assertAlmostEqual(mean(adjusted_slope_errors), 0.013719752333265707)

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

        self.assertEqual(steps, 5797)
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
            dataset_batches_per_epoch=50,
            rms_curve=curve,
            gradient_accumulation_steps=6,
        )

        self.assertEqual(steps, 1374)
        self.assertTrue(probe_schedule_needs_adjusted_probe(steps, 500))

    def test_piecewise_energy_solver_can_extrapolate_below_calibration_reference(self):
        observed = 4e-5
        target_steps = 350
        target = predict_piecewise_energy_final_rms(
            total_steps=target_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=50,
            probe_energy_slope=2.0,
            gradient_accumulation_steps=20,
            later_mean_energy_slope=1.0,
        )

        steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=target,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=50,
            rms_curve=[],
            gradient_accumulation_steps=20,
            probe_energy_slope=2.0,
            later_mean_energy_slope=1.0,
        )

        self.assertEqual(steps, target_steps)

    def test_out_of_range_augmented_estimate_selects_adjusted_probe_settings(self):
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
            dataset_batches_per_epoch=50,
            rms_curve=curve,
            gradient_accumulation_steps=20,
        )
        rounded_steps = round_steps_to_nearest_multiple(steps, 100)
        uncapped_accumulation = choose_gradient_accumulation_steps(
            rounded_steps, 24000, 6, rounding_bias=0.7
        )

        self.assertEqual(steps, 1198)
        self.assertAlmostEqual(details["probe_energy_slope_per_1000_steps"], 2.2019441770164145)
        self.assertEqual(
            details["later_mean_energy_slope_uses_augmented_regression"], 1.0
        )
        self.assertEqual(details["later_mean_energy_slope_uses_legacy_fallback"], 0.0)
        self.assertEqual(rounded_steps, 1200)
        self.assertEqual(uncapped_accumulation, 20)
        self.assertEqual(min(uncapped_accumulation, 50), 20)
        self.assertEqual(equal_squeeze_steps(rounded_steps), (240, 480, 720, 960))

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

    def test_adjusted_probe_is_capped_by_sub_reference_production_horizon(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=120,
            curve_interval=20,
        )

        self.assertEqual(probe_steps, 120)
        self.assertEqual(details["adjusted_probe_training_steps"], 120.0)

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

    def test_adjusted_probe_uses_ratio_adjusted_first_squeeze(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=700,
            curve_interval=20,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )

        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 233.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 333.0)
        self.assertEqual(probe_steps, 340)

    def test_adjusted_probe_budget_counts_epoch_remainder_updates(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=2,
            original_gradient_accumulation_steps=18,
            adjusted_gradient_accumulation_steps=11,
            production_steps=100,
            curve_interval=1,
            minimum_post_squeeze_samples=1,
            dataset_batches_per_epoch=22,
        )

        self.assertEqual(details["adjusted_probe_nominal_microbatch_budget"], 36.0)
        self.assertEqual(details["adjusted_probe_actual_microbatch_budget"], 22.0)
        self.assertEqual(details["adjusted_probe_budget_based_steps"], 2.0)
        self.assertEqual(probe_steps, 2)
        self.assertEqual(details["adjusted_probe_actual_training_microbatches"], 22.0)

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

    def test_adjusted_probe_transfers_relative_rank36_scale_without_short_extrapolation(self):
        primary_curve = (
            (20, 1.0e-5),
            (40, 1.4e-5),
            (60, 1.8e-5),
            (80, 2.2e-5),
            (100, 2.6e-5),
        )
        adjusted_curve = tuple(
            (step, rms * 1.1)
            for step, rms in primary_curve
        )

        reference_rms, energy_slope, details = (
            fit_adjusted_probe_rank36_transfer(
                primary_curve,
                adjusted_curve,
                adjusted_total_steps=400,
                adjusted_probe_steps=100,
                primary_reference_rms=5.0e-5,
                primary_probe_energy_slope=2.0,
            )
        )

        self.assertAlmostEqual(reference_rms, 5.5e-5)
        self.assertAlmostEqual(energy_slope, 2.0)
        self.assertAlmostEqual(details["rank36_transfer_energy_scale"], 1.21)
        self.assertEqual(details["rank36_transfer_sample_count"], 3.0)

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
        self.assertFalse(
            probe_schedule_needs_adjusted_probe(
                1600,
                500,
                first_segment_ratio=2.0,
                final_segment_ratio=1.0,
            )
        )

    def test_adjusted_probe_can_be_forced_outside_probe_window(self):
        self.assertFalse(
            should_run_adjusted_probe("piecewise_energy_v1", 4000, 500)
        )
        self.assertTrue(
            should_run_adjusted_probe(
                "piecewise_energy_v1",
                4000,
                500,
                force_adjusted_probe=True,
            )
        )
        forced_probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=6,
            production_steps=4000,
            curve_interval=20,
        )
        self.assertEqual(forced_probe_steps, 500)
        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 800.0)
        self.assertTrue(
            should_run_adjusted_probe("piecewise_energy_v1", 2500, 500)
        )
        self.assertFalse(
            should_run_adjusted_probe(
                "linear",
                4000,
                500,
                force_adjusted_probe=True,
            )
        )

    def test_ratio_adjusted_schedule_matches_lora_squeeze_weighting(self):
        self.assertEqual(
            squeeze_schedule_steps(
                4000,
                first_segment_ratio=0.5,
                final_segment_ratio=2.0,
            ),
            (363, 1090, 1818, 2545),
        )
        self.assertEqual(
            squeeze_segment_steps(
                4000,
                first_segment_ratio=0.5,
                final_segment_ratio=2.0,
            ),
            (363, 727, 728, 727, 1455),
        )

    def test_ratio_adjusted_boundaries_match_lora_squeeze_schedule(self):
        for total_steps, first_ratio, final_ratio in (
            (503, 1.0, 1.0),
            (4000, 0.75, 1.5),
            (5379, 2.0, 0.5),
        ):
            args = make_args(
                max_train_steps=total_steps,
                lora_squeeze_first_segment_ratio=first_ratio,
                lora_squeeze_final_segment_ratio=final_ratio,
            )
            schedule = LoRASqueezeSchedule(args)
            schedule.set_total_steps(total_steps)

            self.assertEqual(
                squeeze_schedule_steps(
                    total_steps,
                    first_ratio,
                    final_ratio,
                ),
                tuple(schedule.squeeze_steps),
            )
            self.assertEqual(
                squeeze_segment_steps(
                    total_steps,
                    first_ratio,
                    final_ratio,
                ),
                tuple(schedule.segment_steps),
            )

    def test_ratio_adjusted_prediction_uses_exact_integer_segments(self):
        observed = 4e-5
        probe_slope = 2.0
        later_slope = 1.0
        segment_steps = squeeze_segment_steps(
            503,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )
        energy = observed**2 * (
            1.0 + probe_slope * (segment_steps[0] - 500) / 1000.0
        )
        for stage_factor, retention, stage_steps in zip(
            (0.84333973, 0.95492410, 1.08152779, 1.12020838),
            (0.93776704, 0.91973453, 0.89532405, 0.86626541),
            segment_steps[1:],
        ):
            energy *= retention
            energy += (
                observed**2
                * later_slope
                * stage_factor
                * stage_steps
                / 1000.0
            )

        predicted = predict_piecewise_energy_final_rms(
            total_steps=503,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=200,
            probe_energy_slope=probe_slope,
            gradient_accumulation_steps=5,
            later_mean_energy_slope=later_slope,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )

        self.assertAlmostEqual(predicted, math.sqrt(energy))

    def test_ratio_adjusted_solver_hits_the_exact_schedule_target(self):
        observed = 4e-5
        target_steps = 4379
        target = predict_piecewise_energy_final_rms(
            total_steps=target_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=200,
            probe_energy_slope=2.0,
            gradient_accumulation_steps=5,
            later_mean_energy_slope=1.0,
            first_segment_ratio=2.0,
            final_segment_ratio=0.75,
        )

        steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=target,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=200,
            rms_curve=[],
            gradient_accumulation_steps=5,
            probe_energy_slope=2.0,
            later_mean_energy_slope=1.0,
            first_segment_ratio=2.0,
            final_segment_ratio=0.75,
        )

        self.assertEqual(steps, target_steps)
        self.assertAlmostEqual(details["predicted_final_rms"], target)
        self.assertEqual(
            tuple(details[f"segment_{index}_steps"] for index in range(1, 6)),
            tuple(
                float(value)
                for value in squeeze_segment_steps(
                    target_steps,
                    first_segment_ratio=2.0,
                    final_segment_ratio=0.75,
                )
            ),
        )

    def test_production_prediction_is_recomputed_after_step_rounding(self):
        observed = 3.527385845165899e-5
        curve = [
            (
                step,
                observed * math.sqrt(1.0 + 0.002024 * (step - 500)),
            )
            for step in range(100, 501, 20)
        ]
        estimated_steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=250,
            rms_curve=curve,
            gradient_accumulation_steps=6,
        )
        rounded_steps = round_steps_to_nearest_multiple(estimated_steps, 100)

        updated = update_piecewise_production_prediction(
            details,
            production_steps=rounded_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=250,
            gradient_accumulation_steps=6,
        )

        self.assertEqual(estimated_steps, 5797)
        self.assertEqual(rounded_steps, 5800)
        self.assertEqual(
            updated["estimated_steps_predicted_final_rms"],
            details["predicted_final_rms"],
        )
        self.assertAlmostEqual(
            updated["predicted_final_rms"],
            predict_piecewise_energy_final_rms(
                rounded_steps,
                500,
                observed,
                250,
                details["probe_energy_slope_per_1000_steps"],
                6,
            ),
        )
        self.assertEqual(
            tuple(updated[f"segment_{index}_steps"] for index in range(1, 6)),
            tuple(float(value) for value in squeeze_segment_steps(rounded_steps)),
        )

    def test_augmented_later_slope_uses_probe_rms_bpe_and_production_accumulation(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=132,
                observed_rms=3.8782791e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.1028907100460046,
            ),
            1.178902644964147,
        )
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=5,
                probe_energy_slope=2.063401097961805,
            ),
            1.0073731063028069,
        )
        self.assertGreater(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.063401097961805,
            ),
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=4,
                probe_energy_slope=2.063401097961805,
            ),
        )

    def test_augmented_later_slope_extrapolates_out_of_range_features(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=99,
                observed_rms=3.8782791e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.1028907100460046,
            ),
            1.53531826733878,
        )
        self.assertAlmostEqual(
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=20,
                observed_rms=5.3288192e-5,
                gradient_accumulation_steps=20,
                probe_energy_slope=2.2019441770164145,
            ),
            6.952735504945071,
        )

    def test_double_weighted_ten_rosine_revisions_are_in_augmented_calibration(self):
        first_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=99,
            observed_rms=4.469153373812116e-5,
            gradient_accumulation_steps=8,
            probe_energy_slope=2.1870861288967656,
        )
        second_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=95,
            observed_rms=4.4754658783445186e-5,
            gradient_accumulation_steps=9,
            probe_energy_slope=2.216775240715489,
        )
        third_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=104,
            observed_rms=4.29566212128835e-5,
            gradient_accumulation_steps=7,
            probe_energy_slope=2.139091157515416,
        )
        low_accumulation_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=103,
            observed_rms=3.648383790277876e-5,
            gradient_accumulation_steps=2,
            probe_energy_slope=2.085742066127855,
        )
        fourth_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=110,
            observed_rms=4.176764188992321e-5,
            gradient_accumulation_steps=6,
            probe_energy_slope=2.1384678815539027,
        )
        second_low_accumulation_later_slope = (
            estimate_augmented_later_mean_energy_slope(
                dataset_batches_per_epoch=117,
                observed_rms=3.7124011e-5,
                gradient_accumulation_steps=2,
                probe_energy_slope=2.01871264275116,
            )
        )
        fifth_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=112,
            observed_rms=4.4364149e-5,
            gradient_accumulation_steps=9,
            probe_energy_slope=2.1487726165980137,
        )
        sixth_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=119,
            observed_rms=4.293570495580571e-5,
            gradient_accumulation_steps=6,
            probe_energy_slope=2.0936642515103694,
        )
        seventh_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=111,
            observed_rms=4.5584457060263114e-5,
            gradient_accumulation_steps=7,
            probe_energy_slope=2.1592672694926716,
        )
        ga10_later_slope = estimate_augmented_later_mean_energy_slope(
            dataset_batches_per_epoch=103,
            observed_rms=4.6760884669408374e-5,
            gradient_accumulation_steps=10,
            probe_energy_slope=2.166961419148628,
        )
        self.assertAlmostEqual(first_later_slope, 1.2125657661324991)
        self.assertAlmostEqual(second_later_slope, 1.2949596385369848)
        self.assertAlmostEqual(third_later_slope, 1.24090722930083)
        self.assertAlmostEqual(low_accumulation_later_slope, 0.9608762373729053)
        self.assertAlmostEqual(fourth_later_slope, 1.1414868001660226)
        self.assertAlmostEqual(second_low_accumulation_later_slope, 0.864022603473153)
        self.assertAlmostEqual(fifth_later_slope, 1.2058488409396024)
        self.assertAlmostEqual(sixth_later_slope, 1.048040711219461)
        self.assertAlmostEqual(seventh_later_slope, 0.9680133863904204)
        self.assertAlmostEqual(ga10_later_slope, 1.2171498779145546)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3400,
                probe_steps=500,
                observed_rms=4.4364149e-5,
                dataset_batches_per_epoch=112,
                probe_energy_slope=2.1487726165980137,
                gradient_accumulation_steps=9,
            ),
            8.550931987366397e-5,
            delta=1e-11,
        )
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=4200,
                probe_steps=500,
                observed_rms=4.293570495580571e-5,
                dataset_batches_per_epoch=119,
                probe_energy_slope=2.0936642515103694,
                gradient_accumulation_steps=6,
            ),
            8.733721166525736e-5,
            delta=1e-11,
        )
        fifth_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3400,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4364149e-5,
            probe_steps=500,
            dataset_batches_per_epoch=112,
            rms_curve=(),
            gradient_accumulation_steps=9,
            probe_energy_slope=2.1487726165980137,
        )
        newest_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4200,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.293570495580571e-5,
            probe_steps=500,
            dataset_batches_per_epoch=119,
            rms_curve=(),
            gradient_accumulation_steps=6,
            probe_energy_slope=2.0936642515103694,
        )
        seventh_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3600,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.5584457060263114e-5,
            probe_steps=500,
            dataset_batches_per_epoch=111,
            rms_curve=(),
            gradient_accumulation_steps=7,
            probe_energy_slope=2.1592672694926716,
        )
        ga10_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=2900,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.6760884669408374e-5,
            probe_steps=500,
            dataset_batches_per_epoch=103,
            rms_curve=(),
            gradient_accumulation_steps=10,
            probe_energy_slope=2.166961419148628,
        )
        self.assertEqual(fifth_estimated_steps, 3303)
        self.assertEqual(newest_estimated_steps, 3911)
        self.assertEqual(seventh_estimated_steps, 3665)
        self.assertEqual(ga10_estimated_steps, 2956)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3300,
                probe_steps=500,
                observed_rms=4.469153373812116e-5,
                dataset_batches_per_epoch=99,
                probe_energy_slope=2.1870861288967656,
                gradient_accumulation_steps=8,
            ),
            8.507222807420225e-5,
            delta=1e-11,
        )
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3000,
                probe_steps=500,
                observed_rms=4.4754658783445186e-5,
                dataset_batches_per_epoch=95,
                probe_energy_slope=2.216775240715489,
                gradient_accumulation_steps=9,
            ),
            8.322939170853608e-5,
            delta=1e-11,
        )
        first_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.469153373812116e-5,
            probe_steps=500,
            dataset_batches_per_epoch=99,
            rms_curve=(),
            gradient_accumulation_steps=8,
            probe_energy_slope=2.1870861288967656,
        )
        second_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4754658783445186e-5,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            gradient_accumulation_steps=9,
            probe_energy_slope=2.216775240715489,
        )
        self.assertEqual(first_estimated_steps, 3239)
        self.assertEqual(second_estimated_steps, 3073)

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

        self.assertAlmostEqual(mrissi_predicted, 8.0388463787054e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_predicted, 7.979113109175584e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_second_predicted, 8.485904185921547e-5, delta=1e-11)

    def test_gradient_accumulation_uses_biased_nearest_microbatch_budget(self):
        self.assertEqual(choose_gradient_accumulation_steps(3000, 24000, 6), 8)
        self.assertEqual(choose_gradient_accumulation_steps(3500, 24000, 6), 7)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(4500, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5000, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, None, 6), 6)

    def test_epoch_accumulation_groups_include_underfilled_final_update(self):
        self.assertEqual(epoch_accumulation_group_sizes(22, 18), (18, 4))
        self.assertEqual(epoch_accumulation_group_sizes(22, 5), (5, 5, 5, 5, 2))
        self.assertEqual(epoch_accumulation_group_sizes(22, 22), (22,))

    def test_epoch_aware_microbatch_count_tracks_optimizer_step_phase(self):
        self.assertEqual(count_optimizer_step_microbatches(1, 22, 18), 18)
        self.assertEqual(count_optimizer_step_microbatches(2, 22, 18), 22)
        self.assertEqual(count_optimizer_step_microbatches(3, 22, 18), 40)
        self.assertEqual(count_optimizer_step_microbatches(1350, 22, 18), 14850)
        self.assertEqual(optimizer_steps_for_microbatch_budget(14850, 22, 18), 1350)

    def test_automatic_accumulation_matches_actual_epoch_microbatches(self):
        self.assertEqual(
            choose_gradient_accumulation_steps(
                1350,
                24000,
                22,
                rounding_bias=0.7,
                dataset_batches_per_epoch=22,
            ),
            22,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                4000,
                24000,
                6,
                rounding_bias=0.7,
                dataset_batches_per_epoch=95,
            ),
            6,
        )
        self.assertNotEqual(95 % 6, 0)
        self.assertEqual(count_optimizer_step_microbatches(4000, 95, 6), 23750)
        marcia_ga18_microbatches = count_optimizer_step_microbatches(
            1351,
            22,
            18,
        )
        self.assertEqual(marcia_ga18_microbatches, 14868)
        self.assertEqual(
            choose_gradient_accumulation_steps(
                1351,
                marcia_ga18_microbatches,
                22,
                rounding_bias=0.7,
                dataset_batches_per_epoch=22,
            ),
            18,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                6500,
                None,
                6,
                dataset_batches_per_epoch=95,
            ),
            6,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                6500,
                None,
                30,
                dataset_batches_per_epoch=22,
            ),
            30,
        )

    def test_piecewise_plan_reapplies_microbatch_target_after_second_estimate(self):
        observed_rms = 4.4754658783445186e-5
        probe_energy_slope = 2.216775240715489
        later_mean_energy_slope = 1.168901924603102
        final_target_rms = predict_piecewise_energy_final_rms(
            total_steps=3018,
            probe_steps=500,
            observed_rms=observed_rms,
            dataset_batches_per_epoch=95,
            probe_energy_slope=probe_energy_slope,
            gradient_accumulation_steps=9,
            later_mean_energy_slope=later_mean_energy_slope,
        )

        (
            estimated_steps,
            _,
            adjusted_steps,
            uncapped_accumulation,
            adjusted_accumulation,
            details,
        ) = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=final_target_rms,
            observed_rms=observed_rms,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            current_gradient_accumulation_steps=6,
            probe_gradient_accumulation_steps=9,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=100,
            probe_energy_slope=probe_energy_slope,
            later_mean_energy_slope=later_mean_energy_slope,
            later_mean_energy_slope_reference_gradient_accumulation_steps=9,
        )

        self.assertEqual(estimated_steps, 3153)
        self.assertEqual(adjusted_steps, 3200)
        self.assertEqual(uncapped_accumulation, 8)
        self.assertEqual(adjusted_accumulation, 8)
        self.assertEqual(details["gradient_accumulation_is_epoch_compatible"], 0.0)
        self.assertEqual(details["actual_production_microbatches"], 25334.0)
        self.assertEqual(
            adjusted_accumulation,
            choose_gradient_accumulation_steps(
                adjusted_steps,
                24000,
                6,
                rounding_bias=0.7,
                dataset_batches_per_epoch=95,
            ),
        )
        self.assertEqual(
            details["later_mean_energy_slope_gradient_accumulation_steps"],
            float(adjusted_accumulation),
        )
        self.assertAlmostEqual(
            details["observed_later_mean_energy_slope_accumulation_scale"],
            0.9827617371150374,
        )

    def test_marcia_probe_two_transfers_probe_one_rank36_anchor(self):
        probe_one_curve = (
            (20, 8.384812746583882e-6),
            (40, 1.2532975534542193e-5),
            (60, 1.6333030095169823e-5),
            (80, 2.0194843333880055e-5),
            (100, 2.406638620843219e-5),
            (120, 2.8065770467594773e-5),
            (140, 3.2178582850411865e-5),
            (160, 3.591705169945463e-5),
            (180, 3.9572608661434296e-5),
            (200, 4.328498669908549e-5),
        )
        probe_two_curve = (
            (20, 8.431589772589938e-6),
            (40, 1.2581572540427063e-5),
            (60, 1.635862070439193e-5),
            (80, 1.8600551448695187e-5),
            (100, 1.9645133859058497e-5),
            (120, 2.063964607003356e-5),
            (140, 2.1107788438254896e-5),
            (160, 2.2122571828044655e-5),
            (180, 2.308801673276316e-5),
            (200, 2.406842729930015e-5),
        )
        primary_reference_rms, primary_slope, _ = (
            fit_schedule_aware_probe_energy_model(
                probe_one_curve,
                total_steps=4000,
                probe_steps=500,
                curve_interval=20,
            )
        )
        adjusted_reference_rms, adjusted_slope, _ = (
            fit_schedule_aware_probe_energy_model(
                probe_two_curve,
                total_steps=350,
                probe_steps=500,
                curve_interval=20,
            )
        )
        old_observed_later_slope, _ = fit_observed_later_mean_energy_slope(
            probe_two_curve,
            total_steps=350,
            probe_steps=200,
            reference_rms=adjusted_reference_rms,
            curve_interval=20,
        )
        transferred_reference_rms, transferred_slope, transfer_details = (
            fit_adjusted_probe_rank36_transfer(
                probe_one_curve,
                probe_two_curve,
                adjusted_total_steps=350,
                adjusted_probe_steps=200,
                primary_reference_rms=primary_reference_rms,
                primary_probe_energy_slope=primary_slope,
            )
        )
        observed_later_slope, _ = fit_observed_later_mean_energy_slope(
            probe_two_curve,
            total_steps=350,
            probe_steps=200,
            reference_rms=transferred_reference_rms,
            curve_interval=20,
        )

        second_only_plan = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=adjusted_reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=probe_two_curve,
            current_gradient_accumulation_steps=22,
            probe_gradient_accumulation_steps=22,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=50,
            probe_energy_slope=adjusted_slope,
            later_mean_energy_slope=old_observed_later_slope,
            later_mean_energy_slope_reference_gradient_accumulation_steps=22,
        )
        anchored_plan = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=transferred_reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=probe_one_curve,
            current_gradient_accumulation_steps=22,
            probe_gradient_accumulation_steps=22,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=50,
            probe_energy_slope=transferred_slope,
            later_mean_energy_slope=observed_later_slope,
            later_mean_energy_slope_reference_gradient_accumulation_steps=22,
        )

        self.assertGreater(second_only_plan[0], 3000)
        self.assertAlmostEqual(
            transfer_details["rank36_transfer_rms_scale"],
            1.0023300289993526,
        )
        self.assertEqual(transfer_details["rank36_transfer_sample_count"], 3.0)
        self.assertAlmostEqual(observed_later_slope, 0.4108007707940888)
        self.assertEqual(anchored_plan[0], 2744)
        self.assertEqual(anchored_plan[2], 2750)
        self.assertEqual(anchored_plan[4], 11)
        self.assertEqual(
            anchored_plan[5]["gradient_accumulation_is_epoch_compatible"],
            1.0,
        )

    def test_rosine_second_plan_no_longer_locks_probe_two_accumulation(self):
        (
            estimated_steps,
            _,
            adjusted_steps,
            uncapped_accumulation,
            adjusted_accumulation,
            _,
        ) = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4754658783445186e-5,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            current_gradient_accumulation_steps=6,
            probe_gradient_accumulation_steps=9,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=100,
            probe_energy_slope=2.216775240715489,
        )

        self.assertEqual(estimated_steps, 3378)
        self.assertEqual(adjusted_steps, 3400)
        self.assertEqual(uncapped_accumulation, 7)
        self.assertEqual(adjusted_accumulation, 7)

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

    def test_forced_adjusted_probe_requires_piecewise_policy(self):
        with self.assertRaisesRegex(ValueError, "requires.*piecewise_energy_v1"):
            validate_rms_probe_configuration(
                make_args(rms_probe_force_adjusted_probe=True)
            )
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_force_adjusted_probe=True,
                )
            )
        )

    def test_piecewise_policy_accepts_supported_segment_ratios(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_first_segment_ratio=2.0,
                    lora_squeeze_final_segment_ratio=0.75,
                )
            )
        )

    def test_piecewise_policy_uses_actual_ratio_adjusted_first_squeeze(self):
        with self.assertRaisesRegex(ValueError, "first squeeze"):
            validate_rms_probe_configuration(
                make_args(
                    max_train_steps=4000,
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_first_segment_ratio=0.5,
                )
            )

    def test_piecewise_policy_requires_enough_rank36_curve_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two RMS curve samples"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_steps=100,
                    rms_probe_curve_every_n_steps=50,
                )
            )

    def test_piecewise_policy_accepts_non_500_non_interval_aligned_probe(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_steps=150,
                    rms_probe_curve_every_n_steps=20,
                )
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

    def test_second_probe_accepts_production_shorter_than_probe1(self):
        probe_args = build_rms_probe_args(
            make_args(
                rms_probe_scaling_policy="piecewise_energy_v1",
                rms_probe_final_target=8e-5,
            ),
            probe_index=2,
            production_steps=400,
            gradient_accumulation_steps=20,
            training_step_limit=180,
        )

        self.assertEqual(probe_args.max_train_steps, 400)
        self.assertEqual(probe_args._training_step_limit, 180)

if __name__ == "__main__":
    unittest.main()
