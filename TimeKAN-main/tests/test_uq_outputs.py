"""Exercise test() exports and real plotting/MC aggregation without training a model."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp import exp_long_term_forecasting as forecasting


class TestUQOutputs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.truth = np.array([0.96, 0.90, 0.82, 0.74])
        self.median = np.array([0.95, 0.88, 0.80, 0.75])
        self.lower = np.array([0.91, 0.91, 0.77, 0.71])
        self.upper = np.array([0.99, 0.95, 0.86, 0.78])

    def make_experiment(self, loss_type, horizon=1):
        # Keep the CLI's descending head order and cross only some intervals.
        crossed = np.array([True, False, True, False])
        raw_lower = np.where(crossed, self.upper, self.lower)
        raw_upper = np.where(crossed, self.lower, self.upper)
        predictions = np.stack([raw_upper, self.median, raw_lower], axis=-1)
        predictions = predictions.reshape(-1, horizon, 3)
        truth = self.truth.reshape(-1, horizon, 1)

        exp = forecasting.Exp_Long_Term_Forecast.__new__(forecasting.Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            project_root=str(self.root), loss_type=loss_type, pred_len=horizon,
            multi_step_stride=horizon, prediction_target='absolute',
            eval_last_step_only=False, mc_samples=3, mc_alpha=0.1,
            save_input_impact=False,
        )
        exp.quantiles = np.array([0.95, 0.5, 0.05])
        exp.q_median_idx = 1
        exp.model = torch.nn.Identity()
        exp._get_data = Mock(return_value=(object(), object()))

        # Supply deterministic draws while exercising the real MC aggregation.
        # Asymmetric draws distinguish the MC median from both the mean and q0.5.
        mc_draws = iter([-0.04, 0.01, 0.09] * 2)

        def collect_predictions(*_):
            if exp.model.training:
                return predictions + next(mc_draws), truth.copy()
            return predictions.copy(), truth.copy()

        exp._collect_test_predictions = Mock(side_effect=collect_predictions)
        return exp

    def run_outputs(self, loss_type, horizon):
        exp = self.make_experiment(loss_type, horizon)
        setting = f'{loss_type}_h{horizon}'
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(exp, '_mc_interval_from_dropout', wraps=exp._mc_interval_from_dropout) as mc, \
                patch.object(forecasting, 'visual_with_interval', wraps=forecasting.visual_with_interval) as interval_plot, \
                patch.object(forecasting, 'visual', wraps=forecasting.visual) as point_plot, \
                patch.object(forecasting.plt, 'fill_between', wraps=forecasting.plt.fill_between) as fill:
            metrics = exp.test(setting, save_dwt_plot=False)

        self.assertFalse(exp.model.training)
        folder = self.root / 'results' / setting
        csvs = {path.stem: pd.read_csv(path) for path in folder.glob('*.csv')}
        summary = csvs['uq_summary'].set_index('method')
        combined = csvs['prediction_vs_truth']
        np.testing.assert_array_equal(summary['horizon'], [horizon, horizon])

        # Every dedicated CSV must match the flattened combined CSV.
        for column in combined:
            values = csvs[column].to_numpy()
            self.assertEqual(values.shape, (4 // horizon, horizon))
            np.testing.assert_allclose(values.ravel(), combined[column], equal_nan=True)
        np.testing.assert_allclose(combined['true'], self.truth)
        np.testing.assert_allclose(combined['pred_quantile_median'], self.median)
        np.testing.assert_allclose(csvs['pred'], csvs['pred_quantile_median'])
        for key, value in metrics.items():
            self.assertAlmostEqual(csvs['metrics'].loc[0, key], value)
            self.assertAlmostEqual(summary.loc['Quantile', key], value)

        # Exact expectations for the three deterministic MC draws (alpha=0.1).
        np.testing.assert_allclose(combined['pred_mc_lower'], self.median - 0.035)
        np.testing.assert_allclose(combined['pred_mc_median'], self.median + 0.01)
        np.testing.assert_allclose(combined['pred_mc_upper'], self.median + 0.082)
        self.assert_interval_metrics(summary.loc['MC Dropout'], combined, 'pred_mc')
        self.assertAlmostEqual(summary.loc['MC Dropout', 'picp'], 1.0)
        self.assertAlmostEqual(summary.loc['MC Dropout', 'mpiw'], 0.117)

        # Invoke the real plotting functions and verify the actual shaded data.
        interval_plot.assert_called_once()
        point_plot.assert_called_once()
        fill.assert_called_once()
        plotted_true, plotted_median, plotted_lower, plotted_upper, interval_path = interval_plot.call_args.args
        np.testing.assert_allclose(plotted_true, combined['true'])
        np.testing.assert_allclose(plotted_median, csvs['pred'].to_numpy().ravel())
        np.testing.assert_allclose(point_plot.call_args.args[0], plotted_true)
        np.testing.assert_allclose(point_plot.call_args.args[1], plotted_median)
        source = 'pred_quantile' if loss_type == 'pinball' else 'pred_mc'
        np.testing.assert_allclose(plotted_lower, combined[source + '_lower'])
        np.testing.assert_allclose(plotted_upper, combined[source + '_upper'])
        np.testing.assert_array_equal(fill.call_args.args[0], np.arange(len(self.truth)))
        np.testing.assert_allclose(fill.call_args.args[1], plotted_lower)
        np.testing.assert_allclose(fill.call_args.args[2], plotted_upper)
        for path in (interval_path, point_plot.call_args.args[2]):
            self.assertEqual(Path(path).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

        return exp, mc, csvs, summary, combined

    def assert_interval_metrics(self, row, combined, prefix):
        truth = combined['true'].to_numpy()
        lower = combined[prefix + '_lower'].to_numpy()
        upper = combined[prefix + '_upper'].to_numpy()
        self.assertTrue(np.all(lower <= upper))
        width = np.mean(upper - lower)
        coverage = np.mean((lower <= truth) & (truth <= upper))
        np.testing.assert_allclose(
            row[['picp', 'pinaw', 'mpiw']].to_numpy(dtype=float),
            [coverage, width / np.ptp(truth), width],
        )

    def assert_single_mc_pass(self, exp, mc):
        mc.assert_called_once_with(*exp._get_data.return_value)
        self.assertEqual(exp._collect_test_predictions.call_count, 1 + exp.args.mc_samples)

    def test_mse_exports_nan_quantile_bounds_and_plots_mc_interval(self):
        for horizon in (1, 2):
            with self.subTest(horizon=horizon):
                exp, mc, csvs, summary, combined = self.run_outputs('mse', horizon)
                for bound in ('lower', 'upper'):
                    name = 'pred_quantile_' + bound
                    self.assertTrue(csvs[name].isna().to_numpy().all())
                    self.assertTrue(combined[name].isna().all())
                self.assertTrue(summary.loc['Quantile', ['picp', 'pinaw', 'mpiw']].isna().all())
                self.assert_single_mc_pass(exp, mc)

    def test_pinball_preserves_sorted_bounds_in_csv_metrics_and_plot(self):
        for horizon in (1, 2):
            with self.subTest(horizon=horizon):
                exp, mc, _, summary, combined = self.run_outputs('pinball', horizon)
                np.testing.assert_allclose(combined['pred_quantile_lower'], self.lower)
                np.testing.assert_allclose(combined['pred_quantile_upper'], self.upper)
                self.assert_interval_metrics(summary.loc['Quantile'], combined, 'pred_quantile')
                self.assertAlmostEqual(summary.loc['Quantile', 'picp'], 0.75)
                self.assertAlmostEqual(summary.loc['Quantile', 'mpiw'], 0.07)
                self.assert_single_mc_pass(exp, mc)

    def test_mc_runs_once_even_without_saving_outputs(self):
        for loss_type in ('mse', 'pinball'):
            with self.subTest(loss_type=loss_type):
                exp = self.make_experiment(loss_type)
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(exp, '_mc_interval_from_dropout', wraps=exp._mc_interval_from_dropout) as mc:
                    exp.test(loss_type, save_outputs=False)
                self.assert_single_mc_pass(exp, mc)
                self.assertFalse(exp.model.training)
                self.assertFalse((self.root / 'results').exists())

    def test_mc_bound_sorting_is_preserved(self):
        for loss_type in ('mse', 'pinball'):
            with self.subTest(loss_type=loss_type):
                exp = self.make_experiment(loss_type)
                # A defensive sorting case that normal np.quantile draws cannot produce.
                mc_result = tuple(a.reshape(-1, 1, 1) for a in
                                  (self.median, self.upper, self.lower, self.truth))
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(exp, '_mc_interval_from_dropout', return_value=mc_result) as mc, \
                        patch.object(forecasting, 'visual_with_interval') as interval_plot, \
                        patch.object(forecasting, 'visual'):
                    exp.test(loss_type, save_dwt_plot=False)
                mc.assert_called_once()
                folder = self.root / 'results' / loss_type
                combined = pd.read_csv(folder / 'prediction_vs_truth.csv')
                summary = pd.read_csv(folder / 'uq_summary.csv').set_index('method')
                np.testing.assert_allclose(combined['pred_mc_lower'], self.lower)
                np.testing.assert_allclose(combined['pred_mc_upper'], self.upper)
                self.assert_interval_metrics(summary.loc['MC Dropout'], combined, 'pred_mc')
                if loss_type == 'mse':
                    np.testing.assert_allclose(interval_plot.call_args.args[2], self.lower)
                    np.testing.assert_allclose(interval_plot.call_args.args[3], self.upper)


if __name__ == '__main__':
    unittest.main()
