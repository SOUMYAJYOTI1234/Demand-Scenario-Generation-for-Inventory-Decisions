"""Tests for the Phase 4 data pipeline on the synthetic M5-shaped fixture.

The dangerous failures here are silent ones (design doc §8): a leaky split or
a full-dataset scaler produces plausible wrong numbers. These tests pin the
guarantees the evaluation depends on.
"""

import numpy as np
import pandas as pd
import pytest

from demand_vae.data.pipeline import (
    ContextScaler,
    aggregate_weekly,
    build_windows,
    continuous_feature_names,
    temporal_split,
)
from demand_vae.data.validation import assert_no_leakage, check_schema, zero_fraction_report

N_WEEKS = 28  # must match tests/conftest.py


@pytest.fixture()
def panel(synthetic_raw, synthetic_config):
    return aggregate_weekly(synthetic_raw, synthetic_config)


@pytest.fixture()
def pipeline_output(panel, synthetic_config):
    """(splits, scaler, unscaled train context) - the full pipeline, staged."""
    spans = temporal_split(panel, synthetic_config)
    splits = build_windows(panel, synthetic_config, spans)
    unscaled_train = splits["train"].context_cont.copy()
    scaler = ContextScaler().fit(splits["train"].context_cont)
    for split in splits.values():
        split.context_cont = scaler.transform(split.context_cont)
    return splits, scaler, unscaled_train


class TestAggregation:
    def test_schema_check_passes(self, synthetic_raw):
        check_schema(synthetic_raw)  # must not raise

    def test_category_filter_and_series_drop(self, panel):
        # 4 FOODS series minus FOODS_1_003 (no train-period price) = 3 kept;
        # the HOBBIES series must never enter.
        assert panel.n_series == 3
        assert panel.n_dropped_series == 1
        assert not any("HOBBIES" in s for s in panel.series_ids)
        assert not any("FOODS_1_003" in s for s in panel.series_ids)

    def test_weekly_sums_match_daily(self, panel):
        # FOODS_1_001 @ CA_1 sells exactly 1/day -> every weekly sum is 7.
        row = list(panel.series_ids).index("FOODS_1_001_CA_1_evaluation")
        assert panel.n_weeks == N_WEEKS
        np.testing.assert_array_equal(panel.sales[row], np.full(N_WEEKS, 7))

    def test_availability_tracks_prices(self, panel):
        row = list(panel.series_ids).index("FOODS_1_002_CA_1_evaluation")
        assert not panel.available[row, :4].any()  # pre-launch
        assert panel.available[row, 4:].all()

    def test_zero_fraction_report_shape(self, panel):
        report = zero_fraction_report(panel)
        assert len(report) == panel.n_series
        assert {"zero_frac", "mean", "var", "dispersion"} <= set(report.columns)
        assert report["zero_frac"].between(0, 1).all()


class TestRelativePrice:
    def test_relative_price_always_positive(self, panel):
        rel = panel.rel_price[panel.available]
        assert np.isfinite(rel).all()
        assert (rel > 0).all()

    def test_relative_price_is_train_normalized(self, panel):
        # FOODS_1_001 has a constant price, so its relative price is exactly 1.
        row = list(panel.series_ids).index("FOODS_1_001_CA_1_evaluation")
        np.testing.assert_allclose(panel.rel_price[row], 1.0)
        # FOODS_1_002: train weeks 5-12 all at 4.0 -> rel 1.0; the val-period
        # cut to 3.0 must be measured against the TRAIN mean -> 0.75.
        row2 = list(panel.series_ids).index("FOODS_1_002_CA_1_evaluation")
        np.testing.assert_allclose(panel.rel_price[row2, 4:12], 1.0)
        np.testing.assert_allclose(panel.rel_price[row2, 12:], 0.75)


class TestWindowing:
    def test_shapes_match_design(self, pipeline_output, synthetic_config):
        splits, _, _ = pipeline_output
        horizon = synthetic_config.data.horizon_weeks
        n_lags = synthetic_config.data.n_lags
        expected_cont_dim = len(continuous_feature_names(n_lags))  # 5 + L
        for split in splits.values():
            n = len(split)
            assert split.x.shape == (n, horizon)  # H = 4
            assert split.x.dtype == np.int32
            assert split.context_cat.shape == (n, 2)  # item_idx, store_idx
            assert split.context_cont.shape == (n, expected_cont_dim)
            assert np.isfinite(split.context_cont).all()

    def test_window_counts(self, pipeline_output):
        splits, _, _ = pipeline_output
        # Fully available series: 5 train starts each (weeks 1-12, 8-week
        # windows); FOODS_1_002 launches week 5 -> exactly 1 train window.
        assert len(splits["train"]) == 5 + 5 + 1
        # Val and test spans are exactly 8 weeks -> 1 window per series.
        assert len(splits["val"]) == 3
        assert len(splits["test"]) == 3

    def test_pre_launch_windows_dropped(self, pipeline_output, panel):
        splits, _, _ = pipeline_output
        row = list(panel.series_ids).index("FOODS_1_002_CA_1_evaluation")
        train_mask = splits["train"].series_row == row
        assert train_mask.sum() == 1
        # Its only train window must start at week 5 (index 4): lags 5-8, targets 9-12.
        assert splits["train"].first_target_week[train_mask].item() == 8


class TestTemporalSplitNoLeakage:
    def test_assert_no_leakage_passes(self, pipeline_output, panel, synthetic_config):
        splits, scaler, unscaled_train = pipeline_output
        assert_no_leakage(splits, scaler, panel, synthetic_config, unscaled_train)

    def test_window_dates_respect_boundaries(self, pipeline_output, panel, synthetic_config):
        splits, _, _ = pipeline_output
        n_lags = synthetic_config.data.n_lags
        horizon = synthetic_config.data.horizon_weeks
        train_end = pd.Timestamp(synthetic_config.data.split.train_end)
        val_end = pd.Timestamp(synthetic_config.data.split.val_end)

        def span(split):
            first = panel.week_start[split.first_target_week - n_lags]
            last = panel.week_start[split.first_target_week + horizon - 1] + pd.Timedelta(days=6)
            return first, last

        _, train_last = span(splits["train"])
        assert (train_last <= train_end).all()  # train never sees post-boundary data
        val_first, val_last = span(splits["val"])
        assert (val_first > train_end).all() and (val_last <= val_end).all()
        test_first, _ = span(splits["test"])
        assert (test_first > val_end).all()

    def test_no_window_straddles_boundary(self, panel, synthetic_config):
        spans = temporal_split(panel, synthetic_config)
        splits = build_windows(panel, synthetic_config, spans)
        # Between consecutive splits there must be a gap of L+H-1 start
        # positions with no windows: every 8-week window fits one span.
        all_starts = np.concatenate(
            [s.first_target_week - synthetic_config.data.n_lags for s in splits.values()]
        )
        span_bounds = {name: rng for name, rng in spans.items()}
        for name, split in splits.items():
            starts = split.first_target_week - synthetic_config.data.n_lags
            ends = split.first_target_week + synthetic_config.data.horizon_weeks - 1
            lo, hi = span_bounds[name]
            assert (starts >= lo).all() and (ends <= hi).all()
        assert len(all_starts) == sum(len(s) for s in splits.values())


class TestScalerTrainOnly:
    def test_scaler_stats_come_from_train(self, pipeline_output):
        splits, scaler, unscaled_train = pipeline_output
        assert scaler.n_fit_samples_ == len(splits["train"])
        np.testing.assert_allclose(scaler.mean_, unscaled_train[:, 2:].mean(axis=0), atol=1e-6)
        expected_std = unscaled_train[:, 2:].std(axis=0)
        expected_std = np.where(expected_std < 1e-8, 1.0, expected_std)
        np.testing.assert_allclose(scaler.std_, expected_std, atol=1e-6)

    def test_sin_cos_pass_through_unscaled(self, pipeline_output):
        splits, _, _ = pipeline_output
        for split in splits.values():
            assert (np.abs(split.context_cont[:, :2]) <= 1.0 + 1e-6).all()

    def test_leakage_detector_catches_full_dataset_fit(
        self, panel, synthetic_config, pipeline_output
    ):
        # Guard the guard: a scaler fit on train+val+test must be rejected.
        spans = temporal_split(panel, synthetic_config)
        splits = build_windows(panel, synthetic_config, spans)
        everything = np.concatenate([s.context_cont for s in splits.values()])
        bad_scaler = ContextScaler().fit(everything)
        for split in splits.values():
            split.context_cont = bad_scaler.transform(split.context_cont)
        with pytest.raises(AssertionError):
            assert_no_leakage(splits, bad_scaler, panel, synthetic_config)

    def test_transform_requires_fit(self):
        with pytest.raises(RuntimeError):
            ContextScaler().transform(np.zeros((3, 9), dtype=np.float32))
