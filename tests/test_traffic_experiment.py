"""Fast regression tests for the reproducible experiment contract."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from traffic_experiment import (
    CONTEXT_LEN,
    HORIZON,
    DSSSoftmaxForecaster,
    LSTMForecaster,
    compute_metrics,
    count_parameters,
    portable_path,
    prepare_data,
)


class ExperimentContractTests(unittest.TestCase):
    def test_data_split_and_train_only_scaler(self) -> None:
        prepared = prepare_data(batch_size=64)
        self.assertEqual(prepared.raw_counts, {"train": 21168, "validation": 3024, "test": 6048})
        self.assertEqual(prepared.available_sample_counts, {"train": 21144, "validation": 3024, "test": 6048})
        self.assertEqual(prepared.context_len, CONTEXT_LEN)
        self.assertEqual(prepared.horizon, HORIZON)
        self.assertAlmostEqual(float(prepared.scaler.data_min_[0]), float(prepared.raw_splits["train"].min()), places=5)
        self.assertAlmostEqual(float(prepared.scaler.data_max_[0]), float(prepared.raw_splits["train"].max()), places=5)

    def test_windows_are_past_only(self) -> None:
        prepared = prepare_data(batch_size=64)
        dataset = prepared.datasets["test"]
        first_index = int(dataset.target_start_indices[0].item())
        context, target = dataset[0]
        self.assertEqual(tuple(context.shape), (CONTEXT_LEN, 1))
        self.assertEqual(tuple(target.shape), (HORIZON, 1))
        expected_raw = np.asarray(
            prepared.raw_values[first_index - CONTEXT_LEN : first_index],
            dtype=np.float32,
        ).reshape(-1, 1)
        expected_scaled = prepared.scaler.transform(expected_raw).astype(np.float32)
        expected_context = torch.from_numpy(expected_scaled)
        self.assertTrue(torch.allclose(context, expected_context))
        self.assertEqual(int(first_index), int(dataset.target_start))

    def test_model_shapes_and_parameter_counts(self) -> None:
        values = torch.randn(3, CONTEXT_LEN, 1)
        for model in (LSTMForecaster(), DSSSoftmaxForecaster()):
            output = model(values)
            self.assertEqual(tuple(output.shape), (3, HORIZON))
            self.assertGreater(count_parameters(model), 0)
            self.assertTrue(torch.isfinite(output).all())

    def test_metadata_paths_are_portable(self) -> None:
        self.assertEqual(
            portable_path("METR_LA_with_Weather_5min.csv"),
            "METR_LA_with_Weather_5min.csv",
        )

    def test_metric_units(self) -> None:
        metrics = compute_metrics([10.0, 20.0], [12.0, 18.0])
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["mape"], 15.0)


if __name__ == "__main__":
    unittest.main()
