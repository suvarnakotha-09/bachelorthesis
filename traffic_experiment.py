"""Reproducible one-step traffic-speed forecasting experiment.

This module contains a small, self-contained experiment for the repository CSV.
The primary comparison is deliberately univariate: only ``traffic_speed`` is
read and supplied to the models.  The weather columns in the CSV are not model
features.

Protocol
--------
The raw target is split chronologically into 70% train, 10% validation, and
20% test partitions.  A :class:`~sklearn.preprocessing.MinMaxScaler` is fit
only on the *raw* training target.  Validation and test values are transformed
with that already-fit scaler.  A context contains 24 five-minute observations
and the target is the next observation (``horizon=1``). Target indices remain
inside their assigned partition. A validation/test context may use the
immediately preceding, already-observed rows (never future rows), which
preserves the full chronological target set without leakage.

The two trainable comparators are an LSTM and a DSS-softmax traffic wrapper.
The wrapper follows the canonical diagonal state-space recurrence with complex
HiPPO-D initialization, learned positive timescales, length-normalized ZOH
coefficients, feedthrough, and two DSS-softmax blocks.  The original custom
gated recurrence remains available under the explicit ``DSS-inspired`` aliases
for a reproducible ablation, but it is not the primary model in the comparison.
Supervised models consume the MinMax-scaled target; optional TimesFM receives
raw traffic-speed contexts and uses its own documented normalization.  TimesFM
2.5 PyTorch is imported only when explicitly enabled.

The module does not depend on the process working directory.  By default the
CSV and the ``outputs`` directory are resolved relative to this file's
repository directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Public protocol constants.  Keep these visible so a reader can audit the
# experiment without having to infer defaults from the training loop.
SEED = 42
DATA_FILENAME = "METR_LA_with_Weather_5min.csv"
TARGET_COLUMN = "traffic_speed"
CONTEXT_LEN = 24
HORIZON = 1
BATCH_SIZE = 64
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.20
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5
DEFAULT_LEARNING_RATE = 1e-3
FAST_EPOCHS = 1
FAST_MAX_SAMPLES = 256
TIMESFM_BATCH_SIZE = 32
TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
# TimesFM is deliberately disabled unless the caller/CLI opts in.  Keeping the
# default explicit makes it safe to import and test this module without a
# checkpoint or a model download.
RUN_TIMESFM = False


# ---------------------------------------------------------------------------
# Paths, reproducibility, and metadata
# ---------------------------------------------------------------------------


def module_directory() -> Path:
    """Return the directory containing this module.

    The function intentionally uses ``__file__`` rather than the current
    working directory, which may be a notebook directory or an arbitrary shell
    directory.
    """

    return Path(__file__).resolve().parent


def portable_path(path: str | os.PathLike[str]) -> str:
    """Return a repository-relative path when possible for portable metadata."""

    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(module_directory()).as_posix()
    except ValueError:
        return str(candidate)


def resolve_repo_path(path: str | os.PathLike[str] | None, *, default: Path) -> Path:
    """Resolve a relative path against the module/repository, not the cwd.

    For a relative path, both the module directory (the ``bachelorthesis``
    directory) and its parent (the repository root) are sensible bases.  The
    first existing candidate is used.  Absolute paths are returned unchanged.
    """

    if path is None:
        candidate = Path(default).expanduser()
        if not candidate.is_absolute():
            candidate = module_directory() / candidate
        return candidate.resolve()

    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested.resolve()

    bases = (module_directory(), module_directory().parent)
    candidates = [base / requested for base in bases]
    # Avoid returning a surprising duplicate when requested already starts with
    # the repository directory name.
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # The first candidate gives a useful, deterministic error message when the
    # file does not exist.
    return candidates[0].resolve()


def resolve_data_path(data_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the experiment CSV path relative to this module/repository."""

    return resolve_repo_path(data_path, default=Path(DATA_FILENAME))


def resolve_output_dir(output_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and create the output directory relative to the repository."""

    path = resolve_repo_path(output_dir, default=Path("outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and Torch (including CUDA when present).

    ``warn_only=True`` keeps CPU execution usable on backends for which an
    operation has no deterministic implementation while still requesting
    deterministic kernels wherever possible.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on host hardware
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older torch compatibility
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # A backend may reject deterministic mode after initialization.  The
        # experiment remains usable; the selected mode is recorded below.
        pass


def select_device(requested: str | torch.device | None = None) -> torch.device:
    """Select and validate the torch device.

    ``None`` or ``"auto"`` selects CUDA when available and CPU otherwise.  An
    explicit CPU request remains CPU even on a CUDA-capable host.
    """

    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA work so perf-counter timings include completed kernels."""

    if device.type == "cuda" and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.synchronize(device)


def _package_version(package_name: str) -> str | None:
    """Return installed package metadata without importing the package."""

    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive for broken metadata
        return None


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values to JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def version_hardware_metadata(device: torch.device) -> dict[str, Any]:
    """Collect software versions and selected hardware for reproducibility."""

    memory_bytes: int | None = None
    physical_cpu_count: int | None = None
    try:
        import psutil  # lazy: model/data tests do not need this dependency

        memory_bytes = int(psutil.virtual_memory().total)
        physical_cpu_count = psutil.cpu_count(logical=False)
    except Exception:
        memory_bytes = None
        physical_cpu_count = None

    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": [properties.major, properties.minor],
                }
            )

    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_cpu_count,
        "system_memory_bytes": memory_bytes,
        "device": str(device),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "cuda_devices": cuda_devices,
        "versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "matplotlib": _package_version("matplotlib"),
            "psutil": _package_version("psutil"),
            "timesfm": _package_version("timesfm"),
        },
    }


# ---------------------------------------------------------------------------
# Data loading, chronological splitting, scaling, and sequence windows
# ---------------------------------------------------------------------------


@dataclass
class TrafficSeries:
    """The single target series and optional timestamps loaded from the CSV."""

    values: np.ndarray
    timestamps: pd.Series | None
    path: Path
    timestamp_column: str | None


def _choose_timestamp_column(columns: Sequence[str]) -> str | None:
    """Choose a timestamp-like column without loading weather columns."""

    preferred = (
        "Unnamed: 0",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "date",
        "Date",
    )
    for name in preferred:
        if name in columns:
            return name
    return None


def load_traffic_series(data_path: str | os.PathLike[str] | None = None) -> TrafficSeries:
    """Load and chronologically order only ``traffic_speed`` from the CSV.

    The timestamp column is used only for ordering.  Weather columns are never
    selected by ``usecols`` and therefore cannot accidentally enter the primary
    comparison.  The returned values are finite ``float32`` observations in
    timestamp order (or original row order if no parseable timestamp exists).
    """

    path = resolve_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic data CSV was not found: {path}")

    header = pd.read_csv(path, nrows=0)
    if TARGET_COLUMN not in header.columns:
        raise ValueError(
            f"Expected target column {TARGET_COLUMN!r} in {path}; "
            f"found {list(header.columns)!r}"
        )
    timestamp_column = _choose_timestamp_column(list(header.columns))
    usecols = [TARGET_COLUMN] + ([timestamp_column] if timestamp_column else [])
    frame = pd.read_csv(path, usecols=usecols)
    numeric_target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if not np.isfinite(numeric_target.to_numpy(dtype=np.float64)).all():
        bad = int((~np.isfinite(numeric_target.to_numpy(dtype=np.float64))).sum())
        raise ValueError(f"Target contains {bad} missing or non-finite values")

    timestamps: pd.Series | None = None
    order = np.arange(len(frame), dtype=np.int64)
    if timestamp_column is not None:
        parsed = pd.to_datetime(frame[timestamp_column], errors="coerce", dayfirst=True)
        if parsed.notna().all():
            timestamps = parsed.reset_index(drop=True)
            # Stable sorting makes duplicate timestamps deterministic.
            order = np.argsort(parsed.to_numpy(), kind="stable")

    values = numeric_target.to_numpy(dtype=np.float32)[order]
    if timestamps is not None:
        timestamps = timestamps.iloc[order].reset_index(drop=True)
    return TrafficSeries(
        values=np.ascontiguousarray(values),
        timestamps=timestamps,
        path=path,
        timestamp_column=timestamp_column,
    )


def split_indices(
    n_rows: int,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, tuple[int, int]]:
    """Return non-overlapping chronological half-open split boundaries.

    Integer floor boundaries make the counts deterministic and leave any
    remainder in the final test partition.  For the repository CSV (30,240
    rows), the boundaries are exactly 21,168 / 24,192 / 30,240.
    """

    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any((not np.isfinite(float(fraction))) or fraction <= 0 for fraction in fractions):
        raise ValueError("split fractions must be finite and positive")
    if not np.isclose(sum(float(fraction) for fraction in fractions), 1.0):
        raise ValueError("train + validation + test fractions must sum to 1")

    train_end = int(n_rows * float(train_fraction))
    validation_end = train_end + int(n_rows * float(validation_fraction))
    validation_end = min(validation_end, n_rows)
    train_end = min(train_end, validation_end)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n_rows),
    }


def chronological_split(
    values: Sequence[float] | np.ndarray,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, np.ndarray]:
    """Split a one-dimensional sequence chronologically without shuffling."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("chronological_split expects a one-dimensional sequence")
    boundaries = split_indices(
        len(array), train_fraction, validation_fraction, test_fraction
    )
    return {
        name: np.ascontiguousarray(array[start:end])
        for name, (start, end) in boundaries.items()
    }


class SequenceDataset(Dataset):
    """Chronological context/target windows with explicit target boundaries.

    The dataset stores an available-data range ``[start, end)``.  A sample whose
    target starts at global index ``i`` has context ``values[i-context_len:i]``
    and target ``values[i:i+horizon]``.  By default target starts are
    ``start + context_len`` through ``end - horizon`` (inclusive).  Optional
    ``target_start`` and half-open ``target_end`` arguments let validation and
    test targets begin exactly at a chronological split boundary while their
    first contexts use only already-observed rows from the preceding split.
    This is past-only context, not target leakage; no context is ever taken
    from after a target.

    Shapes returned by ``__getitem__`` are ``(context_len, input_size)`` and
    ``(horizon, output_size)``.  In the primary experiment both sizes are one.
    ``max_samples`` limits the number of generated windows without changing the
    raw split or the scaler fit; it is used by fast smoke tests.
    """

    def __init__(
        self,
        values: Sequence[float] | np.ndarray,
        start: int = 0,
        end: int | None = None,
        context_len: int = CONTEXT_LEN,
        horizon: int = HORIZON,
        *,
        input_size: int | None = None,
        max_samples: int | None = None,
        target_start: int | None = None,
        target_end: int | None = None,
    ) -> None:
        if context_len < 1 or horizon < 1:
            raise ValueError("context_len and horizon must be positive")
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim != 2:
            raise ValueError("values must have shape (n,) or (n, features)")
        if not np.isfinite(array).all():
            raise ValueError("SequenceDataset values must all be finite")
        if start < 0 or start > len(array):
            raise ValueError("start is outside the values array")
        resolved_end = len(array) if end is None else int(end)
        if resolved_end < start or resolved_end > len(array):
            raise ValueError("end is outside the values array")
        if resolved_end - start < context_len + horizon:
            raise ValueError(
                "available range is too short for context_len + horizon: "
                f"{resolved_end - start} < {context_len + horizon}"
            )
        if input_size is not None and array.shape[1] != int(input_size):
            raise ValueError(
                f"expected {input_size} input features, found {array.shape[1]}"
            )
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("max_samples must be positive when supplied")

        self.values = torch.from_numpy(np.ascontiguousarray(array))
        self.start = int(start)
        self.end = resolved_end
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.input_size = int(array.shape[1])
        default_target_start = self.start + self.context_len
        default_target_end = self.end - self.horizon + 1
        first_target = (
            default_target_start if target_start is None else int(target_start)
        )
        last_target_exclusive = (
            default_target_end if target_end is None else int(target_end)
        )
        if first_target < default_target_start:
            raise ValueError(
                "target_start would require context before the available range"
            )
        if last_target_exclusive > default_target_end:
            raise ValueError(
                "target_end would require target rows after the available range"
            )
        if first_target >= last_target_exclusive:
            raise ValueError("target range does not contain a complete window")
        target_starts = np.arange(
            first_target, last_target_exclusive, dtype=np.int64
        )
        self.target_start = int(first_target)
        self.target_end = int(last_target_exclusive)
        self._available_sample_count = int(target_starts.size)
        if max_samples is not None:
            target_starts = target_starts[: int(max_samples)]
        if target_starts.size == 0:
            raise ValueError("no valid context/target windows can be generated")
        self.target_start_indices = torch.from_numpy(target_starts)

    @property
    def sample_count(self) -> int:
        """Number of generated windows in this dataset."""

        return int(self.target_start_indices.numel())

    @property
    def available_sample_count(self) -> int:
        """Number of windows before an optional fast-mode sample cap."""

        return self._available_sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_start = int(self.target_start_indices[index].item())
        context = self.values[target_start - self.context_len : target_start]
        target = self.values[target_start : target_start + self.horizon]
        return context.clone(), target.clone()


@dataclass
class PreparedData:
    """Scaled chronological splits and their sequence datasets."""

    raw_values: np.ndarray
    timestamps: pd.Series | None
    timestamp_column: str | None
    data_path: Path
    raw_splits: dict[str, np.ndarray]
    scaled_splits: dict[str, np.ndarray]
    scaler: MinMaxScaler
    datasets: dict[str, SequenceDataset]
    boundaries: dict[str, tuple[int, int]]
    raw_counts: dict[str, int]
    available_sample_counts: dict[str, int]
    sample_counts: dict[str, int]
    context_len: int
    horizon: int
    batch_size: int
    num_workers: int

    def make_loaders(self) -> dict[str, DataLoader]:
        """Create deterministic loaders; every loader explicitly uses ``shuffle=False``."""

        return {
            name: make_data_loader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for name, dataset in self.datasets.items()
        }


def prepare_data(
    data_path: str | os.PathLike[str] | None = None,
    *,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> PreparedData:
    """Load, split, fit the training-only scaler, and construct datasets.

    The scaler sees only ``raw_splits['train']``.  All three partitions are
    transformed after that fit.  Target starts are confined to each split;
    validation/test contexts may reach backward into already-observed rows.
    The optional sample cap changes only the number of windows presented to
    the loaders; it never changes split boundaries or uses validation/test
    values for fitting.
    """

    if context_len < 1 or horizon < 1:
        raise ValueError("context_len and horizon must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    series = load_traffic_series(data_path)
    values = series.values
    boundaries = split_indices(
        len(values), train_fraction, validation_fraction, test_fraction
    )
    raw_splits = {
        name: np.ascontiguousarray(values[start:end])
        for name, (start, end) in boundaries.items()
    }
    for name, split in raw_splits.items():
        if len(split) < horizon:
            raise ValueError(
                f"{name} split has {len(split)} rows, fewer than horizon ({horizon})"
            )

    # Deliberately fit on raw training target only.  No validation/test values,
    # weather values, or future observations are supplied to ``fit``.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_matrix = raw_splits["train"].reshape(-1, 1)
    scaler.fit(train_matrix)
    scaled_all = scaler.transform(values.reshape(-1, 1)).reshape(-1)
    scaled_splits = {
        name: np.ascontiguousarray(scaled_all[start:end])
        for name, (start, end) in boundaries.items()
    }

    datasets: dict[str, SequenceDataset] = {}
    available_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    for name, (split_start, split_end) in boundaries.items():
        # Targets are strictly confined to the split's half-open target range.
        # For validation/test, the first context may begin in the preceding
        # split, but it is always past-only and is transformed with the
        # training-fit scaler.  This preserves the exact target timestamps and
        # avoids discarding usable chronological context.
        first_target = max(int(split_start), context_len)
        last_target_exclusive = int(split_end) - horizon + 1
        if first_target >= last_target_exclusive:
            raise ValueError(
                f"{name} split has no complete target windows: "
                f"target range [{first_target}, {last_target_exclusive})"
            )
        available_start = max(0, first_target - context_len)
        available_end = min(
            len(scaled_all), last_target_exclusive + horizon - 1
        )
        unlimited = SequenceDataset(
            scaled_all,
            start=available_start,
            end=available_end,
            context_len=context_len,
            horizon=horizon,
            input_size=1,
            target_start=first_target,
            target_end=last_target_exclusive,
        )
        available_counts[name] = unlimited.available_sample_count
        dataset = SequenceDataset(
            scaled_all,
            start=available_start,
            end=available_end,
            context_len=context_len,
            horizon=horizon,
            input_size=1,
            max_samples=max_samples,
            target_start=first_target,
            target_end=last_target_exclusive,
        )
        datasets[name] = dataset
        sample_counts[name] = len(dataset)

    return PreparedData(
        raw_values=values,
        timestamps=series.timestamps,
        timestamp_column=series.timestamp_column,
        data_path=series.path,
        raw_splits=raw_splits,
        scaled_splits=scaled_splits,
        scaler=scaler,
        datasets=datasets,
        boundaries=boundaries,
        raw_counts={name: len(split) for name, split in raw_splits.items()},
        available_sample_counts=available_counts,
        sample_counts=sample_counts,
        context_len=int(context_len),
        horizon=int(horizon),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )


def make_data_loader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a loader with the experiment's explicit non-shuffling policy.

    ``shuffle=True`` is rejected rather than silently changing the protocol.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if shuffle:
        raise ValueError("chronological experiment requires DataLoader shuffle=False")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )


def print_data_summary(prepared: PreparedData, *, sample_limit: int | None) -> None:
    """Print exact raw and generated-window counts for auditability."""

    print(
        "[data] CSV:",
        prepared.data_path,
        "| target:",
        TARGET_COLUMN,
        "| weather features used: no",
    )
    print(
        "[data] raw rows: total="
        f"{len(prepared.raw_values)}; "
        + "; ".join(
            f"{name}={count}" for name, count in prepared.raw_counts.items()
        )
    )
    print(
        "[data] sequence samples used: total="
        f"{sum(prepared.sample_counts.values())}; "
        + "; ".join(f"{name}={count}" for name, count in prepared.sample_counts.items())
        + (
            f" (cap={sample_limit} per split)"
            if sample_limit is not None
            else " (no cap)"
        )
    )
    print(
        "[data] sequence samples available before cap: total="
        f"{sum(prepared.available_sample_counts.values())}; "
        + "; ".join(
            f"{name}={count}"
            for name, count in prepared.available_sample_counts.items()
        )
    )


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------


def _as_batched_sequence(
    values: torch.Tensor | Sequence[Any] | np.ndarray,
    input_size: int,
) -> torch.Tensor:
    """Normalize model input to ``(batch, sequence, input_size)`` robustly."""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if tensor.ndim == 1:
        if input_size != 1:
            raise ValueError(
                f"unbatched one-dimensional input requires input_size=1, got {input_size}"
            )
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    elif tensor.ndim == 2:
        # Prefer an unbatched feature sequence when the last dimension is the
        # feature dimension.  For the univariate case, (T, 1) is distinguished
        # from the common (batch, T) form by the relative lengths; a true
        # one-step edge case (batch, 1) is inherently ambiguous, so callers
        # should use the explicit 3-D form when the sequence length is one.
        looks_like_feature_sequence = tensor.shape[-1] == input_size and (
            input_size != 1 or tensor.shape[0] > tensor.shape[1]
        )
        if looks_like_feature_sequence:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 3:
        raise ValueError(
            "model input must have shape (sequence,), (batch, sequence), "
            "or (batch, sequence, features)"
        )
    if tensor.shape[-1] != input_size:
        raise ValueError(
            f"expected final input dimension {input_size}, got {tensor.shape[-1]}"
        )
    if tensor.shape[1] < 1:
        raise ValueError("sequence length must be positive")
    return tensor


class LSTMForecaster(nn.Module):
    """Two-layer, 64-unit LSTM for the one-step univariate experiment.

    The defaults intentionally mirror the supervisor protocol:
    ``input_size=1``, ``hidden_size=64``, ``num_layers=2``, and
    ``dropout=0.2``.  The model returns ``(batch, horizon)`` scaled
    predictions; callers inverse-transform them before computing metrics.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("LSTM dimensions must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.horizon = int(horizon)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_size, self.horizon)

    def forward(self, values: torch.Tensor | Sequence[Any] | np.ndarray) -> torch.Tensor:
        """Return scaled predictions with shape ``(batch, horizon)``."""

        tensor = _as_batched_sequence(values, self.input_size)
        # Match the parameter dtype/device when a caller supplies a CPU tensor
        # or a floating tensor with a different dtype.
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        sequence_output, _ = self.lstm(tensor)
        return self.projection(sequence_output[:, -1, :])


# Descriptive aliases make the model easy to find without changing the class's
# explicit protocol name.
LSTMModel = LSTMForecaster
TrafficLSTMForecaster = LSTMForecaster


class DSSInspiredDiagonalStateSpaceBlock(nn.Module):
    """A small diagonal state-space block inspired by DSS terminology.

    This is intentionally described as *DSS-inspired*.  It is not presented as
    an exact reproduction of a paper's architecture.  Its explicit recurrence
    is:

        h_t = a * h_(t-1) + g(x_t) * Bx_t
        y_t = C h_t

    ``a`` is a learned diagonal decay, ``g`` is a learned input gate, ``B`` is
    the input projection, and ``C`` is the state-to-output projection.  A Python
    loop over time is used so the recurrence is visible and auditable.
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        if input_dim < 1 or state_dim < 1:
            raise ValueError("DSS block dimensions must be positive")
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.output_dim = self.state_dim if output_dim is None else int(output_dim)
        if self.output_dim < 1:
            raise ValueError("output_dim must be positive")

        # B, g, and C deliberately have equation-oriented names.
        self.B = nn.Linear(self.input_dim, self.state_dim, bias=True)
        self.g = nn.Linear(self.input_dim, self.state_dim, bias=True)
        self.C = nn.Linear(self.state_dim, self.output_dim, bias=True)
        initial_decay = 0.90
        initial_logit = math.log(initial_decay / (1.0 - initial_decay))
        self.a_logit = nn.Parameter(torch.full((self.state_dim,), initial_logit))
        self.residual_projection = nn.Linear(
            self.input_dim, self.output_dim, bias=True
        )
        self.norm = nn.LayerNorm(self.output_dim)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor | Sequence[Any] | np.ndarray) -> torch.Tensor:
        """Apply the explicit diagonal recurrence to a sequence."""

        tensor = _as_batched_sequence(values, self.input_dim)
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        batch_size = tensor.shape[0]
        h_previous = tensor.new_zeros((batch_size, self.state_dim))
        a = torch.sigmoid(self.a_logit).view(1, -1)
        outputs: list[torch.Tensor] = []

        for time_index in range(tensor.shape[1]):
            x_t = tensor[:, time_index, :]
            Bx_t = self.B(x_t)
            g_x_t = torch.sigmoid(self.g(x_t))
            # h_t = a * h_(t-1) + g(x_t) * Bx_t
            h_t = a * h_previous + g_x_t * Bx_t
            # y_t = C h_t
            y_t = self.C(h_t)
            outputs.append(y_t)
            h_previous = h_t

        recurrence_output = torch.stack(outputs, dim=1)
        recurrence_output = self.activation(self.norm(recurrence_output))
        residual = self.residual_projection(tensor)
        return recurrence_output + residual


class DSSInspiredDiagonalStateSpaceForecaster(nn.Module):
    """Two-block DSS-inspired diagonal state-space forecasting baseline.

    The block recurrence is documented in
    :class:`DSSInspiredDiagonalStateSpaceBlock`.  This model places two blocks
    in sequence, adds a model-level residual from the input projection, applies
    normalization, and uses a final linear projection.  It accepts common input
    rank conventions and returns ``(batch, horizon)``.
    """

    def __init__(
        self,
        input_size: int = 1,
        model_dim: int = 64,
        state_dim: int = 64,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        if model_dim < 1 or state_dim < 1:
            raise ValueError("DSS model dimensions must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.input_size = int(input_size)
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(self.input_size, self.model_dim)
        self.input_norm = nn.LayerNorm(self.model_dim)
        # Exactly two blocks are part of this baseline specification.
        self.blocks = nn.ModuleList(
            [
                DSSInspiredDiagonalStateSpaceBlock(
                    self.model_dim, self.state_dim, self.model_dim
                ),
                DSSInspiredDiagonalStateSpaceBlock(
                    self.model_dim, self.state_dim, self.model_dim
                ),
            ]
        )
        self.residual_projection = nn.Linear(self.model_dim, self.model_dim)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.projection = nn.Linear(self.model_dim, self.horizon)

    def forward(self, values: torch.Tensor | Sequence[Any] | np.ndarray) -> torch.Tensor:
        """Return scaled one-step (or configured-horizon) predictions."""

        tensor = _as_batched_sequence(values, self.input_size)
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        skip = self.input_projection(tensor)
        hidden = self.input_norm(skip)
        hidden = self.blocks[0](hidden)
        # A second, explicit residual path surrounds the second DSS block.
        hidden = self.blocks[1](hidden + self.residual_projection(skip))
        hidden = self.output_norm(hidden + skip)
        return self.projection(hidden[:, -1, :])


def _hippo_d_eigenvalues(state_dim: int) -> np.ndarray:
    """Construct the canonical DSS HiPPO-D eigenvalue initialization.

    The real HiPPO matrix is built explicitly, its eigenvalues are ordered by
    decreasing imaginary part, and the first ``state_dim`` modes are retained.
    The returned values are complex NumPy values used only at model creation.
    """

    n = int(state_dim)
    if n < 1:
        raise ValueError("state_dim must be positive")
    size = 2 * n
    i, j = np.indices((size, size))
    matrix = np.empty((size, size), dtype=np.float64)
    upper = i < j
    diagonal = i == j
    lower = i > j
    matrix[upper] = 0.5 * np.sqrt((2.0 * i[upper] + 1.0) * (2.0 * j[upper] + 1.0))
    matrix[diagonal] = -0.5
    matrix[lower] = -0.5 * np.sqrt((2.0 * i[lower] + 1.0) * (2.0 * j[lower] + 1.0))
    eigenvalues = np.linalg.eigvals(matrix)
    order = np.argsort(-eigenvalues.imag, kind="stable")
    return np.asarray(eigenvalues[order[:n]], dtype=np.complex64)


def dss_softmax_coefficients(
    eigenvalues: torch.Tensor, dt: torch.Tensor, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DSS-softmax transition and input coefficients.

    The canonical length-normalized discretization is
    ``a = exp(lambda * dt)`` and
    ``b = (a - 1) / (lambda * (exp(L * lambda * dt) - 1))``.
    """

    exponent = eigenvalues.unsqueeze(0) * (sequence_length * dt)
    exponent = torch.complex(
        exponent.real.clamp(min=-30.0, max=30.0),
        exponent.imag.clamp(min=-30.0, max=30.0),
    )
    denominator = torch.exp(exponent) - 1.0 + 0j
    denominator = torch.where(
        denominator.abs() < 1e-7,
        torch.complex(
            torch.full_like(denominator.real, 1e-7),
            torch.zeros_like(denominator.imag),
        ),
        denominator,
    )
    transition_exponent = eigenvalues.unsqueeze(0) * dt
    transition_exponent = torch.complex(
        transition_exponent.real.clamp(min=-30.0, max=30.0),
        transition_exponent.imag.clamp(min=-30.0, max=30.0),
    )
    transition = torch.exp(transition_exponent)
    safe_lambda = torch.where(
        eigenvalues.abs() < 1e-7,
        torch.complex(torch.ones_like(eigenvalues.real), torch.zeros_like(eigenvalues.imag)),
        eigenvalues,
    )
    input_coefficient = (transition - 1.0 + 0j) / safe_lambda.unsqueeze(0) / denominator
    return transition, input_coefficient


class DSSSoftmaxBlock(nn.Module):
    """Canonical DSS-softmax diagonal state-space block.

    The block follows the length-normalized DSS-softmax recurrence of Gupta,
    Gu, and Berant (2022) within an explicit traffic wrapper.  It is not a
    claim to reproduce every detail of the original benchmark code.  For a
    fixed sequence length ``L`` and hidden channel ``h``:

    ``a[h,i] = exp(lambda[i] * dt[h])`` and
    ``b[h,i] = (a[h,i]-1) /
    (lambda[i] * (exp(L*lambda[i]*dt[h])-1))``.

    The state is updated with ``x_i[t] = a*x_i[t-1] + b*u_h[t]`` and the
    layer output is the real part of ``sum_i W[h,i] * x_i[t]``.  This is the
    primary DSS comparison; the custom gated recurrence is retained separately
    under the DSS-inspired aliases.
    """

    def __init__(self, hidden_dim: int, state_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        state_dim = hidden_dim if state_dim is None else int(state_dim)
        if hidden_dim < 1 or state_dim < 1:
            raise ValueError("DSS-softmax dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        eigenvalues = _hippo_d_eigenvalues(state_dim)
        initial = torch.view_as_real(torch.from_numpy(eigenvalues).to(torch.complex64))
        # Store real/imaginary parts so the module remains inspectable and can
        # be moved by ordinary PyTorch device/dtype operations.
        self.lambda_real = nn.Parameter(initial[:, 0].clone())
        self.lambda_imag = nn.Parameter(initial[:, 1].clone())
        log_dt = torch.empty(hidden_dim).uniform_(math.log(1e-3), math.log(1e-1))
        self.log_dt = nn.Parameter(log_dt)
        self.weight_real = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.weight_imag = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.feedthrough = nn.Parameter(torch.zeros(hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor | Sequence[Any] | np.ndarray) -> torch.Tensor:
        tensor = _as_batched_sequence(values, self.hidden_dim)
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        sequence_length = int(tensor.shape[1])
        eigenvalues = torch.complex(self.lambda_real, self.lambda_imag)
        dt = torch.exp(self.log_dt.clamp(min=-12.0, max=2.0)).unsqueeze(-1)
        transition, input_coefficient = dss_softmax_coefficients(
            eigenvalues, dt, sequence_length
        )
        weights = torch.complex(self.weight_real, self.weight_imag)

        batch_size = int(tensor.shape[0])
        state = torch.zeros(
            batch_size,
            self.hidden_dim,
            self.state_dim,
            dtype=torch.complex64,
            device=tensor.device,
        )
        outputs: list[torch.Tensor] = []
        for time_index in range(sequence_length):
            current_input = tensor[:, time_index, :].to(torch.complex64)
            state = transition.unsqueeze(0) * state + input_coefficient.unsqueeze(0) * current_input.unsqueeze(-1)
            state_output = torch.einsum("bhn,hn->bh", state, weights).real
            state_output = state_output + self.feedthrough * tensor[:, time_index, :]
            outputs.append(state_output)
        recurrent_output = torch.stack(outputs, dim=1)
        return self.output_projection(self.activation(recurrent_output))


class DSSSoftmaxForecaster(nn.Module):
    """Two-block traffic wrapper around the canonical DSS-softmax layer."""

    def __init__(
        self,
        input_size: int = 1,
        model_dim: int = 64,
        state_dim: int = 64,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        if input_size < 1 or model_dim < 1 or state_dim < 1 or horizon < 1:
            raise ValueError("DSS-softmax dimensions must be positive")
        self.input_size = int(input_size)
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(self.input_size, self.model_dim)
        self.input_norm = nn.LayerNorm(self.model_dim)
        self.blocks = nn.ModuleList(
            [
                DSSSoftmaxBlock(self.model_dim, self.state_dim),
                DSSSoftmaxBlock(self.model_dim, self.state_dim),
            ]
        )
        self.residual_projection = nn.Linear(self.model_dim, self.model_dim)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.projection = nn.Linear(self.model_dim, self.horizon)

    def forward(self, values: torch.Tensor | Sequence[Any] | np.ndarray) -> torch.Tensor:
        tensor = _as_batched_sequence(values, self.input_size)
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        skip = self.input_projection(tensor)
        hidden = self.input_norm(skip)
        hidden = self.blocks[0](hidden) + hidden
        hidden = self.blocks[1](hidden + self.residual_projection(skip)) + hidden
        hidden = self.output_norm(hidden + skip)
        return self.projection(hidden[:, -1, :])


# The canonical DSS-softmax wrapper is the primary comparison.  The earlier
# custom gated recurrence remains available under explicit DSS-inspired names
# for a separate ablation and for comparison with the legacy implementation.
DSSSoftmaxModel = DSSSoftmaxForecaster
DSSSoftmaxDiagonalStateSpace = DSSSoftmaxForecaster
DSSModel = DSSSoftmaxForecaster
DSSInspiredStateSpaceModel = DSSInspiredDiagonalStateSpaceForecaster
DSSInspiredDiagonalStateSpace = DSSInspiredDiagonalStateSpaceForecaster
DSSInspiredDiagonalSSM = DSSInspiredDiagonalStateSpaceForecaster
DSSInspiredForecaster = DSSInspiredDiagonalStateSpaceForecaster


def count_parameters(model: nn.Module | Any) -> int:
    """Count parameters in a torch model/wrapper.

    The total number of parameters is reported.  The trainable comparators in
    this experiment have all of their parameters enabled for training; for
    TimesFM's wrapper, an underlying ``.model`` module is counted when present.
    """

    candidate = model
    if not isinstance(candidate, torch.nn.Module):
        candidate = getattr(model, "model", None)
    if isinstance(candidate, torch.nn.Module):
        return int(sum(parameter.numel() for parameter in candidate.parameters()))
    if hasattr(model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    return 0


# ---------------------------------------------------------------------------
# Training, metrics, and efficiency measurement
# ---------------------------------------------------------------------------


class PeakRSSMonitor:
    """Measure peak resident memory with a lightweight psutil sampling thread."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = float(interval_seconds)
        self.peak_rss_mb: float | None = None
        self.baseline_rss_mb: float | None = None
        self._process: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return
        self.peak_rss_mb = (
            rss_mb if self.peak_rss_mb is None else max(self.peak_rss_mb, rss_mb)
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)
        self._sample()

    def __enter__(self) -> "PeakRSSMonitor":
        try:
            import psutil  # lazy optional at module import time

            self._process = psutil.Process(os.getpid())
            self._sample()
        except Exception:
            self._process = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="traffic-experiment-rss", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self._sample()
        self._thread = None


def _target_to_2d(target: torch.Tensor) -> torch.Tensor:
    """Convert a batched target to ``(batch, horizon)`` for the model output."""

    if target.ndim == 1:
        return target.unsqueeze(0)
    if target.ndim == 2:
        return target
    if target.ndim == 3 and target.shape[-1] == 1:
        return target.squeeze(-1)
    return target.reshape(target.shape[0], -1)


def _model_tensor_context(context: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a context batch to a device without changing its rank."""

    return context.to(device)


@dataclass
class TrainingResult:
    """Result of train-only fitting with validation checkpoint selection."""

    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    best_validation_mae: float
    history: list[dict[str, float | int]]
    training_seconds: float
    peak_rss_mb: float | None
    optimizer_name: str
    criterion_name: str
    selection_criterion_name: str


@dataclass
class InferenceResult:
    """Predictions and timing information in original target units."""

    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    elapsed_seconds: float
    latency_ms_per_batch: float
    latency_ms_per_sample: float
    num_batches: int
    num_samples: int
    peak_rss_mb: float | None


def _criterion_and_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, nn.Module]:
    """Construct the explicit optimizer and regression criterion."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    criterion: nn.Module = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    return optimizer, criterion


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = 1.0,
) -> float:
    """Run one optimization epoch and return mean training MSE."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    for context, target in loader:
        context = _model_tensor_context(context, device)
        target = _target_to_2d(target).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("training loader is empty")
    return total_loss / total_examples


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss without gradients or checkpoint updates.

    The experiment passes an explicit ``L1Loss`` here, so early stopping uses
    validation MAE on the scaled target (equivalent in ordering to original
    speed-unit MAE because the inverse transform is affine and monotone).
    """

    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for context, target in loader:
            context = _model_tensor_context(context, device)
            target = _target_to_2d(target).to(device)
            prediction = model(context)
            loss = criterion(prediction, target)
            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("validation loader is empty")
    return total_loss / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = 1.0,
) -> TrainingResult:
    """Fit a trainable model using train data and validation-only selection.

    Adam and mean-squared error are explicit.  The best validation checkpoint
    is restored after early stopping; the test loader is never consulted here.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    model.to(device)
    optimizer, criterion = _criterion_and_optimizer(model, learning_rate, weight_decay)
    validation_criterion: nn.Module = nn.L1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0

    with PeakRSSMonitor() as rss_monitor:
        _synchronize_device(device)
        training_start = time.perf_counter()
        for epoch in range(1, int(epochs) + 1):
            train_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                gradient_clip_norm=gradient_clip_norm,
            )
            validation_loss = _validation_loss(
                model, validation_loader, validation_criterion, device
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "validation_loss": float(validation_loss),
                }
            )
            if validation_loss < best_loss:
                best_loss = float(validation_loss)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        _synchronize_device(device)
        training_seconds = time.perf_counter() - training_start

    if best_state is None:  # defensive; a finite validation loss should set it
        raise FloatingPointError("Validation did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_validation_mae=best_loss,
        history=history,
        training_seconds=float(training_seconds),
        peak_rss_mb=rss_monitor.peak_rss_mb,
        optimizer_name="Adam",
        criterion_name="MSELoss",
        selection_criterion_name="L1Loss (scaled validation MAE)",
    )


def compute_metrics(
    targets: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE (percent) in original target units.

    MAPE ignores zero-valued targets to avoid division by zero.  If all
    targets are zero, it is defined as zero when predictions are also zero and
    infinity otherwise.
    """

    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    forecast = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if actual.shape != forecast.shape:
        raise ValueError("targets and predictions must have the same number of values")
    if actual.size == 0:
        raise ValueError("cannot compute metrics for an empty array")
    error = actual - forecast
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    nonzero = np.abs(actual) > np.finfo(np.float64).eps
    if np.any(nonzero):
        mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
    elif np.all(np.abs(error) <= np.finfo(np.float64).eps):
        mape = 0.0
    else:
        mape = float("inf")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def inverse_transform_target(
    values: Sequence[float] | np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Inverse-transform a one-dimensional or batch-one-column array."""

    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    flat = array.reshape(-1, 1)
    restored = scaler.inverse_transform(flat).reshape(original_shape)
    return np.asarray(restored, dtype=np.float64)


def _predict_trainable(
    model: nn.Module,
    loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
) -> InferenceResult:
    """Evaluate a trainable model and measure batch/sample inference latency."""

    model.eval()
    scaled_predictions: list[np.ndarray] = []
    scaled_targets: list[np.ndarray] = []
    num_batches = 0
    num_samples = 0
    with PeakRSSMonitor() as rss_monitor:
        _synchronize_device(device)
        inference_start = time.perf_counter()
        with torch.no_grad():
            for context, target in loader:
                context = _model_tensor_context(context, device)
                prediction = model(context)
                target_2d = _target_to_2d(target)
                if prediction.shape != target_2d.shape:
                    raise ValueError(
                        f"model/target shape mismatch: {tuple(prediction.shape)} "
                        f"versus {tuple(target_2d.shape)}"
                    )
                scaled_predictions.append(prediction.detach().cpu().numpy())
                scaled_targets.append(target_2d.detach().cpu().numpy())
                num_batches += 1
                num_samples += int(target_2d.shape[0])
        _synchronize_device(device)
        elapsed = time.perf_counter() - inference_start

    if not scaled_predictions:
        raise ValueError("test loader is empty")
    prediction_array = np.concatenate(scaled_predictions, axis=0).reshape(-1)
    target_array = np.concatenate(scaled_targets, axis=0).reshape(-1)
    predictions = inverse_transform_target(prediction_array, scaler)
    targets = inverse_transform_target(target_array, scaler)
    return InferenceResult(
        metrics=compute_metrics(targets, predictions),
        predictions=predictions,
        targets=targets,
        elapsed_seconds=float(elapsed),
        latency_ms_per_batch=float(elapsed * 1000.0 / max(1, num_batches)),
        latency_ms_per_sample=float(elapsed * 1000.0 / max(1, num_samples)),
        num_batches=num_batches,
        num_samples=num_samples,
        peak_rss_mb=rss_monitor.peak_rss_mb,
    )


# ---------------------------------------------------------------------------
# Optional TimesFM 2.5 PyTorch loader
# ---------------------------------------------------------------------------


class TimesFM25PyTorchForecaster:
    """Thin optional adapter around the TimesFM 2.5 PyTorch API.

    The adapter presents the same 24-step context windows used by the trainable
    models and returns one prediction per context.  The experiment supplies raw
    traffic-speed contexts so the pretrained model can apply its own documented
    input normalization.  The TimesFM package is intentionally not imported at
    module import time.
    """

    def __init__(self, model: Any, *, context_len: int = CONTEXT_LEN, horizon: int = HORIZON) -> None:
        self.model = model
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        if self.context_len < 1 or self.horizon != 1:
            raise ValueError("TimesFM adapter requires context_len >= 1 and horizon=1")

    @property
    def parameter_count(self) -> int:
        """Return the number of parameters in the underlying TimesFM module."""

        return count_parameters(self.model)

    @staticmethod
    def _context_list(context: torch.Tensor | Sequence[Any] | np.ndarray) -> list[np.ndarray]:
        array = context.detach().cpu().numpy() if isinstance(context, torch.Tensor) else np.asarray(context)
        if array.ndim == 1:
            return [np.asarray(array, dtype=np.float32)]
        if array.ndim == 2:
            if array.shape[-1] == 1 and array.shape[0] > 1:
                return [np.asarray(array[:, 0], dtype=np.float32)]
            return [np.asarray(row, dtype=np.float32) for row in array]
        if array.ndim == 3 and array.shape[-1] == 1:
            return [np.asarray(row[..., 0], dtype=np.float32) for row in array]
        raise ValueError("TimesFM contexts must have shape (T,), (B,T), or (B,T,1)")

    @staticmethod
    def _extract_point_forecast(result: Any, batch_size: int, horizon: int) -> np.ndarray:
        """Normalize point outputs from TimesFM API variants to (B, horizon)."""

        point = result[0] if isinstance(result, (tuple, list)) else result
        array = np.asarray(point)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1) if array.size > 1 else array.reshape(1, 1)
        elif array.ndim == 3:
            # Some releases return [batch, horizon, quantile]; the median is
            # the central TimesFM quantile, while the 2.5 point API returns a
            # two-dimensional point array.
            if array.shape[-1] > 1:
                median_index = min(5, array.shape[-1] - 1)
                array = array[..., median_index]
            else:
                array = array[..., 0]
        if array.ndim != 2:
            array = array.reshape(array.shape[0], -1)
        if array.shape[0] != batch_size:
            raise ValueError(
                f"TimesFM returned batch {array.shape[0]}, expected {batch_size}"
            )
        if array.shape[1] < horizon:
            raise ValueError(
                f"TimesFM returned horizon {array.shape[1]}, expected at least {horizon}"
            )
        return np.asarray(array[:, :horizon], dtype=np.float32)

    def predict(self, contexts: torch.Tensor | Sequence[Any] | np.ndarray) -> np.ndarray:
        """Forecast a batch using ``horizon=1`` and the configured context length."""

        context_list = self._context_list(contexts)
        if not context_list:
            return np.empty((0, self.horizon), dtype=np.float32)
        try:
            result = self.model.forecast(
                horizon=self.horizon, inputs=context_list
            )
        except TypeError:  # older keyword-light wrappers
            result = self.model.forecast(self.horizon, context_list)
        return self._extract_point_forecast(result, len(context_list), self.horizon)


def load_timesfm_25_torch(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> TimesFM25PyTorchForecaster:
    """Load TimesFM 2.5 PyTorch lazily with compilation disabled.

    If ``checkpoint`` is supplied, it must be an existing safetensors file or a
    directory containing ``model.safetensors``.  With no checkpoint, the
    official ``from_pretrained`` path may download weights; therefore this
    function is called only after the user explicitly enables TimesFM.  A
    missing local checkpoint raises a clear ``FileNotFoundError`` rather than
    silently changing the experiment.
    """

    if horizon != 1:
        raise ValueError("The experiment's TimesFM comparison requires horizon=1")
    selected_device = device or select_device(None)
    try:
        import timesfm  # lazy import: disabled runs never download/load TimesFM
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "TimesFM was requested but could not be imported; install the "
            "optional timesfm package or run with run_timesfm=False"
        ) from exc

    model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
    if model_class is None:
        raise RuntimeError(
            "The installed timesfm package does not expose "
            "TimesFM_2p5_200M_torch; use a TimesFM 2.5 PyTorch release"
        )

    checkpoint_path: Path | None = None
    if checkpoint is not None:
        checkpoint_path = resolve_repo_path(checkpoint, default=Path("timesfm-checkpoint"))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "TimesFM checkpoint was requested but does not exist: "
                f"{checkpoint_path}. Supply a local checkpoint or enable "
                "pretrained download explicitly by omitting --timesfm-checkpoint."
            )

    try:
        if checkpoint_path is not None:
            timesfm_model = model_class(torch_compile=False)
            timesfm_model.load_checkpoint(str(checkpoint_path), torch_compile=False)
        else:
            # ``from_pretrained`` is supplied by the 2.5 PyTorch hub mixin.
            timesfm_model = model_class.from_pretrained(
                TIMESFM_MODEL_ID,
                torch_compile=False,
            )
    except FileNotFoundError:
        raise
    except TypeError:
        # Compatibility fallback for releases whose constructor/hub method did
        # not accept the explicit keyword.
        if checkpoint_path is not None:
            timesfm_model = model_class()
            timesfm_model.load_checkpoint(str(checkpoint_path), torch_compile=False)
        else:
            timesfm_model = model_class.from_pretrained(TIMESFM_MODEL_ID)

    # Move the underlying module when the wrapper exposes one.  The 2.5 CPU
    # wrapper stores its device on the inner module and uses it during decode.
    inner_module = getattr(timesfm_model, "model", None)
    if isinstance(inner_module, nn.Module):
        inner_module.to(selected_device)
        if hasattr(inner_module, "device"):
            inner_module.device = selected_device
        if hasattr(inner_module, "device_count"):
            inner_module.device_count = 1
    if hasattr(timesfm_model, "device"):
        try:
            timesfm_model.device = selected_device
        except Exception:
            pass

    forecast_config_class = getattr(timesfm, "ForecastConfig", None)
    if forecast_config_class is not None:
        forecast_config = forecast_config_class(
            max_context=int(context_len),
            max_horizon=int(horizon),
            per_core_batch_size=TIMESFM_BATCH_SIZE,
            normalize_inputs=True,
            infer_is_positive=False,
            force_flip_invariance=False,
        )
        timesfm_model.compile(forecast_config)
    elif hasattr(timesfm_model, "compile"):
        # Last-resort old API; still explicitly disable torch.compile where the
        # wrapper exposes the option.
        timesfm_model.compile()
    if isinstance(inner_module, nn.Module):
        inner_module.eval()
    if hasattr(timesfm_model, "eval"):
        timesfm_model.eval()
    return TimesFM25PyTorchForecaster(
        timesfm_model, context_len=context_len, horizon=horizon
    )


# Short public alias for callers that want the optional loader by its role.
load_timesfm = load_timesfm_25_torch


def _predict_timesfm(
    forecaster: TimesFM25PyTorchForecaster,
    loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device | None = None,
) -> InferenceResult:
    """Evaluate optional TimesFM on raw contexts and measure latency.

    The supervised loaders contain scaled tensors, so contexts are converted
    back to original traffic-speed units before calling TimesFM.  This keeps
    the pretrained model's own normalization independent of the supervised
    MinMaxScaler; targets and predictions are then compared directly in raw
    speed units.
    """

    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    num_batches = 0
    num_samples = 0
    timing_device = device or select_device(None)
    with PeakRSSMonitor() as rss_monitor:
        _synchronize_device(timing_device)
        inference_start = time.perf_counter()
        for context, target in loader:
            context_array = context.detach().cpu().numpy()
            raw_context = inverse_transform_target(context_array, scaler)
            prediction = forecaster.predict(raw_context)
            target_array = _target_to_2d(target).numpy().reshape(-1)
            raw_targets.append(inverse_transform_target(target_array, scaler))
            raw_predictions.append(np.asarray(prediction, dtype=np.float64).reshape(-1))
            num_batches += 1
            num_samples += int(target_array.size)
        _synchronize_device(timing_device)
        elapsed = time.perf_counter() - inference_start

    if not raw_predictions:
        raise ValueError("test loader is empty")
    predictions = np.concatenate(raw_predictions, axis=0)
    targets = np.concatenate(raw_targets, axis=0)
    return InferenceResult(
        metrics=compute_metrics(targets, predictions),
        predictions=predictions,
        targets=targets,
        elapsed_seconds=float(elapsed),
        latency_ms_per_batch=float(elapsed * 1000.0 / max(1, num_batches)),
        latency_ms_per_sample=float(elapsed * 1000.0 / max(1, num_samples)),
        num_batches=num_batches,
        num_samples=num_samples,
        peak_rss_mb=rss_monitor.peak_rss_mb,
    )


# ---------------------------------------------------------------------------
# Results, plotting, and public experiment API
# ---------------------------------------------------------------------------


def save_metric_figure(metric_rows: Sequence[Mapping[str, Any]], path: Path) -> Path | None:
    """Save a high-resolution figure with one subplot per metric unit."""

    if not metric_rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - requirements normally provide it
        print(f"[plot] matplotlib unavailable; skipping figure: {exc}")
        return None

    names = [str(row["model"]) for row in metric_rows]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    specifications = (
        ("mae", "MAE", "Original speed units"),
        ("rmse", "RMSE", "Original speed units"),
        ("mape", "MAPE", "Percent (%)"),
    )
    for axis, (key, title, unit_label) in zip(axes, specifications):
        values = [float(row.get(key, np.nan)) for row in metric_rows]
        axis.bar(positions, values, color=("#4472C4", "#ED7D31", "#70AD47"))
        axis.set_title(title)
        axis.set_ylabel(unit_label)
        axis.set_xticks(positions, names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        # Keep separate units on separate axes; do not overlay a second y-axis.
    figure.suptitle("One-step traffic-speed test metrics (no weather features)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def save_unit_separated_figures(
    metric_rows: Sequence[Mapping[str, Any]], directory: Path
) -> dict[str, Path]:
    """Save two unit-separated figures for the experiment report.

    MAE and RMSE remain in the original traffic-speed units, while MAPE is
    rendered on its own percentage axis. Keeping these as separate files
    prevents a reader from mistaking percentage error for speed-unit error.
    """

    if not metric_rows:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - requirements normally provide it
        print(f"[plot] matplotlib unavailable; skipping unit-separated figures: {exc}")
        return {}

    names = [str(row["model"]) for row in metric_rows]
    positions = np.arange(len(names))
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.36
    axis.bar(
        positions - width / 2,
        [float(row["mae"]) for row in metric_rows],
        width,
        label="MAE",
        color="#4472C4",
    )
    axis.bar(
        positions + width / 2,
        [float(row["rmse"]) for row in metric_rows],
        width,
        label="RMSE",
        color="#ED7D31",
    )
    axis.set_title("One-step test errors in original speed units")
    axis.set_ylabel("Traffic-speed error (original units)")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    speed_path = directory / "mae_rmse_speed.png"
    figure.savefig(speed_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    written["mae_rmse_speed"] = speed_path

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.bar(
        positions,
        [float(row["mape"]) for row in metric_rows],
        color="#70AD47",
    )
    axis.set_title("One-step test percentage error")
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    mape_path = directory / "mape_percent.png"
    figure.savefig(mape_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    written["mape_percent"] = mape_path
    return written


def save_forecast_figure(
    predictions_by_model: Mapping[str, np.ndarray],
    targets: np.ndarray,
    path: Path,
    *,
    max_points: int = 240,
) -> Path | None:
    """Save an aligned one-step forecast plot for the common test targets."""

    if not predictions_by_model or targets.size == 0:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - requirements normally provide it
        print(f"[plot] matplotlib unavailable; skipping forecast figure: {exc}")
        return None

    count = min(int(max_points), int(targets.size))
    x = np.arange(count)
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(x, targets[:count], color="black", linewidth=2.0, label="Actual")
    for name, values in predictions_by_model.items():
        array = np.asarray(values).reshape(-1)
        if array.size:
            axis.plot(x, array[:count], linewidth=1.4, label=str(name))
    axis.set_title("Common one-step traffic-speed test targets")
    axis.set_xlabel("Test sample (five-minute steps)")
    axis.set_ylabel("Traffic speed")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def save_metrics_latex_table(metric_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write a small LaTeX table for optional report generation."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{One-step traffic-speed test metrics generated by the reproducible experiment.}",
        r"\label{tab:performance_results}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & MAE (speed units) & RMSE (speed units) & MAPE (\%) \\",
        r"\hline",
    ]
    for row in metric_rows:
        name = str(row["model"]).replace("_", r"\_")
        lines.append(
            f"{name} & {float(row['mae']):.4f} & {float(row['rmse']):.4f} & "
            f"{float(row['mape']):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_efficiency_latex_table(
    efficiency_rows: Sequence[Mapping[str, Any]], path: Path
) -> Path:
    """Write the measured computational table for optional report generation."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Measured computational records for the common test workload. Process RSS is sampled during each sequential evaluation stage and is not an isolated per-model memory footprint.}",
        r"\label{tab:efficiency_results}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Model & Train time (s) & Inference (ms/1000) & Parameters & Peak RSS (MiB) \\",
        r"\hline",
    ]
    for row in efficiency_rows:
        name = str(row["model"]).replace("_", r"\_")
        training = row.get("training_time_seconds")
        training_text = "N/A" if training is None else f"{float(training):.3f}"
        latency = row.get("inference_latency_ms_per_1000_samples")
        latency_text = "N/A" if latency is None else f"{float(latency):.3f}"
        parameters = row.get("parameter_count")
        parameter_text = "N/A" if parameters is None else f"{int(parameters):,}"
        peak = row.get("peak_rss_mb")
        peak_text = "N/A" if peak is None else f"{float(peak):.2f}"
        lines.append(
            f"{name} & {training_text} & {latency_text} & {parameter_text} & {peak_text} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _metadata_counts(prepared: PreparedData) -> dict[str, Any]:
    return {
        "raw_total": len(prepared.raw_values),
        "raw_by_split": prepared.raw_counts,
        "available_sample_by_split": prepared.available_sample_counts,
        "used_sample_by_split": prepared.sample_counts,
        "boundary_rows_without_target": {
            name: prepared.raw_counts[name] - prepared.available_sample_counts[name]
            for name in prepared.raw_counts
        },
        "boundaries_half_open": {
            name: {"start": start, "end": end}
            for name, (start, end) in prepared.boundaries.items()
        },
    }


@dataclass
class ExperimentConfig:
    """Explicit configuration for a reproducible notebook/script run."""

    data_path: str | os.PathLike[str] | None = None
    output_dir: str | os.PathLike[str] | None = None
    seed: int = SEED
    device: str | torch.device | None = None
    fast: bool = False
    run_timesfm: bool = RUN_TIMESFM
    context_len: int = CONTEXT_LEN
    horizon: int = HORIZON
    batch_size: int = BATCH_SIZE
    train_fraction: float = TRAIN_FRACTION
    validation_fraction: float = VALIDATION_FRACTION
    test_fraction: float = TEST_FRACTION
    epochs: int | None = None
    patience: int | None = None
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = 0.0
    num_workers: int = 0
    max_samples: int | None = None
    timesfm_checkpoint: str | os.PathLike[str] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by :func:`run_experiment`."""

        return {
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "device": self.device,
            "fast": self.fast,
            "run_timesfm": self.run_timesfm,
            "context_len": self.context_len,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "epochs": self.epochs,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "max_samples": self.max_samples,
            "timesfm_checkpoint": self.timesfm_checkpoint,
        }


def run_experiment(
    data_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    seed: int = SEED,
    device: str | torch.device | None = None,
    fast: bool = False,
    run_timesfm: bool = RUN_TIMESFM,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    epochs: int | None = None,
    patience: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    max_samples: int | None = None,
    timesfm_checkpoint: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run the reproducible univariate one-step forecasting experiment.

    Parameters are intentionally ordinary Python values so the function is
    usable from a notebook, a script, or a test without relying on cwd.  In
    fast mode, one epoch and at most ``FAST_MAX_SAMPLES`` windows per split are
    used unless the caller supplies an explicit value.  Full mode uses all
    windows and sensible early-stopping defaults.
    """

    if context_len < 1 or horizon < 1:
        raise ValueError("context_len and horizon must be positive")
    if run_timesfm and horizon != HORIZON:
        raise ValueError("TimesFM comparison is defined for horizon=1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when supplied")
    if fast and max_samples is None:
        max_samples = FAST_MAX_SAMPLES
    resolved_epochs = (FAST_EPOCHS if fast else DEFAULT_EPOCHS) if epochs is None else int(epochs)
    resolved_patience = (1 if fast else DEFAULT_PATIENCE) if patience is None else int(patience)
    if resolved_epochs < 1 or resolved_patience < 1:
        raise ValueError("epochs and patience must be positive")

    set_seed(seed)
    selected_device = select_device(device)
    output_path = resolve_output_dir(output_dir)
    prepared = prepare_data(
        data_path,
        context_len=context_len,
        horizon=horizon,
        batch_size=batch_size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        max_samples=max_samples,
        num_workers=num_workers,
    )
    print_data_summary(prepared, sample_limit=max_samples)
    loaders = prepared.make_loaders()

    metadata: dict[str, Any] = {
        "experiment": "univariate_traffic_speed_one_step",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "fast": bool(fast),
        "run_timesfm": bool(run_timesfm),
        "target_column": TARGET_COLUMN,
        "feature_columns": [TARGET_COLUMN],
        "weather_features_used": False,
        "context_len": int(context_len),
        "horizon": int(horizon),
        "batch_size": int(batch_size),
        "data_loader_shuffle": False,
        "sample_limit_per_split": max_samples,
        "split_fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        },
        "scaler": {
            "class": "MinMaxScaler",
            "feature_range": [0.0, 1.0],
            "fit_split": "train",
            "fit_raw_rows": int(prepared.raw_counts["train"]),
            "fit_raw_index_range": [0, int(prepared.boundaries["train"][1])],
            "training_min": float(prepared.scaler.data_min_[0]),
            "training_max": float(prepared.scaler.data_max_[0]),
        },
        "data": {
            "path": portable_path(prepared.data_path),
            "timestamp_column": prepared.timestamp_column,
            **_metadata_counts(prepared),
        },
        "training": {
            "optimizer": "Adam",
            "criterion": "MSELoss (scaled target)",
            "selection_criterion": "L1Loss (scaled validation MAE)",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip_norm": 1.0,
            "epochs_requested": int(resolved_epochs),
            "patience": int(resolved_patience),
            "early_stopping_split": "validation",
            "checkpoint_selection_split": "validation",
            "test_used_for_selection": False,
        },
        "software_hardware": version_hardware_metadata(selected_device),
        "models": {},
    }

    metric_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}

    model_builders: list[tuple[str, Callable[[], nn.Module]]] = [
        (
            "LSTM",
            lambda: LSTMForecaster(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                horizon=horizon,
            ),
        ),
        (
            "DSS-softmax",
            lambda: DSSSoftmaxForecaster(
                input_size=1,
                model_dim=64,
                state_dim=64,
                horizon=horizon,
            ),
        ),
    ]

    for model_name, builder in model_builders:
        print(f"[train] {model_name} on {selected_device}")
        # Reset before each initialization so model order does not change the
        # reproducible starting weights.
        set_seed(seed)
        model = builder().to(selected_device)
        parameter_count = count_parameters(model)
        training_result = train_model(
            model,
            loaders["train"],
            loaders["validation"],
            selected_device,
            epochs=resolved_epochs,
            patience=resolved_patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        inference_result = _predict_trainable(
            training_result.model,
            loaders["test"],
            prepared.scaler,
            selected_device,
        )
        metric_rows.append(
            {
                "model": model_name,
                "split": "test",
                "context_len": int(context_len),
                "horizon": int(horizon),
                "feature_set": "traffic_speed_only",
                "mae": inference_result.metrics["mae"],
                "rmse": inference_result.metrics["rmse"],
                "mape": inference_result.metrics["mape"],
                "parameter_count": parameter_count,
                "n_test_samples": inference_result.num_samples,
            }
        )
        efficiency_rows.append(
            {
                "model": model_name,
                "trainable": True,
                "device": str(selected_device),
                "inference_batch_size": int(batch_size),
                "training_time_seconds": training_result.training_seconds,
                "training_peak_rss_mb": training_result.peak_rss_mb,
                "inference_latency_ms_per_batch": inference_result.latency_ms_per_batch,
                "inference_latency_ms_per_sample": inference_result.latency_ms_per_sample,
                "inference_latency_ms_per_1000_samples": inference_result.latency_ms_per_sample
                * 1000.0,
                "inference_peak_rss_mb": inference_result.peak_rss_mb,
                "peak_rss_mb": max(
                    value
                    for value in (
                        training_result.peak_rss_mb,
                        inference_result.peak_rss_mb,
                    )
                    if value is not None
                )
                if training_result.peak_rss_mb is not None
                or inference_result.peak_rss_mb is not None
                else None,
                "n_test_batches": inference_result.num_batches,
                "n_test_samples": inference_result.num_samples,
                "parameter_count": parameter_count,
            }
        )
        predictions_by_model[model_name] = inference_result.predictions
        metadata["models"][model_name] = {
            "trainable": True,
            "parameter_count": parameter_count,
            "best_epoch": training_result.best_epoch,
            "best_validation_loss_scaled": training_result.best_validation_loss,
            "best_validation_mae_scaled": training_result.best_validation_mae,
            "training_time_seconds": training_result.training_seconds,
            "inference_latency_ms_per_batch": inference_result.latency_ms_per_batch,
            "inference_latency_ms_per_1000_samples": inference_result.latency_ms_per_sample
            * 1000.0,
            "optimizer": training_result.optimizer_name,
            "criterion": training_result.criterion_name,
            "selection_criterion": training_result.selection_criterion_name,
        }

    if run_timesfm:
        print("[timesfm] explicit TimesFM comparison enabled")
        set_seed(seed)
        # This is intentionally outside the disabled path: a missing package or
        # checkpoint is an error when the user explicitly requested TimesFM.
        checkpoint = timesfm_checkpoint or (
            os.environ.get("TIMESFM_CHECKPOINT") or None
        )
        timesfm = load_timesfm_25_torch(
            checkpoint=checkpoint,
            device=selected_device,
            context_len=context_len,
            horizon=horizon,
        )
        timesfm_loader = make_data_loader(
            prepared.datasets["test"],
            batch_size=TIMESFM_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
        )
        timesfm_result = _predict_timesfm(
            timesfm, timesfm_loader, prepared.scaler, device=selected_device
        )
        timesfm_parameter_count = timesfm.parameter_count
        metric_rows.append(
            {
                "model": "TimesFM 2.5 PyTorch",
                "split": "test",
                "context_len": int(context_len),
                "horizon": int(horizon),
                "feature_set": "traffic_speed_only",
                "mae": timesfm_result.metrics["mae"],
                "rmse": timesfm_result.metrics["rmse"],
                "mape": timesfm_result.metrics["mape"],
                "parameter_count": timesfm_parameter_count,
                "n_test_samples": timesfm_result.num_samples,
            }
        )
        efficiency_rows.append(
            {
                "model": "TimesFM 2.5 PyTorch",
                "trainable": False,
                "device": str(selected_device),
                "inference_batch_size": int(TIMESFM_BATCH_SIZE),
                "training_time_seconds": None,
                "training_peak_rss_mb": None,
                "inference_latency_ms_per_batch": timesfm_result.latency_ms_per_batch,
                "inference_latency_ms_per_sample": timesfm_result.latency_ms_per_sample,
                "inference_latency_ms_per_1000_samples": timesfm_result.latency_ms_per_sample
                * 1000.0,
                "inference_peak_rss_mb": timesfm_result.peak_rss_mb,
                "peak_rss_mb": timesfm_result.peak_rss_mb,
                "n_test_batches": timesfm_result.num_batches,
                "n_test_samples": timesfm_result.num_samples,
                "parameter_count": timesfm_parameter_count,
            }
        )
        predictions_by_model["TimesFM 2.5 PyTorch"] = timesfm_result.predictions
        metadata["models"]["TimesFM 2.5 PyTorch"] = {
            "trainable": False,
            "parameter_count": timesfm_parameter_count,
            "torch_compile": False,
            "forecast_horizon": 1,
            "context_len": int(context_len),
            "inference_batch_size": int(TIMESFM_BATCH_SIZE),
            "input_units": "raw traffic_speed",
            "input_normalization": "TimesFM internal normalization",
            "checkpoint": str(checkpoint) if checkpoint is not None else TIMESFM_MODEL_ID,
            "inference_latency_ms_per_batch": timesfm_result.latency_ms_per_batch,
            "inference_latency_ms_per_1000_samples": timesfm_result.latency_ms_per_sample
            * 1000.0,
        }

    metrics_frame = pd.DataFrame(metric_rows)
    efficiency_frame = pd.DataFrame(efficiency_rows)
    metrics_csv = output_path / "metrics.csv"
    efficiency_csv = output_path / "efficiency.csv"
    metrics_frame.to_csv(metrics_csv, index=False)
    efficiency_frame.to_csv(efficiency_csv, index=False)
    figure_path = output_path / "figures" / "metrics_by_unit.png"
    saved_figure = save_metric_figure(metric_rows, figure_path)
    unit_separated_figures = save_unit_separated_figures(
        metric_rows, output_path / "figures"
    )
    compatibility_figure: Path | None = None
    if saved_figure is not None:
        # Keep a short root-level alias for simple consumers while the
        # canonical, unit-separated figure follows the repository protocol.
        import shutil

        compatibility_figure = output_path / "metrics.png"
        shutil.copy2(saved_figure, compatibility_figure)

    target_indices = prepared.datasets["test"].target_start_indices.detach().cpu().numpy()
    test_targets = prepared.raw_values[target_indices]
    if predictions_by_model:
        first_prediction = next(iter(predictions_by_model.values()))
        test_targets = test_targets[: len(first_prediction)]
    forecast_figure = save_forecast_figure(
        predictions_by_model,
        np.asarray(test_targets, dtype=np.float64),
        output_path / "figures" / "forecast_comparison.png",
    )
    metrics_table = save_metrics_latex_table(
        metric_rows, output_path / "metrics_table.tex"
    )
    efficiency_table = save_efficiency_latex_table(
        efficiency_rows, output_path / "efficiency_table.tex"
    )

    metadata["outputs"] = {
        "metrics_csv": portable_path(metrics_csv),
        "efficiency_csv": portable_path(efficiency_csv),
        "metric_figure": portable_path(saved_figure) if saved_figure is not None else None,
        "metric_figure_alias": portable_path(compatibility_figure)
        if compatibility_figure is not None
        else None,
        "unit_separated_figures": {
            name: portable_path(path) for name, path in unit_separated_figures.items()
        },
        "forecast_figure": portable_path(forecast_figure) if forecast_figure is not None else None,
        "metrics_latex_table": portable_path(metrics_table),
        "efficiency_latex_table": portable_path(efficiency_table),
        "figure_dpi": 300,
    }
    metadata_path = output_path / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default)
        handle.write("\n")

    print(f"[results] metrics: {metrics_csv}")
    print(f"[results] efficiency: {efficiency_csv}")
    print(f"[results] metadata: {metadata_path}")
    if saved_figure is not None:
        print(f"[results] figure: {saved_figure}")
    return {
        "metrics": metrics_frame,
        "efficiency": efficiency_frame,
        "metadata": metadata,
        "predictions": predictions_by_model,
        "paths": metadata["outputs"],
        "prepared_data": prepared,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for full and smoke-test runs."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproducible univariate one-step traffic forecasting experiment "
            "(traffic_speed only; weather is excluded)."
        )
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use one epoch and a small per-split sample cap for a smoke test.",
    )
    parser.add_argument(
        "--run-timesfm",
        action="store_true",
        dest="run_timesfm",
        default=RUN_TIMESFM,
        help="Explicitly load and evaluate optional TimesFM 2.5 PyTorch.",
    )
    parser.add_argument("--data-path", default=None, help="CSV path (relative to repo/module).")
    parser.add_argument("--output-dir", default=None, help="Output directory (relative to repo/module).")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum windows per split (fast mode defaults to 256).",
    )
    parser.add_argument(
        "--timesfm-checkpoint",
        default=None,
        help="Local TimesFM 2.5 safetensors file/directory; omitted means pretrained download.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point; the ``__main__`` guard keeps Windows workers safe."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        fast=args.fast,
        run_timesfm=args.run_timesfm,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        timesfm_checkpoint=args.timesfm_checkpoint,
    )
    return result


if __name__ == "__main__":
    main()
