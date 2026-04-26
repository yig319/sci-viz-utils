"""Small array helpers shared by scientific visualization packages."""

from __future__ import annotations

import numpy as np


def finite_values(data) -> np.ndarray:
    """Return finite values from an array as a 1D float array."""

    values = np.asarray(data, dtype=float)
    return values[np.isfinite(values)]


def normalize_data(data):
    """Scale numeric data to the 0-1 range.

    Constant or non-finite arrays return zeros with the same shape. This helper
    is intentionally simple and is meant for display normalization, not
    statistical preprocessing.
    """

    values = np.asarray(data, dtype=float)
    finite = finite_values(values)
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def NormalizeData(data):
    """Backward-compatible alias for :func:`normalize_data`."""

    return normalize_data(data)


def scale_to_0_1(data, percentiles: tuple[float, float] | None = None):
    """Scale data to 0-1, optionally after percentile clipping."""

    values = np.asarray(data, dtype=float)
    if percentiles is None:
        return normalize_data(values)
    lo, hi = percentile_limits(values, percentiles)
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def smooth_curve(data, window_size):
    """Smooth a 1D curve with a moving average window."""

    values = np.asarray(data, dtype=float)
    window_size = int(window_size)
    if window_size <= 1 or values.size == 0:
        return values
    if window_size > values.size:
        window_size = values.size
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(values, kernel, mode="same")


def percentile_limits(data, percentiles=(1, 99)) -> tuple[float | None, float | None]:
    """Return finite-data percentile limits for display."""

    finite = finite_values(data)
    if finite.size == 0:
        return None, None
    lo, hi = np.percentile(finite, percentiles)
    return float(lo), float(hi)


def sigma_limits(data, sigma: float = 3.0) -> tuple[float | None, float | None]:
    """Return mean +/- ``sigma`` standard deviations for finite data."""

    finite = finite_values(data)
    if finite.size == 0:
        return None, None
    mean = float(np.nanmean(finite))
    std = float(np.nanstd(finite))
    return mean - sigma * std, mean + sigma * std


def to_uint8_frame(frame) -> np.ndarray:
    """Convert one frame to uint8 using robust 0-1 display scaling."""

    scaled = scale_to_0_1(frame, percentiles=(1, 99.5))
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def to_uint8_frames(frames) -> np.ndarray:
    """Convert a frame stack to uint8 frame-by-frame."""

    arr = np.asarray(frames)
    if arr.ndim < 3:
        return to_uint8_frame(arr)
    return np.asarray([to_uint8_frame(frame) for frame in arr])


__all__ = [
    "NormalizeData",
    "finite_values",
    "normalize_data",
    "percentile_limits",
    "scale_to_0_1",
    "sigma_limits",
    "smooth_curve",
    "to_uint8_frame",
    "to_uint8_frames",
]
