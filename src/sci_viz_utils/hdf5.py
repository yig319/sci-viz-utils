"""Generic HDF5 inspection and small dataset-loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np


def _h5py():
    try:
        import h5py

        return h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install sci-viz-utils[hdf5] or h5py to use HDF5 helpers.") from exc


def print_tree(parent, indent: int = 0) -> None:
    """Print a simple HDF5 group/dataset tree from an open HDF5 group/file."""

    prefix = "  " * indent
    for key, item in parent.items():
        print(f"{prefix}{key}")
        if hasattr(item, "items"):
            print_tree(item, indent + 1)


def get_tree(parent) -> list[str]:
    """Return all HDF5 object paths under an open HDF5 group/file."""

    paths: list[str] = []

    def visitor(name, _obj):
        paths.append(name)

    parent.visititems(visitor)
    return paths


def print_h5_structure(
    path: str | Path,
    max_depth: int | None = None,
    show_attrs: bool = True,
    show_sample: bool = False,
    sample_items: int = 5,
) -> None:
    """Print detailed structure of an HDF5 file without loading full datasets."""

    h5py = _h5py()
    path = Path(path)

    def format_attrs(obj: Any, indent: str) -> None:
        if not show_attrs or not obj.attrs:
            return
        print(f"{indent}  attrs:")
        for key, value in obj.attrs.items():
            print(f"{indent}    {key}: {value}")

    def format_dataset(name: str, dataset: Any, indent: str) -> None:
        print(
            f"{indent}{name} [Dataset] "
            f"shape={dataset.shape}, dtype={dataset.dtype}, size={dataset.size}"
        )
        if dataset.chunks is not None:
            print(f"{indent}  chunks: {dataset.chunks}")
        if dataset.compression is not None:
            print(f"{indent}  compression: {dataset.compression}")
        format_attrs(dataset, indent)
        if show_sample and dataset.size > 0:
            try:
                sample = dataset[()] if dataset.shape == () else np.asarray(dataset).reshape(-1)[:sample_items]
                print(f"{indent}  sample: {sample}")
            except Exception as exc:
                print(f"{indent}  sample: <could not read: {exc}>")

    def walk_group(group: Any, indent_level: int = 0) -> None:
        if max_depth is not None and indent_level > max_depth:
            return
        indent = "  " * indent_level
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                print(f"{indent}{key}/ [Group]")
                format_attrs(item, indent)
                walk_group(item, indent_level + 1)
            elif isinstance(item, h5py.Dataset):
                format_dataset(key, item, indent)
            else:
                print(f"{indent}{key} [{type(item).__name__}]")

    with h5py.File(path, "r") as h5:
        print(f"HDF5 file: {path}")
        print("=" * 80)
        format_attrs(h5, "")
        walk_group(h5)


def get_h5_paths(path: str | Path) -> list[str]:
    """Return all HDF5 object paths in a file."""

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        return get_tree(h5)


def list_h5_keys(path: str | Path, group: str | None = None) -> list[str]:
    """Return keys at root or inside a specific group."""

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        target = h5[group] if group else h5
        return list(target.keys())


def load_h5_dataset(path: str | Path, dataset: str) -> np.ndarray:
    """Load one HDF5 dataset into a NumPy array."""

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"Dataset not found: {dataset}")
        return np.asarray(h5[dataset])


def load_h5_group_dataset(
    path: str | Path,
    group_name: str,
    dataset_name: str,
    process_func=None,
) -> np.ndarray:
    """Load ``group_name/dataset_name`` from an HDF5 file.

    Domain packages should wrap this with domain-specific names when a group or
    dataset convention has scientific meaning.
    """

    data = load_h5_dataset(path, f"{group_name}/{dataset_name}")
    return process_func(data) if process_func else data


# ---------------------------------------------------------------------------
# Deprecated / relocated helpers — kept as shims for one release cycle
# ---------------------------------------------------------------------------


def _load_h5_frames_impl(path: str | Path, dataset: str | None = None) -> np.ndarray:
    """Internal implementation retained for the ``load_h5_frames`` deprecation shim."""

    h5py = _h5py()
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if dataset is None:
            candidates: list[str] = []

            def visitor(name, obj):
                if hasattr(obj, "shape") and len(obj.shape) >= 3:
                    candidates.append(name)

            h5.visititems(visitor)
            if not candidates:
                raise ValueError(f"No 3D or 4D dataset found in {path}")
            dataset = candidates[0]
        frames = np.asarray(h5[dataset])
    if frames.ndim == 4 and frames.shape[-1] in (3, 4):
        frames = frames[..., :3].mean(axis=-1)
    return frames


def load_h5_frames(path: str | Path, dataset: str | None = None) -> np.ndarray:
    """Deprecated — use :func:`plume_dynamics.io.load_h5_frames` instead."""

    warnings.warn(
        "sci_viz_utils.hdf5.load_h5_frames is deprecated; use "
        "plume_dynamics.io.load_h5_frames for frame-like HDF5 datasets.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_h5_frames_impl(path, dataset)


def _check_fragmentation_impl(filename, group_name="PLD_Plumes"):
    """Internal implementation retained for the ``check_fragmentation`` deprecation shim."""

    h5py = _h5py()
    with h5py.File(filename, "r") as handle:
        total_size = 0
        allocated_size = 0
        for obj in handle[group_name].values():
            if isinstance(obj, h5py.Dataset):
                total_size += obj.size * obj.dtype.itemsize
                allocated_size += obj.id.get_storage_size()
    if allocated_size == 0:
        return 0.0
    return (allocated_size - total_size) / allocated_size * 100


def check_fragmentation(filename, group_name="PLD_Plumes"):
    """Deprecated — use :func:`plume_dynamics.io.check_fragmentation` instead."""

    warnings.warn(
        "sci_viz_utils.hdf5.check_fragmentation is deprecated; use "
        "plume_dynamics.io.check_fragmentation for HDF5 storage analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _check_fragmentation_impl(filename, group_name)


__all__ = [
    "get_h5_paths",
    "get_tree",
    "list_h5_keys",
    "load_h5_dataset",
    "load_h5_group_dataset",
    "print_h5_structure",
    "print_tree",
]
