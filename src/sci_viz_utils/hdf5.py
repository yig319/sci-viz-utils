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


def show_h5_dataset_name(ds_path, class_name=None):
    """Print top-level keys or keys under ``class_name`` in one HDF5 file."""

    print(list_h5_keys(ds_path, group=class_name))


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


def load_h5_frames(path: str | Path, dataset: str | None = None) -> np.ndarray:
    """Load the first 3D/4D frame-like HDF5 dataset, or a named dataset."""

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


def load_plumes(ds_path, class_name, ds_name, process_func=None):
    """Deprecated compatibility wrapper for older plume notebooks."""

    warnings.warn(
        "sci_viz_utils.hdf5.load_plumes is deprecated; use "
        "plume_dynamics.io.load_plumes for plume data or "
        "sci_viz_utils.hdf5.load_h5_group_dataset for generic grouped datasets.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_h5_group_dataset(ds_path, class_name, ds_name, process_func=process_func)


def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):
    """Deprecated compatibility wrapper for older example notebooks."""

    warnings.warn(
        "sci_viz_utils.hdf5.load_h5_examples is deprecated; use "
        "load_h5_group_dataset for generic grouped HDF5 datasets or a "
        "domain package loader for domain-specific data.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_h5_group_dataset(ds_path, class_name, ds_name, process_func=process_func)


def check_fragmentation(filename, group_name="PLD_Plumes"):
    """Estimate HDF5 storage overhead for datasets under one group.

    The return value is the percent difference between logical dataset size and
    allocated on-disk size. The default group keeps compatibility with older PLD
    notebooks, but the helper itself is generic.
    """

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


__all__ = [
    "check_fragmentation",
    "get_h5_paths",
    "get_tree",
    "list_h5_keys",
    "load_h5_dataset",
    "load_h5_examples",
    "load_h5_frames",
    "load_h5_group_dataset",
    "load_plumes",
    "print_h5_structure",
    "print_tree",
    "show_h5_dataset_name",
]
