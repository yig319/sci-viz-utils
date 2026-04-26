"""Generic HDF5 inspection and small dataset-loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _h5py():
    try:
        import h5py

        return h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install sci-viz-utils[hdf5] or h5py to use HDF5 helpers.") from exc


def print_tree(parent, indent: int = 0) -> None:
    """Print a simple HDF5 group/dataset tree."""

    prefix = "  " * indent
    for key, item in parent.items():
        print(f"{prefix}{key}")
        if hasattr(item, "items"):
            print_tree(item, indent + 1)


def get_tree(parent) -> list[str]:
    """Return all HDF5 object paths under ``parent``."""

    paths: list[str] = []

    def visitor(name, _obj):
        paths.append(name)

    parent.visititems(visitor)
    return paths


def show_h5_dataset_name(ds_path, class_name=None):
    """Print top-level keys or keys under ``class_name`` in one HDF5 file."""

    h5py = _h5py()
    with h5py.File(ds_path) as hf:
        if class_name:
            print(hf[class_name].keys())
        else:
            print(hf.keys())


def load_h5_dataset(path: str | Path, dataset: str):
    """Load one HDF5 dataset into a NumPy array."""

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[dataset])


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
    """Load a named dataset from ``class_name/ds_name`` and optionally process it."""

    data = load_h5_dataset(ds_path, f"{class_name}/{ds_name}")
    return process_func(data) if process_func else data


def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):
    """Load an HDF5 example dataset.

    The ``show`` argument is accepted for compatibility with older utility
    functions; display behavior belongs in calling visualization code.
    """

    return load_plumes(ds_path, class_name, ds_name, process_func=process_func)


def check_fragmentation(filename, group_name="PLD_Plumes"):
    """Estimate HDF5 dataset fragmentation under one group."""

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
    "get_tree",
    "load_h5_dataset",
    "load_h5_examples",
    "load_h5_frames",
    "load_plumes",
    "print_tree",
    "show_h5_dataset_name",
]
