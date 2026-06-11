import warnings

import h5py
import numpy as np

from sci_viz_utils.hdf5 import load_h5_dataset, load_h5_group_dataset


def test_load_h5_group_dataset_loads_group_dataset(tmp_path):
    path = tmp_path / "example.h5"
    data = np.arange(6).reshape(2, 3)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("group")
        group.create_dataset("data", data=data)

    loaded = load_h5_group_dataset(path, "group", "data")

    np.testing.assert_array_equal(loaded, data)


def test_load_h5_group_dataset_applies_process_func(tmp_path):
    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("group")
        group.create_dataset("data", data=np.array([1, 2, 3]))

    loaded = load_h5_group_dataset(path, "group", "data", process_func=lambda values: values * 10)

    np.testing.assert_array_equal(loaded, np.array([10, 20, 30]))


def test_load_h5_dataset(tmp_path):
    path = tmp_path / "example.h5"
    data = np.arange(12).reshape(3, 4)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frames", data=data)

    loaded = load_h5_dataset(path, "frames")

    np.testing.assert_array_equal(loaded, data)


def test_load_h5_dataset_raises_keyerror_for_missing(tmp_path):
    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("exists", data=np.array([1, 2, 3]))

    import pytest

    with pytest.raises(KeyError, match="Dataset not found: missing"):
        load_h5_dataset(path, "missing")


def test_list_h5_keys_at_root(tmp_path):
    from sci_viz_utils.hdf5 import list_h5_keys

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("group_a")
        h5.create_group("group_b")
        h5.create_dataset("scalar", data=np.array(42))

    keys = list_h5_keys(path)

    assert "group_a" in keys
    assert "group_b" in keys
    assert "scalar" in keys


def test_list_h5_keys_inside_group(tmp_path):
    from sci_viz_utils.hdf5 import list_h5_keys

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("PLD_Plumes")
        group.create_dataset("1-SrRuO3", data=np.array([1, 2, 3]))

    keys = list_h5_keys(path, group="PLD_Plumes")

    assert "1-SrRuO3" in keys


def test_get_tree(tmp_path):
    from sci_viz_utils.hdf5 import get_tree

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("root_data", data=np.array([1]))
        group = h5.create_group("group")
        group.create_dataset("nested", data=np.array([2, 3]))

    with h5py.File(path, "r") as h5:
        paths = get_tree(h5)

    assert "root_data" in paths
    assert "group/nested" in paths


def test_get_h5_paths(tmp_path):
    from sci_viz_utils.hdf5 import get_h5_paths

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("root_data", data=np.array([1]))
        group = h5.create_group("group")
        group.create_dataset("nested", data=np.array([2, 3]))

    paths = get_h5_paths(path)

    assert "root_data" in paths
    assert "group/nested" in paths


def test_print_h5_structure_does_not_crash(tmp_path, capsys):
    from sci_viz_utils.hdf5 import print_h5_structure

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("group")
        group.create_dataset("data", data=np.arange(6).reshape(2, 3))

    print_h5_structure(path, show_sample=True)
    captured = capsys.readouterr()

    assert "HDF5 file:" in captured.out
    assert "group/" in captured.out
    assert "data" in captured.out


def test_print_tree(tmp_path, capsys):
    from sci_viz_utils.hdf5 import print_tree

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("group_a")
        h5.create_group("group_b")

    with h5py.File(path, "r") as h5:
        print_tree(h5)

    captured = capsys.readouterr()
    assert "group_a" in captured.out
    assert "group_b" in captured.out


# ---------------------------------------------------------------------------
# Deprecation shim tests — confirm old names still work and emit warnings
# ---------------------------------------------------------------------------


def test_deprecated_load_h5_frames_still_works_and_warns(tmp_path):
    from sci_viz_utils.hdf5 import load_h5_frames

    path = tmp_path / "example.h5"
    data = np.zeros((2, 3, 4, 5), dtype=float)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frames", data=data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_h5_frames(path, dataset="frames")

    np.testing.assert_array_equal(loaded, data)
    assert any(item.category is DeprecationWarning for item in caught)


def test_deprecated_check_fragmentation_still_works_and_warns(tmp_path):
    from sci_viz_utils.hdf5 import check_fragmentation

    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("PLD_Plumes")
        group.create_dataset("data", data=np.arange(100, dtype=np.float64))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_fragmentation(path, group_name="PLD_Plumes")

    assert isinstance(result, float)
    assert any(item.category is DeprecationWarning for item in caught)


def test_deprecated_names_not_in_all():
    from sci_viz_utils import hdf5

    assert "load_h5_frames" not in hdf5.__all__
    assert "check_fragmentation" not in hdf5.__all__
    assert "load_plumes" not in hdf5.__all__
    assert "load_h5_examples" not in hdf5.__all__
    assert "show_h5_dataset_name" not in hdf5.__all__


def test_removed_loaders_are_gone():
    import pytest

    with pytest.raises(ImportError, match="cannot import name 'load_plumes'"):
        from sci_viz_utils.hdf5 import load_plumes  # noqa: F401

    with pytest.raises(ImportError, match="cannot import name 'load_h5_examples'"):
        from sci_viz_utils.hdf5 import load_h5_examples  # noqa: F401

    with pytest.raises(ImportError, match="cannot import name 'show_h5_dataset_name'"):
        from sci_viz_utils.hdf5 import show_h5_dataset_name  # noqa: F401
