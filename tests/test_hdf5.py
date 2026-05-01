import warnings

import h5py
import numpy as np

from sci_viz_utils.hdf5 import load_h5_examples, load_h5_group_dataset, load_plumes


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


def test_deprecated_load_plumes_still_works_and_warns(tmp_path):
    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("PLD_Plumes")
        group.create_dataset("1-SrRuO3", data=np.array([1, 2, 3]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_plumes(path, "PLD_Plumes", "1-SrRuO3")

    np.testing.assert_array_equal(loaded, np.array([1, 2, 3]))
    assert any(item.category is DeprecationWarning for item in caught)


def test_deprecated_load_h5_examples_still_works_and_warns(tmp_path):
    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("examples")
        group.create_dataset("frames", data=np.array([4, 5, 6]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_h5_examples(path, "examples", "frames")

    np.testing.assert_array_equal(loaded, np.array([4, 5, 6]))
    assert any(item.category is DeprecationWarning for item in caught)
