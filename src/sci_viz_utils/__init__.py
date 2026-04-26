"""Shared lightweight scientific visualization and utility helpers."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("sci-viz-utils")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from .arrays import (
    NormalizeData,
    finite_values,
    normalize_data,
    percentile_limits,
    scale_to_0_1,
    sigma_limits,
    smooth_curve,
    to_uint8_frame,
    to_uint8_frames,
)
from .figures import (
    add_scalebar,
    create_axes_grid,
    evaluate_image_histogram,
    imshow_percentile,
    label_panel,
    label_violinplot,
    labelfigs,
    layout_fig,
    make_figure_grid,
    number_to_letters,
    plot_image_map,
    save_figure,
    scalebar,
    set_axis_labels,
    set_cbar,
    set_labels,
    set_style,
    show_image_grid,
    show_images,
    to_scientific_10_power_format,
    trim_axes,
)
from .paths import ensure_dir, find_repo_root, list_files

__all__ = [
    "NormalizeData",
    "add_scalebar",
    "create_axes_grid",
    "ensure_dir",
    "evaluate_image_histogram",
    "finite_values",
    "find_repo_root",
    "imshow_percentile",
    "label_panel",
    "label_violinplot",
    "labelfigs",
    "layout_fig",
    "list_files",
    "make_figure_grid",
    "normalize_data",
    "number_to_letters",
    "percentile_limits",
    "plot_image_map",
    "save_figure",
    "scale_to_0_1",
    "scalebar",
    "set_axis_labels",
    "set_cbar",
    "set_labels",
    "set_style",
    "show_image_grid",
    "show_images",
    "sigma_limits",
    "smooth_curve",
    "to_scientific_10_power_format",
    "to_uint8_frame",
    "to_uint8_frames",
    "trim_axes",
]
