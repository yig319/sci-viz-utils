# sci-viz-utils Usage Guide

`sci-viz-utils` is responsible for shared, domain-neutral notebook utilities:
array display scaling, HDF5 inspection/loading, Matplotlib layout, path helpers,
and lightweight video I/O. AFM, XRD, plume, and RHEED-specific interpretation
should live in the domain packages, not here.

## Install For Development

```bash
cd sci-viz-utils
python -m pip install -e ".[dev,hdf5,video,notebook]"
```

## Workflow: Scale Arrays For Display

Main entry points: `scale_to_0_1`, `to_uint8_frame`, `to_uint8_frames`.
Helpers: `finite_values`, `percentile_limits`, `sigma_limits`, `smooth_curve`.

Use these when a notebook needs robust display contrast or video-compatible
8-bit frames. Inputs are NumPy-like arrays. Outputs keep the same image/frame
shape, except `finite_values` returns a 1D finite-only vector. These functions
are for visualization; do not use them as irreversible scientific preprocessing.

```python
from sci_viz_utils.arrays import scale_to_0_1, to_uint8_frames, smooth_curve

display = scale_to_0_1(image, percentiles=(1, 99.5))
movie_u8 = to_uint8_frames(frames)
smoothed_trace = smooth_curve(raw_trace, window_size=7)
```

Common tuning: use tighter percentiles for noisy images, `sigma_limits` when a
mean-centered signal is more meaningful than percentile contrast, and small
`smooth_curve` windows when cycle edges or plume fronts matter.

## Workflow: Inspect And Load HDF5 Data

Main entry points: `print_h5_structure`, `get_h5_paths`, `list_h5_keys`,
`load_h5_dataset`, `load_h5_group_dataset`, `load_h5_frames`.
Helper: `show_h5_dataset_name` for quick legacy notebook inspection.

Use `print_h5_structure`, `get_h5_paths`, or `list_h5_keys` first when the internal HDF5 structure is
unknown. Use `print_tree` and `get_tree` when you already have an open HDF5 group/file. Use `load_h5_dataset(path, "group/name")` for an exact dataset path.
Use `load_h5_group_dataset(path, group_name, dataset_name)` when a file is
organized as named datasets inside one group. Use `load_h5_frames` when you want
the first 3D/4D frame-like dataset or already know the dataset path.

```python
import h5py
from sci_viz_utils.hdf5 import print_h5_structure, get_h5_paths, load_h5_dataset, load_h5_group_dataset, load_h5_frames

print_h5_structure("experiment.h5", max_depth=2)
paths = get_h5_paths("experiment.h5")

trace = load_h5_dataset("experiment.h5", "signals/intensity")
stack = load_h5_frames("experiment.h5", dataset="frames")
subset = load_h5_group_dataset("experiment.h5", "examples", "frame_stack", process_func=lambda x: x[:10])
```

Compatibility note: `load_plumes` and `load_h5_examples` remain temporarily in
`sci_viz_utils.hdf5` as deprecated wrappers. New plume work should use
`plume_dynamics.io.load_plumes` or `plume_dynamics.io.load_plume_stack`.
Generic grouped HDF5 work should use `load_h5_group_dataset`.

`check_fragmentation(filename, group_name)` estimates storage overhead for all
datasets under one HDF5 group. The default group name is kept for old PLD files,
but the helper is generic.

## Workflow: Build Figures And Image Panels

Main entry points: `layout_fig`, `create_axes_grid`, `show_images`,
`show_image_grid`, `plot_image_map`.
Helpers: `set_style`, `save_figure`, `trim_axes`, `label_panel`, `number_to_letters`,
`scalebar`, `set_cbar`, `set_axis_labels`, `imshow_percentile`.

Use `layout_fig(1)` for one axis, `create_axes_grid` for repeated panels, and
`show_images` for quick notebook image grids. `save_figure` creates parent
folders, so notebooks can save directly into project figure folders.

```python
from sci_viz_utils.figures import set_style, layout_fig, imshow_percentile, scalebar, label_panel, save_figure

set_style()
fig, ax = layout_fig(1)
imshow_percentile(ax, image, percentiles=(1, 99), cmap="magma")
scalebar(ax, image_size=10, scale_size=2, units="um", pixel_size=image.shape[1])
label_panel(ax, 0)
save_figure(fig, "figures/image_panel.png")
```

Common failure modes: `set_cbar` expects an axis with an image/collection; call it
after plotting. Scale bars need a physical image size and the image width in
pixels so the drawn bar has the right length.

## Workflow: Find Files And Make Output Folders

Main entry points: `find_repo_root`, `ensure_dir`, `list_files`.

Use `find_repo_root()` in notebooks that may run from different working
directories. Use `ensure_dir()` before writing outputs. Use `list_files()` with
one or more glob patterns for reproducible data discovery.

```python
from sci_viz_utils.paths import find_repo_root, ensure_dir, list_files

repo = find_repo_root()
figures = ensure_dir(repo / "figures")
h5_files = list_files(repo.parent / "data", ["*.h5", "*.hdf5"])
```

## Workflow: Read And Write Videos

Main entry points: `iter_video_frames`, `write_video`, `make_video`.

Use `iter_video_frames` when a movie is large and you only need sampled frames.
Use `write_video` for a ready frame stack. Use `make_video` when you need a
Matplotlib-rendered comparison video with titles or shared color limits.

```python
from sci_viz_utils.video import iter_video_frames, write_video

frames = list(iter_video_frames("movie.mp4", every=20, max_frames=50, gray=True))
write_video(frames, "preview.mp4", fps=10)
```

## Function Map

This compact map is for lookup after you know the workflow you need.

### `sci_viz_utils.arrays`
Functions: `finite_values(data)`, `normalize_data(data)`, `NormalizeData(data)`, `scale_to_0_1(data, percentiles=None)`, `smooth_curve(data, window_size)`, `percentile_limits(data, percentiles=(1, 99))`, `sigma_limits(data, sigma=3.0)`, `to_uint8_frame(frame)`, `to_uint8_frames(frames)`

### `sci_viz_utils.figures`
Functions: `set_style(name='default')`, `save_figure(fig, path, **kwargs)`, `trim_axes(axs, n_axes)`, `layout_fig(graph=1, mod=None, figsize=None, subplot_style='subplots', spacing=(0.3, 0.3), parent_ax=None, layout='compressed', **kwargs)`, `make_figure_grid(n_plots, *, columns=None, figsize=None, layout='compressed')`, `create_axes_grid(n_plots, n_per_row, plot_height, n_rows=None, figsize='auto')`, `number_to_letters(number)`, `label_panel(ax, number=None, *, style='wb', loc='tl', prefix='', string_add='', size=8, text_pos='center', inset_fraction=(0.15, 0.15), **kwargs)`, `labelfigs(ax, number=None, **kwargs)`, `scalebar(ax, image_size, scale_size, units='', loc='br', pixel_size=None, color='white', linewidth=0, text_color=None, text_offset=0.35, text_position='above', **kwargs)`, `add_scalebar(ax, image_size, scale_size, *, units='nm', loc='br', color='white', linewidth=2.0, **kwargs)`, `set_axis_labels(ax, *, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', logscale=False, legend=None, ticks_both_sides=True, show_ticks=True, label_fontsize=None, title_fontsize=None, ticklabel_fontsize=None, scientific_notation_fontsize=None, tick_padding=10, legend_fontsize=8, legend_loc='best')`, `set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, **kwargs)`, `set_cbar(fig, ax, cbar_label=None, scientific_notation=True, logscale=False, tick_in=True, ticklabel_fontsize=10, labelpad=4, fontsize=10)`, `imshow_percentile(ax, image, percentiles=(1, 99), cmap='viridis', colorbar=True, **kwargs)`, `plot_image_map(ax, data, *, colorbar=True, clim=None, cbar_number_format='%.1e', cmap='viridis')`, `show_image_grid(images, *, labels=None, images_per_row=8, image_height=1.0, show_colorbar=False, clim_sigma=3.0, clim=None, cmap='viridis', scale_0_1=False, scale_range=False, hist_bins=None, show_axis=False, title=None, fig=None, axes=None)`, `show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, clim='auto', cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, fig=None, axes=None, save_path=None)`, `to_scientific_10_power_format(value)`, `label_violinplot(ax, data, label_type='average', text_pos='center', value_format='float', text_size=14, offset_parms=None)`, `evaluate_image_histogram(image, outlier_std=3)`

### `sci_viz_utils.hdf5`
Functions: `print_tree(parent, indent=0)`, `get_tree(parent)`, `print_h5_structure(path, max_depth=None, show_attrs=True, show_sample=False, sample_items=5)`, `get_h5_paths(path)`, `list_h5_keys(path, group=None)`, `show_h5_dataset_name(ds_path, class_name=None)`, `load_h5_dataset(path, dataset)`, `load_h5_group_dataset(path, group_name, dataset_name, process_func=None)`, `load_h5_frames(path, dataset=None)`, `load_plumes(ds_path, class_name, ds_name, process_func=None)`, `load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True)`, `check_fragmentation(filename, group_name='PLD_Plumes')`

### `sci_viz_utils.paths`
Functions: `find_repo_root(start=None)`, `ensure_dir(path)`, `list_files(root, patterns, recursive=True)`

### `sci_viz_utils.video`
Functions: `iter_video_frames(path, every=1, max_frames=None, gray=True)`, `write_video(frames, output, fps=10, cmap='gray')`, `make_video(image_sequences, titles=None, output='video.mp4', fps=5, cmap='viridis', clim='auto')`
