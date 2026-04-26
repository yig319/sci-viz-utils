# sci-viz-utils Usage Guide

`sci-viz-utils` contains generic scientific plotting and utility helpers used
by the AFM, XRD, plume, RHEED, and PFW analysis packages. Keep domain-specific
analysis in those domain packages; use this package for shared foundations.

## Install For Development

```bash
git clone https://github.com/yig319/sci-viz-utils.git
cd sci-viz-utils
python -m pip install -e ".[dev,hdf5,video,notebook]"
```

When working in the local `Pypi_Packages` workspace, install it before the
domain packages:

```bash
python -m pip install -e ../sci-viz-utils
```

## Figure Helpers

```python
import numpy as np
import matplotlib.pyplot as plt
from sci_viz_utils.figures import layout_fig, scalebar, label_panel, save_figure

image = np.random.random((64, 64))
fig, ax = layout_fig(1)
ax.imshow(image, cmap="magma")
scalebar(ax, image_size=10, scale_size=2, units="um", pixel_size=image.shape[1])
label_panel(ax, 0)
save_figure(fig, "figures/example.png")
```

## Image Grids

```python
from sci_viz_utils.figures import show_images

fig, axes = show_images(
    [np.random.random((32, 32)) for _ in range(4)],
    labels="index",
    img_per_row=4,
    show_colorbar=True,
    clim=3,
)
```

## Arrays, HDF5, And Video

```python
from sci_viz_utils.arrays import normalize_data
from sci_viz_utils.hdf5 import load_h5_frames
from sci_viz_utils.video import write_video

scaled = normalize_data(image)
frames = load_h5_frames("movie.h5")
write_video(frames[:20], "preview.mp4", fps=10)
```

## Release

Push to `main` with `#patch`, `#minor`, or `#major` in the commit message to
trigger the GitHub Actions release workflow and PyPI publish step.
