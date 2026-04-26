# sci-viz-utils

Shared lightweight scientific visualization and utility helpers used by
AFM-tools, XRD-utils, PlumeDynamics, RHEED-tools, and PFW-Analysis.

This package owns generic foundations only: figure layout, image grids,
scale bars, colorbar formatting, simple array normalization, lightweight HDF5
inspection, file discovery, and video writing. Domain-specific analysis and
domain-specific plotting entry points stay in their original packages.

Install locally while developing the surrounding packages:

```bash
pip install -e .
```

Import examples:

```python
from sci_viz_utils.figures import layout_fig, show_images, save_figure
from sci_viz_utils.arrays import normalize_data
```
