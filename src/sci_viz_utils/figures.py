"""Generic Matplotlib helpers shared across scientific packages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import patches, patheffects

from .arrays import NormalizeData, percentile_limits, scale_to_0_1, sigma_limits


def set_style(
    name: str = "default",
    *,
    font_size: float | None = None,
    dpi: int | None = None,
    save_dpi: int = 300,
    tick_direction: str = "in",
    ticks_on_all_sides: bool = True,
    show_minor_ticks: bool = True,
    grid: bool = False,
    spine_width: float = 0.8,
    line_width: float = 1.2,
    marker_size: float = 4.0,
    legend_frame: bool = False,
    use_mathtext: bool = True,
) -> None:
    """Apply the shared compact Matplotlib style used by sci-viz packages.

    Parameters
    ----------
    name:
        Named style preset. ``"default"`` is the notebook/analysis style and
        ``"printing"`` uses slightly smaller text with higher on-screen DPI.
    font_size:
        Base font size. When omitted, the selected preset chooses the size.
    dpi, save_dpi:
        Display and saved-figure resolution.
    tick_direction:
        Tick direction passed to Matplotlib. The default ``"in"`` keeps all
        ticks inside the axes.
    ticks_on_all_sides:
        Show x ticks on top and y ticks on the right in addition to bottom/left.
    show_minor_ticks:
        Enable visible minor ticks for line/scatter plots.
    grid:
        Whether axes should show a light grid by default.
    spine_width, line_width, marker_size:
        Baseline widths/sizes for axes, plotted lines, and markers.
    legend_frame:
        Draw legend frames by default.
    use_mathtext:
        Use Matplotlib mathtext for scientific notation offsets.
    """

    presets = {
        "default": {"figure.dpi": 120, "font.size": 10},
        "printing": {"figure.dpi": 150, "font.size": 8},
        "presentation": {"figure.dpi": 120, "font.size": 12},
    }
    if name not in presets:
        raise ValueError(f"Unknown style {name!r}. Choose from {sorted(presets)}.")

    base_font_size = float(font_size if font_size is not None else presets[name]["font.size"])
    figure_dpi = int(dpi if dpi is not None else presets[name]["figure.dpi"])
    label_size = base_font_size
    tick_size = max(base_font_size - 1, 1)
    legend_size = max(base_font_size - 1, 1)
    title_size = base_font_size + 1

    settings = {
        "figure.dpi": figure_dpi,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.dpi": save_dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": base_font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "axes.linewidth": spine_width,
        "axes.grid": grid,
        "axes.axisbelow": True,
        "axes.formatter.use_mathtext": use_mathtext,
        "axes.formatter.limits": (-3, 3),
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": tick_direction,
        "ytick.direction": tick_direction,
        "xtick.top": ticks_on_all_sides,
        "ytick.right": ticks_on_all_sides,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.major.width": spine_width,
        "ytick.major.width": spine_width,
        "xtick.minor.width": spine_width * 0.8,
        "ytick.minor.width": spine_width * 0.8,
        "xtick.minor.visible": show_minor_ticks,
        "ytick.minor.visible": show_minor_ticks,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,
        "patch.linewidth": spine_width,
        "legend.fontsize": legend_size,
        "legend.frameon": legend_frame,
        "legend.borderaxespad": 0.4,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "image.cmap": "viridis",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
    }
    plt.rcParams.update(settings)


def save_figure(fig, path: str | Path, **kwargs) -> Path:
    """Save a Matplotlib figure, creating the parent folder if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches=kwargs.pop("bbox_inches", "tight"), **kwargs)
    return path


def trim_axes(axs, n_axes: int):
    """Remove extra axes from an axes array and return the kept axes."""

    flat_axes = np.asarray(axs, dtype=object).reshape(-1)
    for ax in flat_axes[n_axes:]:
        ax.remove()
    return flat_axes[:n_axes]


def _default_columns(n_plots: int) -> int:
    if n_plots < 3:
        return 2
    if n_plots < 5:
        return 3
    if n_plots < 10:
        return 4
    if n_plots < 17:
        return 5
    if n_plots < 26:
        return 6
    return 7


def layout_fig(
    graph=1,
    mod=None,
    figsize=None,
    subplot_style="subplots",
    spacing=(0.3, 0.3),
    parent_ax=None,
    layout="compressed",
    **kwargs,
):
    """Create a flexible grid of Matplotlib axes.

    Returns one axis for ``graph == 1`` and a flat axes array otherwise. The
    signature intentionally covers the old AFM, Plume, and RHEED helpers.
    """

    graph = max(int(graph), 1)
    mod = _default_columns(graph) if mod is None else max(int(mod), 1)
    nrows = int(math.ceil(graph / mod))
    wspace, hspace = spacing

    if figsize is None:
        figsize = (3 * mod, 3 * nrows)
    elif isinstance(figsize, tuple) and (figsize[0] is None or figsize[1] is None):
        width, height = figsize
        unit_w = kwargs.pop("unit_w", 3)
        unit_h = kwargs.pop("unit_h", 3)
        figsize = (
            width if width is not None else unit_w * mod,
            height if height is not None else unit_h * nrows,
        )

    if parent_ax is not None:
        from matplotlib.gridspec import GridSpec

        fig = parent_ax.figure
        bbox = parent_ax.get_position()
        grid = GridSpec(
            nrows,
            mod,
            figure=fig,
            left=bbox.x0,
            bottom=bbox.y0,
            right=bbox.x1,
            top=bbox.y1,
            wspace=wspace,
            hspace=hspace,
        )
        axes = np.asarray([fig.add_subplot(grid[i // mod, i % mod]) for i in range(graph)])
        return None, axes[0] if graph == 1 else axes

    if subplot_style == "gridspec":
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize)
        grid = GridSpec(nrows, mod, figure=fig, wspace=wspace, hspace=hspace)
        axes = np.asarray([fig.add_subplot(grid[i // mod, i % mod]) for i in range(graph)])
    elif subplot_style == "subplots":
        try:
            fig, axs = plt.subplots(nrows, mod, figsize=figsize, squeeze=False, layout=layout)
        except TypeError:
            fig, axs = plt.subplots(nrows, mod, figsize=figsize, squeeze=False)
        axes = trim_axes(axs, graph)
        if layout is None:
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
    else:
        raise ValueError("subplot_style must be either 'subplots' or 'gridspec'.")

    return fig, axes[0] if graph == 1 else axes


def make_figure_grid(
    n_plots: int,
    *,
    columns: int | None = None,
    figsize: tuple[float, float] | None = None,
    layout: str = "compressed",
):
    """Create a compact Matplotlib figure grid for a known number of panels."""

    if n_plots <= 0:
        raise ValueError("n_plots must be > 0")
    columns = _default_columns(n_plots) if columns is None else int(columns)
    rows = int(math.ceil(n_plots / columns))
    if figsize is None:
        figsize = (3.0 * columns, 3.0 * rows)
    try:
        fig, axs = plt.subplots(rows, columns, figsize=figsize, layout=layout)
    except TypeError:
        fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axes = trim_axes(axs, n_plots)
    return fig, axes


def create_axes_grid(n_plots, n_per_row, plot_height, n_rows=None, figsize="auto"):
    """Create a grid of axes with Plume/M3-style sizing."""

    if n_rows is None:
        n_rows = int(math.ceil(n_plots / n_per_row))
    if figsize == "auto":
        figsize = (16, plot_height * n_rows + 1)
    elif figsize is not None and not isinstance(figsize, tuple):
        raise ValueError("figsize must be a tuple, None, or 'auto'")
    fig, axs = plt.subplots(n_rows, n_per_row, figsize=figsize, squeeze=False)
    return fig, trim_axes(axs, n_plots)


def number_to_letters(number: int) -> str:
    """Convert zero-based panel number to letters: 0 -> a, 25 -> z, 26 -> aa."""

    if number < 0:
        raise ValueError("number must be non-negative")
    letters = ""
    value = int(number)
    while value >= 0:
        value, remainder = divmod(value, 26)
        letters = chr(97 + remainder) + letters
        value -= 1
    return letters


def label_panel(
    ax,
    number: int | None = None,
    *,
    style: str = "wb",
    loc: str = "tl",
    prefix: str = "",
    string_add: str = "",
    size: float = 8,
    text_pos: str = "center",
    inset_fraction: tuple[float, float] = (0.15, 0.15),
    **kwargs,
):
    """Add a small panel label to an axis."""

    if kwargs.pop("add_label", True) is False:
        return None

    formatting_key = {
        "wb": dict(color="w", linewidth=0.75, foreground="k"),
        "b": dict(color="k", linewidth=0.0, foreground="k"),
        "w": dict(color="w", linewidth=0.0, foreground="w"),
        "bw": dict(color="k", linewidth=0.75, foreground="w"),
    }
    if style not in formatting_key:
        raise ValueError(f"style must be one of {sorted(formatting_key)}")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]
    locs = {
        "tl": (xlim[0] + x_inset, ylim[1] - y_inset),
        "tr": (xlim[1] - x_inset, ylim[1] - y_inset),
        "bl": (xlim[0] + x_inset, ylim[0] + y_inset),
        "br": (xlim[1] - x_inset, ylim[0] + y_inset),
        "ct": ((xlim[0] + xlim[1]) / 2, ylim[1] - y_inset),
        "cb": ((xlim[0] + xlim[1]) / 2, ylim[0] + y_inset),
    }
    if loc not in locs:
        raise ValueError("loc must be one of: tl, tr, bl, br, ct, cb.")

    text = prefix + string_add
    if number is not None:
        text += number_to_letters(number)
    formatting = formatting_key[style]
    artist = ax.text(
        *locs[loc],
        text,
        va=text_pos,
        ha="center",
        path_effects=[patheffects.withStroke(linewidth=formatting["linewidth"], foreground=formatting["foreground"])],
        color=formatting["color"],
        size=size,
        **kwargs,
    )
    artist.set_zorder(np.inf)
    return artist


def labelfigs(ax, number=None, **kwargs):
    """Backward-compatible alias for :func:`label_panel`."""

    return label_panel(ax, number=number, **kwargs)


def scalebar(
    ax,
    image_size: float,
    scale_size: float,
    units: str = "",
    loc: str = "br",
    pixel_size: float | None = None,
    color: str = "white",
    linewidth: float = 0,
    text_color: str | None = None,
    text_offset: float = 0.35,
    text_position: str = "above",
    **kwargs,
):
    """Add a scale bar patch and label to an image axis.

    ``image_size`` and ``scale_size`` are physical lengths in the same units.
    ``pixel_size`` is the image width in display pixels.
    """

    valid_locs = {"br", "bl", "tr", "tl"}
    if loc not in valid_locs:
        raise ValueError(f"loc must be one of {sorted(valid_locs)}")
    if text_position not in {"above", "below"}:
        raise ValueError("text_position must be 'above' or 'below'")
    if image_size <= 0 or scale_size <= 0:
        raise ValueError("image_size and scale_size must be positive")

    text_fontsize = kwargs.pop("text_fontsize", kwargs.pop("fontsize", 9))
    text_color = text_color or color
    pixel_size = float(pixel_size or abs(ax.get_xlim()[1] - ax.get_xlim()[0]) or image_size)
    bar_px = max(pixel_size * float(scale_size) / float(image_size), 1.0)
    pad = pixel_size * 0.06
    height = max(pixel_size * 0.015, 1.0)

    x0 = pad if "l" in loc else pixel_size - pad - bar_px
    y0 = pad if "t" in loc else pixel_size - pad - height
    patch = patches.Rectangle(
        (x0, y0),
        bar_px,
        height,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        **kwargs,
    )
    ax.add_patch(patch)
    label = f"{float(scale_size):g} {units}".strip()
    if text_position == "above":
        y_text = y0 - pad * text_offset
        va = "bottom"
    else:
        y_text = y0 + height + pad * text_offset
        va = "top"
    ax.text(
        x0 + bar_px / 2,
        y_text,
        label,
        color=text_color,
        ha="center",
        va=va,
        fontsize=text_fontsize,
    )
    return patch


def add_scalebar(
    ax,
    image_size: float,
    scale_size: float,
    *,
    units: str = "nm",
    loc: str = "br",
    color: str = "white",
    linewidth: float = 2.0,
    **kwargs,
):
    """Draw a simple scalebar on an image axis."""

    return scalebar(
        ax,
        image_size=image_size,
        scale_size=scale_size,
        units=units,
        loc=loc,
        color=color,
        linewidth=linewidth,
        **kwargs,
    )


def set_axis_labels(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    yaxis_style: str | None = "sci",
    logscale: bool = False,
    legend: Sequence[str] | bool | None = None,
    ticks_both_sides: bool = True,
    show_ticks: bool = True,
    minor_ticks: bool | None = None,
    label_fontsize: float | None = None,
    title_fontsize: float | None = None,
    ticklabel_fontsize: float | None = None,
    scientific_notation_fontsize: float | None = None,
    tick_padding: float = 10,
    legend_fontsize: float = 8,
    legend_loc: str = "best",
) -> None:
    """Apply common axis labels, limits, scales, legend, and tick styling."""

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if logscale:
        ax.set_yscale("log")
    if yaxis_style == "sci" and ax.get_yscale() != "log":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useLocale=False)
        if scientific_notation_fontsize is not None:
            ax.yaxis.get_offset_text().set_fontsize(scientific_notation_fontsize)
    if legend:
        if legend is True or legend == "auto":
            ax.legend(fontsize=legend_fontsize, loc=legend_loc)
        else:
            ax.legend(legend, fontsize=legend_fontsize, loc=legend_loc)
    major_length = 5 if show_ticks else 0
    x_minor_visible = bool(plt.rcParams["xtick.minor.visible"]) if minor_ticks is None else bool(minor_ticks)
    y_minor_visible = bool(plt.rcParams["ytick.minor.visible"]) if minor_ticks is None else bool(minor_ticks)
    x_minor_length = 2.2 if show_ticks and x_minor_visible else 0
    y_minor_length = 2.2 if show_ticks and y_minor_visible else 0
    x_direction = plt.rcParams.get("xtick.direction", "in")
    y_direction = plt.rcParams.get("ytick.direction", "in")

    ax.tick_params(axis="x", which="major", direction=x_direction, length=major_length, labelsize=ticklabel_fontsize, pad=tick_padding)
    ax.tick_params(axis="y", which="major", direction=y_direction, length=major_length, labelsize=ticklabel_fontsize, pad=tick_padding)
    ax.tick_params(axis="x", which="minor", direction=x_direction, length=x_minor_length)
    ax.tick_params(axis="y", which="minor", direction=y_direction, length=y_minor_length)
    if not x_minor_visible:
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    if not y_minor_visible:
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    if ticks_both_sides:
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")


def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, **kwargs):
    """M3/Plume-compatible wrapper around :func:`set_axis_labels`."""

    return set_axis_labels(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
        **kwargs,
    )


def set_cbar(
    fig,
    ax,
    cbar_label=None,
    scientific_notation=True,
    logscale=False,
    tick_in=True,
    ticklabel_fontsize=10,
    labelpad=4,
    fontsize=10,
):
    """Add a vertical colorbar using the first collection on ``ax``."""

    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="vertical", pad=0.02, shrink=1)
    if scientific_notation:
        formatter = ticker.LogFormatterMathtext(base=10) if logscale else ticker.ScalarFormatter(useMathText=True)
        if not logscale:
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
        cbar.ax.yaxis.set_major_formatter(formatter)
    if cbar_label:
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.set_label(cbar_label, rotation=0, labelpad=labelpad, fontsize=fontsize)
        cbar.ax.yaxis.set_label_coords(1.5, -0.04)
    cbar.ax.tick_params(direction="in" if tick_in else "out", labelsize=ticklabel_fontsize, which="both")
    return cbar


def imshow_percentile(ax, image, percentiles=(1, 99), cmap="viridis", colorbar=True, **kwargs):
    """Display an image using finite-data percentile contrast."""

    vmin, vmax = percentile_limits(image, percentiles)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_image_map(
    ax,
    data,
    *,
    colorbar: bool = True,
    clim: tuple[float, float] | None = None,
    cbar_number_format: str = "%.1e",
    cmap: str = "viridis",
):
    """Plot one 2D array with clean image axes and optional colorbar."""

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    arr = np.asarray(data)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side != arr.size:
            raise ValueError("1D data can only be reshaped when its length is a square")
        arr = arr.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array or square-length 1D array")
    im = ax.imshow(arr, cmap=plt.get_cmap(cmap), clim=clim)
    ax.set_yticks([])
    ax.set_xticks([])
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax=cax, format=cbar_number_format)
    return im


def show_image_grid(
    images: Sequence[np.ndarray],
    *,
    labels: Sequence[str] | None = None,
    images_per_row: int = 8,
    image_height: float = 1.0,
    show_colorbar: bool = False,
    clim_sigma: float | Sequence[float] | None = 3.0,
    clim=None,
    cmap: str = "viridis",
    scale_0_1: bool = False,
    scale_range=False,
    hist_bins: int | None = None,
    show_axis: bool = False,
    title: str | None = None,
    fig=None,
    axes=None,
):
    """Display a list of images in a compact notebook grid."""

    arrays = [np.asarray(image) for image in images]
    if not arrays:
        raise ValueError("images must contain at least one image")
    if labels is None:
        labels = [str(i) for i in range(len(arrays))]
    elif isinstance(labels, str) and labels == "index":
        labels = [str(i) for i in range(len(arrays))]

    panel_rows = 2 if hist_bins else 1
    n_axes = len(arrays) * panel_rows
    if axes is None:
        fig, axes = layout_fig(n_axes, mod=images_per_row, figsize=(None, image_height * (len(images)//images_per_row+1) * panel_rows))
    else:
        axes = np.atleast_1d(axes).ravel()
        fig = axes[0].figure if fig is None else fig
    axes = np.atleast_1d(axes).ravel()

    for idx, image in enumerate(arrays):
        image_to_show = image
        if scale_range is True or scale_0_1:
            image_to_show = NormalizeData(image_to_show)
        elif isinstance(scale_range, tuple):
            low, high = scale_range
            image_to_show = NormalizeData(image_to_show) * (high - low) + low

        image_ax = axes[idx * panel_rows]
        if labels and idx < len(labels) and str(labels[idx]):
            image_ax.set_title(str(labels[idx]))
        im = image_ax.imshow(image_to_show, cmap=cmap)
        if clim is not None and clim != "auto":
            if isinstance(clim, list):
                im.set_clim(*sigma_limits(image_to_show, clim[idx]))
            elif isinstance(clim, (int, float)):
                im.set_clim(*sigma_limits(image_to_show, float(clim)))
            elif isinstance(clim, tuple):
                im.set_clim(*clim)
        elif show_colorbar and clim_sigma is not None:
            sigma = clim_sigma[idx] if isinstance(clim_sigma, Sequence) and not isinstance(clim_sigma, str) else clim_sigma
            im.set_clim(*sigma_limits(image_to_show, float(sigma)))
        if show_colorbar:
            fig.colorbar(im, ax=image_ax)
        if show_axis:
            image_ax.tick_params(axis="x", direction="in", top=True)
            image_ax.tick_params(axis="y", direction="in", right=True)
        else:
            image_ax.axis("off")
        if hist_bins:
            axes[idx * panel_rows + 1].hist(np.asarray(image_to_show).reshape(-1), bins=hist_bins)

    if title:
        fig.suptitle(title, fontsize=16, y=1.01)
    fig.tight_layout()
    return fig, axes


def show_images(
    images,
    labels=None,
    img_per_row=8,
    img_height=1,
    label_size=12,
    title=None,
    show_colorbar=False,
    clim="auto",
    cmap="viridis",
    scale_range=False,
    hist_bins=None,
    show_axis=False,
    fig=None,
    axes=None,
    save_path=None,
):
    """M3/Plume-compatible wrapper for :func:`show_image_grid`."""

    figure, axes = show_image_grid(
        images,
        labels=labels,
        images_per_row=img_per_row,
        image_height=img_height,
        show_colorbar=show_colorbar,
        clim=clim,
        cmap=cmap,
        scale_range=scale_range,
        hist_bins=hist_bins,
        show_axis=show_axis,
        title=title,
        fig=fig,
        axes=axes,
    )
    for axis in np.atleast_1d(axes).ravel():
        axis.title.set_fontsize(label_size)
    if save_path:
        save_figure(figure, save_path)
    return figure, axes


def to_scientific_10_power_format(value):
    """Format a number as a Matplotlib mathtext scientific notation string."""

    base, exponent = f"{value:.2e}".split("e")
    return rf"${base}\times10^{{{int(exponent)}}}$"


def label_violinplot(
    ax,
    data,
    label_type="average",
    text_pos="center",
    value_format="float",
    text_size=14,
    offset_parms=None,
):
    """Label violin-plot categories with summary text."""

    offset_parms = offset_parms or {"x_type": "fixed", "x_value": 0.02, "y_type": "fixed", "y_value": 0.02}
    values = getattr(data, "values", data)
    for tick, (value, _label) in enumerate(zip(values, ax.get_xticklabels())):
        if label_type in {"average", "average_value"}:
            if value_format == "int":
                label_text = f"{int(value)}"
            elif value_format == "scientific":
                label_text = to_scientific_10_power_format(value)
            else:
                label_text = f"{float(value):.2f}"
        elif label_type == "total_number":
            label_text = f"n: {len(values)}"
        else:
            continue
        x_offset = tick + offset_parms.get("x_value", 0.02)
        y_offset = value + offset_parms.get("y_value", 0.02)
        ha = "center" if text_pos == "center" else "left"
        # ax.text(tick if text_pos == "center" else x_offset, y_offset, label_text, horizontalalignment=ha, size=text_size, weight="semibold")
        ax.text(tick if text_pos == "center" else x_offset, y_offset, label_text, horizontalalignment=ha, size=text_size)


def evaluate_image_histogram(image, outlier_std=3):
    """Plot a simple clipped image histogram and return ``(fig, ax)``."""

    values = np.asarray(image, dtype=float).reshape(-1)
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    lower_clip = mean_val - outlier_std * std_val
    upper_clip = mean_val + outlier_std * std_val
    clipped = values[(values >= lower_clip) & (values <= upper_clip)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(clipped, bins=100, range=(lower_clip, upper_clip), alpha=0.3, edgecolor="black")
    ax.set_title(f"Image Histogram (removing noise outside +/-{outlier_std} sigma)")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    return fig, ax


__all__ = [
    "add_scalebar",
    "create_axes_grid",
    "evaluate_image_histogram",
    "imshow_percentile",
    "label_panel",
    "label_violinplot",
    "labelfigs",
    "layout_fig",
    "make_figure_grid",
    "number_to_letters",
    "plot_image_map",
    "save_figure",
    "scalebar",
    "set_axis_labels",
    "set_cbar",
    "set_labels",
    "set_style",
    "show_image_grid",
    "show_images",
    "to_scientific_10_power_format",
    "trim_axes",
]


