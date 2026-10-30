"""Shared chart style for generated report figures (matplotlib object API, light mode).

Reference palette: series 1 = blue, series 2 = orange, series 3 = green; neutral inks and
surfaces. Every figure carries the research-use disclaimer.
"""

from pathlib import Path

from matplotlib.figure import Figure

from vstent import DISCLAIMER

SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_3 = "#1baf7a"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"


def style_axes(ax) -> None:
    """Light surface, horizontal and vertical grid, only the bottom spine."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=7.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def new_figure(width: float, height: float, title: str, subtitle: str) -> Figure:
    """Figure (inches) with title, subtitle and the disclaimer footer."""
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE)
    fig.text(0.07, 0.94, title, fontsize=11, color=INK_PRIMARY)
    fig.text(0.07, 0.895, subtitle, fontsize=8, color=INK_SECONDARY)
    fig.text(0.07, 0.02, DISCLAIMER, fontsize=7, color=INK_MUTED)
    return fig


def save_figure(fig: Figure, path: str | Path) -> None:
    """Write a PNG on the figure surface colour, creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
