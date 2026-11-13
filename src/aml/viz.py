"""One quiet chart style for every figure (matplotlib, print-friendly)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # scripts render off-screen; notebooks keep their inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Reference categorical palette, fixed order (validated for colour-vision deficiency).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
NORMAL, LAUNDERING = BLUE, ORANGE  # the two classes keep these colours in every chart
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e0", "#ffffff"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, NEUTRAL, RED])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": TEXT,
            "axes.labelcolor": TEXT2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": TEXT,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "axes.prop_cycle": cycler(color=SERIES),
        }
    )


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def note(ax: plt.Axes, text: str, xy: tuple[float, float] = (0.0, -0.22)) -> None:
    """Small secondary-ink note under an axis (source / reading hint)."""
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=7.5, color=TEXT2, ha="left", va="top")
