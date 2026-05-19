"""Shared matplotlib style for publication-quality figures."""
import matplotlib as mpl

# Wong (2011) colorblind-safe palette
BLUE      = "#0072B2"
ORANGE    = "#E69F00"
GREEN     = "#009E73"
VERMILION = "#D55E00"
SKY_BLUE  = "#56B4E9"
PURPLE    = "#CC79A7"

STRATEGY_COLORS = {
    "random":       "#888888",
    "uncertainty":  BLUE,
    "qbc":          ORANGE,
    "hybrid":       GREEN,
    "neural_score": VERMILION,
    "standard_rf":  BLUE,
    "pu_bagging":   GREEN,
}

MODEL_COLORS = [BLUE, ORANGE, GREEN]


def apply_pub_style() -> None:
    """Apply publication-quality rcParams. Call once before any figure is created."""
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   12,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.8,
        "patch.linewidth":    0.5,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
