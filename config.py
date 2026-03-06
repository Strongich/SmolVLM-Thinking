"""
Configuration file for project paths and publication-quality plotting.

This module defines:
- Directory paths for saving plots and text results
- Matplotlib settings for paper-ready plots (Okabe-Ito colorblind palette,
  clean styling, Type 42 fonts for Illustrator compatibility)
"""

from pathlib import Path

import matplotlib as mpl

# Base project directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory paths
PLOTS_DIR = PROJECT_ROOT / "plots"
SAVED_RESULTS_DIR = PROJECT_ROOT / "saved_results"
TEMP_EVAL_PLOTS_DIR = PLOTS_DIR / "temperature_evaluation"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)
SAVED_RESULTS_DIR.mkdir(exist_ok=True)
TEMP_EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# String versions of paths (for compatibility with older code)
PLOTS_PATH = str(PLOTS_DIR)
SAVED_RESULTS_PATH = str(SAVED_RESULTS_DIR)

# ---------------------------------------------------------------------------
# Training prompts
# ---------------------------------------------------------------------------

R1_STYLE_SYSTEM_PROMPT_COT = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. "
    "The reasoning process is enclosed within <think> </think> tags, "
    "and the final answer follows directly after."
)

R1_STYLE_SYSTEM_PROMPT_OPD = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. "
    "Let's think step by step and output the final answer within \\boxed{}."
)

# ---------------------------------------------------------------------------
# Publication-quality plotting configuration
# ---------------------------------------------------------------------------

PAPER_STYLE = {
    # Figure dimensions optimized for single-column papers
    "figure.figsize": (6.0, 3.6),
    "figure.dpi": 150,
    # High-quality output settings
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.format": "pdf",
    # Clean, minimal styling
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    # Typography optimized for papers
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    # Line and marker styling
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,
    # Vector output compatibility
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

# Colorblind-friendly palette (Okabe-Ito scheme)
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion (red-orange)
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#000000",  # Black
]

COLOR_CYCLE = mpl.cycler(color=COLORBLIND_COLORS)

COMPLETE_STYLE = {
    **PAPER_STYLE,
    "axes.prop_cycle": COLOR_CYCLE,
}


def apply_paper_style(font_scale: float = 1.0):
    """Apply the paper-ready matplotlib style."""
    style = dict(COMPLETE_STYLE)
    if font_scale != 1.0:
        for key in (
            "axes.labelsize",
            "axes.titlesize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        ):
            style[key] = style[key] * font_scale
    mpl.rcParams.update(style)


def get_color(index: int) -> str:
    """Get a colorblind-friendly color by index (cycles through palette)."""
    return COLORBLIND_COLORS[index % len(COLORBLIND_COLORS)]


# Dictionary containing all paths for easy access
PATHS = {
    "plots": PLOTS_PATH,
    "saved_results": SAVED_RESULTS_PATH,
}


def get_plot_path(filename: str) -> Path:
    """
    Get the full path for a plot file.

    Args:
        filename (str): Name of the plot file

    Returns:
        Path: Full path to the plot file
    """
    return PLOTS_DIR / filename


def get_results_path(filename: str) -> Path:
    """
    Get the full path for a results file.

    Args:
        filename (str): Name of the results file

    Returns:
        Path: Full path to the results file
    """
    return SAVED_RESULTS_DIR / filename


# Example usage and testing
if __name__ == "__main__":
    print("Project Configuration Paths:")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Plots Directory: {PLOTS_DIR}")
    print(f"Saved Results Directory: {SAVED_RESULTS_DIR}")
    print(f"Plots Directory exists: {PLOTS_DIR.exists()}")
    print(f"Saved Results Directory exists: {SAVED_RESULTS_DIR.exists()}")

    # Example usage of helper functions
    plot_file = get_plot_path("example_plot.png")
    results_file = get_results_path("results.txt")
    print(f"Example plot path: {plot_file}")
    print(f"Example results path: {results_file}")
