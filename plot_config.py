"""Plotting configuration and output path utilities."""

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("output")
EDA_DIR = DEFAULT_OUTPUT_ROOT / "eda"
ANALYSIS_DIR = DEFAULT_OUTPUT_ROOT / "analysis"
RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "results"
TEST_OUTPUT_DIR = Path("test_output")

DEFAULT_DPI = 150


def get_output_path(filename: str, subdir: str = None) -> Path:
    """Build output path, creating directories as needed.

    Args:
        filename: The filename (e.g., "boxplot_pitch.png").
        subdir: Optional subdirectory ("eda", "analysis", "results").

    Returns:
        Full path for the output file.
    """
    if subdir:
        path = DEFAULT_OUTPUT_ROOT / subdir / filename
    else:
        path = DEFAULT_OUTPUT_ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
