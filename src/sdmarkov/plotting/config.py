"""
Centralized plotting configuration for the project.

This module defines:
- Method styles and ordering
- Metric labels and axis labels
- Figure titles (main & supplementary)
- Global matplotlib style defaults

Import this module in notebooks and call `apply_style()` once
before generating figures.
"""

# =========================
# Method styles
# =========================

METHOD_STYLES = {
    "sd_mc": dict(
        label="SD",
        color="#0072B2",
        linestyle="-",
        marker=None,
        alpha=1.0,
    ),
    "random_mc": dict(
        label="Random",
        color="#E7298A",
        linestyle="-",
        marker=None,
        alpha=0.7,
    ),
    "null_mc": dict(
        label="Null",
        color="#7570B3",
        linestyle="--",
        marker=None,
        alpha=1.0,
    ),
    "ref": dict(
        label="Ref",
        color="#D95F02",
        linestyle="-.",
        marker=None,
        alpha=1.0,
    ),
}

# Canonical plotting order for methods
METHOD_ORDER = ["sd_mc", "random_mc", "null_mc", "ref"]


def get_method_style(method):
    """Return a copy of the plotting style for a given method."""
    return METHOD_STYLES[method].copy()


# =========================
# Metric labels
# =========================

METRIC_Y_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "npv": "NPV",
    "kld": "KLD",
    "rmsd": "RMSD",
}

METRIC_X_LABELS = {
    "precision": "Precision = TP / (TP + FP)",
    "recall": "Recall = TP / (TP + FN)",
    "specificity": "Specificity = TN / (TN + FP)",
    "npv": "NPV = TN / (TN + FN)",
    "kld": "KLD",
    "rmsd": "RMSD",
}


# =========================
# Figure titles
# =========================

FIGURE_TITLES = {
    "4": "Strong basins",
    "5": "Attractor reachability",
    "6": "Convergence probabilities",
    "7": "Basin fractions",
    "8": "Average node values",
    "s1": "Attractor states",
    "s2": "Strong basins",
    "s3": "Attractor reachability",
    "s4": "Convergence probabilities",
    "s5": "Basin fractions",
    "s6": "Average node values",
    "s7": "Decision transitions",
    "s8": "Trajectory probabilities",
}


# =========================
# Global matplotlib style
# =========================

import matplotlib.pyplot as plt


def apply_style():
    """Apply global matplotlib rcParams for all figures."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })
