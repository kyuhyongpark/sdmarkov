import warnings
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sdmarkov.plotting.config as pc


def _metric_arrays(df, metric, methods):
    return [
        df.loc[df.method == m, metric].dropna().to_numpy()
        for m in methods
    ]

def plot_violin_figure(
    df,
    metrics,
    *,
    figure_id=None,
    methods=None,
    figsize=(6, 3),
    lim=(0, 1),  # default for all metrics
):
    """
    Main-paper violin plot figure.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing metrics.
    metrics : list[str]
        List of metric column names to plot.
    figure_id : str | None
        Optional figure title ID.
    methods : list[str] | None
        Methods to plot. If None, use default order.
    figsize : tuple
        Figure size.
    lim : tuple or dict
        If tuple: (lo, hi) applied to all metrics.
        If dict: {metric_name: (lo, hi)} allows per-metric limits.
    """

    if methods is None:
        methods = pc.METHOD_ORDER

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(figsize[0], figsize[1] * n_metrics),
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Determine the limits for this metric
        if isinstance(lim, dict):
            lo, hi = lim.get(metric, (0, 1))
        else:
            lo, hi = lim

        arrays = _metric_arrays(df, metric, methods)
        colors = [pc.get_method_style(m)["color"] for m in methods]

        # --- bounds check ---
        for method, arr in zip(methods, arrays):
            if len(arr) == 0:
                continue
            if np.any(arr < lo) or np.any(arr > hi):
                n_bad = np.sum((arr < lo) | (arr > hi))
                warnings.warn(
                    f"[plot_violin_figure] Metric '{metric}', method '{method}': "
                    f"{n_bad} values outside limits {(lo, hi)}.",
                    RuntimeWarning,
                )

        sns.violinplot(
            data=arrays,
            ax=ax,
            palette=colors,
            cut=2,
            inner="box",
        )

        # --- add mean lines (dashed) ---
        for i, arr in enumerate(arrays):
            if len(arr) == 0:
                continue
            mean_val = np.mean(arr)
            ax.hlines(
                mean_val,
                i - 0.35,
                i + 0.35,
                colors="black",
                linestyles="--",
                linewidth=1.5,
                zorder=3,
            )

        positions = np.arange(len(methods))
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [pc.get_method_style(m)["label"] for m in methods]
        )

        ax.set_ylabel(pc.METRIC_Y_LABELS.get(metric, metric))
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.4)

    if figure_id is not None:
        title = pc.FIGURE_TITLES.get(figure_id)
        if title is not None:
            fig.suptitle(title, y=0.93)

    plt.tight_layout(rect=(0, 0, 1, 1))
    return fig


def plot_distribution_figure(
    df,
    metrics,
    *,
    methods=None,
    figure_id=None,
    figsize=(10, 4),
    lim=(0, 1),
):
    """
    Plot KDE and ECDF distributions for multiple metrics and methods.
    Supports per-metric y/x-axis limits via a dict.
    """
    if methods is None:
        methods = pc.METHOD_ORDER

    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_metrics, 2,
        figsize=(figsize[0], figsize[1] * n_metrics),
    )

    # normalize axes shape to (n_metrics, 2)
    if n_metrics == 1:
        axes = np.array([axes])

    for (ax_kde, ax_ecdf), metric in zip(axes, metrics):
        arrays = _metric_arrays(df, metric, methods)

        # --- get per-metric limits ---
        if isinstance(lim, dict):
            lo, hi = lim.get(metric, (0, 1))
        else:
            lo, hi = lim

        # --- bounds check ---
        for method, arr in zip(methods, arrays):
            if len(arr) == 0:
                continue
            n_bad = np.sum((arr < lo) | (arr > hi))
            if n_bad > 0:
                warnings.warn(
                    f"[plot_distribution_figure] Metric '{metric}', method '{method}': "
                    f"{n_bad} values outside limits {(lo, hi)}.",
                    RuntimeWarning,
                )

        # ---------- KDE ----------
        for m, arr in zip(methods, arrays):
            if len(arr) == 0:
                continue
            style = pc.get_method_style(m)
            sns.kdeplot(
                arr,
                ax=ax_kde,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                clip=(lo, hi),
            )

        ax_kde.set_xlim(lo, hi)
        ax_kde.set_xlabel(pc.METRIC_X_LABELS.get(metric, metric))
        ax_kde.set_yticks([])
        ax_kde.set_ylabel("")
        ax_kde.grid(True, alpha=0.4)

        # ---------- ECDF ----------
        for m, arr in zip(methods, arrays):
            if len(arr) == 0:
                continue
            style = pc.get_method_style(m)
            x = np.sort(arr)
            y = np.arange(1, len(arr) + 1) / len(arr)
            # prepend (0,0)
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0)
            # append (max_x_value, 1)
            x = np.append(x, hi)
            y = np.append(y, 1)
            ax_ecdf.plot(
                x, y,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )

        ax_ecdf.set_xlim(lo, hi)
        ax_ecdf.set_xlabel(pc.METRIC_X_LABELS.get(metric, metric))
        ax_ecdf.set_ylim(0, 1)
        ax_ecdf.set_ylabel(f"P({pc.METRIC_Y_LABELS.get(metric, metric)} ≤ x)")
        ax_ecdf.grid(True, alpha=0.4)

    # Get handles and labels from the last ECDF axis
    handles, labels = axes[-1, 1].get_legend_handles_labels()

    # Figure-level legend
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(handles),
    )

    # --- global figure title ---
    if figure_id is not None:
        title = pc.FIGURE_TITLES.get(figure_id)
        if title is not None:
            fig.suptitle(title, y=1)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    return fig

def save_figure(
    fig,
    *,
    figure_id,
    outdir,
    ext="png",
    dpi=300,
):
    """
    Save a matplotlib figure with a standardized name.

    Example:
        Figure_4_Strong_basins.pdf
    """

    # Make filename-safe title
    safe_title = re.sub(r"[^\w\-]+", "_", pc.FIGURE_TITLES[figure_id]).strip("_")

    filename = f"Figure_{figure_id}_{safe_title}.{ext}"
    outpath = Path(outdir) / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath
