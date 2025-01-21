# pylint: disable=invalid-name
""""Plotting utilities."""
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator

from ..config import PVALUE_MAP


def clear_axes(
    ax=None, top=True, right=True, left=False, bottom=False, minorticks_off=True
):
    """A more forcing version of sns.despine.

    From: https://gist.github.com/Neclow/91867667fdc463c0f36aaa7a26e84e0e

    Parameters
    ----------
    ax : matplotlib axes, optional
        Specific axes object to despine.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    minorticks_off: boolean, optional
        If True, remove all minor ticks
    """
    if ax is None:
        axes = plt.gcf().axes
    else:
        axes = [ax]

    for ax_i in axes:
        sns.despine(ax=ax_i, top=top, bottom=bottom, left=left, right=right)

        if minorticks_off:
            ax_i.minorticks_off()

        ax_i.tick_params(
            top=not top,
            bottom=not bottom,
            left=not left,
            right=not right,
            labelleft=not left,
            labelright=not right,
            labeltop=not top,
            labelbottom=not bottom,
        )


def set_size(width, layout="h", fraction=1):
    """Set figure dimensions in inches to avoid scaling in LaTeX.

    From: https://gist.github.com/Neclow/5d0bf39c372596c0de80f7ed5378aa6b

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
        Report: 390 pt
    layout: string
        h: horizontal layout
        v: vertical layout
        s: square layout
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio for aesthetic figures
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if layout == "h":
        fig_height_in = fig_width_in * golden_ratio
    elif layout == "v":
        fig_height_in = fig_width_in / golden_ratio
    elif layout == "s":
        fig_height_in = fig_width_in
    else:
        raise ValueError(f"Unkown value for `layout`: {layout}")

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def annotate_plot(ax, plot_params, pvalues):
    """Annotate a plot with p-values.

    Parameters
    ----------
    ax : matplotlib axes.Axes
        Current axes object
    plot_params : dict
        Plot parameters fed into seaborn
    pvalues : array-like
        list or array of p-values for each pair comparison
    """
    if "hue" in plot_params:
        x_values = sorted(set(plot_params["data"][plot_params["x"]]))
        hue_values = set(plot_params["data"][plot_params["hue"]])
        pairs = [
            ((x_value, pair[0]), (x_value, pair[1]))
            for x_value in x_values
            for pair in combinations(hue_values, 2)
        ]
    else:
        pairs = list(combinations(set(plot_params["data"][plot_params["x"]]), 2))

    annotator = Annotator(ax, pairs=pairs, **plot_params)

    annotator.hide_non_significant = True
    annotator.perform_stat_test = False
    # pylint: disable=protected-access
    annotator._pvalue_format.pvalue_thresholds = PVALUE_MAP
    annotator._pvalue_format._correction_format = "replace"
    # pylint: enable=protected-access

    annotator.configure(text_format="star", loc="inside", fontsize=12)

    annotator.set_pvalues_and_annotate(pvalues=pvalues)
