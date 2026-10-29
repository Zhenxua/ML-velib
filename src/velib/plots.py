"""Static figures for the report (PNG). Two series at most per panel: the chosen model (blue) and
persistence (orange); the majority-class reference is a muted dashed line. Palette validated for
colour-vision deficiency (adjacent CVD ΔE 24.7); every series is also named in a legend.
Each function saves its figure, closes it and returns the path (display it with IPython.display.Image).
The style is applied with plt.rc_context inside each function, so importing this module changes nothing
in the caller's matplotlib settings."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve

MODEL_COLOR, PERSISTENCE_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _font_family() -> list[str]:
    """Segoe UI where it exists (Windows), DejaVu Sans (bundled with matplotlib) everywhere else."""
    try:
        font_manager.findfont("Segoe UI", fallback_to_default=False)
        return ["Segoe UI", "DejaVu Sans"]
    except ValueError:
        return ["DejaVu Sans"]


STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": _font_family(),
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
}


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def pr_curves(panels: dict, path: Path) -> Path:
    """panels: {horizon_min: {"y": array, "model": (name, score, pred), "persistence": (score, pred)}}"""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
        for ax, (h, p) in zip(np.atleast_1d(axes), panels.items()):
            y = np.asarray(p["y"])
            name, m_score, m_pred = p["model"]
            series = [(name, m_score, m_pred, MODEL_COLOR),
                      ("persistence (score: fewer bikes now)", *p["persistence"], PERSISTENCE_COLOR)]
            for label, score, pred, color in series:
                prec, rec, _ = precision_recall_curve(y, score)
                ax.plot(rec, prec, color=color, label=label, drawstyle="steps-post")
                tp = np.sum((pred == 1) & (y == 1))
                op_p, op_r = tp / max(pred.sum(), 1), tp / y.sum()
                ax.plot(op_r, op_p, "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=3)
            ax.axhline(y.mean(), color=MUTED, lw=1.2, ls="--", label=f"majority class (prevalence {y.mean():.2f})")
            ax.set(title=f"Horizon {h} min — test week", xlabel="recall", xlim=(0, 1), ylim=(0, 1.02))
            if ax is np.atleast_1d(axes)[0]:
                ax.set_ylabel("precision")
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.16), fontsize=8)  # above the prevalence line
        fig.text(0.01, 0.01, "Dots: frozen operating points (threshold chosen on validation; persistence: bikes < 2 now).",
                 color=INK_2, fontsize=7.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return _save(fig, path)


def by_hour(model_name: str, model_scored: pd.DataFrame, persistence_scored: pd.DataFrame, horizon: int,
            path: Path) -> Path:
    """Recall and precision per local hour, model vs persistence (two panels, one y-scale each)."""
    def per_hour(s):
        g = s.groupby("hour_local")
        tp = ((s["pred"] == 1) & (s["y"] == 1)).groupby(s["hour_local"]).sum()
        return pd.DataFrame({"recall": tp / g["y"].sum(), "precision": tp / g["pred"].sum().replace(0, np.nan)})

    m, p = per_hour(model_scored), per_hour(persistence_scored)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharex=True)
        for ax, metric in zip(axes, ("recall", "precision")):
            ax.plot(m.index, m[metric], color=MODEL_COLOR, marker="o", ms=4, label=model_name)
            ax.plot(p.index, p[metric], color=PERSISTENCE_COLOR, marker="o", ms=4, label="persistence")
            ax.set(title=f"{metric.capitalize()} by hour (Paris time), {horizon} min",
                   xlabel="hour of day (Europe/Paris)", ylim=(0, 1), xticks=range(0, 24, 3))
        axes[0].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        return _save(fig, path)


def situations(table: pd.DataFrame, model_name: str, horizon: int, path: Path) -> Path:
    """Share predicted "empty" per situation: a recall for the two positive situations,
    a false-alarm rate for the two negative ones."""
    cols = {model_name: f"predicted empty: {model_name}", "persistence": "predicted empty: persistence"}
    x = np.arange(len(table))
    width = 0.38
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for i, (label, col) in enumerate(cols.items()):
            color = MODEL_COLOR if i == 0 else PERSISTENCE_COLOR
            bars = ax.bar(x + (i - 0.5) * width, table[col], width, color=color, edgecolor=SURFACE, linewidth=2,
                          label=label)
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in table[col]], fontsize=7.5, color=INK_2, padding=2)
        ax.set_xticks(x, [f"{s}\n(n={n:,})" for s, n in zip(table.index, table["n"])])
        ax.set(ylim=(0, 1.1), ylabel='share predicted "empty"',
               title=f"Where the errors are — test week, {horizon} min (state now → state at t+h)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=8)  # the "stays available" bars are near 0: free space
        fig.tight_layout()
        return _save(fig, path)
