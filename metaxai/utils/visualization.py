"""
visualization.py — Meta-XAI Visualization Utilities

Provides functions for rendering:
    - Attribution heatmaps overlaid on images
    - Consistency matrix heatmaps
    - Full audit report summary figures
    - ETS gauge charts
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from typing import List, Optional
from metaxai.schemas import ExplanationAuditReport, ConsistencyReport


def plot_attribution_heatmaps(
    image: np.ndarray,
    explanation_maps: list,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 4),
) -> plt.Figure:
    """Plot attribution heatmaps from all XAI methods side by side.

    Args:
        image: Input image as numpy array (H, W, C), float32, [0, 1].
        explanation_maps: List of ExplanationMap objects.
        save_path: Optional path to save the figure. Displays inline if None.
        figsize: Figure size tuple (width, height).

    Returns:
        matplotlib Figure object.
    """
    valid_maps = [m for m in explanation_maps if m.is_valid]
    n_cols = len(valid_maps) + 1  # +1 for original image

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Attribution heatmaps
    for i, emap in enumerate(valid_maps):
        ax = axes[i + 1]
        heatmap = emap.attribution

        # Overlay heatmap on image
        colored_heatmap = cm.jet(heatmap)[:, :, :3]
        blended = 0.6 * np.clip(image, 0, 1) + 0.4 * colored_heatmap
        ax.imshow(np.clip(blended, 0, 1))
        ax.set_title(f"{emap.method.upper()}\n({emap.computation_time_ms:.0f}ms)", fontsize=10)
        ax.axis("off")

    plt.suptitle("Meta-XAI Attribution Maps", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_consistency_matrix(
    report: ConsistencyReport,
    metric: str = "iou",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a pairwise consistency matrix as a heatmap.

    Args:
        report: ConsistencyReport from the Consistency Analyzer.
        metric: One of 'iou', 'cosine', or 'rank'. Default: 'iou'.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    matrix_map = {
        "iou": (report.iou_matrix, "IoU Consistency"),
        "cosine": (report.cosine_matrix, "Cosine Similarity"),
        "rank": (report.rank_corr_matrix, "Rank Correlation (Spearman ρ)"),
    }

    if metric not in matrix_map:
        raise ValueError(f"metric must be one of {list(matrix_map.keys())}")

    matrix, title = matrix_map[metric]
    labels = report.method_names

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Agreement Score")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.upper() for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([l.upper() for l in labels])

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color="black" if matrix[i, j] > 0.4 else "white")

    ax.set_title(f"{title}\n(pairwise_agreement_score = {report.pairwise_agreement_score:.3f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ets_gauge(
    ets: float,
    trust_level: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the ETS as a gauge / speedometer chart.

    Args:
        ets: Explanation Trust Score in [0, 100].
        trust_level: Trust level string ('HIGH'/'MEDIUM'/'LOW'/'REJECT').
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    color_map = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e67e22", "REJECT": "#e74c3c"}
    color = color_map.get(trust_level, "#95a5a6")

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"polar": False})
    ax.axis("off")

    # Draw gauge background
    theta = np.linspace(np.pi, 0, 200)
    r = 0.4
    for i, (start, end, col) in enumerate([
        (0, 25, "#e74c3c"), (25, 50, "#e67e22"), (50, 75, "#f39c12"), (75, 100, "#2ecc71")
    ]):
        t_start = np.pi - (start / 100) * np.pi
        t_end = np.pi - (end / 100) * np.pi
        t = np.linspace(t_start, t_end, 50)
        x = np.cos(t) * r + 0.5
        y = np.sin(t) * r + 0.1
        ax.plot(x, y, color=col, linewidth=15, alpha=0.4)

    # Draw needle
    needle_angle = np.pi - (ets / 100) * np.pi
    nx = np.cos(needle_angle) * (r - 0.05) + 0.5
    ny = np.sin(needle_angle) * (r - 0.05) + 0.1
    ax.annotate("", xy=(nx, ny), xytext=(0.5, 0.1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))

    ax.text(0.5, 0.55, f"ETS: {ets:.1f}/100", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.35, f"[{trust_level}]", ha="center", va="center",
            fontsize=13, color=color, transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def render_audit_report(
    report: ExplanationAuditReport,
    image: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a comprehensive full audit report figure.

    Combines attribution maps, consistency matrix, ETS gauge, and
    hallucination flags into a single publication-quality figure.

    Args:
        report: Complete ExplanationAuditReport from the pipeline.
        image: Input image as numpy array (H, W, C).
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    valid_maps = [m for m in report.explanation_maps if m.is_valid]
    n_methods = len(valid_maps)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, max(n_methods + 1, 3), figure=fig, hspace=0.4, wspace=0.3)

    # Row 0: Attribution maps
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(np.clip(image, 0, 1))
    ax_orig.set_title("Original Image", fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    hallucinated_methods = {h.method for h in report.hallucination_results if h.is_hallucinated}

    for i, emap in enumerate(valid_maps):
        ax = fig.add_subplot(gs[0, i + 1])
        colored = cm.jet(emap.attribution)[:, :, :3]
        blended = 0.55 * np.clip(image, 0, 1) + 0.45 * colored
        ax.imshow(np.clip(blended, 0, 1))
        is_h = emap.method in hallucinated_methods
        label = f"{emap.method.upper()}\n{'⚠ HALLUCINATED' if is_h else '✓ Valid'}"
        color = "red" if is_h else "green"
        ax.set_title(label, fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    # Row 1, Col 0-1: Consistency matrix
    ax_cons = fig.add_subplot(gs[1, :2])
    if len(report.explanation_maps) > 1:
        mat = report.consistency_report.iou_matrix
        labels = report.consistency_report.method_names
        im = ax_cons.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
        ax_cons.set_xticks(range(len(labels)))
        ax_cons.set_yticks(range(len(labels)))
        ax_cons.set_xticklabels([l.upper() for l in labels], rotation=45)
        ax_cons.set_yticklabels([l.upper() for l in labels])
        for ii in range(len(labels)):
            for jj in range(len(labels)):
                ax_cons.text(jj, ii, f"{mat[ii, jj]:.2f}", ha="center", va="center", fontsize=9)
        ax_cons.set_title(f"IoU Consistency Matrix\n(score={report.consistency_report.pairwise_agreement_score:.3f})",
                          fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax_cons)

    # Row 1, Col 2+: ETS Summary
    ax_ets = fig.add_subplot(gs[1, 2:])
    ax_ets.axis("off")
    color_map = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e67e22", "REJECT": "#e74c3c"}
    ets_color = color_map.get(report.trust_level, "#95a5a6")

    summary_lines = [
        (f"ETS: {report.explanation_trust_score:.1f}/100", 18, ets_color, "bold"),
        (f"Trust Level: {report.trust_level}", 14, ets_color, "bold"),
        (f"Predicted: {report.predicted_class} ({report.model_confidence:.2%})", 11, "black", "normal"),
        (f"Consistency: {report.consistency_report.pairwise_agreement_score:.3f}", 10, "#2c3e50", "normal"),
        (f"Robustness:  {report.robustness_score:.3f}", 10, "#2c3e50", "normal"),
        (f"Semantic:    {report.semantic_coherence_score:.3f}", 10, "#2c3e50", "normal"),
        (f"Stability:   {report.stability_score:.3f}", 10, "#2c3e50", "normal"),
        (f"Hallucinated: {report.num_hallucinated}/{len(report.hallucination_results)}", 10,
         "red" if report.num_hallucinated > 0 else "green", "normal"),
    ]

    y_pos = 0.95
    for text, size, color, weight in summary_lines:
        ax_ets.text(0.05, y_pos, text, transform=ax_ets.transAxes,
                    fontsize=size, color=color, fontweight=weight, va="top")
        y_pos -= 0.12

    fig.suptitle(f"Meta-XAI Explanation Audit Report | {report.image_id}",
                 fontsize=14, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
