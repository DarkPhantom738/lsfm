"""Render the README workflow diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent / "results" / "readme_figures" / "sheaf_workflow.png"


def box(ax, x, y, text, color, width=1.55):
    height = 0.82
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        linewidth=1.2, edgecolor="#334155", facecolor=color, zorder=3,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=8.6, color="#172033", weight="bold", zorder=4)
    return x, y, width, height


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25,
        linestyle="--" if dashed else "-", color="#b45309" if dashed else "#475569",
        zorder=2,
    ))


def right(node):
    x, y, width, height = node
    return x + width, y + height / 2


def left(node):
    x, y, _, height = node
    return x, y + height / 2


def top(node):
    x, y, width, height = node
    return x + width / 2, y + height


def bottom(node):
    x, y, width, _ = node
    return x + width / 2, y


def main():
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    y_top = 3.0
    xs = [0.38, 2.30, 4.22, 6.14, 8.06, 9.98, 11.90]
    raw = box(ax, xs[0], y_top, "raw LSFM\nslices", "#ffffff")
    graph = box(ax, xs[1], y_top, "overlap\ngraph", "#e7f0fc")
    registration = box(ax, xs[2], y_top, "local registration\nshift + confidence", "#e7f0fc")
    coordinate = box(ax, xs[3], y_top, "coordinate\nsheaf", "#cfe1fa")
    coord_qc = box(ax, xs[4], y_top, "coordinate QC\ndiscord / no-call", "#fff0cc")
    nifty = box(ax, xs[5], y_top, "NiftyMIC\nreconstruction", "#eef0f3")
    volume = box(ax, xs[6], y_top, "registered\n3D volume", "#d9f5e3")
    for before, after in zip((raw, graph, registration, coordinate, coord_qc, nifty),
                             (graph, registration, coordinate, coord_qc, nifty, volume)):
        arrow(ax, right(before), left(after))

    y_bottom = 0.72
    profiles = box(ax, 0.85, y_bottom, "cortical columns\n+ marker profiles", "#ffffff", width=1.92)
    local = box(ax, 3.50, y_bottom, "local boundary\nestimates", "#e5f7ea", width=1.72)
    laminar = box(ax, 6.10, y_bottom, "laminar\nsheaf", "#c5efcf", width=1.55)
    output = box(ax, 8.56, y_bottom, "layer boundaries\n+ thickness + QC", "#d9f5e3", width=1.88)
    for before, after in zip((profiles, local, laminar), (local, laminar, output)):
        arrow(ax, right(before), left(after))

    elbow_y = 2.33
    arrow(ax, bottom(volume), (bottom(volume)[0], elbow_y))
    arrow(ax, (bottom(volume)[0], elbow_y), (1.81, elbow_y))
    arrow(ax, (1.81, elbow_y), top(profiles))
    arrow(ax, bottom(coord_qc), top(local), dashed=True)
    ax.text(5.9, 2.1, "QC weights local layer estimates", ha="center", fontsize=7.8,
            color="#92400e", style="italic")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
