#!/usr/bin/env python3
"""Create the README figures from the public BigStitcher example."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/bigstitcher/raw3d/Grid1"
XML = ROOT / "data/bigstitcher/aligned3d/grid-3d-stitched-h5/dataset.xml"
OUT = ROOT / "results/readme_figures"


def translation(text: str) -> np.ndarray:
    """Return the x/y translation from BigStitcher's 3×4 affine text."""
    return np.fromstring(text, sep=" ", dtype=float).reshape(3, 4)[:2, 3]


def regular_grid_positions() -> np.ndarray:
    root = ET.parse(XML).getroot()
    values = []
    for registration in root.find("ViewRegistrations")[:6]:
        transform = next(
            item for item in registration.findall("ViewTransform")
            if item.findtext("Name") == "Translation to Regular Grid"
        )
        values.append(translation(transform.findtext("affine")))
    return np.asarray(values)


def public_edges() -> list[tuple[int, int]]:
    root = ET.parse(XML).getroot()
    edges = {
        (
            int(item.attrib["view_setup_a"].split(",")[0]) % 6,
            int(item.attrib["view_setup_b"].split(",")[0]) % 6,
        )
        for item in root.find("StitchingResults").findall("PairwiseResult")
    }
    return sorted(edges)


def normalized_projection(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    projection = np.max(image, axis=0).astype(float)
    low, high = np.percentile(projection, (1, 99.8))
    return np.clip((projection - low) / max(high - low, 1e-9), 0, 1)


def font(size: int) -> ImageFont.FreeTypeFont:
    """Use a standard macOS font for the label-only DANDI figure refresh."""
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)


def make_metric_figures() -> None:
    """Render compact, explicitly labeled before/after metric summaries."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = (
        ("DANDI LSFM coordinate proxy\n(lower is better)", ("Pairwise", "Sheaf"), (22.73, 1.07), "RMSE (pixels)"),
        ("BigStitcher fault test\n(lower is better)", ("Pairwise", "Robust sheaf"), (8.78, 0.54), "RMSE (pixels)"),
        ("Same NiftyMIC SVR\n(higher is better)", ("NiftyMIC only", "Sheaf inputs → NiftyMIC"), (0.633, 0.810), "NCC"),
    )
    colors = ("#e07a5f", "#2a9d8f", "#457b9d")
    for axis, (title, labels, values, ylabel) in zip(axes, panels):
        bars = axis.bar(range(len(values)), values, color=colors[:len(values)])
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(values)), labels, rotation=13, ha="right")
        axis.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.03, f"{value:.3g}", ha="center", fontsize=10)
    figure.suptitle("Controlled before/after evidence for the coordinate sheaf", fontsize=14)
    figure.savefig(OUT / "controlled_before_after_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    labels = ("Local", "Graph", "Sheaf")
    colors = ("#e07a5f", "#8da0cb", "#2a9d8f")
    for axis, title, values, ylabel in (
        (axes[0], "Boundary MAE at matched 80% coverage\n(lower is better)", (0.07968, 0.08146, 0.07961), "normalized depth"),
        (axes[1], "Six-layer Dice at matched 80% coverage\n(higher is better)", (0.60431, 0.59775, 0.60068), "Dice"),
    ):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values) * 1.3)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.04, f"{value:.4f}", ha="center", fontsize=9)
    figure.suptitle("BigBrain cortical-layer proxy: transparent, non-SOTA result", fontsize=13)
    figure.savefig(OUT / "bigbrain_layer_proxy_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def relabel_dandi_before_after() -> None:
    """Refresh the label of an existing DANDI visual without changing pixels."""
    source = ROOT / "results/dandi_coordinate_proxy/coordinate_atlas_proxy.png"
    if not source.exists():
        return
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    draw.rectangle((0, 0, width, 46), fill="white")
    title = "Coordinate sheaf: public DANDI LSFM signal → globally compatible atlas"
    draw.text((width // 2, 8), title, fill="black", font=font(23), anchor="ma")
    # Replace only the legacy name over the fourth panel; all image values stay unchanged.
    draw.rectangle((1595, 42, width, 92), fill="white")
    draw.text((1875, 48), "Sheaf-consistent atlas", fill="black", font=font(17), anchor="ma")
    image.save(OUT / "dandi_lsfm_coordinate_before_after.png")


def main() -> None:
    if not RAW.exists() or not XML.exists():
        raise FileNotFoundError(
            "Download the public BigStitcher example first: "
            "bash code/download_bigstitcher_data.sh"
        )

    projections = [normalized_projection(RAW / f"C1-{73 + index}.tif") for index in range(6)]
    positions = regular_grid_positions()
    edges = public_edges()
    OUT.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 1.15))
    for index, projection in enumerate(projections):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        axis.imshow(projection, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Vertex {index}: physical tile {3 + index}", fontsize=10)
        axis.set_axis_off()
    figure.suptitle(
        "Public BigStitcher input: six raw 3D tile projections (channel 1)",
        fontsize=15,
    )
    figure.savefig(OUT / "bigstitcher_raw_tile_montage.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Render the actual input tiles at their public regular-grid coordinates.
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    origin = positions.min(axis=0)
    tile_width = projections[0].shape[1]
    tile_height = projections[0].shape[0]
    centers = []
    for index, (projection, position) in enumerate(zip(projections, positions)):
        x0, y0 = position - origin
        axis.imshow(
            projection,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(x0, x0 + tile_width, y0 + tile_height, y0),
            alpha=0.78,
            origin="upper",
        )
        center = np.array((x0 + tile_width / 2, y0 + tile_height / 2))
        centers.append(center)
        axis.text(*center, str(index), color="yellow", fontsize=13, fontweight="bold", ha="center", va="center")

    for left, right in edges:
        axis.plot(
            (centers[left][0], centers[right][0]),
            (centers[left][1], centers[right][1]),
            color="#23a6f0",
            linewidth=1.8,
            alpha=0.9,
            zorder=4,
        )
        midpoint = (centers[left] + centers[right]) / 2
        axis.text(*midpoint, f"{left}–{right}", color="white", fontsize=7, ha="center", va="center", zorder=5)

    axis.set_title("Actual input tiles placed on the public grid; blue = 11 measured overlaps", fontsize=13)
    axis.set_xlabel("x (input pixels)")
    axis.set_ylabel("y (input pixels)")
    axis.set_aspect("equal")
    figure.savefig(OUT / "bigstitcher_raw_tiles_with_overlap_edges.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    make_metric_figures()
    relabel_dandi_before_after()


if __name__ == "__main__":
    main()
