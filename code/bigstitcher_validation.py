#!/usr/bin/env python3
"""Coordinate-sheaf validation on BigStitcher's public six-tile overlap graph."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sheaf_solver import CoordinateSheaf, CoordinateSheafEdge

ROOT = Path(__file__).resolve().parent
XML = ROOT / "data" / "bigstitcher" / "aligned3d" / "grid-3d-stitched-h5" / "dataset.xml"
OUT = ROOT / "results" / "bigstitcher_validation"


def parse_affine_translation(text: str) -> np.ndarray:
    """Read the translation part of BigStitcher's serialized 3x4 affine."""
    return np.fromstring(text, sep=" ", dtype=np.float64).reshape(3, 4)[:, 3]


def load_public_overlap_graph(path: Path) -> tuple[list[CoordinateSheafEdge], np.ndarray]:
    """Return real local overlap relations and published global stitch field."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download and extract the official BigStitcher OSF "
            "example archive before running this script."
        )
    root = ET.parse(path).getroot()
    reference = []
    for registration in root.find("ViewRegistrations")[:6]:
        transform = next(
            item for item in registration.findall("ViewTransform") if item.findtext("Name") == "Stitching Transform"
        )
        reference.append(parse_affine_translation(transform.findtext("affine")))
    reference = np.asarray(reference)
    reference -= reference[5]  # fixes only the global-coordinate gauge

    edges = []
    for result in root.find("StitchingResults").findall("PairwiseResult"):
        # Each pair lists all three channels of one tile. Modulo six maps the
        # channel-specific setup ID back to its physical tile.
        i = int(result.attrib["view_setup_a"].split(",")[0]) % 6
        j = int(result.attrib["view_setup_b"].split(",")[0]) % 6
        edges.append(
            CoordinateSheafEdge(
                i=i,
                j=j,
                weight=float(result.findtext("correlation")),
                relative_offset=tuple(parse_affine_translation(result.findtext("shift"))),
            )
        )
    return edges, reference


def pairwise_tree_solution(n_nodes: int, edges: list[CoordinateSheafEdge], anchor: int = 5) -> np.ndarray:
    """Pairwise-only control: maximum-confidence spanning tree, no cycles."""
    parent = list(range(n_nodes))

    def root_of(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    chosen = []
    for edge in sorted(edges, key=lambda item: item.weight, reverse=True):
        left, right = root_of(edge.i), root_of(edge.j)
        if left != right:
            parent[left] = right
            chosen.append(edge)
    if len(chosen) != n_nodes - 1:
        raise RuntimeError("public overlap graph is not connected")

    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_nodes)]
    for edge in chosen:
        offset = np.asarray(edge.relative_offset)
        adjacency[edge.i].append((edge.j, offset))
        adjacency[edge.j].append((edge.i, -offset))
    result = np.full((n_nodes, len(chosen[0].relative_offset)), np.nan)
    result[anchor] = 0.0
    queue = [anchor]
    while queue:
        current = queue.pop(0)
        for neighbor, offset in adjacency[current]:
            if np.isnan(result[neighbor, 0]):
                result[neighbor] = result[current] + offset
                queue.append(neighbor)
    return result


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def solve(edges: list[CoordinateSheafEdge], robust: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes, dimension = 6, len(edges[0].relative_offset)
    mean = np.zeros((n_nodes, dimension))
    variance = np.full_like(mean, 1e5)
    variance[5] = 1e-6  # arbitrary global-coordinate gauge only
    return CoordinateSheaf(n_nodes, edges).solve(
        mean,
        variance,
        consistency_weight=1.0,
        robust_scale=1.0 if robust else None,
        robust_iterations=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-fault", action="store_true", help="add one impossible relation to a real overlap edge")
    args = parser.parse_args()
    edges, reference = load_public_overlap_graph(XML)
    stress_description = "none; all 11 published BigStitcher local overlaps used unchanged"
    if args.inject_fault:
        # Edge 2 is a high-confidence real 5→3 overlap. The added vector is a
        # controlled wrong local registration measured in the same physical
        # coordinate units. It permits a reproducible QC/no-call stress test.
        corruption = np.array([12.0, -10.0, 8.0])
        bad = edges[2]
        edges[2] = replace(bad, relative_offset=tuple(np.asarray(bad.relative_offset) + corruption))
        stress_description = "controlled +[12, -10, +8] error added to published overlap edge 5→3"

    pairwise = pairwise_tree_solution(6, edges)
    quadratic, quadratic_edge, quadratic_node = solve(edges, robust=False)
    glass, edge_discord, node_discord = solve(edges, robust=True)
    no_call_threshold = float(np.quantile(node_discord, 0.80))
    report = {
        "purpose": "public 3D tiled-microscopy coordinate validation; no cortical-layer labels or absolute landmark truth",
        "source": "official BigStitcher OSF example: six 512×512×86, three-channel Drosophila-larva confocal tiles",
        "local_evidence": "11 published pairwise overlap transforms and correlations from BigStitcher's dataset.xml",
        "reference": "published BigStitcher global Stitching Transform; an independent pipeline result, not physical pose ground truth",
        "stress": stress_description,
        "coordinate_rmse_vs_published_stitching_transform": {
            "pairwise_tree_no_cycle_gluing": rmse(pairwise, reference),
            "quadratic_coordinate_sheaf": rmse(quadratic, reference),
            "robust_GLASS_coordinate_sheaf": rmse(glass, reference),
        },
        "robust_qc": {
            "edge_discord": edge_discord.tolist(),
            "node_discord": node_discord.tolist(),
            "evaluation_only_80pct_no_call_nodes": np.flatnonzero(node_discord > no_call_threshold).tolist(),
            "note": "The percentile is for a matched-coverage visualization only. A deployment no-call threshold needs development registration QC.",
        },
        "published_reference_translations": reference.tolist(),
        "robust_glass_translations": glass.tolist(),
        "quadratic_edge_discord": quadratic_edge.tolist(),
        "quadratic_node_discord": quadratic_node.tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "fault" if args.inject_fault else "clean"
    (OUT / f"metrics_{suffix}.json").write_text(json.dumps(report, indent=2))

    names = ("Published BigStitcher", "Pairwise tree", "Quadratic sheaf", "Robust sheaf")
    fields = (reference, pairwise, quadratic, glass)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, field in zip(names, fields):
        axes[0].plot(field[:, 0], field[:, 1], "o-", label=name, alpha=0.85)
        for node, (x_value, y_value, _) in enumerate(field):
            axes[0].annotate(str(node), (x_value, y_value), fontsize=8)
    axes[0].set_title("Global tile-coordinate field")
    axes[0].set_xlabel("x translation (pixels)")
    axes[0].set_ylabel("y translation (pixels)")
    axes[0].legend(fontsize=7)
    axes[0].axis("equal")
    axes[1].bar(np.arange(len(node_discord)), node_discord, color="#3c78d8")
    axes[1].axhline(no_call_threshold, color="#cc0000", linestyle="--", label="80% coverage cutoff")
    axes[1].set_title("Robust sheaf node discord")
    axes[1].set_xlabel("tile")
    axes[1].set_ylabel("mean edge residual")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Public BigStitcher test — {stress_description}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"coordinate_comparison_{suffix}.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
