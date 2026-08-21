#!/usr/bin/env python3
"""Controlled public-LSFM coordinate-sheaf test using DANDI 000108 signal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from numcodecs import Blosc
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

from sheaf_solver import CoordinateSheaf, CoordinateSheafEdge, registration_reliability

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dandi_coordinate_proxy"
ZARR_ID = "bf47be1a-4fed-4105-bcb4-c52534a45b82"  # DANDI 000108, sample 127 YO
RNG = np.random.default_rng(2701)


def fetch_public_projection() -> np.ndarray:
    """Stream one public 128^3 block and return an 8-plane YO projection."""
    root = f"https://dandiarchive.s3.amazonaws.com/zarr/{ZARR_ID}/0"
    meta = requests.get(f"{root}/.zarray", timeout=30).json()
    chunks = tuple(meta["chunks"])
    if chunks != (1, 1, 128, 128, 128):
        raise RuntimeError(f"unexpected public Zarr chunk shape: {chunks}")
    blob = requests.get(f"{root}/0/0/0/0/100", timeout=30)
    blob.raise_for_status()
    arr = np.empty(chunks, dtype=np.dtype(meta["dtype"]))
    Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).decode(blob.content, out=arr)
    image = arr[0, 0, 60:68].mean(axis=0).astype(np.float64)
    lo, hi = np.percentile(image, (1, 99.5))
    return np.clip((image - lo) / max(hi - lo, 1e-8), 0, 1)


def make_observations(reference: np.ndarray, n_nodes: int = 10, failed_node: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Generate known registration offsets on genuine LSFM signal."""
    applied = RNG.uniform(-7.0, 7.0, size=(n_nodes, 2))
    applied[0] = 0.0  # atlas anchor
    observations = []
    for index, offset in enumerate(applied):
        view = nd_shift(reference, offset, order=1, mode="nearest")
        view = np.clip(view + RNG.normal(0, 0.012, view.shape), 0, 1)
        if index != 0:
            y, x = RNG.integers(8, 96, size=2)
            view[y : y + 18, x : x + 20] = np.median(view)
        observations.append(view)
    # Missing signal cannot be recovered by topology; it should be rejected.
    observations[failed_node] = RNG.uniform(0, 1, size=reference.shape)
    # Correction that maps each observation to the node-zero coordinate frame.
    target_correction = applied[0] - applied
    return np.asarray(observations), target_correction


def overlap_measurement(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary local registration evidence plus a conservative match weight."""
    displacement, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=10, normalization=None)
    aligned = nd_shift(moving, displacement, order=1, mode="nearest")
    correlation = float(np.corrcoef(fixed.ravel(), aligned.ravel())[0, 1])
    return displacement.astype(np.float64), float(np.clip(correlation, 0.02, 1.0) ** 2)


def build_overlap_edges(observations: np.ndarray) -> tuple[list[CoordinateSheafEdge], list[dict]]:
    """Build redundant local overlap constraints, including loops."""
    n_nodes = len(observations)
    pairs = {(i, i + 1) for i in range(n_nodes - 1)}
    pairs.update((i, i + 2) for i in range(n_nodes - 2))
    pairs.update({(0, 4), (2, 7), (5, 9)})
    edges, report = [], []
    for i, j in sorted(pairs):
        displacement, weight = overlap_measurement(observations[i], observations[j])
        edges.append(CoordinateSheafEdge(i, j, weight, tuple(float(v) for v in displacement)))
        report.append({"i": i, "j": j, "relative_offset_yx": displacement.tolist(), "match_weight": weight})
    return edges, report


def chained_registration(observations: np.ndarray) -> np.ndarray:
    """Pairwise-only control: compose neighboring shifts, without global cycles."""
    result = np.zeros((len(observations), 2), dtype=np.float64)
    for index in range(1, len(observations)):
        displacement, _ = overlap_measurement(observations[index - 1], observations[index])
        result[index] = result[index - 1] + displacement
    return result


def coordinate_rmse(prediction: np.ndarray, target: np.ndarray, use: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction[use] - target[use]) ** 2)))


def mean_registered_image(observations: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return np.mean([nd_shift(image, offset, order=1, mode="nearest") for image, offset in zip(observations, correction)], axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = fetch_public_projection()
    observations, target = make_observations(reference)
    edges, edge_report = build_overlap_edges(observations)
    local_mean = np.zeros_like(target)
    local_variance = np.full_like(target, 1e5)
    local_variance[0] = 1e-6  # fixes only the unavoidable global translation gauge
    glass, edge_discord, node_discord = CoordinateSheaf(len(observations), edges).solve(
        local_mean,
        local_variance,
        consistency_weight=1.0,
    )
    chained = chained_registration(observations)
    # Evaluation-only matched reporting-coverage point.  This threshold is not
    # a deployment calibration because the public block has no real pose labels.
    accepted = node_discord <= np.quantile(node_discord, 0.80)
    reliability_scale = float(np.quantile(node_discord, 0.80))
    reliability = registration_reliability(node_discord, max(reliability_scale, 1e-6))
    report = {
        "purpose": "controlled coordinate-atlas test on public DANDI 000108 LSFM signal with known synthetic translations; not real-LSFM landmark accuracy",
        "source": "DANDI:000108 sample 127 YO, streamed public image block",
        "method": "local phase-correlation overlap measurements are affine sheaf edge relations; the GLASS coordinate sheaf finds globally gluable translation corrections",
        "synthetic_stress": "independent sub-pixel translations, noise, local occlusion, and one signal-absent tile",
        "n_nodes": int(len(observations)),
        "n_overlap_edges": int(len(edges)),
        "coordinate_rmse_pixels": {
            "pairwise_chain_all_nodes": coordinate_rmse(chained, target, np.ones(len(target), dtype=bool)),
            "glass_all_nodes": coordinate_rmse(glass, target, np.ones(len(target), dtype=bool)),
            "pairwise_chain_matched_80pct": coordinate_rmse(chained, target, accepted),
            "glass_matched_80pct": coordinate_rmse(glass, target, accepted),
        },
        "reporting": {
            "matched_coverage": float(accepted.mean()),
            "no_call_nodes": np.flatnonzero(~accepted).tolist(),
            "node_discord_pixels": node_discord.tolist(),
            "registration_reliability": reliability.tolist(),
            "note": "The 80% point is evaluation-only in this proxy. A real deployment threshold must be selected on development tiles with landmarks or registration QC.",
        },
        "edge_measurements": edge_report,
        "edge_discord_pixels": edge_discord.tolist(),
        "downstream_contract": "For laminar GLASS, multiply local boundary precision by the registered tile's development-calibrated reliability; do not force a layer result where coordinate discord triggers a no-call.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    images = (
        (reference, "Reference LSFM projection"),
        (observations.mean(axis=0), "Unregistered observations"),
        (mean_registered_image(observations, chained), "Pairwise-chain atlas"),
        # A no-call view is deliberately excluded from the reported atlas.
        (mean_registered_image(observations[accepted], glass[accepted]), "Sheaf-consistent atlas (accepted views)"),
    )
    for axis, (image, title) in zip(axes, images):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    rejected = ", ".join(map(str, report["reporting"]["no_call_nodes"])) or "none"
    fig.suptitle(f"Coordinate sheaf: local overlaps → globally compatible atlas (no-call nodes: {rejected})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "coordinate_atlas_proxy.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
