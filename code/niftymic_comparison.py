#!/usr/bin/env python3
"""Controlled DANDI LSFM comparison: NiftyMIC alone versus sheaf coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
from nibabel.processing import resample_from_to
from numcodecs import Blosc
from scipy.ndimage import shift as nd_shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from sheaf_solver import CoordinateSheaf, CoordinateSheafEdge

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "niftymic_comparison"
INPUTS = OUT / "inputs"
ZARR_ID = "bf47be1a-4fed-4105-bcb4-c52534a45b82"  # DANDI 000108 sample 127 YO
RNG = np.random.default_rng(2801)
N_STACKS, N_PLANES = 3, 8
FAILED = (2, 5)  # stack, z-plane; deliberately no usable signal


def fetch_reference() -> np.ndarray:
    root = f"https://dandiarchive.s3.amazonaws.com/zarr/{ZARR_ID}/0"
    meta = requests.get(f"{root}/.zarray", timeout=30).json()
    chunks = tuple(meta["chunks"])
    if chunks != (1, 1, 128, 128, 128):
        raise RuntimeError(f"unexpected public Zarr chunk shape: {chunks}")
    blob = requests.get(f"{root}/0/0/0/0/100", timeout=30)
    blob.raise_for_status()
    arr = np.empty(chunks, dtype=np.dtype(meta["dtype"]))
    Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).decode(blob.content, out=arr)
    source = arr[0, 0, 60 : 60 + N_PLANES].astype(np.float64)
    lo, hi = np.percentile(source, (1, 99.5))
    return np.clip((source - lo) / max(hi - lo, 1e-8), 0, 1)


def write_volume(path: Path, planes_yx: np.ndarray, mask_yx: np.ndarray | None = None) -> None:
    """Write a conventional thick-slice x,y,z NIfTI stack for NiftyMIC."""
    values = np.transpose(planes_yx, (2, 1, 0)).astype(np.float32)
    if mask_yx is not None:
        values = np.transpose(mask_yx, (2, 1, 0)).astype(np.uint8)
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(values, affine), path)


def make_acquisitions(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three short acquisitions of the same local LSFM volume."""
    observations = np.empty((N_STACKS, N_PLANES, *reference.shape[1:]), dtype=np.float64)
    applied = np.zeros((N_STACKS, N_PLANES, 2), dtype=np.float64)
    for stack in range(N_STACKS):
        for z in range(N_PLANES):
            if stack:
                applied[stack, z] = RNG.uniform(-4.5, 4.5, size=2)
            frame = nd_shift(reference[z], applied[stack, z], order=1, mode="nearest")
            frame = np.clip(frame + RNG.normal(0, 0.012, frame.shape), 0, 1)
            if stack:
                y, x = RNG.integers(8, 99, size=2)
                frame[y : y + 15, x : x + 16] = np.median(frame)
            observations[stack, z] = frame
    observations[FAILED] = RNG.uniform(0, 1, size=reference.shape[1:])
    return observations, applied


def local_relation(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float]:
    displacement, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=10, normalization=None)
    aligned = nd_shift(moving, displacement, order=1, mode="nearest")
    corr = float(np.corrcoef(fixed.ravel(), aligned.ravel())[0, 1])
    return displacement.astype(np.float64), float(np.clip(corr, 0.01, 1.0) ** 2)


def widest_path_support(n_nodes: int, edges: list[CoordinateSheafEdge], anchors: list[int]) -> np.ndarray:
    """Best all-reliable-overlap path to an atlas anchor for each node."""
    support = np.zeros(n_nodes, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for edge in edges:
        adjacency[edge.i].append((edge.j, edge.weight))
        adjacency[edge.j].append((edge.i, edge.weight))
    for anchor in anchors:
        support[anchor] = 1.0
        frontier = [(1.0, anchor)]
        while frontier:
            value, node = max(frontier)
            frontier.remove((value, node))
            if value < support[node] - 1e-12:
                continue
            for neighbor, weight in adjacency[node]:
                candidate = min(value, weight)
                if candidate > support[neighbor]:
                    support[neighbor] = candidate
                    frontier.append((candidate, neighbor))
    return support


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (INPUTS / "nifty_only").mkdir(parents=True, exist_ok=True)
    (INPUTS / "glass_assisted").mkdir(parents=True, exist_ok=True)
    (INPUTS / "glass_fixed").mkdir(parents=True, exist_ok=True)
    reference = fetch_reference()
    observations, applied = make_acquisitions(reference)
    n_nodes = N_STACKS * N_PLANES
    node = lambda stack, z: stack * N_PLANES + z
    edges: list[CoordinateSheafEdge] = []
    edge_report = []
    for z in range(N_PLANES):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            offset, weight = local_relation(observations[i, z], observations[j, z])
            edges.append(CoordinateSheafEdge(node(i, z), node(j, z), weight, tuple(offset)))
            edge_report.append({"z": z, "i": i, "j": j, "weight": weight, "offset_yx": offset.tolist()})
    local_mean = np.zeros((n_nodes, 2), dtype=np.float64)
    local_variance = np.full_like(local_mean, 1e5)
    anchors = [node(0, z) for z in range(N_PLANES)]
    local_variance[anchors] = 1e-6
    correction, edge_discord, node_discord = CoordinateSheaf(n_nodes, edges).solve(
        local_mean, local_variance, consistency_weight=1.0
    )
    support = widest_path_support(n_nodes, edges, anchors)
    accepted = support >= 0.25
    accepted[anchors] = True
    correction = correction.reshape(N_STACKS, N_PLANES, 2)
    accepted = accepted.reshape(N_STACKS, N_PLANES)
    support = support.reshape(N_STACKS, N_PLANES)
    node_discord = node_discord.reshape(N_STACKS, N_PLANES)
    glass_observations = np.empty_like(observations)
    for stack in range(N_STACKS):
        for z in range(N_PLANES):
            glass_observations[stack, z] = nd_shift(observations[stack, z], correction[stack, z], order=1, mode="nearest")
    all_mask = np.ones_like(observations, dtype=np.uint8)
    glass_mask = np.broadcast_to(accepted[..., None, None], observations.shape).astype(np.uint8)
    for stack in range(N_STACKS):
        write_volume(INPUTS / "nifty_only" / f"stack_{stack}.nii.gz", observations[stack])
        write_volume(INPUTS / "nifty_only" / f"mask_{stack}.nii.gz", all_mask[stack])
        write_volume(INPUTS / "glass_assisted" / f"stack_{stack}.nii.gz", glass_observations[stack])
        write_volume(INPUTS / "glass_assisted" / f"mask_{stack}.nii.gz", glass_mask[stack])
        write_volume(INPUTS / "glass_fixed" / f"stack_{stack}.nii.gz", glass_observations[stack])
        write_volume(INPUTS / "glass_fixed" / f"mask_{stack}.nii.gz", glass_mask[stack])
    write_volume(INPUTS / "reference.nii.gz", reference)
    expected = -applied
    good = np.ones((N_STACKS, N_PLANES), dtype=bool)
    good[FAILED] = False
    report = {
        "purpose": "controlled public-LSFM NiftyMIC-only versus coordinate-GLASS-assisted-NiftyMIC comparison",
        "source": f"DANDI:000108 sample 127 YO; {N_PLANES} clean planes used as known reference, then re-acquired synthetically three times",
        "conditions": {
            "nifty_only": "raw shifted/noisy/occluded stacks including the signal-absent slice",
            "glass_assisted": "coordinate-sheaf aligned stacks; observations without a reliable overlap path are excluded by the NiftyMIC mask, then standard NiftyMIC re-registers",
            "glass_fixed": "same coordinate-sheaf aligned stacks and mask, but reconstruction-only NiftyMIC does not re-register",
        },
        "failed_observation_stack_z": list(FAILED),
        "glass_qc": {"accepted_observations": int(accepted.sum()), "total_observations": int(accepted.size), "rejected_stack_z": np.argwhere(~accepted).tolist(), "anchor_path_support": support.tolist(), "node_discord_pixels": node_discord.tolist()},
        "coordinate_rmse_pixels_on_nonfailed_views": float(np.sqrt(np.mean((correction[good] - expected[good]) ** 2))),
        "edge_measurements": edge_report,
        "evaluation_note": "NiftyMIC outputs must be generated before running --evaluate. Volume scores are resampled to the known reference grid and are controlled-proxy results only.",
    }
    (OUT / "setup.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float(-10 * np.log10(max(mse, 1e-12)))


def score_volume(output: Path, reference: nib.Nifti1Image) -> dict:
    reconstructed = resampled_volume(output, reference)
    truth = reference.get_fdata().astype(np.float64)
    ssim = float(np.mean([structural_similarity(reconstructed[:, :, z], truth[:, :, z], data_range=1.0) for z in range(truth.shape[2])]))
    return {"ncc": ncc(reconstructed, truth), "psnr": psnr(reconstructed, truth), "mean_slice_ssim": ssim}


def resampled_volume(output: Path, reference: nib.Nifti1Image) -> np.ndarray:
    return resample_from_to(nib.load(output), (reference.shape, reference.affine), order=1).get_fdata().astype(np.float64)


def evaluate() -> None:
    reference = nib.load(INPUTS / "reference.nii.gz")
    outputs = {name: OUT / name / "srr_volume.nii.gz" for name in ("nifty_only", "glass_assisted", "glass_fixed")}
    missing = [str(path) for path in outputs.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing NiftyMIC output(s): " + ", ".join(missing))
    scores = {name: score_volume(path, reference) for name, path in outputs.items()}
    result = {
        "purpose": "controlled NiftyMIC reconstruction comparison; not a real-LSFM benchmark",
        "volume_metrics_against_known_clean_reference": scores,
        "delta_reregistered_glass_minus_nifty": {key: scores["glass_assisted"][key] - scores["nifty_only"][key] for key in scores["nifty_only"]},
        "delta_fixed_coordinate_glass_minus_nifty": {key: scores["glass_fixed"][key] - scores["nifty_only"][key] for key in scores["nifty_only"]},
        "delta_fixed_coordinate_minus_reregistered_glass": {key: scores["glass_fixed"][key] - scores["glass_assisted"][key] for key in scores["glass_assisted"]},
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    z = 0
    truth = reference.get_fdata().astype(np.float64)
    volumes = {name: resampled_volume(path, reference) for name, path in outputs.items()}
    fig = plt.figure(figsize=(14, 5.2))
    axes = fig.subplot_mosaic([["0", "1", "2", "3", "4"], ["mae", "mae", "mae", "mae", "mae"]], height_ratios=(12, 1.5))
    panels = (
        (truth[:, :, z], "Known clean LSFM plane"),
        (volumes["nifty_only"][:, :, z], "NiftyMIC only"),
        (np.abs(volumes["nifty_only"][:, :, z] - truth[:, :, z]), "NiftyMIC absolute error"),
        (volumes["glass_assisted"][:, :, z], "Sheaf coordinates → NiftyMIC"),
        (np.abs(volumes["glass_assisted"][:, :, z] - truth[:, :, z]), "Sheaf-coordinate\nabsolute error"),
    )
    for key, (image, title) in zip(("0", "1", "2", "3", "4"), panels):
        axis = axes[key]
        axis.imshow(image, cmap="magma" if "error" in title else "gray", vmin=0, vmax=0.4 if "error" in title else 1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    mae_nifty = np.mean(np.abs(volumes["nifty_only"] - truth), axis=(0, 1))
    mae_sheaf = np.mean(np.abs(volumes["glass_assisted"] - truth), axis=(0, 1))
    axes["mae"].bar(np.arange(N_PLANES) - 0.2, mae_nifty, width=0.4, label="NiftyMIC", color="#e07a5f")
    axes["mae"].bar(np.arange(N_PLANES) + 0.2, mae_sheaf, width=0.4, label="Sheaf → NiftyMIC", color="#2a9d8f")
    axes["mae"].set_ylabel("MAE")
    axes["mae"].set_xlabel("depth z")
    axes["mae"].set_xticks(range(N_PLANES))
    axes["mae"].legend(fontsize=8, ncol=2, loc="upper center")
    fig.suptitle("Same NiftyMIC engine; sheaf coordinates help at 6/8 depths (displayed: z = 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_slice.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=2801, help="corruption seed; the default is the headline demonstration")
    args = parser.parse_args()
    if args.prepare == args.evaluate:
        parser.error("choose exactly one of --prepare or --evaluate")
    if args.seed != 2801:
        OUT = ROOT / "results" / "niftymic_comparison" / "robustness" / f"seed_{args.seed}"
        INPUTS = OUT / "inputs"
    RNG = np.random.default_rng(args.seed)
    prepare() if args.prepare else evaluate()
