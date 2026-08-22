"""Controlled local-deformation test on a public BigStitcher microscopy tile."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from sheaf_solver import (
    CoordinateSheaf,
    CoordinateSheafEdge,
    DeformationFieldSheaf,
    DeformationSheafEdge,
    bilinear_restriction,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "bigstitcher" / "raw3d" / "Grid1" / "C1-73.tif"
OUT = ROOT / "results" / "deformation_sheaf_validation"
CONTROL_SHAPE = (7, 7)
N_NODES = 3
RNG = np.random.default_rng(941)
CHART_ORIGINS = np.array(((0.0, 0.0), (0.025, -0.018), (-0.020, 0.022)))
CHART_SCALES = np.array(((1.0, 1.0), (0.955, 1.035), (1.045, 0.965)))


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99.8))
    return np.clip((image - low) / max(high - low, 1e-9), 0.0, 1.0)


def source_image() -> np.ndarray:
    if not SOURCE.exists():
        raise FileNotFoundError("download the public BigStitcher data with code/download_bigstitcher_data.sh")
    volume = tifffile.memmap(SOURCE)
    image = normalize(np.max(volume, axis=0).astype(np.float64))
    return image[128:384, 128:384]


def local_grid() -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, CONTROL_SHAPE[0]),
        np.linspace(0.0, 1.0, CONTROL_SHAPE[1]),
        indexing="ij",
    )
    return np.column_stack((y.ravel(), x.ravel()))


def correction_coefficients(node: int) -> np.ndarray:
    y, x = local_grid().T
    if node == 0:
        field = np.zeros((len(y), 2))
    elif node == 1:
        field = np.column_stack((
            3.2 * np.sin(np.pi * y) * np.sin(1.2 * np.pi * x),
            -2.6 * np.sin(1.1 * np.pi * y) * np.sin(np.pi * x),
        ))
    else:
        field = np.column_stack((
            -2.8 * np.sin(1.3 * np.pi * y) * np.sin(np.pi * x),
            3.0 * np.sin(np.pi * y) * np.cos(1.1 * np.pi * x),
        ))
    return field.ravel()


def sample_field(coefficients: np.ndarray, points_yx: np.ndarray) -> np.ndarray:
    return (bilinear_restriction(CONTROL_SHAPE, points_yx) @ coefficients).reshape(-1, 2)


def physical_pixel_field(coefficients: np.ndarray, shape: tuple[int, int], node: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(0.0, 1.0, shape[1]),
        indexing="ij",
    )
    points = (np.column_stack((y.ravel(), x.ravel())) - CHART_ORIGINS[node]) / CHART_SCALES[node]
    return sample_field(coefficients, points).reshape(*shape, 2)


def warp(image: np.ndarray, correction: np.ndarray, direction: float) -> np.ndarray:
    y, x = np.indices(image.shape, dtype=np.float64)
    return map_coordinates(
        image,
        (y + direction * correction[..., 0], x + direction * correction[..., 1]),
        order=1,
        mode="reflect",
    )


def local_overlap_edges(observations: np.ndarray) -> list[DeformationSheafEdge]:
    y, x = np.meshgrid(np.linspace(0.12, 0.88, 7), np.linspace(0.12, 0.88, 7), indexing="ij")
    physical_points = np.column_stack((y.ravel(), x.ravel()))
    edges = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        restriction_i = bilinear_restriction(CONTROL_SHAPE, (physical_points - CHART_ORIGINS[i]) / CHART_SCALES[i])
        restriction_j = bilinear_restriction(CONTROL_SHAPE, (physical_points - CHART_ORIGINS[j]) / CHART_SCALES[j])
        shifts = []
        for point_y, point_x in physical_points:
            center_y = round(point_y * (observations.shape[1] - 1))
            center_x = round(point_x * (observations.shape[2] - 1))
            window = 20
            reference = observations[i, center_y - window:center_y + window, center_x - window:center_x + window]
            moving = observations[j, center_y - window:center_y + window, center_x - window:center_x + window]
            shifts.append(phase_cross_correlation(reference, moving, upsample_factor=5)[0])
        relative = np.asarray(shifts, dtype=np.float64).ravel()
        edges.append(DeformationSheafEdge(i, j, 1.0, restriction_i, restriction_j, relative))
    return edges


def translation_baseline(edges: list[DeformationSheafEdge]) -> np.ndarray:
    offsets = [
        CoordinateSheafEdge(edge.i, edge.j, edge.weight, tuple(edge.relative_displacement.reshape(-1, 2).mean(axis=0)))
        for edge in edges
    ]
    mean = np.zeros((N_NODES, 2))
    variance = np.full_like(mean, 1e5)
    variance[0] = 1e-6
    shifts, _, _ = CoordinateSheaf(N_NODES, offsets).solve(mean, variance)
    return np.repeat(shifts[:, None, :], np.prod(CONTROL_SHAPE), axis=1).reshape(N_NODES, -1)


def metric(image: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(image - reference))),
        "ssim": float(structural_similarity(image, reference, data_range=1.0)),
    }


def fault_qc(truth: np.ndarray, clean_edges: list[DeformationSheafEdge]) -> dict[str, float]:
    edges = list(clean_edges)
    samples_per_edge = edges[-1].restriction_i.shape[0] // 2
    bad_samples = np.zeros(samples_per_edge, dtype=bool)
    bad_samples[-8:] = True
    corrupted = edges[-1].relative_displacement.copy().reshape(-1, 2)
    corrupted[bad_samples] += np.array((5.0, -4.0))
    edges[-1] = replace(edges[-1], relative_displacement=corrupted.ravel())
    mean = np.zeros_like(truth)
    variance = np.full_like(truth, 1e6)
    variance[0] = 1e-6
    solver = DeformationFieldSheaf(N_NODES, CONTROL_SHAPE, edges)
    estimate, _, _ = solver.solve(mean, variance, consistency_weight=1.0, smoothness_weight=0.06)
    residual = solver.coboundary() @ estimate.ravel() - np.concatenate([edge.relative_displacement for edge in edges])
    score = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    truth_mask = np.zeros(len(score), dtype=bool)
    truth_mask[-samples_per_edge:] = bad_samples
    no_call = score >= np.quantile(score, 0.90)
    true_positives = np.count_nonzero(no_call & truth_mask)
    return {
        "injected_bad_patch_fraction": float(np.mean(truth_mask)),
        "no_call_fraction": float(np.mean(no_call)),
        "no_call_precision": float(true_positives / max(np.count_nonzero(no_call), 1)),
        "no_call_recall": float(true_positives / np.count_nonzero(truth_mask)),
        "mean_residual_bad_patch_px": float(np.mean(score[truth_mask])),
        "mean_residual_elsewhere_px": float(np.mean(score[~truth_mask])),
    }


def plot(reference: np.ndarray, translation: np.ndarray, sheaf: np.ndarray, truth: np.ndarray, estimate: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = (
        (reference, "Public microscopy reference"),
        (translation, "Global translation baseline"),
        (sheaf, "Local deformation sheaf"),
        (np.abs(translation - reference), "Translation absolute error"),
        (np.abs(sheaf - reference), "Sheaf absolute error"),
        (np.linalg.norm((estimate - truth).reshape(N_NODES, -1, 2)[2], axis=1).reshape(CONTROL_SHAPE), "Node 2 control-field error"),
    )
    for axis, (image, title) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap="magma" if "error" in title else "gray", vmin=0, vmax=0.3 if "error" in title else 1)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.savefig(OUT / "comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = source_image()
    truth = np.stack([correction_coefficients(node) for node in range(N_NODES)])
    observations = np.stack([
        warp(reference, physical_pixel_field(field, reference.shape, node), 1.0)
        for node, field in enumerate(truth)
    ])
    observations = np.clip(observations + RNG.normal(0.0, 0.012, size=observations.shape), 0.0, 1.0)
    edges = local_overlap_edges(observations)
    local_mean = np.zeros_like(truth)
    local_variance = np.full_like(truth, 1e6)
    local_variance[0] = 1e-6
    estimate, edge_discord, node_discord = DeformationFieldSheaf(N_NODES, CONTROL_SHAPE, edges).solve(
        local_mean, local_variance, consistency_weight=1.0, smoothness_weight=0.06
    )
    translation_field = translation_baseline(edges)
    sheaf_reconstruction = np.mean(
        [
            warp(image, physical_pixel_field(field, reference.shape, node), -1.0)
            for node, (image, field) in enumerate(zip(observations, estimate))
        ], axis=0
    )
    translation_reconstruction = np.mean(
        [
            warp(image, physical_pixel_field(field, reference.shape, node), -1.0)
            for node, (image, field) in enumerate(zip(observations, translation_field))
        ], axis=0
    )
    result = {
        "purpose": "controlled public-microscopy test of local deformation-field recovery; pairwise local displacements are estimated by patch phase correlation after synthetic local warps and noise",
        "source": "BigStitcher public microscopy tile C1-73, maximum-intensity projection",
        "field_control_mae_px": {
            "global_translation": float(np.mean(np.abs(translation_field - truth))),
            "deformation_sheaf": float(np.mean(np.abs(estimate - truth))),
        },
        "reconstruction_against_clean_reference": {
            "global_translation": metric(translation_reconstruction, reference),
            "deformation_sheaf": metric(sheaf_reconstruction, reference),
        },
        "edge_discord_px": edge_discord.tolist(),
        "node_discord_px": node_discord.tolist(),
        "localized_fault_qc": fault_qc(truth, edges),
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / "fields.npz", truth=truth, estimate=estimate, translation=translation_field)
    plot(reference, translation_reconstruction, sheaf_reconstruction, truth, estimate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
