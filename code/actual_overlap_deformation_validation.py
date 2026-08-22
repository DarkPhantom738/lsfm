"""Held-out real-overlap validation for the local deformation sheaf."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

from make_figures import regular_grid_positions
from sheaf_solver import (
    CoordinateSheaf,
    CoordinateSheafEdge,
    DeformationFieldSheaf,
    DeformationSheafEdge,
    bilinear_restriction,
)


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "bigstitcher" / "raw3d" / "Grid1"
OUT = ROOT / "results" / "actual_overlap_deformation_validation"
CONTROL_SHAPE = (3, 3)
PAIRS = ((0, 1), (0, 2), (1, 3), (2, 3))
SMOOTHNESS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def load_images() -> list[np.ndarray]:
    return [np.max(tifffile.memmap(RAW / f"C1-{73 + node}.tif"), axis=0).astype(np.float64) for node in range(4)]


def local_point(global_yx: np.ndarray, position_xy: np.ndarray) -> np.ndarray:
    return np.column_stack(((global_yx[:, 0] - position_xy[1]) / 511.0, (global_yx[:, 1] - position_xy[0]) / 511.0))


def overlap_measurements(images: list[np.ndarray], positions: np.ndarray) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    output = []
    radius = 16
    for i, j in PAIRS:
        low_xy = np.maximum(positions[i], positions[j])
        high_xy = np.minimum(positions[i] + 512, positions[j] + 512)
        y_values = np.linspace(low_xy[1] + radius + 2, high_xy[1] - radius - 2, 6)
        x_values = np.linspace(low_xy[0] + radius + 2, high_xy[0] - radius - 2, 4)
        points, shifts = [], []
        for global_y in y_values:
            for global_x in x_values:
                local_i = np.rint((global_y - positions[i, 1], global_x - positions[i, 0])).astype(int)
                local_j = np.rint((global_y - positions[j, 1], global_x - positions[j, 0])).astype(int)
                patch_i = images[i][local_i[0] - radius:local_i[0] + radius, local_i[1] - radius:local_i[1] + radius]
                patch_j = images[j][local_j[0] - radius:local_j[0] + radius, local_j[1] - radius:local_j[1] + radius]
                points.append((global_y, global_x))
                shifts.append(phase_cross_correlation(patch_i, patch_j, upsample_factor=5)[0])
        output.append((i, j, np.asarray(points), np.asarray(shifts)))
    return output


def selection(size: int, split: str) -> np.ndarray:
    indices = np.arange(size)
    if split == "train":
        return indices % 5 >= 2
    if split == "development":
        return indices % 5 == 1
    if split == "train_development":
        return indices % 5 != 0
    if split == "test":
        return indices % 5 == 0
    raise ValueError(split)


def deformation_edges(data: list[tuple[int, int, np.ndarray, np.ndarray]], positions: np.ndarray, split: str) -> list[DeformationSheafEdge]:
    edges = []
    for i, j, points, shifts in data:
        keep = selection(len(points), split)
        edges.append(DeformationSheafEdge(
            i,
            j,
            1.0,
            bilinear_restriction(CONTROL_SHAPE, local_point(points[keep], positions[i])),
            bilinear_restriction(CONTROL_SHAPE, local_point(points[keep], positions[j])),
            shifts[keep].ravel(),
        ))
    return edges


def deformation_solution(edges: list[DeformationSheafEdge], smoothness: float) -> np.ndarray:
    dimension = 2 * np.prod(CONTROL_SHAPE)
    mean = np.zeros((4, dimension))
    variance = np.full_like(mean, 1e6)
    variance[0] = 1e-6
    return DeformationFieldSheaf(4, CONTROL_SHAPE, edges).solve(mean, variance, 1.0, smoothness)[0]


def deformation_errors(solution: np.ndarray, data: list[tuple[int, int, np.ndarray, np.ndarray]], positions: np.ndarray, split: str) -> np.ndarray:
    errors = []
    for i, j, points, shifts in data:
        keep = selection(len(points), split)
        left = bilinear_restriction(CONTROL_SHAPE, local_point(points[keep], positions[i]))
        right = bilinear_restriction(CONTROL_SHAPE, local_point(points[keep], positions[j]))
        predicted = (right @ solution[j] - left @ solution[i]).reshape(-1, 2)
        errors.extend(np.linalg.norm(predicted - shifts[keep], axis=1))
    return np.asarray(errors)


def translation_errors(edges: list[DeformationSheafEdge], data: list[tuple[int, int, np.ndarray, np.ndarray]], split: str) -> np.ndarray:
    coordinate_edges = [
        CoordinateSheafEdge(edge.i, edge.j, edge.weight, tuple(edge.relative_displacement.reshape(-1, 2).mean(axis=0)))
        for edge in edges
    ]
    mean = np.zeros((4, 2))
    variance = np.full_like(mean, 1e6)
    variance[0] = 1e-6
    shifts, _, _ = CoordinateSheaf(4, coordinate_edges).solve(mean, variance)
    errors = []
    for i, j, _, observed in data:
        keep = selection(len(observed), split)
        errors.extend(np.linalg.norm((shifts[j] - shifts[i]) - observed[keep], axis=1))
    return np.asarray(errors)


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError("download the public BigStitcher data with code/download_bigstitcher_data.sh")
    positions = regular_grid_positions()[:4]
    data = overlap_measurements(load_images(), positions)
    training = deformation_edges(data, positions, "train")
    development = {
        value: deformation_errors(deformation_solution(training, value), data, positions, "development")
        for value in SMOOTHNESS
    }
    chosen_smoothness = min(development, key=lambda value: float(np.mean(development[value])))
    final_edges = deformation_edges(data, positions, "train_development")
    deformation = deformation_errors(
        deformation_solution(final_edges, chosen_smoothness), data, positions, "test"
    )
    translation = translation_errors(final_edges, data, "test")
    report = {
        "purpose": "held-out local-registration agreement on actual public microscopy overlaps; no synthetic image warp is used",
        "source": "four raw BigStitcher confocal tiles, channel 1, with published regular-grid locations",
        "evidence": "patch phase-correlation shifts on true physical overlap regions",
        "split": "per overlap: 60% train, 20% development, 20% untouched test",
        "selected_smoothness_on_development": chosen_smoothness,
        "heldout_shift_residual_px": {
            "global_translation_mean": float(np.mean(translation)),
            "global_translation_median": float(np.median(translation)),
            "deformation_sheaf_mean": float(np.mean(deformation)),
            "deformation_sheaf_median": float(np.median(deformation)),
        },
        "conclusion": "No held-out improvement on this actual mostly-rigid dataset; do not claim a real-data deformation advantage.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.boxplot((translation, deformation), tick_labels=("Global translation", "Deformation sheaf"), showfliers=False)
    axis.set_ylabel("held-out patch-shift residual (pixels)")
    axis.set_title("Actual BigStitcher overlaps: lower is better")
    figure.tight_layout()
    figure.savefig(OUT / "heldout_residuals.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
