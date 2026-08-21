#!/usr/bin/env python3
"""Held-out BigBrain cortical-boundary proxy benchmark."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sheaf_solver import CorticalBoundarySheaf, nearby_column_edges, ordered_projection

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sota_colab" / "data_profiles_sota"
GEOMETRY = DATA / "glass_geometry"
OUT = ROOT / "results" / "bigbrain_layer_proxy"
DEV_SECTIONS = {"s5431", "s6316"}
LAMBDA_GRID = (0.0, 0.03, 0.10, 0.30, 1.0, 3.0)
RNG = np.random.default_rng(2601)


def layer_boundaries(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return five I/II..V/VI boundaries and a complete-profile mask."""
    output = np.full((len(labels), 5), np.nan, dtype=np.float64)
    for row, sequence in enumerate(labels):
        for layer in range(1, 6):
            left = np.flatnonzero(sequence == layer)
            right = np.flatnonzero(sequence == layer + 1)
            if len(left) and len(right):
                output[row, layer - 1] = (left[-1] + right[0]) / (2 * (len(sequence) - 1))
    valid = np.isfinite(output).all(axis=1) & np.all(np.diff(output, axis=1) > 0, axis=1)
    return output, valid


def corrupt_profiles(profiles: np.ndarray, seed: int, severity: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Two deterministic LSFM-like 1D views; returns view stack and dropout rate."""
    rng = np.random.default_rng(seed)
    views = []
    dropout = []
    for view in range(2):
        x = profiles.copy()
        for row in range(len(x)):
            gain = rng.uniform(1.0 - 0.25 * severity, 1.0 + 0.25 * severity)
            offset = rng.normal(0, 0.025 * severity)
            blur = rng.uniform(0.6, 1.5) * severity
            noisy = gaussian_filter1d(x[row], blur) * gain + offset + rng.normal(0, 0.035 * severity, len(x[row]))
            width = int(rng.integers(max(3, int(7 * severity)), min(len(x[row]) - 8, int(19 * severity) + 1)))
            start = int(rng.integers(4, len(x[row]) - width - 4))
            noisy[start : start + width] = np.nan
            good = np.isfinite(noisy)
            noisy[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), noisy[good])
            x[row] = np.clip(noisy, 0, 1)
            dropout.append(width / len(x[row]))
        views.append(x)
    return np.stack(views), np.asarray(dropout, dtype=np.float64).reshape(2, len(profiles)).mean(axis=0)


def make_evidence_model() -> object:
    return make_pipeline(StandardScaler(), PCA(n_components=24, random_state=0), Ridge(alpha=12.0))


def section_geometry(section: str, expected: int) -> list[dict]:
    path = GEOMETRY / f"{section}.json"
    if not path.exists():
        raise RuntimeError(f"missing bundled geometry for {section}; restore the compact benchmark data")
    geometry = json.loads(path.read_text())
    if len(geometry) != expected:
        raise RuntimeError(f"geometry/profile count mismatch for {section}: {len(geometry)} != {expected}")
    return geometry


def profile_warp(profile_i: np.ndarray, profile_j: np.ndarray, max_shift: int = 12) -> np.ndarray:
    """Estimate a band-limited monotone profile correspondence."""
    a = (np.asarray(profile_i, dtype=np.float64) - np.mean(profile_i)) / (np.std(profile_i) + 1e-6)
    b = (np.asarray(profile_j, dtype=np.float64) - np.mean(profile_j)) / (np.std(profile_j) + 1e-6)
    n = len(a)
    if len(b) != n:
        raise ValueError("profile restriction maps need equal-length profiles")
    cost = np.full((n + 1, n + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - max_shift), min(n, i + max_shift) + 1):
            cost[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    if not np.isfinite(cost[n, n]):
        return np.linspace(0.0, 1.0, n)
    pairs = []
    i = j = n
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        candidates = (cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        step = int(np.argmin(candidates))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i, j = i - 1, j - 1
    pairs.reverse()
    x = np.asarray([pair[0] for pair in pairs])
    y = np.asarray([pair[1] for pair in pairs])
    unique_x = np.unique(x)
    mean_y = np.asarray([y[x == value].mean() for value in unique_x])
    return np.interp(np.arange(n), unique_x, mean_y) / max(n - 1, 1)


def affine_profile_restriction(profile_i: np.ndarray, profile_j: np.ndarray, depths_i: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Linearize the image-derived local overlap chart at five boundaries."""
    warp = profile_warp(profile_i, profile_j)
    grid = np.linspace(0.0, 1.0, len(warp))
    step = 1.0 / max(len(warp) - 1, 1)
    slope, offset = [], []
    for depth in depths_i:
        lo, hi = np.clip((depth - step, depth + step), 0.0, 1.0)
        mapped_lo, mapped_hi = np.interp((lo, hi), grid, warp)
        local_slope = float(np.clip((mapped_hi - mapped_lo) / max(hi - lo, 1e-6), 0.35, 2.5))
        mapped = float(np.interp(depth, grid, warp))
        slope.append(local_slope)
        offset.append(-(mapped - local_slope * depth))
    return tuple(slope), tuple(offset)


def section_solve(
    section: str,
    local_mean: np.ndarray,
    local_variance: np.ndarray,
    geometry: list[dict],
    evidence_profiles: np.ndarray,
    lam: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    yx = np.asarray([[g["seed_y"], g["seed_x"]] for g in geometry])
    length = np.asarray([g["length_px"] for g in geometry])
    edges = nearby_column_edges(yx, length, max_neighbors=4, radius_px=110.0)
    if not edges:
        raise RuntimeError(f"no physical sheaf edges for {section}")
    if mode == "affine":
        mean_profile = np.mean(evidence_profiles, axis=0)
        aligned_edges = []
        for edge in edges:
            scales, offsets = affine_profile_restriction(
                mean_profile[edge.i], mean_profile[edge.j], local_mean[edge.i]
            )
            aligned_edges.append(
                replace(
                    edge,
                    scale_i=tuple(scales),
                    scale_j=(1.0,) * 5,
                    relative_offset=tuple(offsets),
                )
            )
        edges = aligned_edges
    sheaf = CorticalBoundarySheaf(len(local_mean), edges)
    if mode == "affine":
        prediction, _, discord = sheaf.solve(local_mean, local_variance, lam, identity_restrictions=False)
    elif mode == "graph":
        prediction, _, discord = sheaf.solve(local_mean, local_variance, lam, identity_restrictions=True)
    else:
        raise ValueError(f"unknown GLASS solve mode: {mode}")
    return prediction, discord


def boundary_mae(prediction: np.ndarray, truth: np.ndarray, use: np.ndarray | None = None) -> float:
    if use is None:
        use = np.ones(len(prediction), dtype=bool)
    return float(np.mean(np.abs(prediction[use] - truth[use]))) if use.any() else float("nan")


def boundaries_to_layer_labels(boundaries: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Turn five ordered internal boundaries into six layer labels per column."""
    depth = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    return (depth[None, :, None] > boundaries[:, None, :]).sum(axis=2).astype(np.int64) + 1


def mean_layer_dice(prediction: np.ndarray, truth_labels: np.ndarray, use: np.ndarray) -> float:
    """Macro Dice for layers I–VI on the reported complete columns only."""
    if not use.any():
        return float("nan")
    predicted_labels = boundaries_to_layer_labels(prediction[use])
    target = truth_labels[use]
    scores = []
    for layer in range(1, 7):
        p, t = predicted_labels == layer, target == layer
        denom = p.sum() + t.sum()
        scores.append(float(2 * np.logical_and(p, t).sum() / denom) if denom else 1.0)
    return float(np.mean(scores))


def summarize_method(prediction: np.ndarray, truth: np.ndarray, discord: np.ndarray, threshold: float) -> dict:
    accepted = discord <= threshold
    all_mae = boundary_mae(prediction, truth)
    accepted_mae = boundary_mae(prediction, truth, accepted)
    rejected_mae = boundary_mae(prediction, truth, ~accepted)
    invalid = float((np.diff(prediction, axis=1) <= 0).mean())
    return {
        "all_boundary_mae_normalized_depth": all_mae,
        "accepted_coverage": float(accepted.mean()),
        "accepted_boundary_mae_normalized_depth": accepted_mae,
        "rejected_boundary_mae_normalized_depth": rejected_mae,
        "invalid_order_rate": invalid,
    }


def matched_coverage_metric(prediction: np.ndarray, truth: np.ndarray, labels: np.ndarray, discord: np.ndarray, coverage: float = 0.80) -> dict:
    """Return a matched-coverage evaluation summary."""
    threshold = float(np.quantile(discord, coverage))
    accepted = discord <= threshold
    return {
        "coverage": float(accepted.mean()),
        "accepted_boundary_mae_normalized_depth": boundary_mae(prediction, truth, accepted),
        "accepted_macro_dice_layers_I_VI": mean_layer_dice(prediction, labels, accepted),
        "evaluation_threshold": threshold,
    }


def main(seed: int = 2601, severity: float = 1.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(DATA / "profiles.npz")
    profiles, labels = archive["P"][:, 0].astype(np.float64), archive["L"].astype(np.int64)
    metadata = json.loads((DATA / "meta.json").read_text())
    split = json.loads((DATA / "splits.json").read_text())
    sections = np.asarray([item["section"] for item in metadata])
    boundaries, complete = layer_boundaries(labels)
    heldout = np.isin(sections, split["holdout_sections"])
    development = np.isin(sections, list(DEV_SECTIONS))
    calibration = ~heldout & ~development & complete
    dev = development & complete
    test = heldout & complete
    if calibration.sum() < 100 or dev.sum() < 10 or test.sum() < 10:
        raise RuntimeError("insufficient complete profiles for GLASS split")

    if severity <= 0:
        raise ValueError("severity must be positive")
    views, dropout = corrupt_profiles(profiles, seed=seed, severity=severity)
    model = make_evidence_model()
    train_x = np.concatenate([views[0, calibration], views[1, calibration]])
    train_y = np.concatenate([boundaries[calibration], boundaries[calibration]])
    model.fit(train_x, train_y)
    pred_a, pred_b = model.predict(views[0]), model.predict(views[1])
    local = ordered_projection((pred_a + pred_b) / 2)
    base_variance = np.var(local[dev] - boundaries[dev], axis=0, ddof=1)
    disagreement = ((pred_a - pred_b) / 2) ** 2
    variance = np.clip(base_variance[None] + disagreement + (0.03 * dropout[:, None]) ** 2, 1e-5, 0.20)

    dev_records = []
    for lam in LAMBDA_GRID:
        for mode, name in (("affine", "glass"), ("graph", "graph")):
            chunks = []
            for section in sorted(DEV_SECTIONS):
                ids = np.flatnonzero((sections == section) & complete)
                geom_all = section_geometry(section, int((sections == section).sum()))
                geometry = [geom_all[metadata[index]["i"]] for index in ids]
                prediction, discord = section_solve(section, local[ids], variance[ids], geometry, views[:, ids], lam, mode)
                chunks.append((prediction, boundaries[ids], discord))
            dev_records.append({"name": name, "lambda": lam, "mae": float(np.mean([boundary_mae(p, t) for p, t, _ in chunks])), "discord": np.concatenate([d for _, _, d in chunks])})
    selected = {}
    for name in ("glass", "graph"):
        candidates = [item for item in dev_records if item["name"] == name]
        selected[name] = min(candidates, key=lambda item: item["mae"])
    threshold = {name: float(np.quantile(selected[name]["discord"], 0.80)) for name in selected}
    local_discord = np.sqrt(np.mean(variance, axis=1))
    local_threshold = float(np.quantile(local_discord[dev], 0.80))

    all_predictions = {"local": [], "graph": [], "glass": []}
    all_truth, all_labels, all_discord = [], [], {"local": [], "graph": [], "glass": []}
    examples = []
    for section in split["holdout_sections"]:
        ids = np.flatnonzero((sections == section) & complete)
        geom_all = section_geometry(section, int((sections == section).sum()))
        geometry = [geom_all[metadata[index]["i"]] for index in ids]
        graph_pred, graph_discord = section_solve(section, local[ids], variance[ids], geometry, views[:, ids], selected["graph"]["lambda"], "graph")
        glass_pred, glass_discord = section_solve(section, local[ids], variance[ids], geometry, views[:, ids], selected["glass"]["lambda"], "affine")
        all_predictions["local"].append(local[ids])
        all_predictions["graph"].append(graph_pred)
        all_predictions["glass"].append(glass_pred)
        all_truth.append(boundaries[ids])
        all_labels.append(labels[ids])
        all_discord["local"].append(local_discord[ids])
        all_discord["graph"].append(graph_discord)
        all_discord["glass"].append(glass_discord)
        if not examples:
            examples = [section, local[ids], graph_pred, glass_pred, boundaries[ids], geometry]

    truth = np.concatenate(all_truth)
    predictions = {name: np.concatenate(value) for name, value in all_predictions.items()}
    discord = {name: np.concatenate(value) for name, value in all_discord.items()}
    metrics = {
        "purpose": "controlled held-section BigBrain proxy for GLASS cellular-sheaf boundary inference; not real-LSFM performance",
        "input_contract": "two synthetically corrupted views of each true pia-to-WM histology profile; GLASS glues local boundary estimates using label-free affine overlap relations measured from adjacent profiles",
        "split": {"calibration_sections": sorted(set(sections[calibration])), "development_sections": sorted(DEV_SECTIONS), "heldout_sections": split["holdout_sections"], "n_calibration": int(calibration.sum()), "n_development": int(dev.sum()), "n_heldout": int(test.sum())},
        "corruption": "independent contiguous dropout bands, blur, gain, and noise in two views; views are a marker-agreement proxy, not real multi-marker LSFM",
        "corruption_seed": seed,
        "corruption_severity": severity,
        "selected_parameters": {"glass_lambda": selected["glass"]["lambda"], "graph_lambda": selected["graph"]["lambda"], "report_coverage_target": 0.80, "glass_discord_threshold": threshold["glass"]},
        "development": {name: {"lambda": selected[name]["lambda"], "boundary_mae_normalized_depth": selected[name]["mae"]} for name in selected},
        "heldout_methods": {
            "local_boundary_evidence": summarize_method(predictions["local"], truth, discord["local"], local_threshold),
            "ordinary_graph_laplacian": summarize_method(predictions["graph"], truth, discord["graph"], threshold["graph"]),
            "glass_cellular_sheaf": summarize_method(predictions["glass"], truth, discord["glass"], threshold["glass"]),
        },
        "interpretation": "This proxy supports GLASS only if it improves both held-out boundary MAE and six-layer Dice over both controls at comparable coverage. It tests the sheaf gluing mechanism, not end-to-end LSFM cortical-layer accuracy.",
    }
    glass = metrics["heldout_methods"]["glass_cellular_sheaf"]
    graph = metrics["heldout_methods"]["ordinary_graph_laplacian"]
    local_summary = metrics["heldout_methods"]["local_boundary_evidence"]
    metrics["glass_delta"] = {
        "deployment_threshold_accepted_mae_vs_local": glass["accepted_boundary_mae_normalized_depth"] - local_summary["accepted_boundary_mae_normalized_depth"],
        "deployment_threshold_accepted_mae_vs_graph": glass["accepted_boundary_mae_normalized_depth"] - graph["accepted_boundary_mae_normalized_depth"],
    }
    heldout_labels = np.concatenate(all_labels)
    matched = {name: matched_coverage_metric(predictions[name], truth, heldout_labels, discord[name]) for name in predictions}
    metrics["matched_80pct_coverage"] = matched
    metrics["glass_delta"]["matched_80pct_accepted_mae_vs_local"] = (
        matched["glass"]["accepted_boundary_mae_normalized_depth"] - matched["local"]["accepted_boundary_mae_normalized_depth"]
    )
    metrics["glass_delta"]["matched_80pct_accepted_mae_vs_graph"] = (
        matched["glass"]["accepted_boundary_mae_normalized_depth"] - matched["graph"]["accepted_boundary_mae_normalized_depth"]
    )
    metrics["glass_delta"]["matched_80pct_macro_dice_vs_local"] = (
        matched["glass"]["accepted_macro_dice_layers_I_VI"] - matched["local"]["accepted_macro_dice_layers_I_VI"]
    )
    metrics["glass_delta"]["matched_80pct_macro_dice_vs_graph"] = (
        matched["glass"]["accepted_macro_dice_layers_I_VI"] - matched["graph"]["accepted_macro_dice_layers_I_VI"]
    )
    metrics["glass_delta"]["passes_two_metric_proxy_gate"] = bool(
        metrics["glass_delta"]["matched_80pct_accepted_mae_vs_local"] < 0
        and metrics["glass_delta"]["matched_80pct_accepted_mae_vs_graph"] < 0
        and metrics["glass_delta"]["matched_80pct_macro_dice_vs_local"] > 0
        and metrics["glass_delta"]["matched_80pct_macro_dice_vs_graph"] > 0
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    section, local_example, graph_example, glass_example, truth_example, geometry_example = examples
    points = np.asarray([[item["seed_y"], item["seed_x"]] for item in geometry_example])
    _, _, right_vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    order = np.argsort((points - points.mean(axis=0)) @ right_vectors[0])
    show = order[: min(70, len(order))]
    x = np.arange(len(show))
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    for boundary in range(5):
        ax = axes[boundary]
        ax.plot(x, truth_example[show, boundary], color="black", alpha=0.65, linewidth=1.4, label="truth")
        ax.scatter(x, local_example[show, boundary], color="#c76d35", s=7, alpha=0.45, label="local")
        ax.plot(x, graph_example[show, boundary], color="#4673b8", alpha=0.75, linewidth=1.0, label="graph")
        ax.plot(x, glass_example[show, boundary], color="#16866d", alpha=0.9, linewidth=1.2, label="sheaf")
        ax.invert_yaxis()
        ax.set_ylabel(("I/II", "II/III", "III/IV", "IV/V", "V/VI")[boundary])
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Columns ordered along pial surface (physical graph uses local neighbors)")
    fig.suptitle(f"Sheaf held-out {section}: each physical cortical-layer boundary")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_boundary_field.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2601, help="synthetic LSFM-like corruption seed")
    parser.add_argument("--severity", type=float, default=1.0, help="artifact severity multiplier; 1.0 is the primary proxy")
    args = parser.parse_args()
    if args.seed != 2601 or args.severity != 1.0:
        OUT = ROOT / "results" / "bigbrain_layer_proxy" / "stress" / f"severity_{args.severity:g}" / f"seed_{args.seed}"
    main(args.seed, args.severity)
