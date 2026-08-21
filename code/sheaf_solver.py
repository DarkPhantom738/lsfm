"""Sparse cellular-sheaf solvers for LSFM coordinates and cortical boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

N_BOUNDARIES = 5


@dataclass(frozen=True)
class SheafEdge:
    """A weighted overlap relation between two cortical columns."""

    i: int
    j: int
    weight: float
    scale_i: float | tuple[float, ...]
    scale_j: float | tuple[float, ...]
    relative_offset: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CoordinateSheafEdge:
    """A weighted local displacement between two overlapping image nodes."""

    i: int
    j: int
    weight: float
    relative_offset: tuple[float, ...]


def ordered_projection(boundaries: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Project 5 boundaries to a strictly increasing sequence in (0, 1)."""
    out = np.asarray(boundaries, dtype=np.float64).copy()
    out = np.clip(out, eps, 1.0 - eps)
    for k in range(1, out.shape[-1]):
        out[..., k] = np.maximum(out[..., k], out[..., k - 1] + eps)
    overflow = np.maximum(out[..., -1] - (1.0 - eps), 0.0)
    if np.any(overflow):
        grid = np.linspace(1, N_BOUNDARIES, N_BOUNDARIES) / N_BOUNDARIES
        out = out - overflow[..., None] * grid
        out = np.clip(out, eps, 1.0 - eps)
    return out


class CorticalBoundarySheaf:
    """Weighted cellular-sheaf solver for a graph of cortical columns."""

    def __init__(self, n_nodes: int, edges: list[SheafEdge]):
        self.n_nodes = int(n_nodes)
        self.edges = list(edges)
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        if any(e.i < 0 or e.j < 0 or e.i >= self.n_nodes or e.j >= self.n_nodes or e.i == e.j for e in self.edges):
            raise ValueError("edge index outside graph or self-edge")
        if any(len(e.relative_offset) != N_BOUNDARIES for e in self.edges):
            raise ValueError(f"each boundary sheaf edge needs {N_BOUNDARIES} relative offsets")

    @property
    def n_variables(self) -> int:
        return self.n_nodes * N_BOUNDARIES

    def coboundary(self, identity_restrictions: bool = False) -> sparse.csr_matrix:
        """Return the block sheaf coboundary matrix."""
        rows, cols, values = [], [], []
        for edge_index, edge in enumerate(self.edges):
            scale_i_values = np.broadcast_to(np.asarray(edge.scale_i, dtype=np.float64), (N_BOUNDARIES,))
            scale_j_values = np.broadcast_to(np.asarray(edge.scale_j, dtype=np.float64), (N_BOUNDARIES,))
            for boundary in range(N_BOUNDARIES):
                row = edge_index * N_BOUNDARIES + boundary
                scale_i = 1.0 if identity_restrictions else scale_i_values[boundary]
                scale_j = 1.0 if identity_restrictions else scale_j_values[boundary]
                rows.extend((row, row))
                cols.extend((edge.i * N_BOUNDARIES + boundary, edge.j * N_BOUNDARIES + boundary))
                values.extend((scale_i, -scale_j))
        return sparse.coo_matrix((values, (rows, cols)), shape=(len(self.edges) * N_BOUNDARIES, self.n_variables)).tocsr()

    def solve(
        self,
        local_mean: np.ndarray,
        local_variance: np.ndarray,
        consistency_weight: float,
        identity_restrictions: bool = False,
        enforce_order: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for boundaries and return edge/node discord."""
        mean = np.asarray(local_mean, dtype=np.float64)
        variance = np.asarray(local_variance, dtype=np.float64)
        if mean.shape != (self.n_nodes, N_BOUNDARIES) or variance.shape != mean.shape:
            raise ValueError(f"expected ({self.n_nodes}, {N_BOUNDARIES}) local mean/variance")
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be nonnegative")
        precision = 1.0 / np.clip(variance, 1e-6, None)
        delta = self.coboundary(identity_restrictions=identity_restrictions)
        edge_weights = np.repeat(np.asarray([max(e.weight, 1e-8) for e in self.edges]), N_BOUNDARIES)
        offsets = np.asarray([e.relative_offset for e in self.edges], dtype=np.float64).ravel()
        weighted_delta = sparse.diags(np.sqrt(edge_weights)) @ delta
        data_precision = sparse.diags(precision.ravel())
        system = data_precision + consistency_weight * (weighted_delta.T @ weighted_delta)
        rhs = precision.ravel() * mean.ravel() + consistency_weight * (delta.T @ (edge_weights * offsets))
        solution = spsolve(system.tocsc(), rhs).reshape(self.n_nodes, N_BOUNDARIES)
        if enforce_order:
            solution = ordered_projection(solution)
        residual = (delta @ solution.ravel() - offsets).reshape(len(self.edges), N_BOUNDARIES)
        edge_discord = np.sqrt(np.mean(residual * residual, axis=1)) if self.edges else np.empty(0)
        node_discord = np.zeros(self.n_nodes, dtype=np.float64)
        node_degree = np.zeros(self.n_nodes, dtype=np.float64)
        for value, edge in zip(edge_discord, self.edges):
            node_discord[edge.i] += value
            node_discord[edge.j] += value
            node_degree[edge.i] += 1
            node_degree[edge.j] += 1
        node_discord = np.divide(node_discord, node_degree, out=np.full_like(node_discord, np.inf), where=node_degree > 0)
        return solution, edge_discord, node_discord


class CoordinateSheaf:
    """Sparse affine sheaf solver for globally compatible image coordinates."""

    def __init__(self, n_nodes: int, edges: list[CoordinateSheafEdge]):
        self.n_nodes = int(n_nodes)
        self.edges = list(edges)
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        if not self.edges:
            raise ValueError("coordinate sheaf needs at least one overlap edge")
        if any(e.i < 0 or e.j < 0 or e.i >= self.n_nodes or e.j >= self.n_nodes or e.i == e.j for e in self.edges):
            raise ValueError("edge index outside graph or self-edge")
        dimensions = {len(e.relative_offset) for e in self.edges}
        if len(dimensions) != 1 or next(iter(dimensions)) < 1:
            raise ValueError("all coordinate edges need the same nonzero offset dimension")
        self.dimension = next(iter(dimensions))

    @property
    def n_variables(self) -> int:
        return self.n_nodes * self.dimension

    def coboundary(self) -> sparse.csr_matrix:
        """Return ``delta`` with rows ``u_j - u_i`` in each shared overlap."""
        rows, cols, values = [], [], []
        for edge_index, edge in enumerate(self.edges):
            for component in range(self.dimension):
                row = edge_index * self.dimension + component
                rows.extend((row, row))
                cols.extend((edge.i * self.dimension + component, edge.j * self.dimension + component))
                values.extend((-1.0, 1.0))
        return sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(len(self.edges) * self.dimension, self.n_variables),
        ).tocsr()

    def solve(
        self,
        local_mean: np.ndarray,
        local_variance: np.ndarray,
        consistency_weight: float = 1.0,
        robust_scale: float | None = None,
        robust_iterations: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return coordinate corrections and overlap/node discord."""
        mean = np.asarray(local_mean, dtype=np.float64)
        variance = np.asarray(local_variance, dtype=np.float64)
        expected = (self.n_nodes, self.dimension)
        if mean.shape != expected or variance.shape != expected:
            raise ValueError(f"expected {expected} local mean/variance")
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be nonnegative")
        if robust_scale is not None and robust_scale <= 0:
            raise ValueError("robust_scale must be positive when supplied")
        if robust_iterations < 1:
            raise ValueError("robust_iterations must be at least one")
        delta = self.coboundary()
        offsets = np.asarray([e.relative_offset for e in self.edges], dtype=np.float64).ravel()
        precision = 1.0 / np.clip(variance, 1e-8, None)
        data_precision = sparse.diags(precision.ravel())
        base_edge_weights = np.asarray([max(e.weight, 1e-8) for e in self.edges])
        robust_edge_weights = np.ones(len(self.edges), dtype=np.float64)
        solution = mean.copy()
        for _ in range(robust_iterations if robust_scale is not None else 1):
            edge_weights = np.repeat(base_edge_weights * robust_edge_weights, self.dimension)
            weighted_delta = sparse.diags(np.sqrt(edge_weights)) @ delta
            system = data_precision + consistency_weight * (weighted_delta.T @ weighted_delta)
            rhs = precision.ravel() * mean.ravel() + consistency_weight * (delta.T @ (edge_weights * offsets))
            solution = spsolve(system.tocsc(), rhs).reshape(expected)
            if robust_scale is not None:
                residual = (delta @ solution.ravel() - offsets).reshape(len(self.edges), self.dimension)
                residual_norm = np.linalg.norm(residual, axis=1)
                robust_edge_weights = np.minimum(1.0, robust_scale / np.maximum(residual_norm, 1e-12))
        residual = (delta @ solution.ravel() - offsets).reshape(len(self.edges), self.dimension)
        edge_discord = np.linalg.norm(residual, axis=1)
        node_discord = np.zeros(self.n_nodes, dtype=np.float64)
        node_degree = np.zeros(self.n_nodes, dtype=np.float64)
        for value, edge in zip(edge_discord, self.edges):
            node_discord[edge.i] += value
            node_discord[edge.j] += value
            node_degree[edge.i] += 1
            node_degree[edge.j] += 1
        node_discord = np.divide(node_discord, node_degree, out=np.full_like(node_discord, np.inf), where=node_degree > 0)
        return solution, edge_discord, node_discord


def registration_reliability(node_discord: np.ndarray, reference_scale: float) -> np.ndarray:
    """Convert coordinate discord into a [0, 1] downstream weight."""
    if reference_scale <= 0:
        raise ValueError("reference_scale must be positive")
    discord = np.asarray(node_discord, dtype=np.float64)
    return np.exp(-0.5 * (discord / reference_scale) ** 2)


def nearby_column_edges(
    seed_yx: np.ndarray,
    lengths: np.ndarray,
    max_neighbors: int = 4,
    radius_px: float = 110.0,
) -> list[SheafEdge]:
    """Connect only tangentially close cortical columns within one section."""
    points = np.asarray(seed_yx, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(lengths):
        raise ValueError("seed_yx must be (N,2) and lengths must have N entries")
    edges: list[SheafEdge] = []
    seen: set[tuple[int, int]] = set()
    for i, point in enumerate(points):
        distance = np.linalg.norm(points - point, axis=1)
        candidates = np.argsort(distance)[1 : max_neighbors + 1]
        for j in candidates:
            if distance[j] > radius_px:
                continue
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            overlap = float(np.exp(-0.5 * (distance[j] / max(radius_px * 0.5, 1.0)) ** 2))
            edges.append(
                SheafEdge(
                    key[0],
                    key[1],
                    overlap,
                    float(max(lengths[key[0]], 1.0)),
                    float(max(lengths[key[1]], 1.0)),
                )
            )
    return edges
