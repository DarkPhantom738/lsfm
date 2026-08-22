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


@dataclass(frozen=True)
class DeformationSheafEdge:
    """Local displacement measurements on one physical overlap patch."""

    i: int
    j: int
    weight: float
    restriction_i: sparse.csr_matrix
    restriction_j: sparse.csr_matrix
    relative_displacement: np.ndarray


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


def bilinear_restriction(control_shape: tuple[int, int], points_yx: np.ndarray) -> sparse.csr_matrix:
    """Sample a two-component control grid at normalized ``(y, x)`` points."""
    height, width = control_shape
    if height < 2 or width < 2:
        raise ValueError("control_shape must be at least (2, 2)")
    points = np.asarray(points_yx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_yx must have shape (N, 2)")
    rows, cols, values = [], [], []
    for point_index, (y, x) in enumerate(np.clip(points, 0.0, 1.0)):
        y_grid, x_grid = y * (height - 1), x * (width - 1)
        y0, x0 = int(np.floor(y_grid)), int(np.floor(x_grid))
        y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
        fy, fx = y_grid - y0, x_grid - x0
        for row, column, value in (
            (y0, x0, (1 - fy) * (1 - fx)),
            (y0, x1, (1 - fy) * fx),
            (y1, x0, fy * (1 - fx)),
            (y1, x1, fy * fx),
        ):
            for component in range(2):
                rows.append(2 * point_index + component)
                cols.append(2 * (row * width + column) + component)
                values.append(value)
    return sparse.coo_matrix(
        (values, (rows, cols)), shape=(2 * len(points), 2 * height * width)
    ).tocsr()


def control_grid_gradient(control_shape: tuple[int, int]) -> sparse.csr_matrix:
    """Return first differences for a two-component displacement control grid."""
    height, width = control_shape
    rows, cols, values = [], [], []
    row_index = 0
    for y in range(height):
        for x in range(width):
            for neighbor_y, neighbor_x in ((y + 1, x), (y, x + 1)):
                if neighbor_y >= height or neighbor_x >= width:
                    continue
                for component in range(2):
                    rows.extend((row_index, row_index))
                    cols.extend((2 * (y * width + x) + component, 2 * (neighbor_y * width + neighbor_x) + component))
                    values.extend((-1.0, 1.0))
                    row_index += 1
    return sparse.coo_matrix((values, (rows, cols)), shape=(row_index, 2 * height * width)).tocsr()


class DeformationFieldSheaf:
    """Sparse sheaf solver for tile-local displacement fields on overlap patches."""

    def __init__(self, n_nodes: int, control_shape: tuple[int, int], edges: list[DeformationSheafEdge]):
        self.n_nodes = int(n_nodes)
        self.control_shape = tuple(int(value) for value in control_shape)
        self.edges = list(edges)
        self.field_dimension = 2 * int(np.prod(self.control_shape))
        if self.n_nodes < 1 or min(self.control_shape) < 2 or not self.edges:
            raise ValueError("need nodes, a >=2x2 control grid, and at least one edge")
        if any(edge.i < 0 or edge.j < 0 or edge.i >= self.n_nodes or edge.j >= self.n_nodes or edge.i == edge.j for edge in self.edges):
            raise ValueError("edge index outside graph or self-edge")
        if any(
            edge.restriction_i.shape[1] != self.field_dimension
            or edge.restriction_j.shape != edge.restriction_i.shape
            or len(edge.relative_displacement) != edge.restriction_i.shape[0]
            for edge in self.edges
        ):
            raise ValueError("each edge must have compatible field restrictions and displacement samples")

    def coboundary(self) -> sparse.csr_matrix:
        """Return rows ``R_j f_j - R_i f_i`` for every overlap displacement sample."""
        blocks = []
        zero = sparse.csr_matrix
        for edge in self.edges:
            blocks.append(sparse.hstack([
                -edge.restriction_i if node == edge.i else edge.restriction_j if node == edge.j else zero((edge.restriction_i.shape[0], self.field_dimension))
                for node in range(self.n_nodes)
            ]))
        return sparse.vstack(blocks).tocsr()

    def solve(
        self,
        local_mean: np.ndarray,
        local_variance: np.ndarray,
        consistency_weight: float = 1.0,
        smoothness_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer local displacement fields and return overlap/node disagreement."""
        mean = np.asarray(local_mean, dtype=np.float64)
        variance = np.asarray(local_variance, dtype=np.float64)
        expected = (self.n_nodes, self.field_dimension)
        if mean.shape != expected or variance.shape != expected:
            raise ValueError(f"expected local mean/variance with shape {expected}")
        if consistency_weight < 0 or smoothness_weight < 0:
            raise ValueError("weights must be nonnegative")
        delta = self.coboundary()
        displacement = np.concatenate([np.asarray(edge.relative_displacement, dtype=np.float64) for edge in self.edges])
        edge_weights = np.concatenate([
            np.full(edge.restriction_i.shape[0], max(edge.weight, 1e-8)) for edge in self.edges
        ])
        precision = 1.0 / np.clip(variance.ravel(), 1e-8, None)
        data_precision = sparse.diags(precision)
        system = data_precision + consistency_weight * (delta.T @ sparse.diags(edge_weights) @ delta)
        if smoothness_weight:
            gradient = sparse.block_diag([control_grid_gradient(self.control_shape)] * self.n_nodes)
            system = system + smoothness_weight * (gradient.T @ gradient)
        rhs = precision * mean.ravel() + consistency_weight * (delta.T @ (edge_weights * displacement))
        solution = spsolve(system.tocsc(), rhs).reshape(expected)
        residual = delta @ solution.ravel() - displacement
        lengths = [edge.restriction_i.shape[0] for edge in self.edges]
        splits = np.split(residual, np.cumsum(lengths)[:-1])
        edge_discord = np.asarray([np.sqrt(np.mean(value * value)) for value in splits])
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
