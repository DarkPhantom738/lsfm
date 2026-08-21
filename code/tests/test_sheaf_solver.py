import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheaf_solver import CoordinateSheaf, CoordinateSheafEdge, CorticalBoundarySheaf, SheafEdge, ordered_projection, registration_reliability


class GlassTests(unittest.TestCase):
    def test_ordered_projection_repairs_crossed_boundaries(self):
        result = ordered_projection(np.array([[0.7, 0.2, 0.2, 0.9, 0.1]]))
        self.assertTrue(np.all(np.diff(result[0]) > 0))
        self.assertGreater(result.min(), 0)
        self.assertLess(result.max(), 1)

    def test_physical_restrictions_differ_from_graph_smoothing(self):
        sheaf = CorticalBoundarySheaf(2, [SheafEdge(0, 1, 1.0, 1.0, 2.0)])
        local = np.array([[0.20, 0.35, 0.50, 0.65, 0.80], [0.20, 0.35, 0.50, 0.65, 0.80]])
        variance = np.full_like(local, 10.0)
        glass, _, _ = sheaf.solve(local, variance, consistency_weight=100.0)
        graph, _, _ = sheaf.solve(local, variance, consistency_weight=100.0, identity_restrictions=True)
        # GLASS equalizes physical depth (1*s0 ~= 2*s1), whereas an ordinary
        # graph Laplacian makes normalized node values equal.
        self.assertLess(np.mean(np.abs(glass[0] - 2 * glass[1])), 0.02)
        self.assertLess(np.mean(np.abs(graph[0] - graph[1])), 0.02)
        self.assertGreater(np.mean(np.abs(glass - graph)), 0.03)

    def test_affine_boundary_relation_is_respected(self):
        # A label-free image overlap may show that node i's boundaries are
        # displaced by +0.10 normalized depth relative to node j. This checks
        # that the inhomogeneous (affine) coboundary is solved, rather than
        # silently falling back to equal-value graph smoothing.
        edge = SheafEdge(0, 1, 1.0, 1.0, 1.0, (0.10,) * 5)
        local = np.array([[0.55, 0.60, 0.65, 0.70, 0.75], [0.15, 0.20, 0.25, 0.30, 0.35]])
        variance = np.full_like(local, 10.0)
        solved, _, _ = CorticalBoundarySheaf(2, [edge]).solve(local, variance, consistency_weight=100.0)
        self.assertLess(np.mean(np.abs((solved[0] - solved[1]) - 0.10)), 0.02)

    def test_coordinate_sheaf_glues_redundant_relative_translations(self):
        # The values are corrections in a shared physical coordinate system.
        truth = np.array([[0.0, 0.0], [2.0, -1.0], [3.5, 1.0], [6.0, 0.5]])
        pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
        edges = [
            CoordinateSheafEdge(i, j, 1.0, tuple(truth[j] - truth[i]))
            for i, j in pairs
        ]
        estimate, edge_discord, node_discord = CoordinateSheaf(4, edges).solve(
            local_mean=np.zeros_like(truth),
            # This fixes the global translation gauge at node zero without
            # imposing any unsupported absolute location on other tiles.
            local_variance=np.array([[1e-6, 1e-6], [1e4, 1e4], [1e4, 1e4], [1e4, 1e4]]),
            consistency_weight=10.0,
        )
        self.assertLess(np.max(np.abs(estimate - truth)), 1e-3)
        self.assertLess(edge_discord.max(), 1e-4)
        self.assertTrue(np.all(registration_reliability(node_discord, 1.0) > 0.99))

    def test_robust_coordinate_sheaf_limits_one_inconsistent_overlap(self):
        truth = np.array([[0.0], [1.0], [2.0], [3.0]])
        pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
        edges = [CoordinateSheafEdge(i, j, 1.0, tuple(truth[j] - truth[i])) for i, j in pairs]
        # The last overlap cannot coexist with the other four around the
        # sheaf's cycles. Robust GLASS should not allow it to set the atlas.
        edges[-1] = CoordinateSheafEdge(1, 3, 1.0, (10.0,))
        variance = np.array([[1e-6], [1e4], [1e4], [1e4]])
        quadratic, _, _ = CoordinateSheaf(4, edges).solve(np.zeros_like(truth), variance)
        robust, _, _ = CoordinateSheaf(4, edges).solve(
            np.zeros_like(truth), variance, robust_scale=0.25, robust_iterations=20
        )
        self.assertLess(np.sqrt(np.mean((robust - truth) ** 2)), np.sqrt(np.mean((quadratic - truth) ** 2)))


if __name__ == "__main__":
    unittest.main()
