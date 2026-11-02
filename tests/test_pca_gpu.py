import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VECTORVIEW_PCA_BACKEND", "numpy")

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None

if np is not None:
    import pca_gpu

# Rotation-invariant checksum of the projected geometry for a fixed input.
# Pairwise distances are unaffected by the arbitrary sign of PCA components,
# so this stays stable across BLAS/LAPACK builds.
GOLDEN_PCA_DISTANCE_DIGEST = "3bbc88e82be408da92df75c9726ef864d6fbf3ce3d36a52fdf879359f42c7c85"


def synthetic_matrix(n=24, dim=12, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def raw_points(matrix):
    return [
        {"id": f"p{i}", "vector": row.tolist(), "payload": {"text": f"doc {i}"}}
        for i, row in enumerate(matrix)
    ]


def distance_digest(coords, decimals=3):
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    rounded = np.round(distances, decimals).tolist()
    return hashlib.sha256(json.dumps(rounded, sort_keys=True).encode()).hexdigest()


@unittest.skipIf(np is None, "numpy is required for projection worker tests")
class ProjectionWorkerTests(unittest.TestCase):
    def test_pca_is_deterministic_and_normalized(self):
        matrix = synthetic_matrix()
        points = raw_points(matrix)

        first = pca_gpu.project_points(points, "pca")
        second = pca_gpu.project_points(points, "pca")

        self.assertEqual(first["total"], len(points))
        self.assertEqual(first["projection"]["method"], "pca")
        self.assertEqual([axis["component"] for axis in first["projection"]["axes"]], ["PC1", "PC2", "PC3"])
        self.assertEqual(first["points"], second["points"])

        for point in first["points"]:
            for axis in ("x", "y", "z"):
                self.assertGreaterEqual(point[axis], -100.0001)
                self.assertLessEqual(point[axis], 100.0001)

        spreads = {axis: max(p[axis] for p in first["points"]) - min(p[axis] for p in first["points"]) for axis in ("x", "y", "z")}
        self.assertTrue(any(spread > 1.0 for spread in spreads.values()), spreads)

    def test_projection_meta_variances_are_ordered(self):
        matrix = synthetic_matrix(n=32, dim=16)
        result = pca_gpu.project_points(raw_points(matrix), "pca")
        axes = result["projection"]["axes"]

        self.assertEqual(len(axes), 3)
        variances = [axis["variance_explained"] for axis in axes]
        for value in variances:
            self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(sum(variances), 100.0001)
        self.assertGreater(variances[0], 0.0)

    def test_pca_geometry_matches_golden(self):
        matrix = synthetic_matrix()
        result = pca_gpu.project_points(raw_points(matrix), "pca")
        coords = [[p["x"], p["y"], p["z"]] for p in result["points"]]
        digest = distance_digest(coords)
        if os.getenv("VV_UPDATE_GOLDEN"):
            print(f"GOLDEN_PCA_DISTANCE_DIGEST={digest}")
            return
        self.assertEqual(digest, GOLDEN_PCA_DISTANCE_DIGEST)

    def test_random_projection_stays_in_range_and_is_deterministic(self):
        matrix = synthetic_matrix()
        points = raw_points(matrix)

        first = pca_gpu.project_points(points, "random")
        second = pca_gpu.project_points(points, "random")

        self.assertEqual(first["projection"]["method"], "random")
        self.assertEqual([axis["component"] for axis in first["projection"]["axes"]], ["R1", "R2", "R3"])
        self.assertEqual(first["points"], second["points"])
        for point in first["points"]:
            for axis in ("x", "y", "z"):
                self.assertGreaterEqual(point[axis], -100.0001)
                self.assertLessEqual(point[axis], 100.0001)

    def test_choose_projection_dim_uses_modal_dimension(self):
        points = [
            {"id": "a", "vector": [0.0] * 8},
            {"id": "b", "vector": [0.0] * 8},
            {"id": "c", "vector": [0.0] * 4},
        ]
        self.assertEqual(pca_gpu.choose_projection_dim(points), 8)

    def test_named_vector_selection(self):
        points = [
            {
                "id": "a",
                "vector": {"text": [0.0] * 8, "image": [0.0] * 4},
            },
            {
                "id": "b",
                "vector": {"text": [1.0] * 8, "image": [1.0] * 4},
            },
        ]
        candidates = pca_gpu.vector_candidates(points[0]["vector"], "image")
        self.assertEqual(len(candidates[0]), 4, "preferred named vector must be tried first")
        selected = pca_gpu.extract_dense_vector(points[0]["vector"], 4, "image")
        self.assertEqual(len(selected), 4)
        fallback = pca_gpu.extract_dense_vector(points[0]["vector"], 8, "image")
        self.assertEqual(len(fallback), 8, "falls back to another candidate when the preferred dim does not match")

    def test_incompatible_vectors_are_skipped(self):
        points = [
            {"id": "a", "vector": [0.0] * 8},
            {"id": "b", "vector": [0.0] * 8},
            {"id": "c", "vector": [0.0] * 8},
            {"id": "d", "vector": ["not", "dense"]},
        ]
        result = pca_gpu.project_points(points, "pca")
        self.assertEqual(result["total"], 3)
        ids = [point["id"] for point in result["points"]]
        self.assertNotIn("d", ids)

    def test_empty_and_vectorless_payloads(self):
        self.assertEqual(pca_gpu.project_points([], "pca")["total"], 0)
        self.assertEqual(pca_gpu.project_points([{"id": "a", "vector": None}], "pca")["total"], 0)

    def test_tsne_and_umap_fall_back_for_tiny_inputs(self):
        matrix = synthetic_matrix(n=3, dim=6)
        for method in ("tsne", "umap"):
            result = pca_gpu.project_points(raw_points(matrix), method)
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["projection"]["method"], method)

    def test_ensure_three_axes_pads_and_trims(self):
        one_d = pca_gpu.ensure_three_axes([1.0, 2.0, 3.0])
        self.assertEqual(one_d.shape, (3, 3))

        two_d = pca_gpu.ensure_three_axes(np.zeros((5, 2)))
        self.assertEqual(two_d.shape, (5, 3))
        self.assertTrue(np.all(two_d[:, 2] == 0.0))

        four_d = pca_gpu.ensure_three_axes(np.ones((5, 4)))
        self.assertEqual(four_d.shape, (5, 3))

    def test_validate_projection_method(self):
        self.assertEqual(pca_gpu.validate_projection_method(""), "pca")
        self.assertEqual(pca_gpu.validate_projection_method("UMAP"), "umap")
        with self.assertRaises(ValueError):
            pca_gpu.validate_projection_method("isomap")

    def test_invalid_backend_is_rejected(self):
        previous = os.environ.get("VECTORVIEW_PCA_BACKEND")
        os.environ["VECTORVIEW_PCA_BACKEND"] = "quantum"
        try:
            with self.assertRaises(ValueError):
                pca_gpu.pca_gpu(synthetic_matrix(n=6, dim=4))
        finally:
            if previous is None:
                os.environ.pop("VECTORVIEW_PCA_BACKEND", None)
            else:
                os.environ["VECTORVIEW_PCA_BACKEND"] = previous

    def test_stdin_cli_roundtrip(self):
        matrix = synthetic_matrix(n=6, dim=4)
        payload = json.dumps({"points": raw_points(matrix), "vector_name": ""})
        env = dict(os.environ)
        env["VECTORVIEW_PCA_BACKEND"] = "numpy"
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "pca_gpu.py"), "--stdin", "pca"],
            input=payload,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        decoded = json.loads(completed.stdout)
        self.assertEqual(decoded["total"], 6)
        self.assertEqual(decoded["projection"]["method"], "pca")
        self.assertEqual(len(decoded["points"]), 6)


if __name__ == "__main__":
    unittest.main()
