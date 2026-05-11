#!/usr/bin/env python3
"""
VectorView projection worker.
Called by Go:
  - python3 pca_gpu.py <collection> <limit> <qdrant_url> [projection]
  - python3 pca_gpu.py --stdin [projection]
Outputs JSON to stdout: {"points": [...], "total": N, "projection": {...}}
"""

import json
import sys
import time

import numpy as np


def log(msg):
    print(f"[pca_gpu] {msg}", file=sys.stderr)


def fetch_points(collection, limit, qdrant_url):
    """Scroll all points from Qdrant with vectors."""
    import urllib.error
    import urllib.request

    all_points = []
    offset = None
    page_size = min(200, limit)

    while True:
        body = {
            "limit": page_size,
            "with_payload": True,
            "with_vectors": True,
        }
        if offset is not None:
            body["offset"] = offset

        data = json.dumps(body).encode()
        url = f"{qdrant_url}/collections/{collection}/points/scroll"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            log(f"HTTP error {e.code}: {e.read().decode()}")
            sys.exit(1)

        points = result["result"]["points"]
        all_points.extend(points)
        log(f"Fetched {len(all_points)} / {limit}")

        next_page = result["result"].get("next_page_offset")
        if not next_page or len(all_points) >= limit:
            break
        offset = next_page

    return all_points[:limit]


def pca_gpu(matrix):
    """Run PCA to 3D using GPU if available, fall back to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            log(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            t = torch.tensor(matrix, dtype=torch.float32).cuda()
            t -= t.mean(dim=0)
            _, _, vt = torch.linalg.svd(t, full_matrices=False)
            coords = (t @ vt[:3].T).cpu().numpy()
            log("GPU PCA complete")
            return coords

        log("CUDA not available, using torch CPU SVD")
        t = torch.tensor(matrix, dtype=torch.float32)
        t -= t.mean(dim=0)
        _, _, vt = torch.linalg.svd(t, full_matrices=False)
        coords = (t @ vt[:3].T).numpy()
        return coords
    except ImportError:
        pass

    try:
        import cupy as cp

        log("Using CuPy GPU")
        t = cp.array(matrix, dtype=cp.float32)
        t -= t.mean(axis=0)
        _, _, vt = cp.linalg.svd(t, full_matrices=False)
        coords = cp.asnumpy(t @ vt[:3].T)
        log("CuPy PCA complete")
        return coords
    except ImportError:
        pass

    log("Using NumPy randomized SVD (CPU)")
    matrix = matrix.astype(np.float32)
    matrix -= matrix.mean(axis=0)

    n, dim = matrix.shape
    k = 3
    n_iter = 4
    n_oversampling = 10

    rng = np.random.default_rng(42)
    q = rng.standard_normal((dim, k + n_oversampling)).astype(np.float32)
    z = matrix @ q
    for _ in range(n_iter):
        z = matrix @ (matrix.T @ z)
    q, _ = np.linalg.qr(z)
    b = q.T @ matrix
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    coords = matrix @ vt[:k].T
    log("NumPy randomized SVD complete")
    return coords


def vector_candidates(raw_vector):
    if isinstance(raw_vector, list):
        return [raw_vector]
    if not isinstance(raw_vector, dict):
        return []

    keys = sorted(raw_vector.keys())
    ordered_keys = []
    if "vector" in raw_vector:
        ordered_keys.append("vector")
    ordered_keys.extend([k for k in keys if k != "vector"])

    candidates = []
    for key in ordered_keys:
        value = raw_vector.get(key)
        if isinstance(value, list):
            candidates.append(value)
            continue
        if isinstance(value, dict):
            nested = value.get("vector")
            if isinstance(nested, list):
                candidates.append(nested)
    return candidates


def choose_projection_dim(points):
    dim_counts = {}
    for p in points:
        seen_dims = set()
        for vec in vector_candidates(p.get("vector")):
            dim = len(vec)
            if dim <= 0 or dim in seen_dims:
                continue
            seen_dims.add(dim)
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
    if not dim_counts:
        return 0
    return max(dim_counts.items(), key=lambda item: (item[1], item[0]))[0]


def extract_dense_vector(raw_vector, target_dim=None):
    for vec in vector_candidates(raw_vector):
        if target_dim is None or len(vec) == target_dim:
            return vec
    return None


def validate_projection_method(raw):
    method = str(raw or "").strip().lower()
    if not method:
        method = "pca"
    if method not in {"pca", "random", "tsne"}:
        raise ValueError(f"invalid projection: {raw!r} (expected pca, random, tsne)")
    return method


def ensure_three_axes(coords):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")
    elif coords.shape[1] > 3:
        coords = coords[:, :3]
    return coords


def random_projection(centered_matrix):
    dim = centered_matrix.shape[1]
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((dim, 3)).astype(np.float32)
    basis_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis_norm[basis_norm < 1e-8] = 1.0
    basis /= basis_norm
    return centered_matrix @ basis


def tsne_projection(matrix):
    n = matrix.shape[0]
    if n <= 3:
        log("t-SNE needs at least 4 points for stable perplexity; using random projection")
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        return random_projection(centered)

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("t-SNE projection requires scikit-learn installed in Python env") from exc

    perplexity = min(30.0, max(2.0, float(n - 1) / 3.0))
    tsne = TSNE(
        n_components=3,
        init="pca",
        random_state=42,
        learning_rate="auto",
        perplexity=perplexity,
    )
    return tsne.fit_transform(matrix)


def build_projection_meta(method, centered_matrix, projected_coords):
    projected_coords = ensure_three_axes(projected_coords)

    if method == "pca":
        total_variance = 0.0
        if centered_matrix.shape[0] > 1:
            total_variance = float(np.var(centered_matrix, axis=0, ddof=1).sum())
        prefix = "PC"
    elif method == "random":
        total_variance = float(np.var(projected_coords, axis=0, ddof=1).sum()) if projected_coords.shape[0] > 1 else 0.0
        prefix = "R"
    else:
        total_variance = float(np.var(projected_coords, axis=0, ddof=1).sum()) if projected_coords.shape[0] > 1 else 0.0
        prefix = "T"

    axes = []
    for idx in range(3):
        component_variance = 0.0
        if projected_coords.shape[0] > 1:
            component_variance = float(np.var(projected_coords[:, idx], ddof=1))
        variance_explained = (component_variance / total_variance * 100.0) if total_variance > 1e-12 else 0.0
        axes.append(
            {
                "component": f"{prefix}{idx + 1}",
                "variance_explained": variance_explained,
            }
        )

    return {
        "method": method,
        "axes": axes,
    }


def normalize_coords(coords):
    coords = ensure_three_axes(coords)
    for axis in range(3):
        col = coords[:, axis]
        mn, mx = col.min(), col.max()
        span = mx - mn
        if span < 1e-8:
            coords[:, axis] = 0.0
        else:
            coords[:, axis] = ((col - mn) / span - 0.5) * 200.0
    return coords


def project_points(raw_points, projection_method):
    if not raw_points:
        return {"points": [], "total": 0, "projection": None}

    target_dim = choose_projection_dim(raw_points)
    if target_dim <= 0:
        log("Skipped all points: no dense vectors found")
        return {"points": [], "total": 0, "projection": None}

    valid = []
    skipped = 0
    for p in raw_points:
        vec = extract_dense_vector(p.get("vector"), target_dim)
        if not vec:
            skipped += 1
            continue
        try:
            valid.append((p, np.asarray(vec, dtype=np.float32)))
        except (TypeError, ValueError):
            skipped += 1

    if skipped:
        log(f"Skipped {skipped} points with non-dense or incompatible vectors")
    if not valid:
        return {"points": [], "total": 0, "projection": None}

    points, vectors = zip(*valid)
    matrix = np.vstack(vectors)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    log(f"Matrix shape: {matrix.shape} (dim={target_dim})")

    t1 = time.time()
    if projection_method == "pca":
        coords = pca_gpu(matrix)
    elif projection_method == "random":
        coords = random_projection(centered)
        log("Random projection complete")
    else:
        coords = tsne_projection(matrix)
        log("t-SNE projection complete")
    log(f"Projection done in {time.time()-t1:.1f}s")

    coords = ensure_three_axes(coords)
    projection_meta = build_projection_meta(projection_method, centered, coords)
    coords = normalize_coords(coords)

    out = []
    for i, p in enumerate(points):
        payload = {k: v for k, v in (p.get("payload") or {}).items()}
        out.append(
            {
                "id": p["id"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "payload": payload,
            }
        )

    return {"points": out, "total": len(out), "projection": projection_meta}


def main():
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--stdin":
            projection_method = validate_projection_method(sys.argv[2] if len(sys.argv) >= 3 else "pca")
            payload = json.load(sys.stdin)
            raw_points = payload.get("points") if isinstance(payload, dict) else None
            if not isinstance(raw_points, list):
                raise ValueError("stdin payload must be object with points array")
            result = project_points(raw_points, projection_method)
            print(json.dumps(result))
            return

        if len(sys.argv) < 4:
            print(json.dumps({"error": "usage: pca_gpu.py <collection> <limit> <qdrant_url> [projection]"}))
            sys.exit(1)

        collection = sys.argv[1]
        limit = int(sys.argv[2])
        qdrant_url = sys.argv[3].rstrip("/")
        projection_method = validate_projection_method(sys.argv[4] if len(sys.argv) >= 5 else "pca")

        t0 = time.time()
        log(f"Fetching {limit} points from {collection} @ {qdrant_url}")
        raw_points = fetch_points(collection, limit, qdrant_url)
        log(f"Fetch done in {time.time()-t0:.1f}s — {len(raw_points)} points")

        result = project_points(raw_points, projection_method)
        log(f"Total time: {time.time()-t0:.1f}s")
        print(json.dumps(result))
    except Exception as exc:
        log(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
