#!/usr/bin/env python3
"""
VectorView GPU PCA worker.
Called by Go: python3 pca_gpu.py <collection> <limit> <qdrant_url>
Outputs JSON to stdout: {"points": [{"id":..,"x":..,"y":..,"z":..,"payload":{..}}, ...]}
"""

import sys, json, os, time
import numpy as np

def log(msg):
    print(f"[pca_gpu] {msg}", file=sys.stderr)

def fetch_points(collection, limit, qdrant_url):
    """Scroll all points from Qdrant with vectors."""
    import urllib.request
    import urllib.error

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
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
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
            _, _, Vt = torch.linalg.svd(t, full_matrices=False)
            coords = (t @ Vt[:3].T).cpu().numpy()
            log("GPU PCA complete")
            return coords
        else:
            log("CUDA not available, using torch CPU SVD")
            t = torch.tensor(matrix, dtype=torch.float32)
            t -= t.mean(dim=0)
            _, _, Vt = torch.linalg.svd(t, full_matrices=False)
            coords = (t @ Vt[:3].T).numpy()
            return coords
    except ImportError:
        pass

    try:
        import cupy as cp
        log(f"Using CuPy GPU")
        t = cp.array(matrix, dtype=cp.float32)
        t -= t.mean(axis=0)
        _, _, Vt = cp.linalg.svd(t, full_matrices=False)
        coords = cp.asnumpy(t @ Vt[:3].T)
        log("CuPy PCA complete")
        return coords
    except ImportError:
        pass

    # CPU numpy fallback with randomized SVD
    log("Using NumPy randomized SVD (CPU)")
    matrix = matrix.astype(np.float32)
    matrix -= matrix.mean(axis=0)

    # Randomized SVD — much faster than full SVD for large dims
    n, dim = matrix.shape
    k = 3
    n_iter = 4
    n_oversampling = 10

    # Random projection
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((dim, k + n_oversampling)).astype(np.float32)
    Z = matrix @ Q
    for _ in range(n_iter):
        Z = matrix @ (matrix.T @ Z)
    Q, _ = np.linalg.qr(Z)
    B = Q.T @ matrix
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    coords = (matrix @ Vt[:k].T)
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


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "usage: pca_gpu.py <collection> <limit> <qdrant_url>"}))
        sys.exit(1)

    collection = sys.argv[1]
    limit      = int(sys.argv[2])
    qdrant_url = sys.argv[3].rstrip("/")

    t0 = time.time()
    log(f"Fetching {limit} points from {collection} @ {qdrant_url}")
    raw_points = fetch_points(collection, limit, qdrant_url)
    log(f"Fetch done in {time.time()-t0:.1f}s — {len(raw_points)} points")

    if not raw_points:
        print(json.dumps({"points": [], "total": 0}))
        return

    target_dim = choose_projection_dim(raw_points)
    if target_dim <= 0:
        log("Skipped all points: no dense vectors found")
        print(json.dumps({"points": [], "total": 0}))
        return

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
        print(json.dumps({"points": [], "total": 0}))
        return

    points, vectors = zip(*valid)
    matrix = np.vstack(vectors)
    log(f"Matrix shape: {matrix.shape} (dim={target_dim})")

    t1 = time.time()
    coords = pca_gpu(matrix)
    log(f"PCA done in {time.time()-t1:.1f}s")

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")
    elif coords.shape[1] > 3:
        coords = coords[:, :3]

    # Normalize to [-100, 100] cube
    for axis in range(3):
        col = coords[:, axis]
        mn, mx = col.min(), col.max()
        r = mx - mn
        if r < 1e-8:
            coords[:, axis] = 0.0
        else:
            coords[:, axis] = ((col - mn) / r - 0.5) * 200.0

    # Build output — strip vector from payload to keep JSON small
    out = []
    for i, p in enumerate(points):
        payload = {k: v for k, v in (p.get("payload") or {}).items()}
        out.append({
            "id":      p["id"],
            "x":       float(coords[i, 0]),
            "y":       float(coords[i, 1]),
            "z":       float(coords[i, 2]),
            "payload": payload,
        })

    log(f"Total time: {time.time()-t0:.1f}s")
    print(json.dumps({"points": out, "total": len(out)}))


if __name__ == "__main__":
    main()
