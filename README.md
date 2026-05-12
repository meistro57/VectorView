<div align="center">

```
 ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗     ██╗   ██╗██╗███████╗██╗    ██╗
 ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗    ██║   ██║██║██╔════╝██║    ██║
 ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝    ██║   ██║██║█████╗  ██║ █╗ ██║
 ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║
  ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║     ╚████╔╝ ██║███████╗╚███╔███╔╝
   ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝      ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝
```

**Navigate the latent space. See what you know.**

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white)](https://golang.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20db-dc244c?style=flat-square)](https://qdrant.tech)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=flat-square&logo=threedotjs)](https://threejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Tests](https://github.com/meistro57/VectorView/actions/workflows/test.yml/badge.svg)](https://github.com/meistro57/VectorView/actions/workflows/test.yml)

</div>

<img width="420" height="465" alt="image" src="https://github.com/user-attachments/assets/c4be347a-75ca-447b-a112-f3aa664b3a53" />

<img width="1882" height="1683" alt="image" src="https://github.com/user-attachments/assets/aa76ef8a-c5ac-4fb6-ba6e-cbd702cbe781" />

---

VectorView is a **Go-first local app** that turns your [Qdrant](https://qdrant.tech) vector collections into a live, interactive **3D particle universe** — rendered in the browser with Three.js and a custom GLSL shader engine. The server is a single Go binary with an optional Python PCA worker for fast large-collection projection. No Node.js build step. No config hell. One command, one port, instant visualization.

It was born out of the [meta_bridge](https://github.com/meistro57/meta-bridge) / Knowledge Archaeology Engine ecosystem as a way to *see* what's actually living inside a vector database — not just query it blindly, but watch clusters form, spot outliers, and navigate latent space like a physical territory.

> *"Traversing latent space is like taking a walk through the mind of the model."*

---

## ✨ Features

**Visualization**
- Live 3D particle cloud rendered via custom WebGL shaders — additive blending, pulsing glow, dual-layer bloom
- Unified projection pipeline: both `/api/points` and `/api/search` use `pca_gpu.py` (PyTorch/CuPy/NumPy) with selectable `pca`, `random`, `tsne`, or `umap` modes
- Projection auto-selects the dominant dense vector dimension and skips incompatible vectors, then returns per-axis variance metadata for HUD readout
- Color-coded clusters derived from source payload keys (`file_source`, `source_id`, `source_collection`, `source_file`, `source`) with lowercase/UPPERCASE compatibility
- Exponential fog, starfield background, and a subtle grid anchor the scene in deep space
- Cluster convex hull overlays (translucent + wireframe) for dominant visible clusters
- Density-aware fog tuning updates per load to emphasize sparse vs dense structures
- Particle trail compositor (off-screen ping-pong textures) for ghosted motion during navigation
- Cluster distance matrix heatmap in the HUD (top clusters by centroid distance)
- Outlier detection marks sparse low-density points in a distinct color
- Friendly error overlay when Qdrant is unreachable

**Exploration**
- Orbit, zoom, and pan with mouse — smooth damped controls
- View cube overlay in the 3D window mirrors camera orientation, supports one-click axis snapping, and adds clickable corner markers for instant isometric views
- Click any particle to inspect its full Qdrant payload in the HUD
- Signal Scanner: run **Find Similar** on the selected point to fetch nearest neighbors from Qdrant
- Similarity radius scan: run **RADIUS SCAN** to highlight all neighbors above a cosine threshold
- Similarity highlight mode: selected signal pulses, neighbor signals brighten, unrelated points fade
- External highlight trigger support via `/api/highlight` for programmatic scanner overlays from other tools
- Live ingest mode: optional SSE stream (`/api/stream`) with animated drop-in bursts and live point append in random projection mode (full sync fallback for PCA/t-SNE/UMAP)
- Durable stream history via Redis Streams with opt-in replay for reconnecting clients
- Similar Signals side list with rank, score, snippet, and click-to-focus for loaded points
- Hover preview — inspector updates as you sweep when no point is pinned
- Payload text search — keyword scan returns a filtered sub-cloud, re-projected live
- Payload compatibility layer in UI: title/snippet/source/inspector fields resolve mixed-case keys (for example `source_id` and `SOURCE_ID`)

**Controls**
- Real-time sliders: point count, point size, opacity, bloom strength, auto-rotation speed, trail persistence, timeline reveal, hue shift, saturation, lightness
- Payload-driven size mapping (`score`, `confidence`, `chunk_length`, `text_length`) for semantic salience sizing, including nested payload metadata keys
- Theme switcher: Deep Space, Bioluminescent, Amber Archaeology, Terminal Green
- Timeline controls for ingestion-order reveal (manual scrub + autoplay)
- Collection picker — shows point count/vector dim, disables non-projectable collections, and switches without restarting
- Projection selector — switch between PCA / random / t-SNE and reload current view with the selected method
- Collection metadata panel — collection name, point count, vector size, distance metric, projection status
- Projection axes panel — shows top 3 projection components and variance explained percentages
- Loading overlay with progress percentage while vectors are fetched and projected
- Top HUD load-state badge (FETCH / PROJECTING / RENDERING / READY / ERROR) for at-a-glance projection status
- Responsive HUD collapse for narrow viewports (mobile/tablet)
- Keyboard shortcuts: `R` reload, `Space` pause/resume rotation, `Esc` clear inspector selection
- Reload button — re-pull latest vectors on demand
- Live mode toggle — subscribe to stream events and keep a WebSocket heartbeat alive during long ingest sessions
- Screenshot button — export current viewport as PNG

**Architecture**
- Single Go binary with `//go:embed` — ships the entire frontend inside the executable
- Python projection worker (`pca_gpu.py`) for both full-collection and filtered-result projection with GPU/CPU fallbacks
- Raw HTTP Qdrant client — no SDK bloat, same pattern as meta_bridge
- `.env` support via `godotenv` — drop your existing config and go
- GitHub Actions CI workflow runs `go test ./...` on pushes to `main` and on pull requests

---

## 🚀 Quickstart

### Prerequisites

- [Go 1.22+](https://golang.org/dl/)
- [Python 3.9+](https://www.python.org/downloads/) with `numpy` installed (`torch`/`cupy` optional for GPU acceleration, `scikit-learn` optional for t-SNE mode)
- [Qdrant](https://qdrant.tech) running locally (default: `http://localhost:6333`)
- A populated collection to explore

### Install & Run

```bash
git clone https://github.com/meistro57/VectorView.git
cd VectorView
go mod tidy
python3 -m pip install numpy
# optional: required only for projection=tsne
python3 -m pip install scikit-learn
# optional: required only for projection=umap
python3 -m pip install umap-learn
go run .
```

Open your browser at **[http://localhost:7433](http://localhost:7433)**

### Build a standalone binary

```bash
go build -o vectorview .
./vectorview
```

### Test requirements

- Go 1.22+
- Python 3.9+ with `numpy` available (needed by projection paths that are compiled into test-time build checks)
- No live Qdrant dependency is required for `go test ./...` right now

```bash
go test ./...
```

Latest local verification:

```bash
$ go test ./...
?   	github.com/meistro57/vectorview	[no test files]
```

This still confirms the app compiles and links cleanly end-to-end.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust:

```env
# Qdrant connection
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Port VectorView serves on
VECTORVIEW_PORT=7433

# Max points pulled per collection
# PCA is O(n × dim) — keep this sane for large collections
VECTORVIEW_MAX_POINTS=2000

# Optional Redis projection cache (recommended for instant reloads)
VECTORVIEW_REDIS_URL=
VECTORVIEW_CACHE_TTL_SECONDS=600

# Live streaming + ingest integrations
VECTORVIEW_STREAM_HEARTBEAT_SECONDS=15
VECTORVIEW_STREAM_REPLAY_COUNT=0
VECTORVIEW_STREAM_MAX_EVENTS_PER_SECOND=30
VECTORVIEW_WS_PING_SECONDS=20
VECTORVIEW_META_BRIDGE_LIVE=true
VECTORVIEW_META_BRIDGE_COLLECTIONS=mb_chunks,mb_claims
VECTORVIEW_META_BRIDGE_POLL_SECONDS=3
VECTORVIEW_REDIS_PUBSUB_CHANNEL=vectorview.telemetry
VECTORVIEW_REDIS_STREAM_KEY=vectorview.events
VECTORVIEW_REDIS_STREAM_MAXLEN=20000

# Semantic search embedding provider
VECTORVIEW_SEMANTIC_PROVIDER=ollama
VECTORVIEW_OLLAMA_URL=http://localhost:11434
VECTORVIEW_EMBED_MODEL=nomic-embed-text
```

Environment variables override `.env` — works cleanly with Docker and systemd.

---

## 🖥️ Interface

```
┌─────────────────────────────────────────────────────────────┐
│  VECTORVIEW  │ COLLECTION  │ POINTS  │ VISIBLE │ LOAD │ FPS  │ [⚙] │  ← Top HUD
├────────┬────────────────────────────────────────┬───────────┤
│Controls│                                        │ Inspector │
│        │                                        │           │
│ Size   │         3 D   P A R T I C L E          │ id:       │
│ Opacity│              C L O U D                 │ type:     │
│ Bloom  │                                        │ source:   │
│ Speed  │                                        │ text:     │
│        │                                        │           │
│ Legend │                                        │           │
├────────┴────────────────────────────────────────┴───────────┤
│  [ Search memory text... ]               [ SCAN ]           │  ← Bottom HUD
└─────────────────────────────────────────────────────────────┘
```

| Control | Action |
|---|---|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll wheel | Zoom |
| View cube face click | Snap camera to ±X / ±Y / ±Z |
| View cube corner click | Snap camera to diagonal isometric view (±X ±Y ±Z) |
| Click particle | Pin signal in inspector |
| FIND SIMILAR | Run top-K nearest-neighbor scan in current collection |
| RADIUS SCAN | Highlight all neighbors above selected cosine threshold |
| CLEAR SCAN | Exit highlight mode and restore normal cloud |
| Point Count slider | Reload cloud with selected sample size (500–75,000) |
| Hue / Saturation / Lightness | Live palette remapping without reloading |
| Hover particle | Quick preview (when no pinned signal) |
| Search + SCAN | Filter to matching points |
| Collection picker | Switch active collection |
| Theme selector | Swap full scene/HUD palette |
| Timeline Reveal + Auto | Scrub/animate ingestion-order reveal |
| 📸 SNAPSHOT | Export current viewport as PNG |
| ↺ RELOAD | Re-pull vectors from Qdrant |
| R / Space / Esc | Reload / pause rotation / clear inspector |

---

## 🧠 How the 3D Projection Works

VectorView uses a **unified projection pipeline**:

1. **`/api/points` path (full collection):** Go spawns `pca_gpu.py`, which scrolls vectors from Qdrant, detects the most common dense vector dimension in the sample (or a selected named vector), keeps vectors matching that dimension, runs the selected projection (`pca`, `random`, `tsne`, or `umap`), and returns normalized 3D coordinates.
2. **`/api/search` path (filtered subset):** Go fetches keyword-matched points, then calls the same `pca_gpu.py` worker via stdin with the selected projection method, so behavior matches full-load projection.
3. **Normalize + metadata** — coordinates are scaled into a ±100 unit cube and responses include projection metadata (`method`, top 3 component labels, variance explained %) for HUD axis readout.
4. **Redis caching + incremental append** — when `VECTORVIEW_REDIS_URL` is set, `/api/points` responses are cached by collection + limit + projection + vector name. Random projection loads can request `append_from=N` to add only newly requested points without recomputing the existing projected subset.

The result: semantically similar points cluster together in 3D space. The geometry you see **is** the structure of your knowledge base.

---

## 📡 Signal Scanner

Signal Scanner turns the inspector into a neighborhood probe:

1. Load a collection.
2. Click a point to pin it in **Signal Inspector**.
3. Press **FIND SIMILAR** for top-K neighbors, or **RADIUS SCAN** for all neighbors above a cosine threshold.
4. VectorView fetches neighbors from Qdrant using the selected point's vector.
5. The selected signal pulses, similar signals brighten, and unrelated points dim.
6. Review neighbors in **Similar Signals**; click one to focus it when present in the loaded sample.

### Similarity assumptions and limits

- Endpoint uses the selected point vector directly against `/points/search` in the same collection.
- If vectors are named, VectorView can send a requested `vector_name`; otherwise it falls back to the first detected named vector key from the selected point payload.
- `limit` defaults to `12` and is capped to `500` server-side.
- If neighbors are returned by Qdrant but not present in the currently loaded point sample, they still appear in the Similar Signals list and are marked as outside the visible sample.
- Search bar supports **KEYWORD** mode (`/api/search`) and **SEMANTIC** mode (`/api/semantic-search`). Semantic mode embeds the query with Ollama (`VECTORVIEW_OLLAMA_URL`, `VECTORVIEW_EMBED_MODEL`) and then runs nearest-neighbor search.
- Semantic mode supports cross-collection querying via `target_collection`: run the query from one active collection and project matches from another.

---

## 🔌 API

VectorView exposes a small REST API that the frontend uses — useful for scripting or integration:

```
GET  /api/collections                                                             → list all Qdrant collections with metadata + projection readiness
GET  /api/load-progress?id=progress_id                                            → live projection/fetch progress for active /api/points request
GET  /api/points?collection=X&limit=N&projection=pca|random|tsne|umap&vector_name=V&append_from=N&progress_id=token → full projection, cache-aware load, or incremental random append
GET  /api/search?collection=X&q=term&limit=N&projection=pca|random|tsne|umap&vector_name=V                           → payload keyword search + Python worker reprojection (mixed-case payload key support)
GET  /api/semantic-search?collection=X&target_collection=Y&q=term&limit=N&projection=pca|random|tsne|umap&vector_name=V → embed query text via Ollama; if target_collection is set, search in that collection and project its hits
GET  /api/collections/{collection}/points/{point_id}/similar?limit=N              → nearest-neighbor top-K scan from selected point vector
POST /api/collections/{collection}/points/{point_id}/similar                      → same as above (JSON body supports {"limit": N})
GET  /api/collections/{collection}/points/{point_id}/similar-radius?radius=R&limit=N → cosine-threshold neighborhood scan
POST /api/collections/{collection}/points/{point_id}/similar-radius               → same as above (JSON body supports {"radius": R, "limit": N})
POST /api/highlight                                                                 → publish external highlight event ({"collection":"X","ids":[...],"focus_id":"..."})
GET  /api/highlight?collection=X&since=event_id                                     → poll latest external highlight event (204 when unchanged)
GET  /api/stream?collection=X&heartbeat_seconds=N&replay=K&max_events_per_second=M  → SSE stream for ingest/highlight/telemetry with optional replay + server-side throttling
GET  /api/ws                                                                         → WebSocket heartbeat endpoint (server ping frames)

`/api/stream` notes:
- `replay=K` replays up to `K` most recent events from Redis Streams (`VECTORVIEW_REDIS_STREAM_KEY`) before live events.
- `max_events_per_second=M` applies server-side rate limiting; when bursty, newest event wins (coalesced backpressure behavior).
- Replay is capped at 1000 events per request.
```

Example response from `/api/collections`:
```json
[
  {
    "name": "meistro_brain",
    "points_count": 847,
    "vector_size": 768,
    "distance_metric": "Cosine",
    "projection_ready": true,
    "projection_note": "ok"
  }
]
```

`/api/search` currently scans these payload fields (both lowercase and UPPERCASE variants):

- `text`, `content`, `chunk_text`, `title`, `summary`
- `claims`, `concepts`, `questions`
- `source_id`, `file_source`, `source_file`, `tone`

Example response from `/api/points`:
```json
{
  "points": [
    { "id": 123, "x": 14.2, "y": -7.8, "z": 3.1, "payload": { "entity_type": "chunk", "text": "..." } }
  ],
  "total": 847,
  "projection": {
    "method": "pca",
    "axes": [
      { "component": "PC1", "variance_explained": 38.4 },
      { "component": "PC2", "variance_explained": 22.1 },
      { "component": "PC3", "variance_explained": 14.7 }
    ]
  }
}
```

Example response from `/api/collections/{collection}/points/{point_id}/similar`:
```json
{
  "selected_id": "abc123",
  "collection": "meistro_brain",
  "limit": 12,
  "vector_name": "default",
  "neighbors": [
    {
      "id": "def456",
      "score": 0.874,
      "payload": {
        "title": "Example title",
        "text": "Example text...",
        "date": "2025-01-17",
        "file_source": "example.json"
      }
    }
  ]
}
```

Example response from `/api/collections/{collection}/points/{point_id}/similar-radius`:
```json
{
  "selected_id": "abc123",
  "collection": "meistro_brain",
  "limit": 400,
  "similarity_radius": 0.92,
  "vector_name": "default",
  "neighbors": [
    {
      "id": "ghi789",
      "score": 0.934,
      "payload": {
        "title": "Nearby signal",
        "text": "Within cosine threshold..."
      }
    }
  ]
}
```

---

## 🗂️ Project Structure

```
VectorView/
├── main.go          # Go server — Qdrant client, API handlers, projection orchestration, embed
├── pca_gpu.py       # Python projection worker for /api/points, /api/search, /api/semantic-search (PCA/random/t-SNE/UMAP)
├── static/
│   └── index.html   # Entire frontend — Three.js, GLSL shaders, HUD
├── go.mod
├── go.sum
├── .github/
│   └── workflows/
│       └── test.yml # CI: run go test ./... on push/PR
├── .env             # Local config (gitignored)
├── .env.example     # Config template
├── Makefile         # make run / make build
└── ROADMAP.md       # Where this is going
```

---

## 🤝 Ecosystem

VectorView is part of the **Meistro Knowledge Archaeology** stack:

| Project | Role |
|---|---|
| [meta_bridge](https://github.com/meistro57/meta-bridge) | Ingestion, chunking, claim extraction → Qdrant |
| **VectorView** | 3D visual exploration of Qdrant collections |
| Vectoreologist | TUI-based archaeological reasoning over vector topology |
| KAE | Autonomous knowledge graph builder (Wikipedia, arXiv, Gutenberg) |
| Chat Bridge | Multi-provider AI orchestration with Qdrant RAG |

---

## 📄 License

MIT — do what you want, build something weird.

---

<div align="center">

Built by [meistro57](https://github.com/meistro57) · Powered by Go + Qdrant + Three.js

*"The map is not the territory — but this one's pretty close."*

</div>
