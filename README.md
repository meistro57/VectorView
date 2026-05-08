<div align="center">

```
 ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗     ██╗   ██╗██╗███████╗██╗    ██╗
 ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗    ██║   ██║██║██╔════╝██║    ██║
 ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝    ██║   ██║██║█████╗  ██║ █╗ ██║
 ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║
  ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║     ╚████╔╝ ██║███████╗╚███╔███╔╝
   ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝      ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝
```
<img width="420" height="465" alt="image" src="https://github.com/user-attachments/assets/c4be347a-75ca-447b-a112-f3aa664b3a53" />

**Navigate the latent space. See what you know.**

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white)](https://golang.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20db-dc244c?style=flat-square)](https://qdrant.tech)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=flat-square&logo=threedotjs)](https://threejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---
<img width="1916" height="907" alt="image" src="https://github.com/user-attachments/assets/35e24889-af89-4063-bce9-6dff11aabc5c" />


VectorView is a **Go-first local app** that turns your [Qdrant](https://qdrant.tech) vector collections into a live, interactive **3D particle universe** — rendered in the browser with Three.js and a custom GLSL shader engine. The server is a single Go binary with an optional Python PCA worker for fast large-collection projection. No Node.js build step. No config hell. One command, one port, instant visualization.

It was born out of the [meta_bridge](https://github.com/meistro57/meta-bridge) / Knowledge Archaeology Engine ecosystem as a way to *see* what's actually living inside a vector database — not just query it blindly, but watch clusters form, spot outliers, and navigate latent space like a physical territory.

> *"Traversing latent space is like taking a walk through the mind of the model."*

---

## ✨ Features

**Visualization**
- Live 3D particle cloud rendered via custom WebGL shaders — additive blending, pulsing glow, dual-layer bloom
- Hybrid PCA pipeline: `/api/points` uses `pca_gpu.py` (PyTorch/CuPy/NumPy), `/api/search` uses inline Go PCA for smaller result sets
- Projection auto-selects the dominant dense vector dimension and skips incompatible vectors
- Color-coded clusters derived from source payload keys (`file_source`, `source_id`, `source_collection`, `source_file`, `source`) with lowercase/UPPERCASE compatibility
- Exponential fog, starfield background, and a subtle grid anchor the scene in deep space

**Exploration**
- Orbit, zoom, and pan with mouse — smooth damped controls
- View cube overlay in the 3D window mirrors camera orientation and supports one-click axis snapping
- Click any particle to inspect its full Qdrant payload in the HUD
- Signal Scanner: run **Find Similar** on the selected point to fetch nearest neighbors from Qdrant
- Similarity highlight mode: selected signal pulses, neighbor signals brighten, unrelated points fade
- Similar Signals side list with rank, score, snippet, and click-to-focus for loaded points
- Hover preview — inspector updates as you sweep when no point is pinned
- Payload text search — keyword scan returns a filtered sub-cloud, re-projected live
- Payload compatibility layer in UI: title/snippet/source/inspector fields resolve mixed-case keys (for example `source_id` and `SOURCE_ID`)

**Controls**
- Real-time sliders: point count, point size, opacity, bloom strength, auto-rotation speed, hue shift, saturation, lightness
- Collection picker — shows point count/vector dim, disables non-projectable collections, and switches without restarting
- Reload button — re-pull latest vectors on demand

**Architecture**
- Single Go binary with `//go:embed` — ships the entire frontend inside the executable
- Python PCA worker (`pca_gpu.py`) for full-collection projection with GPU/CPU fallbacks
- Raw HTTP Qdrant client — no SDK bloat, same pattern as meta_bridge
- `.env` support via `godotenv` — drop your existing config and go

---

## 🚀 Quickstart

### Prerequisites

- [Go 1.22+](https://golang.org/dl/)
- [Python 3.9+](https://www.python.org/downloads/) with `numpy` installed (`torch`/`cupy` optional for GPU acceleration)
- [Qdrant](https://qdrant.tech) running locally (default: `http://localhost:6333`)
- A populated collection to explore

### Install & Run

```bash
git clone https://github.com/meistro57/VectorView.git
cd VectorView
go mod tidy
python3 -m pip install numpy
go run .
```

Open your browser at **[http://localhost:7433](http://localhost:7433)**

### Build a standalone binary

```bash
go build -o vectorview .
./vectorview
```

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
```

Environment variables override `.env` — works cleanly with Docker and systemd.

---

## 🖥️ Interface

```
┌─────────────────────────────────────────────────────────────┐
│  VECTORVIEW  │ COLLECTION  │ POINTS  │ VISIBLE │ FPS  │ [⚙] │  ← Top HUD
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
| Click particle | Pin signal in inspector |
| FIND SIMILAR | Run nearest-neighbor scan in current collection |
| CLEAR SCAN | Exit highlight mode and restore normal cloud |
| Point Count slider | Reload cloud with selected sample size |
| Hue / Saturation / Lightness | Live palette remapping without reloading |
| Hover particle | Quick preview (when no pinned signal) |
| Search + SCAN | Filter to matching points |
| Collection picker | Switch active collection |
| ↺ RELOAD | Re-pull vectors from Qdrant |

---

## 🧠 How the 3D Projection Works

VectorView uses a **hybrid PCA pipeline**:

1. **`/api/points` path (full collection):** Go spawns `pca_gpu.py`, which scrolls vectors from Qdrant, detects the most common dense vector dimension in the sample, keeps vectors matching that dimension, runs PCA through PyTorch/CuPy when available (NumPy randomized SVD fallback), and returns normalized 3D coordinates.
2. **`/api/search` path (filtered subset):** Go performs inline power-iteration PCA for fast small-result reprojection without spawning Python, using the same dominant-dimension dense-vector selection logic.
3. **Normalize** — both paths scale coordinates into a ±100 unit cube for comfortable viewing.

The result: semantically similar points cluster together in 3D space. The geometry you see **is** the structure of your knowledge base.

---

## 📡 Signal Scanner

Signal Scanner turns the inspector into a neighborhood probe:

1. Load a collection.
2. Click a point to pin it in **Signal Inspector**.
3. Press **FIND SIMILAR**.
4. VectorView fetches nearest neighbors from Qdrant using the selected point's vector.
5. The selected signal pulses, similar signals brighten, and unrelated points dim.
6. Review neighbors in **Similar Signals**; click one to focus it when present in the loaded sample.

### Similarity assumptions and limits

- Endpoint uses the selected point vector directly against `/points/search` in the same collection.
- If vectors are named, VectorView picks the first detected named vector key from the selected point payload and sends `{name, vector}`.
- `limit` defaults to `12` and is capped to `200` server-side.
- If neighbors are returned by Qdrant but not present in the currently loaded point sample, they still appear in the Similar Signals list and are marked as outside the visible sample.
- Current search endpoint remains payload-text search; it does not yet combine keyword scan + nearest-neighbor reranking.

---

## 🔌 API

VectorView exposes a small REST API that the frontend uses — useful for scripting or integration:

```
GET  /api/collections                                                → list all Qdrant collections with metadata + projection readiness
GET  /api/points?collection=X&limit=N                                → full-collection projection via Python PCA worker
GET  /api/search?collection=X&q=term                                 → payload keyword search + inline Go PCA reprojection (mixed-case payload key support)
GET  /api/collections/{collection}/points/{point_id}/similar?limit=N → nearest-neighbor scan from selected point vector
POST /api/collections/{collection}/points/{point_id}/similar         → same as above (JSON body supports {"limit": N})
```

Example response from `/api/collections`:
```json
[
  {
    "name": "meistro_brain",
    "points_count": 847,
    "vector_size": 768,
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
  "total": 847
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

---

## 🗂️ Project Structure

```
VectorView/
├── main.go          # Go server — Qdrant client, API handlers, inline search PCA, embed
├── pca_gpu.py       # Python PCA worker for /api/points (PyTorch/CuPy/NumPy fallback)
├── static/
│   └── index.html   # Entire frontend — Three.js, GLSL shaders, HUD
├── go.mod
├── go.sum
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
