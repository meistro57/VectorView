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

</div>

---

VectorView is a **zero-dependency Go binary** that turns your [Qdrant](https://qdrant.tech) vector collections into a live, interactive **3D particle universe** — rendered in the browser with Three.js and a custom GLSL shader engine. No Python runtime. No Node.js build step. No config hell. One binary, one port, one command.

It was born out of the [meta_bridge](https://github.com/meistro57/meta-bridge) / Knowledge Archaeology Engine ecosystem as a way to *see* what's actually living inside a vector database — not just query it blindly, but watch clusters form, spot outliers, and navigate latent space like a physical territory.

> *"Traversing latent space is like taking a walk through the mind of the model."*

---

## ✨ Features

**Visualization**
- Live 3D particle cloud rendered via custom WebGL shaders — additive blending, pulsing glow, dual-layer bloom
- Pure Go PCA (power iteration) — projects N-dimensional vectors to 3D with zero Python dependency
- Color-coded clusters by `entity_type`, `source_id`, or any payload field
- Exponential fog, starfield background, and a subtle grid anchor the scene in deep space

**Exploration**
- Orbit, zoom, and pan with mouse — smooth damped controls
- Click any particle to inspect its full Qdrant payload in the HUD
- Hover preview — point inspector updates as you sweep across the cloud
- Payload text search — keyword scan returns a filtered sub-cloud, re-projected live

**Controls**
- Real-time sliders: point size, opacity, bloom strength, auto-rotation speed
- Collection picker — switch between all Qdrant collections without restarting
- Reload button — re-pull latest vectors on demand

**Architecture**
- Single Go binary with `//go:embed` — ships the entire frontend inside the executable
- Paginated Qdrant scroll — handles large collections gracefully (configurable limit)
- Raw HTTP Qdrant client — no SDK bloat, same pattern as meta_bridge
- `.env` support via `godotenv` — drop your existing config and go

---

## 🚀 Quickstart

### Prerequisites

- [Go 1.22+](https://golang.org/dl/)
- [Qdrant](https://qdrant.tech) running locally (default: `http://localhost:6333`)
- A populated collection to explore

### Install & Run

```bash
git clone https://github.com/meistro57/VectorView.git
cd VectorView
go mod tidy
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
│  [ Search payload text... ]              [ SCAN ]           │  ← Bottom HUD
└─────────────────────────────────────────────────────────────┘
```

| Control | Action |
|---|---|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll wheel | Zoom |
| Click particle | Inspect full payload |
| Hover particle | Quick preview |
| Search + SCAN | Filter to matching points |
| Collection picker | Switch active collection |
| ↺ RELOAD | Re-pull vectors from Qdrant |

---

## 🧠 How the 3D Projection Works

VectorView implements **Principal Component Analysis via power iteration** entirely in Go — no cgo, no native libs, no Python subprocess.

1. **Scroll** — all points are fetched from Qdrant with `with_vectors: true`
2. **Center** — vectors are mean-centered across all dimensions
3. **Power iterate** — the top 3 eigenvectors of the covariance matrix are found iteratively
4. **Deflate** — each component is subtracted before finding the next (Gram-Schmidt style)
5. **Project** — every point's position in 3D space = its dot product against the 3 principal components
6. **Normalize** — coordinates are scaled to fill a ±100 unit cube for comfortable viewing

The result: semantically similar points cluster together in 3D space. The geometry you see **is** the structure of your knowledge base.

---

## 🔌 API

VectorView exposes a small REST API that the frontend uses — useful for scripting or integration:

```
GET  /api/collections          → list all Qdrant collections with metadata
GET  /api/points?collection=X&limit=N  → PCA-projected points as JSON
GET  /api/search?collection=X&q=term   → payload keyword search, projected
```

Example response from `/api/points`:
```json
{
  "points": [
    { "id": 123, "x": 14.2, "y": -7.8, "z": 3.1, "payload": { "entity_type": "chunk", "text": "..." } }
  ],
  "total": 847
}
```

---

## 🗂️ Project Structure

```
VectorView/
├── main.go          # Go server — Qdrant client, PCA, HTTP handlers, embed
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
