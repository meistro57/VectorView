# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Compatibility layer for mixed-case payload schemas across the UI (`source_id`/`SOURCE_ID`, etc.) so clustering, inspector rendering, and similar-signal metadata resolve consistently.
- Expanded payload keyword search coverage in `/api/search` to include additional fields (`summary`, `claims`, `concepts`, `questions`, `source_id`, `file_source`, `source_file`, `tone`) with lowercase and UPPERCASE key variants.
- Interactive view cube overlay in the 3D viewport that tracks camera orientation, uses corrected axis-face mapping (+Z is top/bottom), lets users snap to primary axes with a face click, and adds clickable cube corners for diagonal isometric snaps.
- Friendly Qdrant connectivity error overlay with retry action for collection load, projection load, search, and similarity scan failures.
- Live loading progress percentage during `/api/points` fetch/projection via new `/api/load-progress` API and frontend polling.
- Keyboard shortcuts: `R` reload points, `Space` pause/resume rotation, `Esc` clear pinned inspector selection.
- Collection metadata sidebar in the HUD showing collection name, point count, vector size, distance metric, and projection readiness note.
- Responsive HUD behavior for narrow screens (top bar wrapping, collapsed side panels, compact viewcube placement).
- Cosine-threshold similarity radius scanning (`/similar-radius`) with adjustable UI slider and highlighted neighborhood mode.
- Cluster distance matrix mini-heatmap in HUD based on centroid distances of top visible clusters.
- Outlier detection pass that flags sparse low-density points in a distinct color and legend entry.
- Projection axis readout in the HUD showing `PC1/PC2/PC3` variance explained percentages from both `/api/points` and `/api/search` responses.
- Projection selector with server-side `pca`, `random`, and `tsne` modes wired across both full collection load (`/api/points`) and filtered search projection (`/api/search`).

### Changed
- `/api/collections` now includes `distance_metric` metadata when available.
- `/api/search` projection now runs through the Python projection worker for parity with `/api/points`, using shared `projection` response metadata.
- Similarity limit cap increased to `500`.
- README and roadmap updated to reflect newly shipped stability and vector-search features.
