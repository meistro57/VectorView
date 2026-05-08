# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Compatibility layer for mixed-case payload schemas across the UI (`source_id`/`SOURCE_ID`, etc.) so clustering, inspector rendering, and similar-signal metadata resolve consistently.
- Expanded payload keyword search coverage in `/api/search` to include additional fields (`summary`, `claims`, `concepts`, `questions`, `source_id`, `file_source`, `source_file`, `tone`) with lowercase and UPPERCASE key variants.
- Interactive view cube overlay in the 3D viewport that tracks camera orientation, uses corrected axis-face mapping (+Z is top/bottom), lets users snap to primary axes with a face click, and adds clickable cube corners for diagonal isometric snaps.

### Changed
- README documentation updated to describe mixed-case payload compatibility, expanded `/api/search` field coverage, corrected view cube axis orientation, and new corner-click isometric navigation controls.
