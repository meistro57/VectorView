# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Compatibility layer for mixed-case payload schemas across the UI (`source_id`/`SOURCE_ID`, etc.) so clustering, inspector rendering, and similar-signal metadata resolve consistently.
- Expanded payload keyword search coverage in `/api/search` to include additional fields (`summary`, `claims`, `concepts`, `questions`, `source_id`, `file_source`, `source_file`, `tone`) with lowercase and UPPERCASE key variants.
- Interactive view cube overlay in the 3D viewport that tracks camera orientation and lets users snap to primary axes with a face click.

### Changed
- README documentation updated to describe mixed-case payload compatibility, expanded `/api/search` field coverage, and the new view cube axis-snap navigation control.
