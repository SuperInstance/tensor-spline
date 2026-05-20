# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2025-05-20

### Added
- `SplineLinear` — nn.Linear replacement with Eisenstein lattice control points.
- `LowRankLinear` / `LowRankClassifier` — learned U@V factorization layers.
- `HierarchicalSplineLinear` / `HierarchicalSplineClassifier` — multi-scale coarse+fine interpolation.
- `inject_spline`, `inject_low_rank`, `inject_hierarchical_spline` — model surgery utilities.
- `compression_ratio` — compute compression stats for a model.
- `recommend_variant` — task-aware compression selection.
- JavaScript/TypeScript port (`npm/dist/index.js`).
- `py.typed` marker for PEP 561 type-checking support.
- `__all__` exports in `tensor_spline.__init__`.
- Input validation in `SplineLinear.__init__` (in_features, out_features, n_control_points).
- `SplineLinear.toString()` in JS port for human-readable representation.
- CHANGELOG.md and CONTRIBUTING.md.
