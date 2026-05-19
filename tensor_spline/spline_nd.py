"""
N-dimensional Eisenstein lattice spline layers.

Extends the 2-D SplineLinear to arbitrary input dimensions using:
  - EisensteinGrid: precomputed lattice tables for 2D, 3D, 4D
  - SplineLinearND: pure-numpy core + optional PyTorch wrapper
  - HierarchicalSplineND: coarse-to-fine multi-scale decomposition
  - Benchmark utilities comparing spline vs dense parameter counts

Design constraint: all core interpolation math uses pure NumPy for
portability (no PyTorch required). PyTorch integration is provided
as an optional layer when torch is available.

PLATO tile serialization:
    Every class implements ``to_tile_dict()`` / ``from_tile_dict()`` for
    round-tripping through the PLATO TrainingTile store format.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EisensteinGrid — precomputed lattice lookup tables
# ---------------------------------------------------------------------------

class EisensteinGrid:
    """
    Precomputed Eisenstein lattice positions for dimensions 2 through 4.

    For 2-D the lattice is the standard Eisenstein ring ℤ[ω] with
    ω = e^{2πi/3}, placed in the plane using Cartesian coordinates
    (a − b/2,  b√3/2).

    For 3-D and 4-D we extend by taking the Cartesian product of the
    2-D hexagonal lattice with a regular (equally-spaced) axis for each
    additional dimension.  This preserves hexagonal packing in the
    dominant 2-D subspace while covering the extra dimensions uniformly.

    Points are sorted by ascending Euclidean distance from the origin
    and normalised so the outermost selected point has unit norm.

    Args:
        n_points: Number of control points.
        ndim:     Lattice dimensionality (2, 3, or 4).

    Example::

        >>> grid = EisensteinGrid(n_points=16, ndim=3)
        >>> grid.positions.shape
        (16, 3)
    """

    _SQRT3_HALF: float = math.sqrt(3.0) / 2.0

    def __init__(self, n_points: int, ndim: int = 2) -> None:
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        if ndim not in (2, 3, 4):
            raise ValueError(f"ndim must be 2, 3, or 4; got {ndim}")
        self.n_points = n_points
        self.ndim = ndim
        self.positions: np.ndarray = self._build(n_points, ndim)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hex_disk(radius: int) -> List[Tuple[float, ...]]:
        """Return 2-D Eisenstein lattice points within *radius* rings."""
        sqrt3_half = math.sqrt(3.0) / 2.0
        pts: List[Tuple[float, ...]] = []
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                x = float(a) - float(b) * 0.5
                y = float(b) * sqrt3_half
                pts.append((x, y))
        return pts

    def _build(self, n: int, ndim: int) -> np.ndarray:
        if ndim == 2:
            return self._build_2d(n)
        # For ndim > 2: product of hexagonal disk with uniform extra axes.
        return self._build_nd(n, ndim)

    def _build_2d(self, n: int) -> np.ndarray:
        R = max(int(math.ceil(math.sqrt(n / 3.0))) + 2, 3)
        pts = self._hex_disk(R)
        # Sort by distance, break ties deterministically
        pts.sort(key=lambda p: (round(p[0]**2 + p[1]**2, 9), p[0], p[1]))
        sel = np.array(pts[:n], dtype=np.float64)
        mx = np.linalg.norm(sel, axis=1).max()
        if mx > 0:
            sel /= mx
        return sel

    def _build_nd(self, n: int, ndim: int) -> np.ndarray:
        """Build N-D lattice by taking the top-k closest product points."""
        extra = ndim - 2
        # Generate 2-D hex disk
        R2 = max(int(math.ceil(math.sqrt(n / 3.0))) + 2, 3)
        hex_pts = self._hex_disk(R2)
        hex_pts.sort(key=lambda p: (round(p[0]**2 + p[1]**2, 9), p[0], p[1]))

        # For extra dimensions: use small set of equally-spaced values
        # We want enough product points to have >= n total.
        # A conservative count: take top hex_k hex points × grid^extra
        hex_k = min(len(hex_pts), max(n, 16))
        hex_arr = np.array(hex_pts[:hex_k], dtype=np.float64)  # (hex_k, 2)

        # Extra axis values: zero-centred, equally spaced
        extra_ticks = 3  # start with 3 ticks per extra dim: -1, 0, 1
        axes = [np.linspace(-1.0, 1.0, extra_ticks) for _ in range(extra)]

        # Build product grid for extra dimensions
        if extra == 1:
            extra_grid = axes[0]  # (extra_ticks,)
            # Product: each hex_pt paired with each extra value
            candidates = []
            for hi in range(hex_k):
                for ei in range(extra_ticks):
                    pt = list(hex_arr[hi]) + [extra_grid[ei]]
                    dist_sq = sum(x * x for x in pt)
                    candidates.append((dist_sq, pt))
        else:  # extra == 2
            extra_grid = np.array(np.meshgrid(axes[0], axes[1], indexing='ij'))
            extra_flat = extra_grid.reshape(2, -1).T  # (extra_ticks^2, 2)
            candidates = []
            for hi in range(hex_k):
                for ei in range(extra_flat.shape[0]):
                    pt = list(hex_arr[hi]) + list(extra_flat[ei])
                    dist_sq = sum(x * x for x in pt)
                    candidates.append((dist_sq, pt))

        candidates.sort(key=lambda t: (round(t[0], 9),) + tuple(t[1]))
        sel = np.array([c[1] for c in candidates[:n]], dtype=np.float64)

        # Normalise
        mx = np.linalg.norm(sel, axis=1).max()
        if mx > 0:
            sel /= mx
        return sel

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def nearest_k(
        self,
        point: np.ndarray,
        k: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest control points to a query.

        Args:
            point: shape ``(ndim,)``.
            k:     Number of neighbours.

        Returns:
            ``(distances, indices)`` each of shape ``(k,)``.
        """
        k = min(k, self.n_points)
        dists = np.linalg.norm(self.positions - point, axis=1)
        if k >= self.n_points:
            idx = np.argsort(dists)
        else:
            idx = np.argpartition(dists, k)[:k]
            idx = idx[np.argsort(dists[idx])]
        return dists[idx], idx

    def to_tile_dict(self) -> Dict[str, Any]:
        """Serialize to PLATO tile-compatible dict."""
        return {
            "type": "eisenstein_grid",
            "n_points": self.n_points,
            "ndim": self.ndim,
            "positions": self.positions.tolist(),
        }

    @classmethod
    def from_tile_dict(cls, d: Dict[str, Any]) -> "EisensteinGrid":
        grid = cls.__new__(cls)
        grid.n_points = d["n_points"]
        grid.ndim = d["ndim"]
        grid.positions = np.array(d["positions"], dtype=np.float64)
        return grid

    def __repr__(self) -> str:
        return f"EisensteinGrid(n_points={self.n_points}, ndim={self.ndim})"


# ---------------------------------------------------------------------------
# SplineLinearND — core interpolation in pure NumPy
# ---------------------------------------------------------------------------

class SplineLinearND:
    """
    N-dimensional spline weight parameterization (pure NumPy).

    Given input shape ``(*input_shape)`` and output shape ``(*output_shape)``,
    the weight tensor is an ``prod(output_shape) × prod(input_shape)`` matrix
    whose every element is interpolated from ``n_control_points`` scalar
    values placed on an N-D Eisenstein grid.

    This is the NumPy-only counterpart to :class:`SplineLinear`.  It
    computes the same IDW interpolation without requiring PyTorch, making
    it suitable for deployment on NPU / embedded targets.

    Args:
        input_shape:  Shape of the input tensor's feature dimensions.
        output_shape: Shape of the output tensor's feature dimensions.
        n_control_points: Number of lattice control points.
        ndim: Lattice dimensionality (2–4). Defaults to 2.

    Example::

        >>> layer = SplineLinearND(input_shape=(64,), output_shape=(32,),
        ...                        n_control_points=16)
        >>> W = layer.materialize_weights()
        >>> W.shape
        (32, 64)
        >>> layer.compression_ratio()
        128.0
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        n_control_points: int = 16,
        ndim: int = 2,
    ) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.n_control_points = n_control_points
        self.ndim = ndim

        self.in_size = int(np.prod(input_shape))
        self.out_size = int(np.prod(output_shape))

        # Control-point values: the learned parameters
        bound = 1.0 / math.sqrt(n_control_points)
        self.control_values = np.random.uniform(-bound, bound, n_control_points)

        # Build lattice
        self.grid = EisensteinGrid(n_control_points, ndim)

        # Precompute weight-matrix sampling positions in [-1, 1]^ndim
        self._query_positions = self._build_query_positions()

        # Precompute IDW kernel (fixed — depends only on geometry)
        self._kernel = self._build_kernel()

    def _build_query_positions(self) -> np.ndarray:
        """
        Map each weight element (out_i, in_j) to an ndim-D coordinate
        in [-1, 1]^ndim.  For ndim==2 this is the same scheme as
        SplineLinear; for higher ndim we distribute coordinates across
        dimensions using interleaving.
        """
        n_total = self.out_size * self.in_size
        if self.ndim == 2:
            row_coords = np.linspace(-1.0, 1.0, self.out_size)
            col_coords = np.linspace(-1.0, 1.0, self.in_size)
            grid_y, grid_x = np.meshgrid(row_coords, col_coords, indexing='ij')
            pos = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            return pos.astype(np.float64)

        # ndim > 2: split the out_size × in_size elements across ndim axes
        # Use Hilbert-like interleaving: axis k gets coordinates that cycle
        # through the range at a rate proportional to 2^k.
        positions = np.zeros((n_total, self.ndim), dtype=np.float64)
        # Assign dims by splitting the linear index
        # Use a simple scheme: distribute across dims via modular decomposition
        # Total elements = out_size * in_size
        # We want to map each linear index to ndim coordinates
        # Use prime-based striding for good distribution
        primes = [2, 3, 5, 7]
        for d in range(self.ndim):
            stride = primes[d % len(primes)]
            perm = np.arange(n_total) * stride % n_total
            positions[:, d] = np.linspace(-1.0, 1.0, n_total)[perm]
        return positions

    def _build_kernel(self) -> np.ndarray:
        """
        Precompute the IDW kernel matrix.

        Returns:
            Normalised kernel of shape ``(out_size * in_size, n_control_points)``.
        """
        dists = np.linalg.norm(
            self._query_positions[:, None, :] - self.grid.positions[None, :, :],
            axis=2,
        )  # (n_query, n_cp)
        kernel = 1.0 / (dists ** 2 + 1e-6)
        kernel /= kernel.sum(axis=1, keepdims=True)
        return kernel

    def materialize_weights(self) -> np.ndarray:
        """
        Compute the full weight matrix from control-point values.

        Returns:
            Float64 array of shape ``(out_size, in_size)``.
        """
        w_flat = self._kernel @ self.control_values  # (out_size * in_size,)
        return w_flat.reshape(self.out_size, self.in_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Linear transform: ``x @ W.T + bias`` (no bias in base class).

        Args:
            x: Input array of shape ``(*, in_size)``.

        Returns:
            Output array of shape ``(*, out_size)``.
        """
        W = self.materialize_weights()
        return x @ W.T

    def set_control_values(self, values: np.ndarray) -> None:
        """Set control-point values from external array."""
        assert values.shape == (self.n_control_points,)
        self.control_values = values.copy()

    def num_params(self) -> int:
        """Total trainable parameters (control values only)."""
        return self.n_control_points

    def num_dense_params(self) -> int:
        """Parameters an equivalent dense layer would require."""
        return self.out_size * self.in_size

    def compression_ratio(self) -> float:
        return float(self.num_dense_params()) / max(self.num_params(), 1)

    # ------------------------------------------------------------------
    # PLATO tile serialization
    # ------------------------------------------------------------------

    def to_tile_dict(self) -> Dict[str, Any]:
        return {
            "type": "spline_linear_nd",
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "n_control_points": self.n_control_points,
            "ndim": self.ndim,
            "control_values": self.control_values.tolist(),
        }

    @classmethod
    def from_tile_dict(cls, d: Dict[str, Any]) -> "SplineLinearND":
        layer = cls(
            input_shape=tuple(d["input_shape"]),
            output_shape=tuple(d["output_shape"]),
            n_control_points=d["n_control_points"],
            ndim=d.get("ndim", 2),
        )
        layer.control_values = np.array(d["control_values"], dtype=np.float64)
        return layer

    def __repr__(self) -> str:
        return (
            f"SplineLinearND(input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, "
            f"n_cp={self.n_control_points}, ndim={self.ndim}, "
            f"compression={self.compression_ratio():.0f}:1)"
        )


# ---------------------------------------------------------------------------
# HierarchicalSplineND — multi-scale coarse-to-fine
# ---------------------------------------------------------------------------

class HierarchicalSplineND:
    """
    Multi-scale N-dimensional spline with coarse-to-fine refinement.

    The weight tensor is decomposed into patches.  A coarse spline
    captures global structure across the entire tensor; fine splines
    capture local detail within each patch.  The outputs are blended
    with a learnable (sigmoid-constrained) alpha parameter.

    This is the pure-NumPy counterpart to HierarchicalSplineLinear,
    generalised to N dimensions.

    Args:
        input_shape:    Input feature dimensions.
        output_shape:   Output feature dimensions.
        coarse_pts:     Control points for the coarse spline.
        fine_pts:       Control points per fine patch.
        patch_sizes:    Patch size per spatial dimension.  If a single int
                        is given it is replicated to all dims.
        ndim:           Lattice dimensionality (2–4).
        alpha_init:     Initial blend factor (0.5 = equal coarse/fine).

    Example::

        >>> layer = HierarchicalSplineND(
        ...     input_shape=(128,), output_shape=(64,),
        ...     coarse_pts=16, fine_pts=8, patch_sizes=32,
        ... )
        >>> W = layer.materialize_weights()
        >>> W.shape
        (64, 128)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        coarse_pts: int = 16,
        fine_pts: int = 8,
        patch_sizes: int = 32,
        ndim: int = 2,
        alpha_init: float = 0.5,
    ) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.in_size = int(np.prod(input_shape))
        self.out_size = int(np.prod(output_shape))
        self.ndim = ndim
        self.coarse_pts = coarse_pts
        self.fine_pts = fine_pts

        # Patch decomposition (2-D decomposition of the weight matrix)
        self.patch_size = patch_sizes
        self.n_in_patches = max(1, math.ceil(self.in_size / patch_sizes))
        self.n_out_patches = max(1, math.ceil(self.out_size / patch_sizes))
        self.total_patches = self.n_in_patches * self.n_out_patches

        # Coarse spline
        bound_c = 1.0 / math.sqrt(coarse_pts)
        self.coarse_control = np.random.uniform(-bound_c, bound_c, coarse_pts)
        self.coarse_grid = EisensteinGrid(coarse_pts, ndim)

        # Fine splines: one set of control values per patch
        bound_f = 1.0 / math.sqrt(fine_pts)
        self.fine_control = np.random.uniform(
            -bound_f, bound_f, (self.total_patches, fine_pts)
        )
        self.fine_grid = EisensteinGrid(fine_pts, ndim)

        # Blend factor
        self.alpha = float(alpha_init)

        # Precompute kernels
        self._coarse_kernel = None
        self._fine_kernels = None
        self._build_kernels()

    def _build_kernels(self) -> None:
        """Precompute IDW kernels for coarse and fine levels."""
        # Coarse: full weight matrix
        n_total = self.out_size * self.in_size
        row_c = np.linspace(-1.0, 1.0, self.out_size)
        col_c = np.linspace(-1.0, 1.0, self.in_size)
        gy, gx = np.meshgrid(row_c, col_c, indexing='ij')
        if self.ndim == 2:
            coarse_query = np.stack([gx.ravel(), gy.ravel()], axis=1)
        else:
            # For higher ndim, replicate 2D query and add zeros
            coarse_query = np.zeros((n_total, self.ndim))
            coarse_query[:, 0] = gx.ravel()
            coarse_query[:, 1] = gy.ravel()

        dists = np.linalg.norm(
            coarse_query[:, None, :] - self.coarse_grid.positions[None, :, :],
            axis=2,
        )
        kernel = 1.0 / (dists ** 2 + 1e-6)
        kernel /= kernel.sum(axis=1, keepdims=True)
        self._coarse_kernel = kernel

        # Fine: per-patch
        self._fine_kernels = []
        ps = self.patch_size
        for pi in range(self.n_in_patches):
            for pj in range(self.n_out_patches):
                in_start = pi * ps
                in_end = min(in_start + ps, self.in_size)
                out_start = pj * ps
                out_end = min(out_start + ps, self.out_size)
                patch_rows = out_end - out_start
                patch_cols = in_end - in_start
                n_patch = patch_rows * patch_cols

                if n_patch == 0:
                    self._fine_kernels.append(None)
                    continue

                pr = np.linspace(-1.0, 1.0, patch_rows)
                pc = np.linspace(-1.0, 1.0, patch_cols)
                pgy, pgx = np.meshgrid(pr, pc, indexing='ij')

                if self.ndim == 2:
                    fine_query = np.stack([pgx.ravel(), pgy.ravel()], axis=1)
                else:
                    fine_query = np.zeros((n_patch, self.ndim))
                    fine_query[:, 0] = pgx.ravel()
                    fine_query[:, 1] = pgy.ravel()

                fd = np.linalg.norm(
                    fine_query[:, None, :] - self.fine_grid.positions[None, :, :],
                    axis=2,
                )
                fk = 1.0 / (fd ** 2 + 1e-6)
                fk /= fk.sum(axis=1, keepdims=True)
                self._fine_kernels.append(fk)

    def materialize_weights(self) -> np.ndarray:
        """
        Build the full weight matrix from hierarchical control points.

        Returns:
            Float64 array of shape ``(out_size, in_size)``.
        """
        # Coarse pass
        coarse_flat = self._coarse_kernel @ self.coarse_control

        # Fine pass
        W = np.zeros((self.out_size, self.in_size), dtype=np.float64)
        ps = self.patch_size
        alpha = 1.0 / (1.0 + math.exp(-self.alpha))  # sigmoid

        for pi in range(self.n_in_patches):
            for pj in range(self.n_out_patches):
                patch_idx = pj * self.n_in_patches + pi
                fk = self._fine_kernels[patch_idx]
                if fk is None:
                    continue

                in_start = pi * ps
                in_end = min(in_start + ps, self.in_size)
                out_start = pj * ps
                out_end = min(out_start + ps, self.out_size)

                fine_vals = fk @ self.fine_control[patch_idx]
                patch_w = fine_vals.reshape(out_end - out_start, in_end - in_start)

                # Extract coarse contribution for this patch
                coarse_patch = coarse_flat.reshape(self.out_size, self.in_size)[
                    out_start:out_end, in_start:in_end
                ]

                W[out_start:out_end, in_start:in_end] = (
                    alpha * coarse_patch + (1.0 - alpha) * patch_w
                )

        return W

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.materialize_weights()
        return x @ W.T

    def num_params(self) -> int:
        return self.coarse_pts + self.total_patches * self.fine_pts + 1  # +1 alpha

    def num_dense_params(self) -> int:
        return self.out_size * self.in_size

    def compression_ratio(self) -> float:
        return float(self.num_dense_params()) / max(self.num_params(), 1)

    # ------------------------------------------------------------------
    # PLATO tile serialization
    # ------------------------------------------------------------------

    def to_tile_dict(self) -> Dict[str, Any]:
        return {
            "type": "hierarchical_spline_nd",
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "coarse_pts": self.coarse_pts,
            "fine_pts": self.fine_pts,
            "patch_size": self.patch_size,
            "ndim": self.ndim,
            "alpha": self.alpha,
            "coarse_control": self.coarse_control.tolist(),
            "fine_control": self.fine_control.tolist(),
        }

    @classmethod
    def from_tile_dict(cls, d: Dict[str, Any]) -> "HierarchicalSplineND":
        layer = cls(
            input_shape=tuple(d["input_shape"]),
            output_shape=tuple(d["output_shape"]),
            coarse_pts=d["coarse_pts"],
            fine_pts=d["fine_pts"],
            patch_sizes=d["patch_size"],
            ndim=d.get("ndim", 2),
            alpha_init=d.get("alpha", 0.5),
        )
        layer.coarse_control = np.array(d["coarse_control"], dtype=np.float64)
        layer.fine_control = np.array(d["fine_control"], dtype=np.float64)
        return layer

    def __repr__(self) -> str:
        return (
            f"HierarchicalSplineND("
            f"in={self.input_shape}, out={self.output_shape}, "
            f"coarse={self.coarse_pts}, fine={self.fine_pts}, "
            f"patches={self.total_patches}, "
            f"compression={self.compression_ratio():.0f}:1)"
        )


# ---------------------------------------------------------------------------
# PyTorch wrapper (optional)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class SplineLinearNDTorch(nn.Module):
        """
        PyTorch wrapper around :class:`SplineLinearND`.

        Provides ``nn.Module`` compatibility with autograd for the control
        values.  The IDW kernel is precomputed and registered as a buffer;
        only ``control_values`` carries gradients.

        Args:
            input_shape:      Input feature dimensions (flattened to 1-D for the matmul).
            output_shape:     Output feature dimensions.
            n_control_points: Number of lattice control points.
            ndim:             Lattice dimensionality (2–4).
            bias:             If True, add a learnable bias.
        """

        def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            n_control_points: int = 16,
            ndim: int = 2,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self._core = SplineLinearND(
                input_shape, output_shape, n_control_points, ndim,
            )
            self.in_size = self._core.in_size
            self.out_size = self._core.out_size

            self.control_values = nn.Parameter(
                torch.from_numpy(self._core.control_values.copy()).float()
            )
            self.register_buffer(
                "_kernel", torch.from_numpy(self._core._kernel.copy()).float()
            )

            if bias:
                self.bias = nn.Parameter(torch.zeros(self.out_size))
            else:
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            W_flat = self._kernel @ self.control_values
            W = W_flat.reshape(self.out_size, self.in_size)
            return torch.nn.functional.linear(x, W, self.bias)

        def compression_ratio(self) -> float:
            actual = self._core.n_control_points
            if self.bias is not None:
                actual += self.out_size
            dense = self.out_size * self.in_size
            if self.bias is not None:
                dense += self.out_size
            return float(dense) / max(actual, 1)

except ImportError:
    pass  # PyTorch not available — NumPy-only mode


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_spline_nd(
    dimensions: List[Tuple[int, int]] = None,
    n_control_points_list: List[int] = None,
    ndim_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compare SplineLinearND parameter counts against dense equivalents.

    For each combination of (in_size, out_size), n_control_points, and ndim,
    reports the compression ratio and parameter counts.

    Args:
        dimensions:             List of ``(in_size, out_size)`` pairs.
        n_control_points_list:  Control-point counts to evaluate.
        ndim_values:            Lattice dimensionalities to evaluate.

    Returns:
        Dict with ``"results"`` (list of per-config dicts) and ``"summary"``.
    """
    if dimensions is None:
        dimensions = [(64, 32), (128, 64), (256, 128), (512, 256), (512, 512)]
    if n_control_points_list is None:
        n_control_points_list = [4, 8, 16, 32, 64]
    if ndim_values is None:
        ndim_values = [2, 3, 4]

    results: List[Dict[str, Any]] = []

    for in_s, out_s in dimensions:
        dense = in_s * out_s
        for n_cp in n_control_points_list:
            for nd in ndim_values:
                try:
                    layer = SplineLinearND(
                        input_shape=(in_s,),
                        output_shape=(out_s,),
                        n_control_points=n_cp,
                        ndim=nd,
                    )
                    ratio = layer.compression_ratio()
                    # Verify weights materialise correctly
                    W = layer.materialize_weights()
                    assert W.shape == (out_s, in_s), (
                        f"Shape mismatch: expected ({out_s}, {in_s}), got {W.shape}"
                    )
                    results.append({
                        "in_size": in_s,
                        "out_size": out_s,
                        "n_control_points": n_cp,
                        "ndim": nd,
                        "dense_params": dense,
                        "spline_params": n_cp,
                        "compression_ratio": round(ratio, 1),
                        "weight_shape_ok": True,
                    })
                except Exception as e:
                    results.append({
                        "in_size": in_s,
                        "out_size": out_s,
                        "n_control_points": n_cp,
                        "ndim": nd,
                        "error": str(e),
                    })

    # Summary: best compression per dimension combo
    summary_lines: List[str] = []
    for in_s, out_s in dimensions:
        dense = in_s * out_s
        subset = [r for r in results
                  if r.get("in_size") == in_s and r.get("out_size") == out_s
                  and "error" not in r]
        if subset:
            best = max(subset, key=lambda r: r["compression_ratio"])
            summary_lines.append(
                f"  ({in_s}, {out_s}): best ratio {best['compression_ratio']}:1 "
                f"with {best['n_control_points']} pts in {best['ndim']}D  "
                f"(dense={dense:,})"
            )

    return {
        "results": results,
        "summary": "\n".join(summary_lines),
    }


def benchmark_hierarchical(
    dimensions: List[Tuple[int, int]] = None,
    coarse_pts: int = 16,
    fine_pts: int = 8,
    patch_size: int = 32,
) -> Dict[str, Any]:
    """
    Benchmark HierarchicalSplineND vs dense for various layer sizes.
    """
    if dimensions is None:
        dimensions = [(64, 32), (128, 64), (256, 128), (512, 256), (512, 512)]

    results = []
    for in_s, out_s in dimensions:
        try:
            layer = HierarchicalSplineND(
                input_shape=(in_s,),
                output_shape=(out_s,),
                coarse_pts=coarse_pts,
                fine_pts=fine_pts,
                patch_sizes=patch_size,
            )
            W = layer.materialize_weights()
            assert W.shape == (out_s, in_s)
            results.append({
                "in_size": in_s,
                "out_size": out_s,
                "dense_params": in_s * out_s,
                "hier_params": layer.num_params(),
                "compression_ratio": round(layer.compression_ratio(), 1),
                "n_patches": layer.total_patches,
            })
        except Exception as e:
            results.append({
                "in_size": in_s,
                "out_size": out_s,
                "error": str(e),
            })

    return {"results": results}
