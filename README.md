# tensor-spline

## The problem with neural network weights

A typical `nn.Linear(512, 512)` layer stores 262,144 float32 parameters. That's 1MB per layer. A transformer model has hundreds of these layers. The weights are stored as independent floating-point numbers — every weight is a separate learned value with no relationship to its neighbors.

But here's the thing: weight matrices aren't random. Adjacent weights in a trained layer are correlated. If weight `[i, j]` is 0.7, weight `[i, j+1]` is probably somewhere near 0.7 too. The weight surface is smooth.

Spline interpolation exploits this. Instead of storing 262,144 independent values, you store a small number of *control points* and interpolate between them.

## How SplineLinear works

A regular `nn.Linear` layer:

```python
# Standard: every weight is independent
W[i][j] = learned_float
```

A `SplineLinear` layer:

```python
# Spline: weights are interpolated from control points
W[i][j] = interpolate(control_points, position(i, j))
```

The control points sit on an Eisenstein (hexagonal) lattice. Each weight position `(i, j)` maps to a 2D coordinate. The weight value at that position is interpolated from nearby control points. During training, gradients flow back through the interpolation to the control points — standard autograd handles everything.

The math is inverse-distance-squared weighting:

```
W(p) = Σ_k c_k · d_k⁻² / Σ_k d_k⁻²

where d_k = ||p − L_k|| + ε
```

Each control point `c_k` contributes to the weight at position `p` proportionally to how close it is. Nearby control points dominate. Far-away ones barely matter.

## Compression ratio

For a `nn.Linear(512, 512)` layer with 16 control points:

```
Dense:    512 × 512 = 262,144 parameters
Spline:   16 control points + 512 bias = 528 parameters
Ratio:    262,144 / 528 = 497× compression
```

That's not a typo. You're replacing 262K parameters with 528 and the layer still works. The weight surface is smooth enough that 16 control points capture the essential shape.

## What's actually happening?

The interpolation creates a smooth surface that passes near (but not through) the control points. Think of it like a rubber sheet with pegs at the control-point locations — the sheet stretches to minimize the total bending energy. The optimizer only tweaks the peg heights; the sheet shape follows automatically.

Three basis functions are available:

| Basis | How it works | Best for |
|-------|-------------|----------|
| `eisenstein` | Inverse-distance² on hexagonal lattice | General use |
| `bspline` | Bicubic B-spline via `grid_sample` | Strong locality |
| `gaussian` | RBF with learned bandwidth σ | Adaptive smoothness |

## Install and use

```bash
pip install tensor-spline
```

```python
import torch
import torch.nn as nn
from tensor_spline import SplineLinear, inject_spline, compression_ratio

# Replace a single layer
layer = SplineLinear(512, 512, n_control_points=16, basis="eisenstein")
x = torch.randn(4, 512)
print(layer(x).shape)                    # torch.Size([4, 512])
print(layer.compression_ratio())          # ~497×
print(layer.num_trainable_params())       # 528

# Inject into an entire model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
injection_map = inject_spline(model, n_control_points=16)
stats = compression_ratio(model)
print(f"Compressed {stats['ratio']:.0f}× ({stats['n_spline_layers']} layers)")
```

## The Eisenstein lattice

Control points are placed on the densest possible 2D grid — the Eisenstein lattice (hexagonal packing, proven optimal by Thue 1910). Points at positions `a + bω` where `ω = e^(2πi/3)`.

```python
from tensor_spline import EisensteinLattice

lattice = EisensteinLattice(7)
print(lattice.positions().shape)          # torch.Size([7, 2])
# First point is always the origin (closest to center)
print(lattice.positions()[0])             # tensor([0., 0.])
```

## Hierarchical and low-rank variants

`tensor-spline` includes two additional compression strategies:

- **HierarchicalSplineLinear** — multi-scale: coarse control points capture global shape, fine control points capture local detail. More expressive than flat spline for the same parameter budget.
- **LowRankLinear** — standard U×V matrix factorization. Simpler than spline but effective when weights have low intrinsic rank.

```python
from tensor_spline import recommend_variant

# Task-aware recommendation
rec = recommend_variant(
    param_budget=1024,      # max trainable params per layer
    task="classification",  # classification vs generation vs embedding
    model_size="small",     # small, medium, large
)
print(rec["variant"], rec["reason"])
```

## Why does this work?

Neural network weights are smooth surfaces. Smooth surfaces need far fewer control points than independent samples to represent accurately. This is the same principle behind JPEG compression (store low-frequency components, drop high-frequency noise), vector graphics (bezier curves instead of pixel grids), and NURBS modeling in CAD.

The Eisenstein lattice maximizes control-point coverage because hexagonal packing is the densest arrangement in 2D. Each control point "covers" the maximum possible area, so you need fewer of them.

The optimizer only updates control-point values. The interpolation handles the rest. Less parameters means faster training, less memory, and natural regularization — the spline can't overfit to noise because it doesn't have enough degrees of freedom.

## Related Repos

- **[flux-tensor-midi](https://github.com/SuperInstance/flux-tensor-midi)** — INT8-saturated MIDI for neural synthesis (uses Eisenstein lattice)
- **[constraint-instrument](https://github.com/SuperInstance/constraint-instrument)** — Constraint-based music generation (uses spline surfaces)
- **[plato-room-musician](https://github.com/SuperInstance/plato-room-musician)** — Sonify fleet activity via MIDI
- **[penrose-memory](https://github.com/SuperInstance/penrose-memory)** — Aperiodic memory palace (related aperiodic math)

## License

MIT
