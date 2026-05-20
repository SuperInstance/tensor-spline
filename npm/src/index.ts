/**
 * tensor-spline — Compressed neural network layers for JavaScript/TypeScript.
 *
 * Weight parameterization using Eisenstein lattice control points,
 * plus standard low-rank factorization. Framework-agnostic pure JS.
 *
 * Layers:
 *   SplineLinear:        weights interpolated from hex lattice control points
 *   LowRankLinear:       learned U@V factorization
 *   HierarchicalSplineLinear: multi-scale coarse+fine interpolation
 *
 * Ported from the Python tensor-spline library.
 *
 * @module tensor-spline
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Basis function type for SplineLinear */
export type Basis = 'eisenstein' | 'bspline' | 'gaussian';

/** Compression method descriptor */
export interface CompressionMethod {
  name: string;
  method: string;
  bestFor: string;
  compression: number;
  accuracyRetention: number;
  notes?: string;
}

/** Compression ratio report */
export interface CompressionReport {
  originalParams: number;
  splineParams: number;
  ratio: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Basis constant: Eisenstein (inverse-distance-squared on hex lattice) */
export const EISENSTEIN_BASIS: Basis = 'eisenstein';
/** Basis constant: Bicubic B-spline on regular grid */
export const BSPLINE_BASIS: Basis = 'bspline';
/** Basis constant: Gaussian RBF on Eisenstein lattice */
export const GAUSSIAN_BASIS: Basis = 'gaussian';

const SQRT3_HALF = Math.sqrt(3) / 2;
const VALID_BASES: Set<Basis> = new Set(['eisenstein', 'bspline', 'gaussian']);

export const VARIANT_GUIDE: Record<string, CompressionMethod> = {
  smooth: {
    name: 'spline',
    method: 'spline',
    bestFor: 'Continuous signals, drift detection, regression',
    compression: 20.0,
    accuracyRetention: 0.95,
    notes: 'IDW interpolation excels at smooth functions',
  },
  sharp: {
    name: 'low-rank',
    method: 'lowrank',
    bestFor: 'Classification, sharp boundaries, categorical',
    compression: 10.0,
    accuracyRetention: 0.80,
    notes: 'Learned factorization handles discontinuities',
  },
  adaptive: {
    name: 'lora',
    method: 'lora',
    bestFor: 'Fine-tuning pretrained models, multi-task',
    compression: 5.0,
    accuracyRetention: 0.85,
    notes: 'Best when you have a base model to adapt',
  },
};

// ---------------------------------------------------------------------------
// EisensteinLattice
// ---------------------------------------------------------------------------

/**
 * Places N control points on a hexagonal (Eisenstein) lattice.
 *
 * The Eisenstein lattice is spanned by {1, ω} where ω = e^(2πi/3).
 * Lattice point (a, b) maps to Cartesian (a − b/2, b·√3/2).
 * Points are selected as the N closest to the origin, normalized
 * so the outermost sits at unit Euclidean norm.
 */
export class EisensteinLattice {
  public readonly nPoints: number;
  private _positions: number[][]; // (n, 2)

  constructor(nPoints: number) {
    if (nPoints < 1) throw new Error(`nPoints must be >= 1, got ${nPoints}`);
    this.nPoints = nPoints;
    this._positions = EisensteinLattice.build(nPoints);
  }

  private static build(n: number): number[][] {
    const R = Math.max(Math.ceil(Math.sqrt(n / 3)) + 2, 3);
    const candidates: [number, number, number][] = [];

    for (let a = -R; a <= R; a++) {
      for (let b = -R; b <= R; b++) {
        const x = a - b * 0.5;
        const y = b * SQRT3_HALF;
        const distSq = x * x + y * y;
        candidates.push([distSq, x, y]);
      }
    }

    candidates.sort((a, b) => {
      const d = Math.round(a[0] * 1e9) - Math.round(b[0] * 1e9);
      if (d !== 0) return d;
      return a[1] !== b[1] ? a[1] - b[1] : a[2] - b[2];
    });

    const selected = candidates.slice(0, n);
    let maxDist = 0;
    for (const [, x, y] of selected) {
      const d = Math.sqrt(x * x + y * y);
      if (d > maxDist) maxDist = d;
    }

    if (maxDist > 0) {
      return selected.map(([, x, y]) => [x / maxDist, y / maxDist]);
    }
    return selected.map(([, x, y]) => [x, y]);
  }

  /** Control-point coordinates as (nPoints × 2) array */
  get positions(): number[][] {
    return this._positions;
  }

  /**
   * Find the k nearest control points to a 2-D query point.
   * Returns { distances, indices } sorted nearest-first.
   */
  nearestK(point: number[], k: number = 4): { distances: number[]; indices: number[] } {
    k = Math.min(k, this.nPoints);
    const dists = this._positions.map(([px, py]) =>
      Math.sqrt((px - point[0]) ** 2 + (py - point[1]) ** 2)
    );
    const indexed = dists.map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d);
    const top = indexed.slice(0, k);
    return {
      distances: top.map(t => t.d),
      indices: top.map(t => t.i),
    };
  }
}

// ---------------------------------------------------------------------------
// SplineLinear
// ---------------------------------------------------------------------------

/**
 * Drop-in replacement for a dense linear layer with lattice-parameterized weights.
 *
 * The full weight matrix W (outFeatures × inFeatures) is materialized from
 * control-point scalar values. Only nControlPoints scalars (plus optional bias)
 * are trained instead of outFeatures × inFeatures independent weights.
 *
 * Basis functions:
 * - "eisenstein": Normalized inverse-distance-squared weighting on hex lattice
 * - "bspline":    Bicubic B-spline on regular grid
 * - "gaussian":   Gaussian RBF on hex lattice with learnable bandwidth
 */
export class SplineLinear {
  public readonly inFeatures: number;
  public readonly outFeatures: number;
  public readonly nControlPoints: number;
  public readonly basis: Basis;

  /** Control-point scalar values (the learnable parameters) */
  public controlValues: number[];

  /** Log-bandwidth (only for gaussian basis) */
  public logBandwidth: number;

  /** Bias vector (null if bias=false) */
  public bias: number[] | null;

  // Precomputed internals
  private _latticePositions: number[][] | null;
  private _gridPositions: number[][]; // (out*in, 2)
  private _gridK: number; // for bspline

  constructor(
    inFeatures: number,
    outFeatures: number,
    nControlPoints: number = 16,
    basis: Basis = 'eisenstein',
    bias: boolean = true
  ) {
    if (!VALID_BASES.has(basis)) {
      throw new Error(`basis must be one of ${[...VALID_BASES]}, got '${basis}'`);
    }

    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.nControlPoints = nControlPoints;
    this.basis = basis;

    // Kaiming-style init
    const bound = 1 / Math.sqrt(nControlPoints);
    this.controlValues = Array.from({ length: nControlPoints }, () =>
      (Math.random() * 2 - 1) * bound
    );

    this.logBandwidth = 0; // exp(0) = 1 at init

    // Lattice for eisenstein/gaussian
    if (basis === 'eisenstein' || basis === 'gaussian') {
      const lattice = new EisensteinLattice(nControlPoints);
      this._latticePositions = lattice.positions;
    } else {
      this._latticePositions = null;
    }

    // Grid side for bspline
    this._gridK = Math.ceil(Math.sqrt(nControlPoints));

    // Weight-matrix sampling grid: maps every W[i,j] to [-1,1]²
    const gridPositions: number[][] = [];
    for (let i = 0; i < outFeatures; i++) {
      const y = outFeatures > 1 ? -1 + (2 * i) / (outFeatures - 1) : 0;
      for (let j = 0; j < inFeatures; j++) {
        const x = inFeatures > 1 ? -1 + (2 * j) / (inFeatures - 1) : 0;
        gridPositions.push([x, y]);
      }
    }
    this._gridPositions = gridPositions;

    // Bias
    this.bias = bias ? new Array(outFeatures).fill(0) : null;
  }

  /**
   * Materialize the full weight matrix from control-point values.
   * Returns a flat Float64Array of length outFeatures * inFeatures (row-major).
   */
  materializeWeights(): Float64Array {
    const n = this.outFeatures * this.inFeatures;
    const wFlat = new Float64Array(n);

    if (this.basis === 'eisenstein') {
      this._basisEisenstein(wFlat);
    } else if (this.basis === 'bspline') {
      this._basisBspline(wFlat);
    } else {
      this._basisGaussian(wFlat);
    }

    return wFlat;
  }

  /**
   * Get the weight matrix as a 2D array (outFeatures × inFeatures).
   */
  getWeightMatrix(): number[][] {
    const flat = this.materializeWeights();
    const mat: number[][] = [];
    for (let i = 0; i < this.outFeatures; i++) {
      mat.push(Array.from(flat.subarray(i * this.inFeatures, (i + 1) * this.inFeatures)));
    }
    return mat;
  }

  /**
   * Forward pass: compute output = x @ W^T + bias
   * @param x Input array of length inFeatures (or array of arrays for batch)
   */
  forward(x: number[]): number[];
  forward(x: number[][]): number[][];
  forward(x: number[] | number[][]): number[] | number[][] {
    if (x.length === 0) return x;
    const W = this.materializeWeights();

    if (Array.isArray(x[0])) {
      // Batch input
      const batch = x as number[][];
      return batch.map(sample => this._applyLinear(sample, W));
    }
    return this._applyLinear(x as number[], W);
  }

  private _applyLinear(x: number[], W: Float64Array): number[] {
    const out = new Array(this.outFeatures);
    for (let i = 0; i < this.outFeatures; i++) {
      let sum = 0;
      const rowOff = i * this.inFeatures;
      for (let j = 0; j < this.inFeatures; j++) {
        sum += x[j] * W[rowOff + j];
      }
      out[i] = this.bias ? sum + this.bias[i] : sum;
    }
    return out;
  }

  /** Total trainable parameter count */
  numTrainableParams(): number {
    let count = this.nControlPoints;
    if (this.basis === 'gaussian') count += 1; // logBandwidth
    if (this.bias) count += this.outFeatures;
    return count;
  }

  /** Parameters an equivalent dense layer would use */
  numEquivalentDenseParams(): number {
    let n = this.outFeatures * this.inFeatures;
    if (this.bias) n += this.outFeatures;
    return n;
  }

  /** Compression ratio vs dense layer */
  compressionRatio(): number {
    return this.numEquivalentDenseParams() / Math.max(this.numTrainableParams(), 1);
  }

  // --- Basis implementations ---

  private _basisEisenstein(wFlat: Float64Array): void {
    const nPts = this._gridPositions.length;
    const lattice = this._latticePositions!;
    const nCP = this.nControlPoints;

    for (let p = 0; p < nPts; p++) {
      const [px, py] = this._gridPositions[p];
      let sumKernel = 0;
      let sumWeighted = 0;
      for (let k = 0; k < nCP; k++) {
        const dx = px - lattice[k][0];
        const dy = py - lattice[k][1];
        const distSq = dx * dx + dy * dy + 1e-6;
        const kernel = 1.0 / distSq;
        sumKernel += kernel;
        sumWeighted += kernel * this.controlValues[k];
      }
      wFlat[p] = sumWeighted / sumKernel;
    }
  }

  private _basisBspline(wFlat: Float64Array): void {
    // Simplified bicubic B-spline: use bilinear sampling on a regular grid
    const k = this._gridK;
    const nPts = this._gridPositions.length;

    // Build grid from control values (zero-pad if needed)
    const grid = new Float64Array(k * k);
    for (let i = 0; i < this.nControlPoints; i++) {
      grid[i] = this.controlValues[i];
    }

    for (let p = 0; p < nPts; p++) {
      const [gx, gy] = this._gridPositions[p];
      // Map [-1,1] to [0, k-1]
      const fx = (gx + 1) * 0.5 * (k - 1);
      const fy = (gy + 1) * 0.5 * (k - 1);

      // Bicubic interpolation using cubic Hermite spline
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const fx2 = fx - x0;
      const fy2 = fy - y0;

      // Cubic basis function: B(t) = (1/6)((-t³+3t²-3t+1)max + (3t³-6t²+4)mid + (-3t³+3t²+3t+1)mid2 + (t³)max2)
      let val = 0;
      for (let dy = -1; dy <= 2; dy++) {
        for (let dx = -1; dx <= 2; dx++) {
          const sx = clamp(x0 + dx, 0, k - 1);
          const sy = clamp(y0 + dy, 0, k - 1);
          const w = cubicWeight(dx - fx2) * cubicWeight(dy - fy2);
          val += grid[sy * k + sx] * w;
        }
      }
      wFlat[p] = val;
    }
  }

  private _basisGaussian(wFlat: Float64Array): void {
    const sigma = Math.max(Math.exp(this.logBandwidth), 1e-3);
    const sigma2x2 = 2 * sigma * sigma;
    const nPts = this._gridPositions.length;
    const lattice = this._latticePositions!;
    const nCP = this.nControlPoints;

    for (let p = 0; p < nPts; p++) {
      const [px, py] = this._gridPositions[p];
      let sumKernel = 0;
      let sumWeighted = 0;
      for (let k = 0; k < nCP; k++) {
        const dx = px - lattice[k][0];
        const dy = py - lattice[k][1];
        const distSq = dx * dx + dy * dy;
        const kernel = Math.exp(-distSq / sigma2x2);
        sumKernel += kernel;
        sumWeighted += kernel * this.controlValues[k];
      }
      wFlat[p] = sumWeighted / (sumKernel + 1e-8);
    }
  }
}

// ---------------------------------------------------------------------------
// LowRankLinear
// ---------------------------------------------------------------------------

/**
 * Low-rank factorized linear layer: W ≈ U @ V + bias.
 *
 * Params: inFeatures * rank + rank * outFeatures (+ outFeatures bias).
 * For 512×512 at rank 16: 262,144 → 16,384 (16:1 compression).
 */
export class LowRankLinear {
  public readonly inFeatures: number;
  public readonly outFeatures: number;
  public readonly rank: number;

  /** U matrix: (inFeatures × rank) */
  public U: number[][];

  /** V matrix: (rank × outFeatures) */
  public V: number[][];

  /** Bias vector (null if bias=false) */
  public bias: number[] | null;

  constructor(
    inFeatures: number,
    outFeatures: number,
    rank: number = 16,
    bias: boolean = true,
    initScale: number = 0.01
  ) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.rank = rank;

    this.U = makeMatrix(inFeatures, rank, () => randn() * initScale);
    this.V = makeMatrix(rank, outFeatures, () => randn() * initScale);
    this.bias = bias ? new Array(outFeatures).fill(0) : null;
  }

  /**
   * Forward pass: x @ U @ V + bias
   */
  forward(x: number[]): number[];
  forward(x: number[][]): number[][];
  forward(x: number[] | number[][]): number[] | number[][] {
    if (Array.isArray(x[0])) {
      return (x as number[][]).map(sample => this._apply(sample));
    }
    return this._apply(x as number[]);
  }

  private _apply(x: number[]): number[] {
    // x @ U → (rank,)
    const hidden = new Array(this.rank).fill(0);
    for (let r = 0; r < this.rank; r++) {
      for (let j = 0; j < this.inFeatures; j++) {
        hidden[r] += x[j] * this.U[j][r];
      }
    }
    // hidden @ V → (outFeatures,)
    const out = new Array(this.outFeatures).fill(0);
    for (let o = 0; o < this.outFeatures; o++) {
      for (let r = 0; r < this.rank; r++) {
        out[o] += hidden[r] * this.V[r][o];
      }
      if (this.bias) out[o] += this.bias[o];
    }
    return out;
  }

  numLowRankParams(): number {
    let count = this.inFeatures * this.rank + this.rank * this.outFeatures;
    if (this.bias) count += this.outFeatures;
    return count;
  }

  numEquivalentDenseParams(): number {
    let n = this.inFeatures * this.outFeatures;
    if (this.bias) n += this.outFeatures;
    return n;
  }

  compressionRatio(): number {
    return this.numEquivalentDenseParams() / Math.max(this.numLowRankParams(), 1);
  }
}

// ---------------------------------------------------------------------------
// HierarchicalSplineLinear
// ---------------------------------------------------------------------------

/**
 * Two-level spline: coarse grid for global structure + fine patches for local detail.
 *
 * W[i,j] = α * interpolate_coarse(i,j) + (1-α) * interpolate_fine(patch, local)
 * α is learnable (starts at alphaInit).
 */
export class HierarchicalSplineLinear {
  public readonly inFeatures: number;
  public readonly outFeatures: number;
  public readonly patchSize: number;

  /** Coarse control-point values */
  public coarseValues: number[];

  /** Fine control-point values: (totalPatches × finePts) */
  public fineValues: number[][];

  /** Learnable blend factor */
  public alpha: number;

  /** Bias vector */
  public bias: number[] | null;

  public readonly nInPatches: number;
  public readonly nOutPatches: number;
  public readonly totalPatches: number;

  private coarseLattice: EisensteinLattice;
  private fineLattice: EisensteinLattice;

  constructor(
    inFeatures: number,
    outFeatures: number,
    coarsePts: number = 16,
    finePts: number = 8,
    patchSize: number = 32,
    alphaInit: number = 0.5,
    bias: boolean = true
  ) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.patchSize = patchSize;

    this.coarseLattice = new EisensteinLattice(coarsePts);
    this.fineLattice = new EisensteinLattice(finePts);

    this.coarseValues = Array.from({ length: coarsePts }, () => randn() * 0.01);

    this.nInPatches = Math.max(1, Math.ceil(inFeatures / patchSize));
    this.nOutPatches = Math.max(1, Math.ceil(outFeatures / patchSize));
    this.totalPatches = this.nInPatches * this.nOutPatches;

    this.fineValues = Array.from({ length: this.totalPatches }, () =>
      Array.from({ length: finePts }, () => randn() * 0.01)
    );

    this.alpha = alphaInit;
    this.bias = bias ? new Array(outFeatures).fill(0) : null;
  }

  /**
   * Materialize the full weight matrix.
   */
  materializeWeights(): Float64Array {
    const coarsePositions = this.coarseLattice.positions;
    const finePositions = this.fineLattice.positions;
    const ps = this.patchSize;
    const sigAlpha = 1 / (1 + Math.exp(-this.alpha)); // sigmoid

    // Coarse interpolation over entire weight matrix
    const coarseW = new Float64Array(this.outFeatures * this.inFeatures);
    for (let i = 0; i < this.outFeatures; i++) {
      const y = this.outFeatures > 1 ? -1 + (2 * i) / (this.outFeatures - 1) : 0;
      for (let j = 0; j < this.inFeatures; j++) {
        const x = this.inFeatures > 1 ? -1 + (2 * j) / (this.inFeatures - 1) : 0;
        coarseW[i * this.inFeatures + j] = idwInterpolate(
          [x, y], coarsePositions, this.coarseValues
        );
      }
    }

    // Fine interpolation per patch
    const result = new Float64Array(this.outFeatures * this.inFeatures);

    for (let pi = 0; pi < this.nInPatches; pi++) {
      for (let pj = 0; pj < this.nOutPatches; pj++) {
        const patchIdx = pj * this.nInPatches + pi;
        const inStart = pi * ps;
        const inEnd = Math.min(inStart + ps, this.inFeatures);
        const outStart = pj * ps;
        const outEnd = Math.min(outStart + ps, this.outFeatures);

        const patchRows = outEnd - outStart;
        const patchCols = inEnd - inStart;

        for (let i = 0; i < patchRows; i++) {
          for (let j = 0; j < patchCols; j++) {
            const globalI = outStart + i;
            const globalJ = inStart + j;

            // Local coords normalized to [-1, 1]
            const ly = patchRows > 1 ? -1 + (2 * i) / (patchRows - 1) : 0;
            const lx = patchCols > 1 ? -1 + (2 * j) / (patchCols - 1) : 0;

            const fineW = idwInterpolate(
              [lx, ly], finePositions, this.fineValues[patchIdx]
            );

            const coarseVal = coarseW[globalI * this.inFeatures + globalJ];
            result[globalI * this.inFeatures + globalJ] =
              sigAlpha * coarseVal + (1 - sigAlpha) * fineW;
          }
        }
      }
    }

    return result;
  }

  /**
   * Forward pass: x @ W^T + bias
   */
  forward(x: number[]): number[];
  forward(x: number[][]): number[][];
  forward(x: number[] | number[][]): number[] | number[][] {
    const W = this.materializeWeights();
    if (Array.isArray(x[0])) {
      return (x as number[][]).map(sample => {
        const out = new Array(this.outFeatures);
        for (let i = 0; i < this.outFeatures; i++) {
          let sum = 0;
          for (let j = 0; j < this.inFeatures; j++) {
            sum += sample[j] * W[i * this.inFeatures + j];
          }
          out[i] = this.bias ? sum + this.bias[i] : sum;
        }
        return out;
      });
    }
    const sample = x as number[];
    const out = new Array(this.outFeatures);
    for (let i = 0; i < this.outFeatures; i++) {
      let sum = 0;
      for (let j = 0; j < this.inFeatures; j++) {
        sum += sample[j] * W[i * this.inFeatures + j];
      }
      out[i] = this.bias ? sum + this.bias[i] : sum;
    }
    return out;
  }

  numControlParams(): number {
    return this.coarseValues.length + this.totalPatches * this.fineValues[0].length + 1;
  }

  numEquivalentDenseParams(): number {
    let n = this.inFeatures * this.outFeatures;
    if (this.bias) n += this.outFeatures;
    return n;
  }

  compressionRatio(): number {
    const actual = this.numControlParams() + (this.bias ? this.outFeatures : 0);
    return this.numEquivalentDenseParams() / Math.max(actual, 1);
  }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/**
 * Compute compression ratio for a SplineLinear configuration.
 */
export function compressionRatio(
  inFeatures: number,
  outFeatures: number,
  controlPoints: number,
  basis: Basis = 'eisenstein',
  bias: boolean = true
): CompressionReport {
  let actual = controlPoints;
  if (basis === 'gaussian') actual += 1;
  if (bias) actual += outFeatures;

  let dense = inFeatures * outFeatures;
  if (bias) dense += outFeatures;

  return {
    originalParams: dense,
    splineParams: actual,
    ratio: dense / Math.max(actual, 1),
  };
}

/**
 * Recommend the best compression variant based on task description.
 *
 * Keywords like "drift", "sensor", "regression" → "spline"
 * Keywords like "classify", "detect", "intent" → "lowrank"
 * Otherwise → "lowrank" (safe default)
 */
export function recommendVariant(taskDescription: string): string {
  const desc = taskDescription.toLowerCase();

  const smoothKeywords = ['drift', 'sensor', 'continuous', 'regression', 'smooth', 'anomaly', 'signal'];
  if (smoothKeywords.some(kw => desc.includes(kw))) return 'spline';

  const sharpKeywords = ['classify', 'detect', 'intent', 'topic', 'priority', 'category'];
  if (sharpKeywords.some(kw => desc.includes(kw))) return 'lowrank';

  return 'lowrank';
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/** Serialize a SplineLinear layer to a plain object */
export function serializeSplineLinear(layer: SplineLinear): Record<string, unknown> {
  return {
    type: 'spline_linear',
    inFeatures: layer.inFeatures,
    outFeatures: layer.outFeatures,
    nControlPoints: layer.nControlPoints,
    basis: layer.basis,
    controlValues: layer.controlValues,
    logBandwidth: layer.logBandwidth,
    bias: layer.bias,
  };
}

/** Deserialize a SplineLinear layer from a plain object */
export function deserializeSplineLinear(data: Record<string, unknown>): SplineLinear {
  const layer = new SplineLinear(
    data.inFeatures as number,
    data.outFeatures as number,
    data.nControlPoints as number,
    data.basis as Basis,
    (data.bias as number[] | null) !== null
  );
  layer.controlValues = data.controlValues as number[];
  if (layer.basis === 'gaussian') layer.logBandwidth = data.logBandwidth as number;
  if (data.bias) layer.bias = data.bias as number[];
  return layer;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function idwInterpolate(
  point: number[],
  positions: number[][],
  values: number[]
): number {
  let sumKernel = 0;
  let sumWeighted = 0;
  for (let k = 0; k < positions.length; k++) {
    const dx = point[0] - positions[k][0];
    const dy = point[1] - positions[k][1];
    const distSq = dx * dx + dy * dy + 1e-6;
    const kernel = 1.0 / distSq;
    sumKernel += kernel;
    sumWeighted += kernel * values[k];
  }
  return sumWeighted / sumKernel;
}

function cubicWeight(t: number): number {
  return bSplineBasis(Math.abs(t));
}

function bSplineBasis(d: number): number {
  if (d < 1) return (2 / 3) + d * d * (d / 2 - 1);
  if (d < 2) return (1 / 6) * (2 - d) * (2 - d) * (2 - d);
  return 0;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function makeMatrix(rows: number, cols: number, fill: () => number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, fill));
}

function randn(): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
