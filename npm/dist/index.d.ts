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
/** Basis constant: Eisenstein (inverse-distance-squared on hex lattice) */
export declare const EISENSTEIN_BASIS: Basis;
/** Basis constant: Bicubic B-spline on regular grid */
export declare const BSPLINE_BASIS: Basis;
/** Basis constant: Gaussian RBF on Eisenstein lattice */
export declare const GAUSSIAN_BASIS: Basis;
export declare const VARIANT_GUIDE: Record<string, CompressionMethod>;
/**
 * Places N control points on a hexagonal (Eisenstein) lattice.
 *
 * The Eisenstein lattice is spanned by {1, ω} where ω = e^(2πi/3).
 * Lattice point (a, b) maps to Cartesian (a − b/2, b·√3/2).
 * Points are selected as the N closest to the origin, normalized
 * so the outermost sits at unit Euclidean norm.
 */
export declare class EisensteinLattice {
    readonly nPoints: number;
    private _positions;
    constructor(nPoints: number);
    private static build;
    /** Control-point coordinates as (nPoints × 2) array */
    get positions(): number[][];
    /**
     * Find the k nearest control points to a 2-D query point.
     * Returns { distances, indices } sorted nearest-first.
     */
    nearestK(point: number[], k?: number): {
        distances: number[];
        indices: number[];
    };
}
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
export declare class SplineLinear {
    readonly inFeatures: number;
    readonly outFeatures: number;
    readonly nControlPoints: number;
    readonly basis: Basis;
    /** Control-point scalar values (the learnable parameters) */
    controlValues: number[];
    /** Log-bandwidth (only for gaussian basis) */
    logBandwidth: number;
    /** Bias vector (null if bias=false) */
    bias: number[] | null;
    private _latticePositions;
    private _gridPositions;
    private _gridK;
    constructor(inFeatures: number, outFeatures: number, nControlPoints?: number, basis?: Basis, bias?: boolean);
    /**
     * Materialize the full weight matrix from control-point values.
     * Returns a flat Float64Array of length outFeatures * inFeatures (row-major).
     */
    materializeWeights(): Float64Array;
    /**
     * Get the weight matrix as a 2D array (outFeatures × inFeatures).
     */
    getWeightMatrix(): number[][];
    /**
     * Forward pass: compute output = x @ W^T + bias
     * @param x Input array of length inFeatures (or array of arrays for batch)
     */
    forward(x: number[]): number[];
    forward(x: number[][]): number[][];
    private _applyLinear;
    /** Total trainable parameter count */
    numTrainableParams(): number;
    /** Parameters an equivalent dense layer would use */
    numEquivalentDenseParams(): number;
    /** Compression ratio vs dense layer */
    compressionRatio(): number;
    private _basisEisenstein;
    private _basisBspline;
    private _basisGaussian;
}
/**
 * Low-rank factorized linear layer: W ≈ U @ V + bias.
 *
 * Params: inFeatures * rank + rank * outFeatures (+ outFeatures bias).
 * For 512×512 at rank 16: 262,144 → 16,384 (16:1 compression).
 */
export declare class LowRankLinear {
    readonly inFeatures: number;
    readonly outFeatures: number;
    readonly rank: number;
    /** U matrix: (inFeatures × rank) */
    U: number[][];
    /** V matrix: (rank × outFeatures) */
    V: number[][];
    /** Bias vector (null if bias=false) */
    bias: number[] | null;
    constructor(inFeatures: number, outFeatures: number, rank?: number, bias?: boolean, initScale?: number);
    /**
     * Forward pass: x @ U @ V + bias
     */
    forward(x: number[]): number[];
    forward(x: number[][]): number[][];
    private _apply;
    numLowRankParams(): number;
    numEquivalentDenseParams(): number;
    compressionRatio(): number;
}
/**
 * Two-level spline: coarse grid for global structure + fine patches for local detail.
 *
 * W[i,j] = α * interpolate_coarse(i,j) + (1-α) * interpolate_fine(patch, local)
 * α is learnable (starts at alphaInit).
 */
export declare class HierarchicalSplineLinear {
    readonly inFeatures: number;
    readonly outFeatures: number;
    readonly patchSize: number;
    /** Coarse control-point values */
    coarseValues: number[];
    /** Fine control-point values: (totalPatches × finePts) */
    fineValues: number[][];
    /** Learnable blend factor */
    alpha: number;
    /** Bias vector */
    bias: number[] | null;
    readonly nInPatches: number;
    readonly nOutPatches: number;
    readonly totalPatches: number;
    private coarseLattice;
    private fineLattice;
    constructor(inFeatures: number, outFeatures: number, coarsePts?: number, finePts?: number, patchSize?: number, alphaInit?: number, bias?: boolean);
    /**
     * Materialize the full weight matrix.
     */
    materializeWeights(): Float64Array;
    /**
     * Forward pass: x @ W^T + bias
     */
    forward(x: number[]): number[];
    forward(x: number[][]): number[][];
    numControlParams(): number;
    numEquivalentDenseParams(): number;
    compressionRatio(): number;
}
/**
 * Compute compression ratio for a SplineLinear configuration.
 */
export declare function compressionRatio(inFeatures: number, outFeatures: number, controlPoints: number, basis?: Basis, bias?: boolean): CompressionReport;
/**
 * Recommend the best compression variant based on task description.
 *
 * Keywords like "drift", "sensor", "regression" → "spline"
 * Keywords like "classify", "detect", "intent" → "lowrank"
 * Otherwise → "lowrank" (safe default)
 */
export declare function recommendVariant(taskDescription: string): string;
/** Serialize a SplineLinear layer to a plain object */
export declare function serializeSplineLinear(layer: SplineLinear): Record<string, unknown>;
/** Deserialize a SplineLinear layer from a plain object */
export declare function deserializeSplineLinear(data: Record<string, unknown>): SplineLinear;
