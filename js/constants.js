export const COLORS = [
    '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#6366f1',
    '#84cc16', '#eab308', '#d946ef', '#0ea5e9', '#fca5a5'
];

export const MAX_PKL_BYTES = 512 * 1024 * 1024;

export function createDefaultState() {
    return {
        kValues: [],
        epsilon: 0.0,
        cache: [],
        otCache: [],
        pointOtGammaCache: [],
        pointOtAggCache: [],
        truePis: null,
        globalHistory: {},
        pcaPoints: [],
        pca3dPoints: [],
        kernelPcaPoints: [],
        dendrograms: [],
        kernelPcaPerplexity: null,
        dendrogramLinkage: null,
        computingPca3d: false,
        computingKernelPca: false,
        computingDendrograms: false,
        computingPointOt: false,
        pointOtEpsilon: null,
        plotSize: 400,
        plotFeedback: {},
        plotAnnotations: {},
        plotAnnotationEmphasis: {},
        kLocks: []
    };
}
