/**
 * Cheap geometric "cluster shape" descriptors from PCA coords + assignments.
 * No vision model — captures separation, compactness, and cluster-size balance.
 */

function clusterCentroids(points2d, assignments, k) {
    const sumX = new Float64Array(k);
    const sumY = new Float64Array(k);
    const count = new Int32Array(k);
    const n = assignments.length;
    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        if (c < 0 || c >= k) continue;
        count[c]++;
        sumX[c] += points2d[i][0];
        sumY[c] += points2d[i][1];
    }
    const cx = new Float64Array(k);
    const cy = new Float64Array(k);
    for (let c = 0; c < k; c++) {
        const d = count[c] || 1;
        cx[c] = sumX[c] / d;
        cy[c] = sumY[c] / d;
    }
    return { cx, cy, count };
}

/** Mean silhouette in [-1, 1] (sampled for speed on large n). */
export function computeMeanSilhouette(points2d, assignments, k, maxSamples = 400) {
    const n = assignments?.length ?? 0;
    if (!points2d || points2d.length !== n || k <= 1 || n < k + 2) return 0;

    const indices = [];
    if (n <= maxSamples) {
        for (let i = 0; i < n; i++) indices.push(i);
    } else {
        const used = new Set();
        while (indices.length < maxSamples) {
            const j = (Math.random() * n) | 0;
            if (!used.has(j)) {
                used.add(j);
                indices.push(j);
            }
        }
    }

    const dist = (i, j) => {
        const dx = points2d[i][0] - points2d[j][0];
        const dy = points2d[i][1] - points2d[j][1];
        return Math.sqrt(dx * dx + dy * dy);
    };

    let sumS = 0;
    let counted = 0;
    for (const i of indices) {
        const ci = assignments[i];
        let a = 0;
        let aCount = 0;
        const bByCluster = new Float64Array(k).fill(Infinity);

        for (const j of indices) {
            if (i === j) continue;
            const cj = assignments[j];
            const d = dist(i, j);
            if (cj === ci) {
                a += d;
                aCount++;
            } else if (d < bByCluster[cj]) {
                bByCluster[cj] = d;
            }
        }
        if (aCount === 0) continue;
        a /= aCount;
        let b = Infinity;
        for (let c = 0; c < k; c++) {
            if (c === ci) continue;
            if (bByCluster[c] < b) b = bByCluster[c];
        }
        if (!Number.isFinite(b)) continue;
        const s = b > a ? (b - a) / b : (b - a) / a;
        sumS += s;
        counted++;
    }
    return counted > 0 ? sumS / counted : 0;
}

/** Per-step stats; higher `score` = nicer blob shapes (compact + separated + balanced). */
export function computeStepShapeStats(points2d, assignments, k) {
    const n = assignments?.length ?? 0;
    if (!points2d || points2d.length !== n || k <= 1 || n < k + 2) {
        return { separation: 0, compactness: 0, balance: 0, score: 0, k };
    }

    const { cx, cy, count } = clusterCentroids(points2d, assignments, k);

    let within = 0;
    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        const dx = points2d[i][0] - cx[c];
        const dy = points2d[i][1] - cy[c];
        within += dx * dx + dy * dy;
    }
    within = within / Math.max(n, 1);

    let minInter = Infinity;
    let meanInter = 0;
    let pairs = 0;
    for (let i = 0; i < k; i++) {
        if (count[i] === 0) continue;
        for (let j = i + 1; j < k; j++) {
            if (count[j] === 0) continue;
            const dx = cx[i] - cx[j];
            const dy = cy[i] - cy[j];
            const d = Math.sqrt(dx * dx + dy * dy);
            minInter = Math.min(minInter, d);
            meanInter += d;
            pairs++;
        }
    }
    if (!Number.isFinite(minInter)) minInter = 0;
    meanInter = pairs > 0 ? meanInter / pairs : 0;

    const separation = minInter / (Math.sqrt(within) + 1e-9);
    const compactness = 1 / (1 + Math.sqrt(within));

    // Penalize one huge cluster + tiny satellites (low entropy of cluster sizes)
    let entropy = 0;
    for (let c = 0; c < k; c++) {
        const p = count[c] / n;
        if (p > 0) entropy -= p * Math.log(p);
    }
    const maxEntropy = Math.log(k);
    const balance = maxEntropy > 0 ? entropy / maxEntropy : 0;

    const silhouette = computeMeanSilhouette(points2d, assignments, k);
    const silBonus = Math.max(0, (silhouette + 0.05) / 1.05);
    const score = separation * 0.4 + compactness * 0.2 + balance * 0.15 + silBonus * 0.25;
    const nonempty = Array.from(count).filter(x => x > 0).length;

    return {
        k,
        nonemptyClusters: nonempty,
        separation: +separation.toFixed(3),
        compactness: +compactness.toFixed(3),
        balance: +balance.toFixed(3),
        silhouette: +silhouette.toFixed(3),
        score: +score.toFixed(4),
        sizes: Array.from(count)
    };
}

export function buildTrajectoryClusterDiagnostics(pcaPointsByStep, clusters, kValues) {
    const steps = clusters.map((c, t) => {
        const k = kValues[t] ?? c.centers?.length ?? 1;
        const stats = computeStepShapeStats(pcaPointsByStep[t], c.assignments, c.centers?.length ?? k);
        return {
            step: t + 1,
            K: k,
            ...stats
        };
    });
    return {
        steps,
        meanSilhouette: steps.length
            ? +(steps.reduce((s, x) => s + x.silhouette, 0) / steps.length).toFixed(3)
            : 0
    };
}

export function formatClusterDiagnosticsForPrompt(diagnostics) {
    if (!diagnostics?.steps?.length) return '';
    const lines = diagnostics.steps.map(s =>
        `Step ${s.step}: K=${s.K}, ${s.nonemptyClusters} non-empty clusters, sizes=[${s.sizes.join(',')}], ` +
        `silhouette=${s.silhouette}, separation=${s.separation}`
    );
    return (
        '[Measured cluster structure in PCA space — use to verify what you see]\n' +
        `Mean silhouette: ${diagnostics.meanSilhouette}\n` +
        lines.join('\n')
    );
}

export function aggregateTrajectoryShapeStats(pcaPointsByStep, clusters) {
    const steps = clusters.map((c, t) =>
        computeStepShapeStats(pcaPointsByStep[t], c.assignments, c.centers?.length ?? stateK(c))
    );
    const score = steps.length
        ? steps.reduce((s, x) => s + x.score, 0) / steps.length
        : 0;
    return { steps, score: +score.toFixed(4) };
}

function stateK(cluster) {
    if (!cluster?.centers) return 1;
    return cluster.centers.length;
}
