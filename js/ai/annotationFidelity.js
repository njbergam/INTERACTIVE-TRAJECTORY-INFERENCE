/**
 * Score how well k-means assignments honor user-drawn cluster circles on PCA / kernel PCA.
 */

function pointInEllipse(nx, ny, e) {
    const dx = (nx - e.cx) / Math.max(e.rx, 1e-6);
    const dy = (ny - e.cy) / Math.max(e.ry, 1e-6);
    return dx * dx + dy * dy <= 1;
}

function boundsForPoints(points) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
        minX = Math.min(minX, p[0]);
        maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]);
        maxY = Math.max(maxY, p[1]);
    }
    const range = Math.max(maxX - minX, maxY - minY) || 1;
    return { minX, minY, range };
}

function globalBounds(pcaPointsByStep) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const proj of pcaPointsByStep) {
        if (!proj) continue;
        for (const p of proj) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }
    }
    const range = Math.max(maxX - minX, maxY - minY) || 1;
    return { minX, minY, range };
}

function toNorm(p, bounds) {
    return [(p[0] - bounds.minX) / bounds.range, (p[1] - bounds.minY) / bounds.range];
}

function dominantCluster(assignments, indices) {
    const counts = new Map();
    for (const i of indices) {
        const c = assignments[i];
        if (c == null || c < 0) continue;
        counts.set(c, (counts.get(c) || 0) + 1);
    }
    let best = -1;
    let bestN = 0;
    for (const [c, n] of counts) {
        if (n > bestN) {
            bestN = n;
            best = c;
        }
    }
    return { cluster: best, count: bestN, total: indices.length };
}

function evaluateCirclesOnPlot({
    plotId,
    ellipses,
    normCoords,
    assignments
}) {
    const circles = [];
    const dominants = [];

    for (let ci = 0; ci < ellipses.length; ci++) {
        const e = ellipses[ci];
        const inside = [];
        for (let i = 0; i < normCoords.length; i++) {
            const [nx, ny] = normCoords[i];
            if (pointInEllipse(nx, ny, e)) inside.push(i);
        }
        if (inside.length < 4) {
            circles.push({
                plotId,
                circleIndex: ci + 1,
                nPoints: inside.length,
                purity: 0,
                dominantCluster: -1,
                skipped: true,
                reason: 'too few points inside circle'
            });
            continue;
        }
        const { cluster, count, total } = dominantCluster(assignments, inside);
        const purity = total > 0 ? count / total : 0;
        circles.push({
            plotId,
            circleIndex: ci + 1,
            nPoints: total,
            purity: +purity.toFixed(3),
            dominantCluster: cluster,
            skipped: false
        });
        if (cluster >= 0) dominants.push(cluster);
    }

    let collisionPairs = 0;
    for (let i = 0; i < dominants.length; i++) {
        for (let j = i + 1; j < dominants.length; j++) {
            if (dominants[i] === dominants[j]) collisionPairs++;
        }
    }
    const nPairs = dominants.length > 1 ? (dominants.length * (dominants.length - 1)) / 2 : 0;
    const distinctness = nPairs > 0 ? 1 - collisionPairs / nPairs : 1;

    const active = circles.filter(c => !c.skipped);
    const purityMin = active.length ? Math.min(...active.map(c => c.purity)) : 1;
    const purityMean = active.length
        ? active.reduce((s, c) => s + c.purity, 0) / active.length
        : 1;

    const plotScore = active.length
        ? purityMean * 0.55 + purityMin * 0.25 + distinctness * 0.2
        : 1;

    return { circles, distinctness, purityMin, purityMean, plotScore, collisionPairs };
}

function buildNormCoordsForPca(stepIndex, pcaPointsByStep, independentScale) {
    const points = pcaPointsByStep[stepIndex];
    if (!points?.length) return { normCoords: [], pointIndices: [] };
    const bounds = independentScale
        ? boundsForPoints(points)
        : globalBounds(pcaPointsByStep);
    const normCoords = [];
    const pointIndices = [];
    for (let i = 0; i < points.length; i++) {
        normCoords.push(toNorm(points[i], bounds));
        pointIndices.push(i);
    }
    return { normCoords, pointIndices };
}

function buildNormCoordsForKpca(stepIndex, kernelPcaPointsByStep, pcaPointsByStep, independentScale) {
    const kp = kernelPcaPointsByStep?.[stepIndex];
    if (!kp) return { normCoords: [], pointIndices: [] };
    const coords = Array.isArray(kp.coords) ? kp.coords : kp;
    const idx = Array.isArray(kp.sampledIndices) ? kp.sampledIndices : coords.map((_, i) => i);
    if (!coords?.length) return { normCoords: [], pointIndices: [] };

    let bounds;
    if (independentScale) {
        bounds = boundsForPoints(coords);
    } else {
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const row of kernelPcaPointsByStep) {
            if (!row) continue;
            const pts = Array.isArray(row.coords) ? row.coords : row;
            if (!pts) continue;
            for (const p of pts) {
                minX = Math.min(minX, p[0]);
                maxX = Math.max(maxX, p[0]);
                minY = Math.min(minY, p[1]);
                maxY = Math.max(maxY, p[1]);
            }
        }
        bounds = {
            minX,
            minY,
            range: Math.max(maxX - minX, maxY - minY) || 1
        };
    }

    const normCoords = [];
    const pointIndices = [];
    for (let localI = 0; localI < coords.length; localI++) {
        normCoords.push(toNorm(coords[localI], bounds));
        pointIndices.push(idx[localI]);
    }
    return { normCoords, pointIndices };
}

/** Min K per step from circle counts (0-based step index). */
export function minKFromAnnotationSnapshot(snapshot, T) {
    const minK = Array(T).fill(1);
    if (!snapshot?.byPlotId) return minK;
    for (const [plotId, ellipses] of Object.entries(snapshot.byPlotId)) {
        const m = plotId.match(/^(?:pca|kpca):(\d+)$/);
        if (!m || !ellipses?.length) continue;
        const t = parseInt(m[1], 10) - 1;
        if (t < 0 || t >= T) continue;
        minK[t] = Math.max(minK[t], ellipses.length);
    }
    return minK;
}

/**
 * @returns {{ overall: number, totalCircles: number, plots: object[], failures: object[] }}
 */
export function evaluateAnnotationFidelity({
    annotationSnapshot,
    pcaPointsByStep,
    kernelPcaPointsByStep = null,
    clusters,
    independentScale = false
}) {
    if (!annotationSnapshot?.byPlotId || !clusters?.length) {
        return { overall: 1, totalCircles: 0, plots: [], failures: [], satisfied: true };
    }

    const plots = [];
    const failures = [];
    let totalCircles = 0;
    let scoreSum = 0;
    let scoreWeight = 0;

    for (const [plotId, ellipses] of Object.entries(annotationSnapshot.byPlotId)) {
        if (!ellipses?.length) continue;
        const m = plotId.match(/^(pca|kpca):(\d+)$/);
        if (!m) continue;
        const kind = m[1];
        const step = parseInt(m[2], 10);
        const t = step - 1;
        const cluster = clusters[t];
        if (!cluster?.assignments) continue;

        totalCircles += ellipses.length;

        let normCoords;
        let pointIndices;
        if (kind === 'pca') {
            ({ normCoords, pointIndices } = buildNormCoordsForPca(t, pcaPointsByStep, independentScale));
        } else {
            ({ normCoords, pointIndices } = buildNormCoordsForKpca(
                t,
                kernelPcaPointsByStep,
                pcaPointsByStep,
                independentScale
            ));
        }

        const assignments = pointIndices.map(i => cluster.assignments[i]);
        const result = evaluateCirclesOnPlot({
            plotId,
            ellipses,
            normCoords,
            assignments
        });
        plots.push({ step, kind, ...result });
        scoreSum += result.plotScore * ellipses.length;
        scoreWeight += ellipses.length;

        for (const c of result.circles) {
            if (c.skipped) {
                failures.push({
                    plotId,
                    step,
                    circle: c.circleIndex,
                    issue: c.reason
                });
                continue;
            }
            if (c.purity < 0.82) {
                failures.push({
                    plotId,
                    step,
                    circle: c.circleIndex,
                    issue: `only ${Math.round(c.purity * 100)}% of points share one color (need ~one cluster per circle)`,
                    purity: c.purity,
                    dominantCluster: c.dominantCluster
                });
            }
        }
        if (result.collisionPairs > 0) {
            failures.push({
                plotId,
                step,
                issue: `${result.collisionPairs} pair(s) of circles map to the same cluster color — circles should be different colors`
            });
        }
    }

    const overall = scoreWeight > 0 ? scoreSum / scoreWeight : 1;
    const satisfied = overall >= 0.88 && failures.length === 0;

    return {
        overall: +overall.toFixed(3),
        totalCircles,
        plots,
        failures,
        satisfied
    };
}

/** Large penalty on explore combined score (lower is better). */
export function annotationFidelityPenalty(fidelity, { strict = true, emphasis = false } = {}) {
    if (!fidelity?.totalCircles) return 0;
    const gap = 1 - Math.max(0, Math.min(1, fidelity.overall));
    const failCount = fidelity.failures?.length ?? 0;
    const scale = emphasis ? 1.55 : 1;
    let pen = 620 * gap * fidelity.totalCircles * scale;
    if (strict) pen += 140 * failCount * scale;
    if (!fidelity.satisfied && fidelity.overall < 0.75) pen += 280 * scale;
    if (emphasis && !fidelity.satisfied) pen += 220;
    return pen;
}

export function formatAnnotationFulfillmentReport(fidelity, { threshold = 0.88 } = {}) {
    if (!fidelity?.totalCircles) return '';

    const pct = Math.round(fidelity.overall * 100);
    const lines = [];

    if (fidelity.satisfied && fidelity.overall >= threshold) {
        lines.push(
            `Cluster sketches: satisfied (${pct}% match). Each circled region is mostly one color and circles use different colors.`
        );
        return lines.join('\n');
    }

    lines.push(
        `Cluster sketches: NOT fully satisfied (${pct}% match). I prioritized your yellow circles but the result still misses the goal.`
    );

    const byPlot = new Map();
    for (const f of fidelity.failures || []) {
        const key = f.plotId || `step ${f.step}`;
        if (!byPlot.has(key)) byPlot.set(key, []);
        byPlot.get(key).push(f);
    }
    for (const [plotId, items] of byPlot) {
        for (const f of items) {
            const circ = f.circle != null ? ` circle ${f.circle}` : '';
            lines.push(`  · ${plotId}${circ}: ${f.issue}`);
        }
    }

    for (const p of fidelity.plots || []) {
        const label = `Step ${p.step} ${p.kind === 'kpca' ? 'kernel PCA' : 'PCA'}`;
        if (p.purityMin != null && p.purityMin < 0.82) {
            lines.push(
                `  · ${label}: weakest region is ${Math.round(p.purityMin * 100)}% color-pure inside the circle.`
            );
        }
    }

    lines.push('Try raising K on those steps, redrawing circles, or running explore again with a short note like “match my circles exactly”.');
    return lines.join('\n');
}
