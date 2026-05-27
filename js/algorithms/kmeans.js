export function kMeans(matrix, k, iters = 30, n_init = 3) {
    const [n, d] = matrix.shape;
    const data = matrix.data;
    if (k > n) k = n;

    let bestInertia = Infinity;
    let bestResult = null;

    for (let init = 0; init < n_init; init++) {
        let centers = [];
        const firstIdx = Math.floor(Math.random() * n);
        centers.push(Array.from(data.slice(firstIdx * d, firstIdx * d + d)));

        for (let c = 1; c < k; c++) {
            let distances = new Float64Array(n); let sumSqDist = 0;
            for (let i = 0; i < n; i++) {
                let minDist = Infinity;
                for (let j = 0; j < c; j++) {
                    let dist = 0;
                    for (let dim = 0; dim < d; dim++) {
                        const diff = data[i * d + dim] - centers[j][dim];
                        dist += diff * diff;
                    }
                    if (dist < minDist) minDist = dist;
                }
                distances[i] = minDist;
                sumSqDist += minDist;
            }

            let target = Math.random() * sumSqDist, cumulative = 0, selectedIdx = n - 1;
            for (let i = 0; i < n; i++) {
                cumulative += distances[i];
                if (cumulative >= target) { selectedIdx = i; break; }
            }
            centers.push(Array.from(data.slice(selectedIdx * d, selectedIdx * d + d)));
        }

        let counts = new Int32Array(k), assignments = new Int32Array(n);
        for (let iter = 0; iter < iters; iter++) {
            counts.fill(0);
            let newCenters = Array.from({ length: k }, () => new Float64Array(d));
            for (let i = 0; i < n; i++) {
                let minDist = Infinity, minIdx = 0;
                for (let c = 0; c < k; c++) {
                    let dist = 0;
                    for (let j = 0; j < d; j++) {
                        const diff = data[i * d + j] - centers[c][j];
                        dist += diff * diff;
                    }
                    if (dist < minDist) { minDist = dist; minIdx = c; }
                }
                assignments[i] = minIdx;
                counts[minIdx]++;
                for (let j = 0; j < d; j++) newCenters[minIdx][j] += data[i * d + j];
            }
            for (let c = 0; c < k; c++) {
                if (counts[c] > 0) for (let j = 0; j < d; j++) centers[c][j] = newCenters[c][j] / counts[c];
            }
        }

        let inertia = 0;
        for (let i = 0; i < n; i++) {
            let cIdx = assignments[i];
            for (let j = 0; j < d; j++) {
                const diff = data[i * d + j] - centers[cIdx][j];
                inertia += diff * diff;
            }
        }

        if (inertia < bestInertia) {
            bestInertia = inertia;
            bestResult = {
                centers,
                counts: new Int32Array(counts),
                marginals: Array.from(counts).map(c => c / n),
                assignments: new Int32Array(assignments),
                inertia
            };
        }
    }

    // Sort by y-axis if possible for consistent colors.
    let order = Array.from({ length: k }, (_, i) => i).sort((a, b) => (bestResult.centers[a][1] ?? 0) - (bestResult.centers[b][1] ?? 0));
    let orderedCenters = order.map(i => bestResult.centers[i]);
    let orderedCounts = new Int32Array(k).map((_, i) => bestResult.counts[order[i]]);
    let orderedMarginals = order.map(i => bestResult.marginals[i]);

    let orderedAssignments = new Int32Array(n);
    let revOrder = {};
    order.forEach((oldIdx, newIdx) => revOrder[oldIdx] = newIdx);
    for (let i = 0; i < n; i++) orderedAssignments[i] = revOrder[bestResult.assignments[i]];

    return {
        centers: orderedCenters,
        counts: orderedCounts,
        marginals: orderedMarginals,
        assignments: orderedAssignments,
        inertia: bestResult.inertia
    };
}

export function findElbow(inertiaCache) {
    const ks = Object.keys(inertiaCache).map(Number).sort((a, b) => a - b);
    if (ks.length < 3) return ks[0];

    const points = ks.map(k => ({ x: k, y: inertiaCache[k].inertia }));
    const first = points[0];
    const last = points[points.length - 1];

    const m = (last.y - first.y) / (last.x - first.x);
    const c = first.y - m * first.x;

    let maxDist = -1;
    let elbowK = ks[0];
    points.forEach(p => {
        const dist = Math.abs(m * p.x - p.y + c) / Math.sqrt(m * m + 1);
        if (dist > maxDist) {
            maxDist = dist;
            elbowK = p.x;
        }
    });
    return elbowK;
}

