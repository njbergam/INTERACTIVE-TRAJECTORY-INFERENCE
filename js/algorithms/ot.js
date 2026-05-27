function wassersteinCostMatrix(c1, c2) {
    const k1 = c1.centers.length, k2 = c2.centers.length;
    const d = c1.centers[0]?.length ?? 0;
    if (d !== (c2.centers[0]?.length ?? -1)) {
        throw new Error('Wasserstein OT requires matching center dimensions.');
    }
    const C = Array.from({ length: k1 }, () => new Float64Array(k2));
    for (let i = 0; i < k1; i++) {
        for (let j = 0; j < k2; j++) {
            let dist = 0;
            for (let dim = 0; dim < d; dim++) {
                const diff = c1.centers[i][dim] - c2.centers[j][dim];
                dist += diff * diff;
            }
            C[i][j] = dist;
        }
    }
    return C;
}

function exactOTFromCost(C, a, b) {
    const k1 = C.length, k2 = C[0].length;
    const P = Array.from({ length: k1 }, () => new Float64Array(k2));
    const aRem = Array.from(a), bRem = Array.from(b);
    let totalCost = 0;

    const pairs = [];
    for (let i = 0; i < k1; i++) for (let j = 0; j < k2; j++) pairs.push({ i, j, cost: C[i][j] });
    pairs.sort((x, y) => x.cost - y.cost);

    for (const { i, j, cost } of pairs) {
        const amount = Math.min(aRem[i], bRem[j]);
        if (amount > 1e-10) {
            P[i][j] = amount;
            aRem[i] -= amount;
            bRem[j] -= amount;
            totalCost += amount * cost;
        }
    }
    return { P, cost: totalCost };
}

function sinkhornFromCost(C, a, b, epsilon, iters = 50) {
    const k1 = C.length, k2 = C[0].length;
    let maxC = 0;
    for (let i = 0; i < k1; i++) for (let j = 0; j < k2; j++) if (C[i][j] > maxC) maxC = C[i][j];
    const scale = maxC > 0 ? maxC : 1;
    const K = Array.from({ length: k1 }, (_, i) =>
        new Float64Array(k2).map((_, j) => Math.exp(-(C[i][j] / scale) / epsilon))
    );

    const u = new Float64Array(k1), v = new Float64Array(k2);
    for (let iter = 0; iter < iters; iter++) {
        for (let i = 0; i < k1; i++) {
            u[i] = a[i] / (K[i].reduce((sum, val, j) => sum + val * v[j], 0) + 1e-15);
        }
        for (let j = 0; j < k2; j++) {
            v[j] = b[j] / (K.reduce((sum, row, i) => sum + row[j] * u[i], 0) + 1e-15);
        }
    }

    return Array.from({ length: k1 }, (_, i) => new Float64Array(k2).map((_, j) => u[i] * K[i][j] * v[j]));
}

export function exactOT(c1, c2) {
    const a = Array.from(c1.marginals), b = Array.from(c2.marginals);
    return exactOTFromCost(wassersteinCostMatrix(c1, c2), a, b);
}

export function sinkhorn(c1, c2, epsilon, iters = 50) {
    const a = Array.from(c1.marginals), b = Array.from(c2.marginals);
    return sinkhornFromCost(wassersteinCostMatrix(c1, c2), a, b, epsilon, iters);
}

// Point-level entropic Sinkhorn OT with uniform point masses.
// Returns a full coupling matrix `P` of shape [nA][nB] where each row/col
// sums to 1/nA and 1/nB respectively.
export function sinkhornPointwise(
    dataA,
    nA,
    dA,
    dataB,
    nB,
    dB,
    epsilon,
    iters = 50,
    { maxEntries = 120000 } = {}
) {
    if (epsilon === 0) throw new Error('Point-level OT requires epsilon > 0 (entropic Sinkhorn).');
    if (dA !== dB) throw new Error('Point-level OT requires matching feature dimensions.');
    const entries = nA * nB;
    if (entries > maxEntries) {
        throw new Error(`Point-level OT too large for in-browser compute (${entries} entries).`);
    }

    const C = Array.from({ length: nA }, () => new Float64Array(nB));
    for (let i = 0; i < nA; i++) {
        for (let j = 0; j < nB; j++) {
            let dist = 0;
            const aOff = i * dA;
            const bOff = j * dB;
            for (let dim = 0; dim < dA; dim++) {
                const diff = dataA[aOff + dim] - dataB[bOff + dim];
                dist += diff * diff;
            }
            C[i][j] = dist;
        }
    }

    const a = Array.from({ length: nA }, () => 1 / nA);
    const b = Array.from({ length: nB }, () => 1 / nB);
    const P = sinkhornFromCost(C, a, b, epsilon, iters);
    return { P, entries };
}

// Point-level OT "exact" (greedy) for epsilon=0 (cost matrix + greedy plan).
// Note: This matches the existing `exactOT` behavior in this project (not a full LP solver).
export function exactOTPointwise(
    dataA,
    nA,
    dA,
    dataB,
    nB,
    dB,
    { maxEntries = 60000 } = {}
) {
    if (dA !== dB) throw new Error('Point-level OT requires matching feature dimensions.');
    const entries = nA * nB;
    if (entries > maxEntries) {
        throw new Error(`Point-level exact OT too large for in-browser compute (${entries} entries).`);
    }

    const C = Array.from({ length: nA }, () => new Float64Array(nB));
    for (let i = 0; i < nA; i++) {
        for (let j = 0; j < nB; j++) {
            let dist = 0;
            const aOff = i * dA;
            const bOff = j * dB;
            for (let dim = 0; dim < dA; dim++) {
                const diff = dataA[aOff + dim] - dataB[bOff + dim];
                dist += diff * diff;
            }
            C[i][j] = dist;
        }
    }

    const a = Array.from({ length: nA }, () => 1 / nA);
    const b = Array.from({ length: nB }, () => 1 / nB);
    const { P, cost } = exactOTFromCost(C, a, b);
    return { P, cost, entries };
}

function clusterDistMatrix(centers) {
    const k = centers.length;
    const C = Array.from({ length: k }, () => new Float64Array(k));
    for (let i = 0; i < k; i++) {
        for (let j = i; j < k; j++) {
            let dist = 0;
            const ci = centers[i], cj = centers[j];
            for (let dim = 0; dim < ci.length; dim++) {
                const diff = ci[dim] - cj[dim];
                dist += diff * diff;
            }
            C[i][j] = dist;
            C[j][i] = dist;
        }
    }
    return C;
}

function outerProductPlan(p, q) {
    return Array.from({ length: p.length }, (_, i) =>
        new Float64Array(q.length).map((_, j) => p[i] * q[j])
    );
}

function multiplyC1TC2(C1, T, C2) {
    const k1 = C1.length, k2 = C2.length;
    const out = Array.from({ length: k1 }, () => new Float64Array(k2));
    for (let i = 0; i < k1; i++) {
        for (let j = 0; j < k2; j++) {
            let sum = 0;
            for (let ip = 0; ip < k1; ip++) {
                for (let jp = 0; jp < k2; jp++) {
                    sum += C1[i][ip] * T[ip][jp] * C2[jp][j];
                }
            }
            out[i][j] = sum;
        }
    }
    return out;
}

function gromovLinearizedCost(C1, C2, p, q, T) {
    const k1 = C1.length, k2 = C2.length;
    const constC1 = new Float64Array(k1);
    const constC2 = new Float64Array(k2);
    for (let i = 0; i < k1; i++) {
        let s = 0;
        for (let ip = 0; ip < k1; ip++) s += C1[i][ip] * C1[i][ip] * p[ip];
        constC1[i] = s;
    }
    for (let j = 0; j < k2; j++) {
        let s = 0;
        for (let jp = 0; jp < k2; jp++) s += C2[j][jp] * C2[j][jp] * q[jp];
        constC2[j] = s;
    }
    const cross = multiplyC1TC2(C1, T, C2);
    return Array.from({ length: k1 }, (_, i) =>
        new Float64Array(k2).map((_, j) => constC1[i] + constC2[j] - 2 * cross[i][j])
    );
}

function gwObjective(C1, C2, T) {
    const k1 = C1.length, k2 = C2.length;
    let cost = 0;
    for (let i = 0; i < k1; i++) {
        for (let j = 0; j < k2; j++) {
            for (let ip = 0; ip < k1; ip++) {
                for (let jp = 0; jp < k2; jp++) {
                    const d = C1[i][ip] - C2[j][jp];
                    cost += d * d * T[i][j] * T[ip][jp];
                }
            }
        }
    }
    return cost;
}

function planFrobeniusNormDiff(T, Tnew) {
    let diff = 0;
    for (let i = 0; i < T.length; i++) {
        for (let j = 0; j < T[i].length; j++) {
            const d = T[i][j] - Tnew[i][j];
            diff += d * d;
        }
    }
    return Math.sqrt(diff);
}

export function gromovWasserstein(c1, c2, epsilon = 0, maxIter = 80, tol = 1e-7) {
    const C1 = clusterDistMatrix(c1.centers);
    const C2 = clusterDistMatrix(c2.centers);
    const p = Array.from(c1.marginals);
    const q = Array.from(c2.marginals);

    let T = outerProductPlan(p, q);
    for (let iter = 0; iter < maxIter; iter++) {
        const M = gromovLinearizedCost(C1, C2, p, q, T);
        const Tnew = epsilon === 0 ? exactOTFromCost(M, p, q).P : sinkhornFromCost(M, p, q, epsilon);
        const change = planFrobeniusNormDiff(T, Tnew);
        T = Tnew;
        if (change < tol) break;
    }
    return { P: T, cost: gwObjective(C1, C2, T), method: 'gw' };
}

