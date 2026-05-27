import { COLORS } from '../constants.js';

function hexToRgb(hex) {
    const h = (hex || '').replace('#', '');
    if (h.length !== 6) return { r: 148, g: 163, b: 184 }; // slate-400 fallback
    return {
        r: parseInt(h.slice(0, 2), 16),
        g: parseInt(h.slice(2, 4), 16),
        b: parseInt(h.slice(4, 6), 16)
    };
}

function clamp01(x) {
    return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
}

function stableArgsortBy(arr, scoreFn) {
    return arr
        .map((v, idx) => ({ v, idx, s: scoreFn(v, idx) }))
        .sort((a, b) => (a.s - b.s) || (a.idx - b.idx))
        .map(x => x.v);
}

/**
 * Reduce crossings by reordering clusters vertically per step.
 * Uses a few barycenter sweeps (Sugiyama-style) based on coupling mass.
 *
 * Returns orderByStep[t] = array of cluster indices in top->bottom order.
 */
function computeOrderByStep(clusters, couplings, { iters = 6 } = {}) {
    const T = clusters.length;
    const orderByStep = Array.from({ length: T }, (_, t) =>
        Array.from({ length: clusters[t].centers.length }, (_, i) => i)
    );
    if (T <= 1) return orderByStep;

    const posFromOrder = (order) => {
        const pos = new Array(order.length);
        for (let r = 0; r < order.length; r++) pos[order[r]] = r;
        return pos;
    };

    const updateByIncoming = (t) => {
        // incoming couplings from t-1 -> t
        const P = couplings[t - 1];
        const prevPos = posFromOrder(orderByStep[t - 1]);
        const kPrev = orderByStep[t - 1].length;
        const kCur = orderByStep[t].length;
        const bc = new Array(kCur).fill(0);
        const w = new Array(kCur).fill(0);
        for (let i = 0; i < kPrev; i++) {
            const pi = prevPos[i];
            const row = P?.[i];
            if (!row) continue;
            for (let j = 0; j < kCur; j++) {
                const m = row[j] || 0;
                if (m <= 0) continue;
                bc[j] += m * pi;
                w[j] += m;
            }
        }
        const cur = orderByStep[t].slice();
        orderByStep[t] = stableArgsortBy(cur, (j) => (w[j] > 0 ? (bc[j] / w[j]) : j + 0.01 * j));
    };

    const updateByOutgoing = (t) => {
        // outgoing couplings from t -> t+1
        const P = couplings[t];
        const nextPos = posFromOrder(orderByStep[t + 1]);
        const kCur = orderByStep[t].length;
        const kNext = orderByStep[t + 1].length;
        const bc = new Array(kCur).fill(0);
        const w = new Array(kCur).fill(0);
        for (let i = 0; i < kCur; i++) {
            const row = P?.[i];
            if (!row) continue;
            for (let j = 0; j < kNext; j++) {
                const m = row[j] || 0;
                if (m <= 0) continue;
                bc[i] += m * nextPos[j];
                w[i] += m;
            }
        }
        const cur = orderByStep[t].slice();
        orderByStep[t] = stableArgsortBy(cur, (i) => (w[i] > 0 ? (bc[i] / w[i]) : i + 0.01 * i));
    };

    for (let iter = 0; iter < iters; iter++) {
        // Left -> right: order by incoming barycenter
        for (let t = 1; t < T; t++) updateByIncoming(t);
        // Right -> left: order by outgoing barycenter
        for (let t = T - 2; t >= 0; t--) updateByOutgoing(t);
    }
    return orderByStep;
}

/**
 * Pick per-step color indices to encourage continuity.
 * For step t+1, assign each cluster the color of the source cluster that transports most mass into it.
 * Collisions are resolved greedily (largest mass first), then remaining clusters get unused colors.
 */
function computeColorIndexByStep(clusters, couplings) {
    const T = clusters.length;
    const colorIndexByStep = Array.from({ length: T }, () => []);
    if (!T) return colorIndexByStep;

    // Step 0: stable base order.
    const k0 = clusters[0]?.centers?.length ?? 0;
    colorIndexByStep[0] = Array.from({ length: k0 }, (_, i) => i);

    for (let t = 0; t < T - 1; t++) {
        const k1 = clusters[t].centers.length;
        const k2 = clusters[t + 1].centers.length;
        const P = couplings[t];
        const prev = colorIndexByStep[t] || Array.from({ length: k1 }, (_, i) => i);

        const best = Array.from({ length: k2 }, (_, j) => {
            let bi = 0;
            let bm = -Infinity;
            for (let i = 0; i < k1; i++) {
                const m = P?.[i]?.[j] ?? 0;
                if (m > bm) {
                    bm = m;
                    bi = i;
                }
            }
            return { j, i: bi, mass: bm };
        }).sort((a, b) => (b.mass || 0) - (a.mass || 0));

        const used = new Set();
        const next = new Array(k2).fill(null);

        for (const { j, i } of best) {
            const c = prev[i] ?? i;
            if (!used.has(c)) {
                next[j] = c;
                used.add(c);
            }
        }

        let cursor = 0;
        for (let j = 0; j < k2; j++) {
            if (next[j] !== null) continue;
            while (used.has(cursor)) cursor++;
            next[j] = cursor;
            used.add(cursor);
            cursor++;
        }

        colorIndexByStep[t + 1] = next;
    }
    return colorIndexByStep;
}

export function drawGraph(elements, clusters, couplings, drawnEdgePathsOut) {
    const { canvas, ctx } = elements;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawnEdgePathsOut.length = 0;

    const T = clusters.length;
    const paddingX = 84;
    const paddingY = 64;
    const stepX = (canvas.width - 2 * paddingX) / (T > 1 ? T - 1 : 1);
    const orderByStep = computeOrderByStep(clusters, couplings, { iters: 6 });
    const rankByStep = orderByStep.map(order => {
        const rank = new Array(order.length);
        for (let r = 0; r < order.length; r++) rank[order[r]] = r;
        return rank;
    });
    const getY = (t, clusterIdx, k) => {
        const r = rankByStep[t]?.[clusterIdx] ?? clusterIdx;
        return paddingY + (k > 1 ? r * ((canvas.height - 2 * paddingY) / (k - 1)) : (canvas.height / 2 - paddingY));
    };

    const colorIndexByStep = computeColorIndexByStep(clusters, couplings);

    // Sleeker rendering defaults.
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = 'source-over';

    for (let t = 0; t < T - 1; t++) {
        const P = couplings[t];
        const k1 = clusters[t].centers.length;
        const k2 = clusters[t + 1].centers.length;

        let maxP = 0;
        for (let i = 0; i < k1; i++) for (let j = 0; j < k2; j++) if (P[i][j] > maxP) maxP = P[i][j];

        // Draw edges behind nodes.
        for (let i = 0; i < k1; i++) {
            for (let j = 0; j < k2; j++) {
                const mass = P[i][j];
                if (mass < 1e-6) continue;

                const x1 = paddingX + t * stepX, y1 = getY(t, i, k1);
                const x2 = paddingX + (t + 1) * stepX, y2 = getY(t + 1, j, k2);

                const intensity = clamp01(mass / (maxP || mass || 1));
                const path = new Path2D();
                path.moveTo(x1, y1);
                path.bezierCurveTo(x1 + stepX / 2, y1, x2 - stepX / 2, y2, x2, y2);

                const srcColorIdx = colorIndexByStep[t]?.[i] ?? i;
                const dstColorIdx = colorIndexByStep[t + 1]?.[j] ?? j;
                const srcRgb = hexToRgb(COLORS[srcColorIdx % COLORS.length]);
                const dstRgb = hexToRgb(COLORS[dstColorIdx % COLORS.length]);

                // Gradient emphasizes continuity across steps (source -> destination).
                const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                const a0 = 0.10 + 0.55 * intensity;
                const a1 = 0.08 + 0.48 * intensity;
                grad.addColorStop(0, `rgba(${srcRgb.r}, ${srcRgb.g}, ${srcRgb.b}, ${a0})`);
                grad.addColorStop(1, `rgba(${dstRgb.r}, ${dstRgb.g}, ${dstRgb.b}, ${a1})`);
                ctx.strokeStyle = grad;

                // Smoother width scaling than linear.
                const lineWidth = Math.max(0.9, 1.2 + Math.pow(intensity, 0.65) * 14);
                ctx.lineWidth = lineWidth;
                ctx.shadowColor = 'rgba(15, 23, 42, 0.35)';
                ctx.shadowBlur = 8;
                ctx.stroke(path);
                ctx.shadowBlur = 0;

                drawnEdgePathsOut.push({
                    path,
                    mass,
                    renderedWidth: lineWidth,
                    stepIdx: t,
                    sourceIdx: i
                });
            }
        }
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "600 12.5px system-ui, -apple-system, sans-serif";

    for (let t = 0; t < T; t++) {
        const k = clusters[t].centers.length;
        const maxCount = Math.max(...clusters[t].counts);

        for (let i = 0; i < k; i++) {
            const count = clusters[t].counts[i];
            const x = paddingX + t * stepX, y = getY(t, i, k);
            const radius = 15 + Math.sqrt(count / (maxCount || 1)) * 20;

            ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI);
            const ci = colorIndexByStep[t]?.[i] ?? i;
            const rgb = hexToRgb(COLORS[ci % COLORS.length]);

            // Subtle radial fill for a more "designed" look.
            const rg = ctx.createRadialGradient(x - radius * 0.25, y - radius * 0.25, radius * 0.1, x, y, radius);
            rg.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.98)`);
            rg.addColorStop(1, `rgba(${Math.round(rgb.r * 0.78)}, ${Math.round(rgb.g * 0.78)}, ${Math.round(rgb.b * 0.78)}, 0.98)`);
            ctx.fillStyle = rg;
            ctx.shadowColor = 'rgba(0,0,0,0.35)';
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.strokeStyle = 'rgba(15, 23, 42, 0.85)'; // slate-900-ish
            ctx.lineWidth = 3.5;
            ctx.stroke();

            ctx.fillStyle = 'rgba(255,255,255,0.96)';
            ctx.fillText(count.toString(), x, y);
        }

        ctx.fillStyle = 'rgba(148, 163, 184, 0.95)';
        ctx.font = "600 12px system-ui, -apple-system, sans-serif";
        ctx.fillText(`State ${t + 1}`, paddingX + t * stepX, 22);
        ctx.font = "600 12.5px system-ui, -apple-system, sans-serif";
    }

    ctx.restore();
}

