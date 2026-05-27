import { COLORS } from '../constants.js';

function projectPoints(points2d, width, height, pad = 14) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points2d) {
        minX = Math.min(minX, p[0]);
        maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]);
        maxY = Math.max(maxY, p[1]);
    }
    const range = Math.max(maxX - minX, maxY - minY) || 1;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;
    const toScreen = (p) => [
        pad + ((p[0] - minX) / range) * innerW,
        pad + ((p[1] - minY) / range) * innerH
    ];
    return { toScreen, minX, minY, range };
}

function convexHullMonotone(screenPts) {
    if (screenPts.length < 3) return screenPts.slice();
    const pts = screenPts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

    const lower = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
            lower.pop();
        }
        lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
            upper.pop();
        }
        upper.push(p);
    }
    upper.pop();
    lower.pop();
    return lower.concat(upper);
}

/**
 * PCA plot tuned for vision models: hulls, larger dots, numbered centroids, legend.
 */
export function drawPcaClusterVisionPlot(ctx, width, height, points2d, cluster) {
    const k = cluster?.centers?.length ?? 0;
    const assignments = cluster?.assignments;
    const n = assignments?.length ?? 0;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    if (!points2d || n === 0 || k < 1) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data', width / 2, height / 2);
        return;
    }

    const { toScreen } = projectPoints(points2d, width, height);
    const byCluster = Array.from({ length: k }, () => []);
    const centroids = Array(k).fill(null).map(() => [0, 0]);
    const counts = new Int32Array(k);

    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        if (c < 0 || c >= k) continue;
        const scr = toScreen(points2d[i]);
        byCluster[c].push(scr);
        centroids[c][0] += scr[0];
        centroids[c][1] += scr[1];
        counts[c]++;
    }

    for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
            centroids[c][0] /= counts[c];
            centroids[c][1] /= counts[c];
        }
    }

    // Convex hulls (filled, behind points)
    for (let c = 0; c < k; c++) {
        const pts = byCluster[c];
        if (pts.length < 3) continue;
        const hull = convexHullMonotone(pts);
        if (hull.length < 3) continue;
        const color = COLORS[c % COLORS.length];
        ctx.beginPath();
        ctx.moveTo(hull[0][0], hull[0][1]);
        for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i][0], hull[i][1]);
        ctx.closePath();
        ctx.fillStyle = color + '33';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Points
    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        if (c < 0 || c >= k) continue;
        const [x, y] = toScreen(points2d[i]);
        ctx.beginPath();
        ctx.arc(x, y, 2.8, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[c % COLORS.length];
        ctx.fill();
    }

    // Centroids + numeric labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let c = 0; c < k; c++) {
        if (counts[c] === 0) continue;
        const [x, y] = centroids[c];
        const color = COLORS[c % COLORS.length];
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.fillStyle = '#f8fafc';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.font = 'bold 10px system-ui, sans-serif';
        ctx.fillStyle = '#111827';
        ctx.fillText(String(c), x, y + 0.5);
    }

    // Mini legend (top-left below title margin)
    const legendY = height - 10;
    ctx.font = '9px system-ui, sans-serif';
    ctx.textAlign = 'left';
    let lx = 8;
    for (let c = 0; c < k && c < 8; c++) {
        if (counts[c] === 0) continue;
        ctx.fillStyle = COLORS[c % COLORS.length];
        ctx.fillRect(lx, legendY - 7, 8, 8);
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(`${c}:${counts[c]}`, lx + 10, legendY - 2);
        lx += 38;
    }
}
