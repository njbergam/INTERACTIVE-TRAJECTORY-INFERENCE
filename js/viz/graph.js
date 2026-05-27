import { COLORS } from '../constants.js';

export function drawGraph(elements, clusters, couplings, drawnEdgePathsOut) {
    const { canvas, ctx } = elements;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawnEdgePathsOut.length = 0;

    const T = clusters.length;
    const paddingX = 80;
    const paddingY = 60;
    const stepX = (canvas.width - 2 * paddingX) / (T > 1 ? T - 1 : 1);
    const getY = (t, i, k) => paddingY + (k > 1 ? i * ((canvas.height - 2 * paddingY) / (k - 1)) : (canvas.height / 2 - paddingY));

    for (let t = 0; t < T - 1; t++) {
        const P = couplings[t];
        const k1 = clusters[t].centers.length;
        const k2 = clusters[t + 1].centers.length;

        let maxP = 0;
        for (let i = 0; i < k1; i++) for (let j = 0; j < k2; j++) if (P[i][j] > maxP) maxP = P[i][j];

        for (let i = 0; i < k1; i++) {
            for (let j = 0; j < k2; j++) {
                const mass = P[i][j];
                if (mass < 1e-6) continue;

                const x1 = paddingX + t * stepX, y1 = getY(t, i, k1);
                const x2 = paddingX + (t + 1) * stepX, y2 = getY(t + 1, j, k2);

                const intensity = mass / (maxP || mass || 1);
                const path = new Path2D();
                path.moveTo(x1, y1);
                path.bezierCurveTo(x1 + stepX / 2, y1, x2 - stepX / 2, y2, x2, y2);

                const hex = COLORS[i % COLORS.length];
                const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);

                ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${intensity * 0.8})`;
                const lineWidth = Math.max(1, intensity * 25);
                ctx.lineWidth = lineWidth;
                ctx.stroke(path);

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
    ctx.font = "bold 13px sans-serif";

    for (let t = 0; t < T; t++) {
        const k = clusters[t].centers.length;
        const maxCount = Math.max(...clusters[t].counts);

        for (let i = 0; i < k; i++) {
            const count = clusters[t].counts[i];
            const x = paddingX + t * stepX, y = getY(t, i, k);
            const radius = 15 + Math.sqrt(count / (maxCount || 1)) * 20;

            ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fill();
            ctx.strokeStyle = '#111827'; ctx.lineWidth = 4; ctx.stroke();
            ctx.fillStyle = '#ffffff'; ctx.fillText(count.toString(), x, y);
        }

        ctx.fillStyle = '#9ca3af';
        ctx.fillText(`Abstract State ${t + 1}`, paddingX + t * stepX, 20);
    }
}

