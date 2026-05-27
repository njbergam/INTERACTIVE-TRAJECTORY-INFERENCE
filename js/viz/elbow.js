export function drawLocalElbowPlot(canvas, cacheData, currentK) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const knownKs = Object.keys(cacheData).map(Number).sort((a, b) => a - b);
    if (knownKs.length === 0) return;

    const padX = 20, padY = 20;
    const w = canvas.width - padX * 2;
    const h = canvas.height - padY * 2;

    const maxKSlider = 15, minKSlider = 1;
    const maxInertia = Math.max(...knownKs.map(k => cacheData[k].inertia));

    const getX = k => padX + ((k - minKSlider) / (maxKSlider - minKSlider)) * w;
    const getY = inv => padY + h - ((inv / (maxInertia || 1)) * h);

    ctx.strokeStyle = '#4b5563'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(padX, canvas.height - padY); ctx.lineTo(canvas.width - padX, canvas.height - padY); ctx.stroke();

    ctx.beginPath(); ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 3;
    knownKs.forEach((k, i) => {
        const x = getX(k), y = getY(cacheData[k].inertia);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    knownKs.forEach(k => {
        const x = getX(k), y = getY(cacheData[k].inertia);
        const isActive = k === currentK;
        ctx.beginPath(); ctx.arc(x, y, isActive ? 8 : 4, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? '#3b82f6' : '#d1d5db';
        ctx.fill();
    });
}

export function drawGlobalElbowPlot(
    target,
    rawSequences,
    globalHistory,
    currentSumK,
    currentSumInertia,
    currentConfigStr,
    drawnGlobalPointsOut,
    { mode = 'both' } = {}
) {
    const cvs = target.canvas;
    const ctx = target.ctx;
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    drawnGlobalPointsOut.length = 0;

    const showInertia = mode === 'both' || mode === 'inertia';
    const showTruePi = mode === 'both' || mode === 'truePi';

    const bestInertiaByX = {};
    const bestTruePiByX = {};

    for (let key in globalHistory) {
        const entry = globalHistory[key];
        if (showInertia) {
            const cost = entry.sumInertia;
            if (Number.isFinite(cost)) {
                if (bestInertiaByX[entry.sumK] === undefined || cost < bestInertiaByX[entry.sumK].cost) {
                    bestInertiaByX[entry.sumK] = { cost, configStr: key };
                }
            }
        }
        if (showTruePi) {
            const cost = entry.truePiTrajectoryCost;
            if (Number.isFinite(cost)) {
                if (bestTruePiByX[entry.sumK] === undefined || cost < bestTruePiByX[entry.sumK].cost) {
                    bestTruePiByX[entry.sumK] = { cost, configStr: key };
                }
            }
        }
    }

    const pointsInertia = Object.keys(bestInertiaByX).map(Number).sort((a, b) => a - b).map(x => ({
        x, cost: bestInertiaByX[x].cost, configStr: bestInertiaByX[x].configStr
    }));
    const pointsTruePi = Object.keys(bestTruePiByX).map(Number).sort((a, b) => a - b).map(x => ({
        x, cost: bestTruePiByX[x].cost, configStr: bestTruePiByX[x].configStr
    }));

    if ((showInertia && pointsInertia.length === 0) && (showTruePi && pointsTruePi.length === 0)) return;

    const padX = 70, padY = 40;
    const w = cvs.width - padX - 30;
    const h = cvs.height - padY - 30;

    const minX = rawSequences.length;
    const maxX = rawSequences.length * 15;

    const currentTruePiCost = globalHistory[currentConfigStr]?.truePiTrajectoryCost;

    const costsForScale = [];
    if (showInertia) costsForScale.push(...pointsInertia.map(p => p.cost));
    if (showTruePi) costsForScale.push(...pointsTruePi.map(p => p.cost));
    if (Number.isFinite(currentSumInertia)) costsForScale.push(currentSumInertia);
    if (showTruePi && Number.isFinite(currentTruePiCost)) costsForScale.push(currentTruePiCost);

    const minCost = Math.min(...costsForScale);
    const maxCost = Math.max(...costsForScale);
    const minY = minCost * 0.9;
    const maxY = maxCost * 1.1;

    const getX = x => padX + ((x - minX) / (maxX - minX)) * w;
    const getY = y => padY + h - ((y - minY) / (maxY - minY || 1)) * h;

    ctx.font = "10px sans-serif"; ctx.fillStyle = "#6b7280"; ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
        const val = minX + (i / 5) * (maxX - minX);
        const x = getX(val);
        ctx.beginPath(); ctx.moveTo(x, padY + h); ctx.lineTo(x, padY + h + 5); ctx.stroke();
        ctx.fillText(Math.round(val), x, padY + h + 15);
    }

    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) {
        const val = minY + (i / 5) * (maxY - minY);
        const y = getY(val);
        ctx.beginPath(); ctx.moveTo(padX, y); ctx.lineTo(padX - 5, y); ctx.stroke();
        ctx.fillText(Math.round(val), padX - 8, y + 4);
    }

    ctx.fillStyle = '#9ca3af'; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Sum of Clusters (K_total)", padX + w / 2, cvs.height - 5);

    ctx.save();
    ctx.translate(20, cvs.height / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Cost (lower is better)", 0, 0);
    ctx.restore();

    // Legend (only when both are drawn on one plot)
    if (mode === 'both') {
        ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = "#9ca3af";
        let lx = padX - 10;
        let ly = padY - 28;
        if (showInertia) {
            ctx.fillStyle = "#9ca3af";
            ctx.fillText("• k-means inertia", lx, ly);
            lx += 105;
        }
        if (showTruePi) {
            ctx.fillStyle = "#f59e0b";
            ctx.fillText("• true `pis` trajectory cost", lx, ly);
        }
    }

    const drawCurve = (pts, color, currentCost, isTruePi) => {
        if (!pts || pts.length === 0) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        pts.forEach((p, i) => {
            const x = getX(p.x), y = getY(p.cost);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        pts.forEach(p => {
            const px = getX(p.x), py = getY(p.cost);
            drawnGlobalPointsOut.push({ cx: px, cy: py, cost: p.cost, sumK: p.x, configStr: p.configStr });
            ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
            // Highlight current config with blue (inertia) / orange outline (true)
            if (p.configStr === currentConfigStr) {
                ctx.fillStyle = isTruePi ? "#fbbf24" : "#3b82f6";
            } else {
                ctx.fillStyle = color;
            }
            ctx.fill();
        });

        if (Number.isFinite(currentCost)) {
            const cx = getX(currentSumK);
            const cy = getY(currentCost);
            ctx.beginPath();
            ctx.setLineDash([5, 5]);
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx, padY + h);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.beginPath();
            ctx.arc(cx, cy, 7, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }
    };

    if (showInertia) drawCurve(pointsInertia, "#9ca3af", currentSumInertia, false);
    if (showTruePi) drawCurve(pointsTruePi, "#f59e0b", currentTruePiCost, true);
}

