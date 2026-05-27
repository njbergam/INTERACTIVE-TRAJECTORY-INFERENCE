/**
 * User-drawn ellipses on PCA / kernel PCA plots (one cluster per circle).
 */

import { state } from '../state.js';
import { minKFromAnnotationSnapshot } from '../ai/annotationFidelity.js';

const DRAWABLE_PREFIXES = ['pca:', 'kpca:'];

const LABELS = {
    pca: (s) => `Step ${s} PCA`,
    kpca: (s) => `Step ${s} kernel PCA`
};

function ensureStore() {
    if (!state.plotAnnotations || typeof state.plotAnnotations !== 'object') {
        state.plotAnnotations = {};
    }
}

export function isDrawablePlotId(plotId) {
    return DRAWABLE_PREFIXES.some(p => plotId.startsWith(p));
}

function labelForPlotId(plotId) {
    const m = plotId.match(/^(pca|kpca):(\d+)$/);
    if (!m) return plotId;
    return LABELS[m[1]](m[2]);
}

function getEllipses(plotId) {
    ensureStore();
    if (!Array.isArray(state.plotAnnotations[plotId])) state.plotAnnotations[plotId] = [];
    return state.plotAnnotations[plotId];
}

function normEllipseFromDrag(x0, y0, x1, y1, w, h) {
    const cx = (x0 + x1) / 2 / w;
    const cy = (y0 + y1) / 2 / h;
    const rx = Math.max(0.02, Math.abs(x1 - x0) / 2 / w);
    const ry = Math.max(0.02, Math.abs(y1 - y0) / 2 / h);
    return { cx, cy, rx, ry };
}

function redrawOverlay(overlay, plotId) {
    const w = overlay.width;
    const h = overlay.height;
    if (w < 2 || h < 2) return;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    for (const e of getEllipses(plotId)) {
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.95)';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.ellipse(e.cx * w, e.cy * h, e.rx * w, e.ry * h, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(250, 204, 21, 0.15)';
        ctx.fill();
    }
}

function resizeOverlay(overlay, dataCanvas) {
    const rect = dataCanvas.getBoundingClientRect();
    const w = Math.max(2, Math.round(rect.width) || dataCanvas.width);
    const h = Math.max(2, Math.round(rect.height) || dataCanvas.height);
    overlay.width = w;
    overlay.height = h;
    overlay.style.width = '100%';
    overlay.style.height = '100%';
}

function wireDrawing(overlay, plotId) {
    if (overlay.dataset.wired === '1') return;
    overlay.dataset.wired = '1';

    let drag = null;

    const posFromEvent = (e) => {
        const r = overlay.getBoundingClientRect();
        const sx = overlay.width / (r.width || 1);
        const sy = overlay.height / (r.height || 1);
        return {
            x: (e.clientX - r.left) * sx,
            y: (e.clientY - r.top) * sy
        };
    };

    const drawPreview = () => {
        if (!drag) return;
        const w = overlay.width;
        const h = overlay.height;
        redrawOverlay(overlay, plotId);
        const ctx = overlay.getContext('2d');
        const ell = normEllipseFromDrag(drag.x0, drag.y0, drag.x1, drag.y1, w, h);
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.ellipse(ell.cx * w, ell.cy * h, ell.rx * w, ell.ry * h, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
    };

    const finishDrag = (e) => {
        if (!drag) return;
        const p = posFromEvent(e);
        drag.x1 = p.x;
        drag.y1 = p.y;
        const w = overlay.width;
        const h = overlay.height;
        const ell = normEllipseFromDrag(drag.x0, drag.y0, drag.x1, drag.y1, w, h);
        if (ell.rx * w > 6 && ell.ry * h > 6) {
            getEllipses(plotId).push(ell);
        }
        drag = null;
        window.removeEventListener('pointermove', onWindowMove);
        window.removeEventListener('pointerup', onWindowUp);
        window.removeEventListener('pointercancel', onWindowUp);
        redrawOverlay(overlay, plotId);
        updateDrawHint(stackForPlotId(plotId));
    };

    const onWindowMove = (e) => {
        if (!drag) return;
        const p = posFromEvent(e);
        drag.x1 = p.x;
        drag.y1 = p.y;
        drawPreview();
    };

    const onWindowUp = (e) => {
        finishDrag(e);
    };

    const onPointerDown = (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        overlay.setPointerCapture?.(e.pointerId);
        const p = posFromEvent(e);
        drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
        window.addEventListener('pointermove', onWindowMove);
        window.addEventListener('pointerup', onWindowUp);
        window.addEventListener('pointercancel', onWindowUp);
    };

    overlay.addEventListener('pointerdown', onPointerDown);

    overlay.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const list = getEllipses(plotId);
        if (list.length) list.pop();
        redrawOverlay(overlay, plotId);
        updateDrawHint(stackForPlotId(plotId));
    });
}

function stackForPlotId(plotId) {
    return document.querySelector(`.plot-feedback-wrap[data-plot-id="${plotId}"] .plot-draw-stack`);
}

function updateDrawHint(stack) {
    if (!stack) return;
    const hint = stack.querySelector('.plot-draw-hint');
    if (!hint) return;
    const plotId = stack.closest('.plot-feedback-wrap')?.dataset?.plotId;
    const n = plotId ? getEllipses(plotId).length : 0;
    hint.textContent = n ? `${n} region(s) — drag to add, right-click undo` : 'drag to circle a cluster';
}

/** Stack data canvas + annotation overlay inside feedback wrap. */
export function attachAnnotationLayer(wrap, plotId) {
    if (!wrap || !isDrawablePlotId(plotId)) return;

    let stack = wrap.querySelector('.plot-draw-stack');
    let dataCanvas = stack?.querySelector('canvas.plot-data-canvas')
        || stack?.querySelector('canvas:not(.plot-annotation-layer)');

    if (!stack && dataCanvas) {
        stack = document.createElement('div');
        stack.className = 'plot-draw-stack';
        dataCanvas.classList.add('plot-data-canvas');
        dataCanvas.parentNode.insertBefore(stack, dataCanvas);
        stack.appendChild(dataCanvas);
    }

    if (!stack || !dataCanvas) return;

    if (!dataCanvas.classList.contains('plot-data-canvas')) {
        dataCanvas.classList.add('plot-data-canvas');
    }

    let hint = stack.querySelector('.plot-draw-hint');
    if (!hint) {
        hint = document.createElement('div');
        hint.className = 'plot-draw-hint';
        hint.textContent = 'drag to circle a cluster';
        stack.appendChild(hint);
    }

    let overlay = stack.querySelector('canvas.plot-annotation-layer');
    if (!overlay) {
        overlay = document.createElement('canvas');
        overlay.className = 'plot-annotation-layer';
        overlay.title = 'Drag to circle a cluster; right-click removes last circle';
        stack.appendChild(overlay);
        wireDrawing(overlay, plotId);
    }

    const sync = () => {
        resizeOverlay(overlay, dataCanvas);
        redrawOverlay(overlay, plotId);
        updateDrawHint(stack);
    };
    sync();
    requestAnimationFrame(sync);
}

export function refreshAnnotationLayers(root = document) {
    root.querySelectorAll('.plot-feedback-wrap[data-plot-id]').forEach(wrap => {
        const plotId = wrap.dataset.plotId;
        if (isDrawablePlotId(plotId)) attachAnnotationLayer(wrap, plotId);
    });
}

export function captureAnnotationSnapshot() {
    ensureStore();
    const byPlotId = {};
    for (const [plotId, ellipses] of Object.entries(state.plotAnnotations)) {
        if (ellipses?.length) byPlotId[plotId] = ellipses.map(e => ({ ...e }));
    }
    return { byPlotId };
}

function clearAnnotationDomForPlotIds(plotIdFilter) {
    document.querySelectorAll('.plot-feedback-wrap[data-plot-id]').forEach(wrap => {
        const plotId = wrap.dataset.plotId;
        if (!plotIdFilter(plotId)) return;
        const stack = wrap.querySelector('.plot-draw-stack');
        const overlay = stack?.querySelector('canvas.plot-annotation-layer');
        if (overlay) {
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);
        }
        const hint = stack?.querySelector('.plot-draw-hint');
        if (hint) hint.textContent = 'drag to circle a cluster';
    });
}

/** Remove all PCA + kernel PCA sketches (e.g. new dataset). */
export function clearPlotAnnotations() {
    state.plotAnnotations = {};
    clearAnnotationDomForPlotIds(() => true);
}

/** Kernel PCA embedding changed — drop only kpca:* circles. */
export function clearKernelPcaPlotAnnotations() {
    ensureStore();
    for (const plotId of Object.keys(state.plotAnnotations)) {
        if (plotId.startsWith('kpca:')) delete state.plotAnnotations[plotId];
    }
    clearAnnotationDomForPlotIds(plotId => plotId.startsWith('kpca:'));
}

export function formatAnnotationPromptBlock(snapshot) {
    if (!snapshot?.byPlotId) return '';
    const lines = [];
    let total = 0;
    for (const [plotId, ellipses] of Object.entries(snapshot.byPlotId)) {
        if (!ellipses?.length) continue;
        total += ellipses.length;
        lines.push(
            `- ${labelForPlotId(plotId)}: ${ellipses.length} yellow circle(s) — ` +
            'MANDATORY: every point inside each circle = ONE solid cluster color; each circle = a DIFFERENT color than the others.'
        );
    }
    if (!lines.length) return '';
    return (
        '[HIGHEST PRIORITY — user cluster sketches]\n' +
        'The user drew yellow circles on PCA/kernel PCA plots. This overrides generic elbow/parsimony goals when they conflict.\n' +
        `Total regions: ${total}. Rules:\n` +
        '1) Inside each circle: ≥~85% of points must share one cluster assignment (color-pure).\n' +
        '2) Across circles on the same plot: each circle must map to a different cluster (different colors).\n' +
        '3) Use at least as many clusters K as there are circles on that step.\n' +
        'If you cannot satisfy this, say so explicitly in reasoning.\n' +
        lines.join('\n')
    );
}

/** Boost separation weight and minimum K per step from circle counts. */
export function mergeAnnotationsIntoPlan(plan, snapshot, T) {
    if (!snapshot?.byPlotId || !plan) return plan;
    const annotationMinK = minKFromAnnotationSnapshot(snapshot, T);
    const totalRegions = Object.values(snapshot.byPlotId).reduce((s, a) => s + (a?.length || 0), 0);
    if (!totalRegions) return plan;

    const targetK = plan.targetK.slice();
    const notes = [...(plan.notes || [])];
    for (let t = 0; t < T; t++) {
        if (plan.fixedK[t] !== null) continue;
        const need = annotationMinK[t];
        if (need > 1) {
            targetK[t] = targetK[t] != null ? Math.max(targetK[t], need) : need;
            notes.push(`Step ${t + 1}: at least K=${need} (user drew ${need} cluster region(s)).`);
        }
    }

    const w = { ...plan.weights };
    w.wSep = Math.max(w.wSep ?? 0.15, 0.55);

    return {
        ...plan,
        weights: w,
        targetK,
        annotationMinK,
        prioritizeAnnotations: true,
        notes: [...new Set(notes)]
    };
}

export function getAnnotationsForStep(snapshot, step1Based) {
    if (!snapshot?.byPlotId) return [];
    return [
        ...(snapshot.byPlotId[`pca:${step1Based}`] || []),
        ...(snapshot.byPlotId[`kpca:${step1Based}`] || [])
    ];
}

export function drawAnnotationsOnCtx(ctx, width, height, ellipses) {
    if (!ellipses?.length) return;
    for (const e of ellipses) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 6]);
        ctx.beginPath();
        ctx.ellipse(e.cx * width, e.cy * height, e.rx * width, e.ry * height, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}
