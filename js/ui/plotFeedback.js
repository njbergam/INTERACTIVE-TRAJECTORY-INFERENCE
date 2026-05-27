import { attachAnnotationLayer, isDrawablePlotId } from './plotAnnotations.js';

/**
 * Per-plot thumbs up / down — captured when AI explore starts, then cleared.
 *
 * Plot id: pca:1, kpca:1, dendro:1, elbow-local:1, elbow-global:inertia, …
 */

import { state } from '../state.js';

const LABELS = {
    pca: (s) => `Step ${s} PCA scatter`,
    kpca: (s) => `Step ${s} kernel PCA`,
    dendro: (s) => `Step ${s} dendrogram`,
    'elbow-local': (s) => `Step ${s} local elbow (K vs inertia)`,
    'elbow-global:inertia': () => 'Global cost frontier (sum of k-means inertias)',
    'elbow-global:truepi': () => 'Global cost frontier (true OT pis trajectory cost)'
};

function ensureFeedbackMap() {
    if (!state.plotFeedback || typeof state.plotFeedback !== 'object') {
        state.plotFeedback = {};
    }
}

export function plotIdStep(kind, step1Based) {
    return `${kind}:${step1Based}`;
}

export function getPlotVote(plotId) {
    ensureFeedbackMap();
    const v = state.plotFeedback[plotId];
    return v === 'up' || v === 'down' ? v : null;
}

/** Snapshot before explore clears UI — { ups: string[], downs: string[] } */
export function capturePlotFeedbackSnapshot() {
    ensureFeedbackMap();
    const ups = [];
    const downs = [];
    for (const [id, vote] of Object.entries(state.plotFeedback)) {
        if (vote === 'up') ups.push(id);
        if (vote === 'down') downs.push(id);
    }
    return { ups, downs };
}

export function clearPlotFeedback() {
    state.plotFeedback = {};
    document.querySelectorAll('.plot-feedback-wrap').forEach(wrap => syncWrapVisual(wrap, wrap.dataset.plotId));
}

function setPlotVote(plotId, vote, wrapEl) {
    ensureFeedbackMap();
    const current = state.plotFeedback[plotId];
    if (current === vote) {
        delete state.plotFeedback[plotId];
    } else {
        state.plotFeedback[plotId] = vote;
    }
    if (wrapEl) syncWrapVisual(wrapEl, plotId);
    else {
        document.querySelectorAll(`.plot-feedback-wrap[data-plot-id="${plotId}"]`).forEach(w => syncWrapVisual(w, plotId));
    }
}

function syncWrapVisual(wrap, plotId) {
    const vote = getPlotVote(plotId);
    wrap.classList.toggle('plot-vote-up', vote === 'up');
    wrap.classList.toggle('plot-vote-down', vote === 'down');

    const upBtn = wrap.querySelector('.plot-feedback-btn--up');
    const downBtn = wrap.querySelector('.plot-feedback-btn--down');
    if (upBtn) {
        upBtn.classList.toggle('active', vote === 'up');
        upBtn.setAttribute('aria-pressed', vote === 'up' ? 'true' : 'false');
    }
    if (downBtn) {
        downBtn.classList.toggle('active', vote === 'down');
        downBtn.setAttribute('aria-pressed', vote === 'down' ? 'true' : 'false');
    }
}

function createFeedbackToolbar(wrap, plotId) {
    const bar = document.createElement('div');
    bar.className = 'plot-feedback-toolbar';

    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'plot-feedback-btn plot-feedback-btn--up';
    upBtn.textContent = '👍';
    upBtn.title = 'Thumbs up — prefer this view in the next AI explore';
    upBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        setPlotVote(plotId, 'up', wrap);
    });

    const downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.className = 'plot-feedback-btn plot-feedback-btn--down';
    downBtn.textContent = '👎';
    downBtn.title = 'Thumbs down — avoid this look in the next AI explore';
    downBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        setPlotVote(plotId, 'down', wrap);
    });

    bar.appendChild(upBtn);
    bar.appendChild(downBtn);
    syncWrapVisual(wrap, plotId);
    return bar;
}

export function wrapPlotCanvas(canvas, plotId) {
    if (!canvas || !plotId) return canvas;

    const parent = canvas.parentElement;
    if (parent?.classList?.contains('plot-feedback-wrap') && parent.dataset.plotId === plotId) {
        syncWrapVisual(parent, plotId);
        if (isDrawablePlotId(plotId)) attachAnnotationLayer(parent, plotId);
        return parent;
    }

    const wrap = document.createElement('div');
    wrap.className = 'plot-feedback-wrap';
    wrap.dataset.plotId = plotId;

    if (parent) {
        parent.insertBefore(wrap, canvas);
    }

    canvas.classList.add('plot-data-canvas');

    const stack = document.createElement('div');
    stack.className = 'plot-draw-stack';
    stack.appendChild(canvas);
    wrap.appendChild(stack);
    wrap.appendChild(createFeedbackToolbar(wrap, plotId));
    syncWrapVisual(wrap, plotId);

    if (isDrawablePlotId(plotId)) {
        attachAnnotationLayer(wrap, plotId);
    }
    return wrap;
}

export function ensureCanvasEmphasisWrap(canvas, plotId) {
    return wrapPlotCanvas(canvas, plotId);
}

function labelForPlotId(plotId) {
    if (plotId === 'elbow-global:inertia') return LABELS['elbow-global:inertia']();
    if (plotId === 'elbow-global:truepi') return LABELS['elbow-global:truepi']();
    const m = plotId.match(/^([a-z-]+):(\d+)$/);
    if (!m) return plotId;
    const fn = LABELS[m[1]];
    return fn ? fn(m[2]) : plotId;
}

export function formatFeedbackPromptBlock(snapshot) {
    if (!snapshot) return '';
    const { ups = [], downs = [] } = snapshot;
    if (!ups.length && !downs.length) return '';

    const lines = [];
    if (ups.length) {
        lines.push('👍 User LIKES these views (optimize K to preserve/improve them):');
        ups.forEach(id => lines.push(`  - ${labelForPlotId(id)}`));
    }
    if (downs.length) {
        lines.push('👎 User DISLIKES these views (change K to fix them):');
        downs.forEach(id => lines.push(`  - ${labelForPlotId(id)}`));
    }
    return `[Plot feedback for this explore run]\n${lines.join('\n')}`;
}

export function mergePlotFeedbackIntoPlan(plan, snapshot) {
    if (!snapshot) return plan;
    const { ups = [], downs = [] } = snapshot;
    if (!ups.length && !downs.length) return plan;

    const notes = [...(plan.notes || [])];
    if (ups.length) notes.push(`Liked: ${ups.map(labelForPlotId).join('; ')}.`);
    if (downs.length) notes.push(`Disliked: ${downs.map(labelForPlotId).join('; ')}.`);

    const w = { ...plan.weights };
    for (const id of ups) {
        if (id.startsWith('elbow-global:truepi')) w.wTrue += 0.14;
        if (id.startsWith('elbow-global:inertia')) w.wInertia += 0.14;
        if (id.startsWith('elbow-local:')) w.wInertia += 0.1;
        if (id.startsWith('pca:') || id.startsWith('kpca:')) w.wSep += 0.12;
        if (id.startsWith('dendro:')) w.wSep += 0.08;
    }
    for (const id of downs) {
        if (id.startsWith('elbow-global:truepi')) w.wTrue = Math.max(0.05, w.wTrue - 0.08);
        if (id.startsWith('elbow-global:inertia')) w.wInertia = Math.max(0.05, w.wInertia - 0.08);
        if (id.startsWith('pca:') || id.startsWith('kpca:')) w.wSep += 0.06;
    }
    const s = w.wInertia + w.wTrue;
    if (s > 0) {
        w.wInertia /= s;
        w.wTrue /= s;
    }

    return { ...plan, weights: w, notes, plotFeedback: snapshot };
}

/** PCA steps with thumbs up (for vision montage focus). */
export function getPositivePcaSteps(snapshot) {
    const ups = snapshot?.ups ?? [];
    return ups
        .filter(id => id.startsWith('pca:'))
        .map(id => parseInt(id.split(':')[1], 10))
        .filter(n => Number.isFinite(n));
}

export function feedbackAdjustmentForShape(shapeStats, snapshot) {
    if (!shapeStats?.steps?.length || !snapshot) return 0;
    let delta = 0;
    for (const s of shapeStats.steps) {
        const step = s.step;
        for (const prefix of ['pca', 'kpca', 'dendro']) {
            const id = `${prefix}:${step}`;
            if (snapshot.ups?.includes(id)) delta -= 12 * (s.score + Math.max(0, s.silhouette));
            if (snapshot.downs?.includes(id)) delta += 18 * (1 - Math.max(0, (s.silhouette + 0.2) / 1.2));
        }
    }
    return delta;
}
