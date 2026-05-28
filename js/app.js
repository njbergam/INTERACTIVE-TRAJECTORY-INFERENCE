import { MAX_PKL_BYTES, COLORS } from './constants.js';
import { elements } from './dom.js';
import { rawSequences, state, drawnEdgePaths, dataPickerAvailable, setRawSequences, setDataPickerAvailable } from './state.js';

import { kMeans, findElbow } from './algorithms/kmeans.js';
import { exactOT, sinkhorn, sinkhornPointwise, gromovWasserstein } from './algorithms/ot.js';

import { extractSequencesFromZip } from './data/npz.js';
import { extractSequencesFromPkl } from './data/pyodide.js';

import {
    runPCARaw,
    drawScatterPlots,
    computeKernelPcaProjection,
    computeDendrogram,
    KERNEL_PCA_MAX_POINTS,
    setPlotSectionExpandHandler
} from './viz/pca.js';
import { runPCARaw3D } from './viz/pca3d.js';
import { drawGraph } from './viz/graph.js';
import { drawLocalElbowPlot, drawGlobalElbowPlot } from './viz/elbow.js';

import { ensureResizeControl, getResizePlotSize, showHomePage } from './ui/layout.js';
import {
    wrapPlotCanvas,
    plotIdStep,
    ensureCanvasEmphasisWrap,
    capturePlotFeedbackSnapshot,
    clearPlotFeedback,
    formatFeedbackPromptBlock,
    mergePlotFeedbackIntoPlan,
    getPositivePcaSteps,
    feedbackAdjustmentForShape
} from './ui/plotFeedback.js';
import {
    ensureKLocks,
    toggleKLock,
    syncKLockButton,
    syncKLockSlider,
    applyKLocksToExplorePlan,
    enforceLockedKInVector,
    formatKLocksPromptBlock,
    isStepKLocked
} from './ui/kLocks.js';
import {
    captureAnnotationSnapshot,
    clearPlotAnnotations,
    clearKernelPcaPlotAnnotations,
    formatAnnotationPromptBlock,
    mergeAnnotationsIntoPlan,
    refreshAnnotationLayers,
    snapshotHasAnnotationEmphasis
} from './ui/plotAnnotations.js';
import {
    evaluateAnnotationFidelity,
    annotationFidelityPenalty,
    formatAnnotationFulfillmentReport
} from './ai/annotationFidelity.js';
import { promptConstraintPenalty } from './ai/promptPlan.js';
import {
    proposeExploreCandidate,
    seedExploreCandidates,
    filterUntriedConfigs,
    rememberTriedConfig,
    hasTriedConfig,
    exploreComplexityPenalty,
    kConfigKey,
    dequeueBestCandidate,
    warmKMeansCacheForExplore,
    rankCandidatesByHeuristic
} from './ai/exploreSearch.js';
import { aggregateTrajectoryShapeStats, buildTrajectoryClusterDiagnostics } from './ai/shapeStats.js';
import { buildClusterVisionMontage, analyzeClusterMontage, visionQualityToScore } from './ai/visionExplore.js';
import { getLlmSettings, saveLlmSettings, applyPreset, listPresets, testLlmConnection } from './ai/llmClient.js';
import {
    llmInitialPlan,
    llmNextProposal,
    fallbackPlan,
    mergeLlmWeights,
    formatPlanSummary,
    applyPromptConstraints
} from './ai/llmExplore.js';

let lastLoadedFileName = null;
let lastClusters = null;
let lastIndependentScale = false;
let analysisJobId = 0;
/** Step indices whose Kernel PCA / dendrogram section was expanded at least once. */
const kernelPcaStepsRequested = new Set();
const kernelPcaComputingSteps = new Set();
const dendrogramStepsRequested = new Set();
const dendrogramComputingSteps = new Set();
const pca3dStepsRequested = new Set();
const pca3dComputingSteps = new Set();
let pointOtJobId = 0;

const drawnGlobalInertiaPoints = [];
const drawnGlobalTruePiPoints = [];

function showCombinedPlotModal(show) {
    if (!elements.combinedPlotModal) return;
    elements.combinedPlotModal.style.display = show ? 'block' : 'none';
}

function fitCanvasToRect(canvas, { maxW = 1400, maxH = 900 } = {}) {
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(2, Math.min(maxW, Math.round(rect.width || canvas.width || maxW)));
    const h = Math.max(2, Math.min(maxH, Math.round((rect.height || 0) || (w * 0.6))));
    if (canvas.width === w && canvas.height === h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
}

function fitHiDPICanvasToRect(canvas, { maxW = 1400, maxH = 900 } = {}) {
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(2, Math.min(maxW, Math.round(rect.width || canvas.clientWidth || maxW)));
    const cssH = Math.max(2, Math.min(maxH, Math.round((rect.height || 0) || (cssW * 0.6))));
    const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const w = Math.round(cssW * dpr);
    const h = Math.round(cssH * dpr);
    if (canvas.width === w && canvas.height === h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
}

function drawArrow(ctx, x1, y1, x2, y2, { head = 9 } = {}) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const hx = x2 - ux * head;
    const hy = y2 - uy * head;
    const px = -uy;
    const py = ux;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(hx + px * (head * 0.55), hy + py * (head * 0.55));
    ctx.lineTo(hx - px * (head * 0.55), hy - py * (head * 0.55));
    ctx.closePath();
    ctx.fill();
}

function hexToRgb(hex) {
    const h = (hex || '').replace('#', '');
    if (h.length !== 6) return { r: 148, g: 163, b: 184 };
    return {
        r: parseInt(h.slice(0, 2), 16),
        g: parseInt(h.slice(2, 4), 16),
        b: parseInt(h.slice(4, 6), 16)
    };
}

function computeCouplingsForClusters(clusters) {
    const couplings = [];
    for (let t = 0; t < clusters.length - 1; t++) {
        const k1 = state.kValues[t];
        const k2 = state.kValues[t + 1];
        const useGW = rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1];
        const pairKey = useGW ? `${k1}-${k2}-gw-${state.epsilon}` : `${k1}-${k2}-ot`;
        if (state.otCache[t][pairKey] === undefined) {
            state.otCache[t][pairKey] = useGW
                ? gromovWasserstein(clusters[t], clusters[t + 1], state.epsilon)
                : exactOT(clusters[t], clusters[t + 1]);
        }
        const result = state.otCache[t][pairKey];
        couplings.push(
            useGW ? result.P : (state.epsilon === 0 ? result.P : sinkhorn(clusters[t], clusters[t + 1], state.epsilon))
        );
    }
    return couplings;
}

// Match the transport graph's color continuity heuristic (see js/viz/graph.js).
function computeColorIndexByStepForTransportGraph(clusters, couplings) {
    const T = clusters.length;
    const colorIndexByStep = Array.from({ length: T }, () => []);
    if (!T) return colorIndexByStep;

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

function generateCombinedTrajectoryPlot() {
    try {
        showCombinedPlotModal(true);

        if (!rawSequences.length || !lastClusters?.length) {
            elements.status.textContent = 'Combined plot: load a dataset first.';
            return;
        }
        const clusters = lastClusters;
        const T = clusters.length;
        if (T < 2) {
            elements.status.textContent = 'Combined plot requires at least 2 timesteps.';
            return;
        }

        const canvas = elements.combinedPlotCanvas;
        const ctx = elements.combinedPlotCtx;
        if (!canvas || !ctx) {
            elements.status.textContent = 'Combined plot canvas not found.';
            return;
        }
        // Ensure crisp rendering on HiDPI screens.
        fitHiDPICanvasToRect(canvas, { maxW: 1600, maxH: 980 });

        // Require consistent dimensions for a single PCA over all points.
        const d0 = rawSequences[0].shape?.[1] ?? null;
        if (!d0 || rawSequences.some(s => (s.shape?.[1] ?? null) !== d0)) {
            elements.status.textContent = 'Combined plot requires equal feature dimension across steps (no GW edges).';
            return;
        }

        elements.status.textContent = 'Generating combined PCA plot…';

        const couplings = computeCouplingsForClusters(clusters);
        const colorIndexByStep = computeColorIndexByStepForTransportGraph(clusters, couplings);

        // Concatenate all points and run one PCA.
        const offsets = [];
        let totalN = 0;
        for (let t = 0; t < T; t++) {
            offsets[t] = totalN;
            totalN += rawSequences[t].shape[0];
        }
        const all = new Float64Array(totalN * d0);
        for (let t = 0; t < T; t++) {
            const src = rawSequences[t].data;
            all.set(src, offsets[t] * d0);
        }
        const coords2d = runPCARaw(all, totalN, d0);

        // Compute bounds.
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of coords2d) {
            minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
        }
        const pad = 52;
        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;
        const scale = Math.min((canvas.width - 2 * pad) / rangeX, (canvas.height - 2 * pad) / rangeY);
        const tx = (x) => pad + (x - minX) * scale;
        const ty = (y) => canvas.height - pad - (y - minY) * scale;

    // Precompute centroid positions in PCA space (per step, per cluster).
    const centroidXY = Array.from({ length: T }, (_, t) => {
        const k = clusters[t].centers.length;
        return Array.from({ length: k }, () => ({ x: 0, y: 0, n: 0 }));
    });
    for (let t = 0; t < T; t++) {
        const n = rawSequences[t].shape[0];
        const off = offsets[t];
        const assn = clusters[t].assignments;
        for (let i = 0; i < n; i++) {
            const c = assn[i];
            const p = coords2d[off + i];
            const acc = centroidXY[t][c];
            acc.x += p[0];
            acc.y += p[1];
            acc.n += 1;
        }
        for (const acc of centroidXY[t]) {
            const denom = acc.n || 1;
            acc.x /= denom;
            acc.y /= denom;
        }
    }

        // Draw background.
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#0b1220';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw centroid-to-centroid transport edges (major flows), styled like the OT graph.
        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.95;
        for (let t = 0; t < T - 1; t++) {
            const P = couplings[t];
            const k1 = clusters[t].centers.length;
            const k2 = clusters[t + 1].centers.length;
            let maxP = 0;
            for (let i = 0; i < k1; i++) for (let j = 0; j < k2; j++) if ((P?.[i]?.[j] ?? 0) > maxP) maxP = P[i][j];
            const denom = maxP || 1;

            for (let i = 0; i < k1; i++) {
                const a = centroidXY[t]?.[i];
                if (!a) continue;
                for (let j = 0; j < k2; j++) {
                    const mass = P?.[i]?.[j] ?? 0;
                    if (mass < 1e-6) continue;
                    const b = centroidXY[t + 1]?.[j];
                    if (!b) continue;

                    const intensity = Math.max(0, Math.min(1, mass / denom));
                    // Keep only "major" edges but scale threshold with intensity.
                    if (intensity < 0.03) continue;

                    const srcColorIdx = colorIndexByStep[t]?.[i] ?? i;
                    const dstColorIdx = colorIndexByStep[t + 1]?.[j] ?? j;
                    const srcRgb = hexToRgb(COLORS[srcColorIdx % COLORS.length]);
                    const dstRgb = hexToRgb(COLORS[dstColorIdx % COLORS.length]);

                    const x1 = tx(a.x), y1 = ty(a.y);
                    const x2 = tx(b.x), y2 = ty(b.y);

                    const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                    const a0 = 0.08 + 0.55 * intensity;
                    const a1 = 0.06 + 0.48 * intensity;
                    grad.addColorStop(0, `rgba(${srcRgb.r}, ${srcRgb.g}, ${srcRgb.b}, ${a0})`);
                    grad.addColorStop(1, `rgba(${dstRgb.r}, ${dstRgb.g}, ${dstRgb.b}, ${a1})`);

                    ctx.strokeStyle = grad;
                    ctx.fillStyle = `rgba(${dstRgb.r}, ${dstRgb.g}, ${dstRgb.b}, ${0.10 + 0.55 * intensity})`;
                    ctx.lineWidth = Math.max(0.9, 1.1 + Math.pow(intensity, 0.65) * 7.5);
                    ctx.shadowColor = 'rgba(0,0,0,0.25)';
                    ctx.shadowBlur = 7;
                    drawArrow(ctx, x1, y1, x2, y2, { head: 8 + 6 * intensity });
                    ctx.shadowBlur = 0;
                }
            }
        }
        ctx.restore();

        // Draw points, colored to match the transport graph (continuity heuristic).
        const pointR = Math.max(2.2, (window.devicePixelRatio || 1) * 1.75);
        for (let t = 0; t < T; t++) {
            const n = rawSequences[t].shape[0];
            const off = offsets[t];
            const assn = clusters[t].assignments;
            for (let i = 0; i < n; i++) {
                const c = assn[i];
                const colorIdx = colorIndexByStep[t]?.[c] ?? c;
                const hex = COLORS[colorIdx % COLORS.length];
                const rgb = hexToRgb(hex);
                const p = coords2d[off + i];
                ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.55)`;
                ctx.beginPath();
                ctx.arc(tx(p[0]), ty(p[1]), pointR, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw centroid markers on top.
        for (let t = 0; t < T; t++) {
            const k = clusters[t].centers.length;
            for (let i = 0; i < k; i++) {
                const colorIdx = colorIndexByStep[t]?.[i] ?? i;
                const rgb = hexToRgb(COLORS[colorIdx % COLORS.length]);
                const c = centroidXY[t][i];
                const x = tx(c.x);
                const y = ty(c.y);
                ctx.beginPath();
                ctx.arc(x, y, 5.5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.95)`;
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.85)';
                ctx.lineWidth = 1.6;
                ctx.stroke();
            }
        }
        ctx.globalAlpha = 1;
        elements.status.textContent = 'Combined plot generated.';
    } catch (err) {
        const msg = err?.message ? String(err.message) : String(err);
        elements.status.textContent = `Combined plot error: ${msg}`;
        console.error(err);
        showCombinedPlotModal(true);
    }
}

function wireCombinedPlotUi() {
    const btn = elements.generateCombinedPlotBtn;
    if (btn && btn.dataset.wired !== '1') {
        btn.dataset.wired = '1';
        btn.onclick = () => {
            try {
                btn.disabled = true;
                btn.textContent = 'Generating…';
                generateCombinedTrajectoryPlot();
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate combined plot';
            }
        };
    }

    const modal = elements.combinedPlotModal;
    if (modal && modal.dataset.wired !== '1') {
        modal.dataset.wired = '1';
        modal.addEventListener('click', (e) => {
            const el = e.target;
            if (!(el instanceof HTMLElement)) return;
            if (el.dataset.action === 'close') showCombinedPlotModal(false);
        });
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') showCombinedPlotModal(false);
        });
    }
}

let aiExploreRunning = false;
let aiExploreCancelToken = 0;
let aiExploreProgressEl = null;
let aiExploreTimerLabelEl = null;
let aiExploreBtnEl = null;
let aiExploreDurationEl = null;
let aiExplorePromptEl = null;
let aiExploreWidgetEl = null;
let aiExploreTranscriptEl = null;
let aiExploreMinimizeBtnEl = null;
let aiExploreLiveMsgEl = null;
let aiExploreMinimized = false;
let aiExploreAbortController = null;

function isExploreCancelled(myToken, signal) {
    return myToken !== aiExploreCancelToken || !!signal?.aborted;
}

function requestStopAiExplore() {
    if (!aiExploreRunning) return;
    aiExploreCancelToken++;
    aiExploreAbortController?.abort();
    setAiExploreLiveStatus('Stopping… cancelling LLM / search.');
    if (aiExploreBtnEl) {
        aiExploreBtnEl.disabled = false;
        aiExploreBtnEl.textContent = 'Stopping…';
    }
    finalizeAiExploreLiveStatus();
}

function setAiExploreWidgetVisible(visible) {
    if (aiExploreWidgetEl) {
        aiExploreWidgetEl.classList.toggle('visible', !!visible);
    }
}

function setAiExploreMinimized(minimized) {
    aiExploreMinimized = !!minimized;
    if (aiExploreWidgetEl) {
        aiExploreWidgetEl.classList.toggle('minimized', aiExploreMinimized);
    }
    if (aiExploreMinimizeBtnEl) {
        aiExploreMinimizeBtnEl.textContent = aiExploreMinimized ? '+' : '−';
        aiExploreMinimizeBtnEl.title = aiExploreMinimized ? 'Restore AI explore panel' : 'Minimize AI explore panel';
    }
}

function scrollAiTranscriptToBottom() {
    if (!aiExploreTranscriptEl) return;
    aiExploreTranscriptEl.scrollTop = aiExploreTranscriptEl.scrollHeight;
}

function appendAiExploreMessage(kind, text) {
    if (!aiExploreTranscriptEl) return null;

    const wrap = document.createElement('div');
    wrap.className = `ai-explore-msg ${kind}`;

    const label = document.createElement('div');
    label.className = 'ai-explore-msg-label';
    label.textContent = kind === 'user' ? 'You' : (kind === 'status' ? 'Status' : 'AI explore');

    const body = document.createElement('div');
    body.textContent = text;

    wrap.appendChild(label);
    wrap.appendChild(body);
    aiExploreTranscriptEl.appendChild(wrap);
    scrollAiTranscriptToBottom();
    return wrap;
}

function setAiExploreLiveStatus(text) {
    if (!aiExploreTranscriptEl) return;
    if (!aiExploreLiveMsgEl) {
        const wrap = document.createElement('div');
        wrap.className = 'ai-explore-msg status';
        const label = document.createElement('div');
        label.className = 'ai-explore-msg-label';
        label.textContent = 'Status';
        const body = document.createElement('div');
        wrap.appendChild(label);
        wrap.appendChild(body);
        aiExploreTranscriptEl.appendChild(wrap);
        aiExploreLiveMsgEl = body;
    }
    aiExploreLiveMsgEl.textContent = text;
    scrollAiTranscriptToBottom();
}

function finalizeAiExploreLiveStatus() {
    aiExploreLiveMsgEl = null;
}

function clearAiExploreTranscript() {
    if (aiExploreTranscriptEl) aiExploreTranscriptEl.innerHTML = '';
    aiExploreLiveMsgEl = null;
}

function formatKVector(kValues) {
    return `[${kValues.join(', ')}]`;
}

function buildAiExploreSummary({ kValues, details, prompt, startingK, trials, cancelled }) {
    if (!details) {
        return cancelled ? 'Explore cancelled.' : 'Explore finished.';
    }
    const lines = [];
    if (cancelled) lines.push('Stopped early.');
    const sumK = kValues.reduce((a, b) => a + b, 0);
    const startSumK = startingK.reduce((a, b) => a + b, 0);
    lines.push(`Chose K=${formatKVector(kValues)} (ΣK=${sumK}, was ${formatKVector(startingK)} ΣK=${startSumK}).`);
    const trueTxt = details.trueCost !== null ? `, true OT cost ${details.trueCost.toFixed(0)}` : '';
    const shapeTxt = details.shapeStats?.score != null ? `, shape ${details.shapeStats.score.toFixed(2)}` : '';
    const visionTxt = details.visionQuality != null ? `, vision ${(details.visionQuality * 10).toFixed(1)}/10` : '';
    const ann = details.annotationFidelity;
    const annTxt = ann?.totalCircles
        ? `, sketches ${Math.round(ann.overall * 100)}%${ann.satisfied ? ' ✓' : ' (incomplete)'}`
        : '';
    lines.push(`Scores: inertia ${details.sumInertia.toFixed(0)}${trueTxt}${shapeTxt}${visionTxt}${annTxt} (${trials} trials).`);
    const p = (prompt || '').trim();
    if (p) lines.push(`Prompt: “${p.length > 80 ? p.slice(0, 77) + '…' : p}”`);
    return lines.join(' ');
}

function clampInt(x, lo, hi) {
    const v = Math.round(x);
    return Math.max(lo, Math.min(hi, Number.isFinite(v) ? v : lo));
}

function applyKValuesToUI(kValues, explorePlan = null) {
    const next = explorePlan ? enforceLockedKInVector(kValues, explorePlan) : kValues.slice();
    state.kValues = next;
    state.kValues.forEach((k, i) => {
        const slider = elements.kContainer.querySelector(`input[type="range"][data-step="${i}"]`);
        const label = elements.kContainer.querySelector(`label[data-step="${i}"]`);
        const locked = isStepKLocked(i);
        if (locked) {
            if (label) label.textContent = `Step ${i + 1} Clusters: ${k} (locked)`;
            if (slider) slider.value = String(k);
            return;
        }
        if (slider) slider.value = String(k);
        if (label) label.textContent = `Step ${i + 1} Clusters: ${k}`;
    });
}

function scoreConfigFast(kValues, {
    weights,
    allowTrueCost,
    plan = null,
    visionAssessment = null,
    plotFeedback = null,
    annotationSnapshot = null
}) {
    const T = rawSequences.length;
    let sumInertia = 0;
    const clusters = [];
    for (let t = 0; t < T; t++) {
        const k = kValues[t];
        if (!state.cache[t][k]) state.cache[t][k] = kMeans(rawSequences[t], k);
        const c = state.cache[t][k];
        clusters.push(c);
        sumInertia += c.inertia;
    }
    const shapeStats = aggregateTrajectoryShapeStats(state.pcaPoints, clusters);
    const geomShape = shapeStats.score;
    const visionQuality = visionAssessment
        ? visionQualityToScore(visionAssessment, visionAssessment.diagnostics)
        : null;

    const entry = state.globalHistory[kValues.join(',')];
    const trueCost = allowTrueCost && Number.isFinite(entry?.truePiTrajectoryCost)
        ? entry.truePiTrajectoryCost
        : null;
    const sumK = Math.max(1, kValues.reduce((a, b) => a + b, 0));
    const inertiaTerm = sumInertia / sumK;

    // Transport entropy: lower is generally better (sharper, less diffuse couplings).
    // We compute normalized entropy per step in [0,1] using the current cluster-level couplings.
    const couplingEntropy01 = (P) => {
        if (!P?.length || !P[0]?.length) return 0;
        const k1 = P.length;
        const k2 = P[0].length;
        let sum = 0;
        let H = 0;
        for (let i = 0; i < k1; i++) {
            const row = P[i];
            for (let j = 0; j < k2; j++) {
                const v = row[j] || 0;
                if (v > 0) {
                    sum += v;
                    H -= v * Math.log(v);
                }
            }
        }
        if (!(sum > 0)) return 0;
        // Convert to entropy of normalized probabilities.
        H = (H / sum) + Math.log(sum); // ( -Σ v log v )/sum + log(sum)
        const denom = Math.log(Math.max(2, k1 * k2));
        return denom > 0 ? Math.max(0, Math.min(1, H / denom)) : 0;
    };

    let meanEntropy = 0;
    if (clusters.length > 1) {
        let steps = 0;
        for (let t = 0; t < clusters.length - 1; t++) {
            const k1 = kValues[t];
            const k2 = kValues[t + 1];
            const useGW = rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1];
            const pairKey = useGW ? `${k1}-${k2}-gw-${state.epsilon}` : `${k1}-${k2}-ot`;
            if (state.otCache[t][pairKey] === undefined) {
                state.otCache[t][pairKey] = useGW
                    ? gromovWasserstein(clusters[t], clusters[t + 1], state.epsilon)
                    : exactOT(clusters[t], clusters[t + 1]);
            }
            const result = state.otCache[t][pairKey];
            const P = useGW ? result.P : (state.epsilon === 0 ? result.P : sinkhorn(clusters[t], clusters[t + 1], state.epsilon));
            meanEntropy += couplingEntropy01(P);
            steps++;
        }
        meanEntropy = steps ? meanEntropy / steps : 0;
    }

    let combined =
        trueCost !== null && allowTrueCost
            ? weights.wInertia * inertiaTerm + weights.wTrue * trueCost
            : inertiaTerm;
    combined -= weights.wSep * (geomShape / Math.sqrt(sumK / kValues.length));
    if (visionQuality !== null) {
        combined -= weights.wSep * 1.2 * visionQuality * 10;
    }
    combined += exploreComplexityPenalty(kValues, plan);
    combined += feedbackAdjustmentForShape(shapeStats, plotFeedback);
    // Mild preference for lower-entropy transport maps.
    combined += 0.22 * meanEntropy;
    if (plan) combined += promptConstraintPenalty(kValues, plan);

    let annotationFidelity = null;
    if (annotationSnapshot?.byPlotId && Object.keys(annotationSnapshot.byPlotId).length) {
        annotationFidelity = evaluateAnnotationFidelity({
            annotationSnapshot,
            pcaPointsByStep: state.pcaPoints,
            kernelPcaPointsByStep: state.kernelPcaPoints,
            clusters,
            independentScale: lastIndependentScale
        });
        const annotationEmphasis = snapshotHasAnnotationEmphasis(annotationSnapshot);
        combined += annotationFidelityPenalty(annotationFidelity, { emphasis: annotationEmphasis });
        const annSepBoost = annotationEmphasis ? 5.4 : 4.2;
        combined -= weights.wSep * annSepBoost * annotationFidelity.overall * Math.max(1, annotationFidelity.totalCircles);
    }

    const sepScore = geomShape + (visionQuality ?? 0) * 3.5
        + (annotationFidelity?.overall ?? 0) * 4;
    return {
        combined,
        sumInertia,
        trueCost,
        sepScore,
        meanTransportEntropy: meanEntropy,
        shapeStats,
        visionQuality,
        visionAssessment: visionAssessment || null,
        annotationFidelity,
        clusters
    };
}

function refreshExploreVisualization(clusters) {
    if (!clusters?.length) return;
    fitMainCanvases();
    state.plotSize = getResizePlotSize(state.plotSize);
    elements.elbowCanvases.forEach((canvas, t) => drawLocalElbowPlot(canvas, state.cache[t], state.kValues[t]));
    lastClusters = clusters;
    drawScatterPlots(elements.scatterContainer, state.pcaPoints, clusters, state.plotSize, scatterPlotOptions());
    requestAnimationFrame(() => refreshAnnotationLayers(elements.scatterContainer));
    const couplings = [];
    for (let t = 0; t < clusters.length - 1; t++) {
        const k1 = state.kValues[t];
        const k2 = state.kValues[t + 1];
        const useGW = rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1];
        const pairKey = useGW ? `${k1}-${k2}-gw-${state.epsilon}` : `${k1}-${k2}-ot`;
        if (state.otCache[t][pairKey] === undefined) {
            state.otCache[t][pairKey] = useGW
                ? gromovWasserstein(clusters[t], clusters[t + 1], state.epsilon)
                : exactOT(clusters[t], clusters[t + 1]);
        }
        const result = state.otCache[t][pairKey];
        couplings.push(
            useGW ? result.P : (state.epsilon === 0 ? result.P : sinkhorn(clusters[t], clusters[t + 1], state.epsilon))
        );
    }
    drawGraph(elements, clusters, couplings, drawnEdgePaths);
    updateGlobalCostFrontiers(clusters);
}

function updateGlobalCostFrontiers(clusters) {
    if (!clusters?.length) return;

    const sumInertia = clusters.reduce((sum, c) => sum + c.inertia, 0);
    const currentConfigStr = state.kValues.join(',');
    const sumK = state.kValues.reduce((a, b) => a + b, 0);
    const existing = state.globalHistory[currentConfigStr];
    const shouldComputeTrueCost = Array.isArray(state.truePis) && state.truePis.length > 0 &&
        (!existing || existing.truePiTrajectoryCost === null || existing.truePiTrajectoryCost === undefined);
    const truePiTrajectoryCost = shouldComputeTrueCost
        ? computeCentroidDiscretizationCostFromTruePis(clusters)
        : existing?.truePiTrajectoryCost ?? null;

    if (!existing) {
        state.globalHistory[currentConfigStr] = { sumK, sumInertia, truePiTrajectoryCost };
    } else {
        existing.sumK = sumK;
        existing.sumInertia = sumInertia;
        if (truePiTrajectoryCost !== null && truePiTrajectoryCost !== undefined) {
            existing.truePiTrajectoryCost = truePiTrajectoryCost;
        }
    }

    if (elements.globalInertiaCanvas && elements.globalInertiaCtx) {
        drawGlobalElbowPlot(
            { canvas: elements.globalInertiaCanvas, ctx: elements.globalInertiaCtx },
            rawSequences,
            state.globalHistory,
            sumK,
            sumInertia,
            currentConfigStr,
            drawnGlobalInertiaPoints,
            { mode: 'inertia' }
        );
    }

    if (Array.isArray(state.truePis) && state.truePis.length > 0) {
        if (elements.globalTruePiCanvas && elements.globalTruePiCtx) {
            drawGlobalElbowPlot(
                { canvas: elements.globalTruePiCanvas, ctx: elements.globalTruePiCtx },
                rawSequences,
                state.globalHistory,
                sumK,
                sumInertia,
                currentConfigStr,
                drawnGlobalTruePiPoints,
                { mode: 'truePi' }
            );
        }
    } else {
        drawnGlobalTruePiPoints.length = 0;
        if (elements.globalTruePiCtx && elements.globalTruePiCanvas) {
            elements.globalTruePiCtx.clearRect(0, 0, elements.globalTruePiCanvas.width, elements.globalTruePiCanvas.height);
        }
    }
}

function setExploreVisualActive(active) {
    elements.vizContainer?.classList.toggle('explore-active', !!active);
    elements.controls?.classList.toggle('explore-active', !!active);
}

function computeEmbeddingSeparationScore(points2d, assignments, k) {
    // points2d: Array<[x,y]> length n, assignments length n in [0..k-1]
    // Score is higher when clusters are compact + well-separated in the embedding.
    const n = assignments.length;
    if (!points2d || points2d.length !== n) return 0;
    if (k <= 1) return 0;

    const sumX = new Float64Array(k);
    const sumY = new Float64Array(k);
    const count = new Int32Array(k);
    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        const p = points2d[i];
        if (!p || c < 0 || c >= k) continue;
        count[c]++;
        sumX[c] += p[0];
        sumY[c] += p[1];
    }

    const cx = new Float64Array(k);
    const cy = new Float64Array(k);
    for (let c = 0; c < k; c++) {
        const denom = count[c] || 1;
        cx[c] = sumX[c] / denom;
        cy[c] = sumY[c] / denom;
    }

    // within-cluster variance
    let within = 0;
    for (let i = 0; i < n; i++) {
        const c = assignments[i];
        const p = points2d[i];
        if (!p || c < 0 || c >= k) continue;
        const dx = p[0] - cx[c];
        const dy = p[1] - cy[c];
        within += dx * dx + dy * dy;
    }
    within = within / Math.max(n, 1);

    // minimum inter-center distance
    let minInter = Infinity;
    for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) {
            const dx = cx[i] - cx[j];
            const dy = cy[i] - cy[j];
            const d2 = dx * dx + dy * dy;
            if (d2 < minInter) minInter = d2;
        }
    }
    if (!Number.isFinite(minInter)) minInter = 0;

    // Higher better: separation / compactness
    return (minInter + 1e-9) / (within + 1e-9);
}

function scoreConfig(kValues, { weights, allowTrueCost, plan = null }) {
    const T = rawSequences.length;
    let sumInertia = 0;
    let sepScore = 0;

    // Use cached kMeans results when possible.
    const clusters = [];
    for (let t = 0; t < T; t++) {
        const k = kValues[t];
        if (!state.cache[t][k]) state.cache[t][k] = kMeans(rawSequences[t], k);
        const c = state.cache[t][k];
        clusters.push(c);
        sumInertia += c.inertia;

        // "visual" metric from PCA and (if available) kernel PCA.
        const pcaSep = computeEmbeddingSeparationScore(state.pcaPoints?.[t], c.assignments, k);
        let kpcaSep = 0;
        const kp = state.kernelPcaPoints?.[t];
        if (kp && Array.isArray(kp.coords) && Array.isArray(kp.sampledIndices)) {
            // kernel PCA is sometimes sampled; score only over sampled indices.
            const idx = kp.sampledIndices;
            const pts = kp.coords;
            const assn = idx.map(i => c.assignments[i]);
            kpcaSep = computeEmbeddingSeparationScore(pts, assn, k);
        }
        sepScore += pcaSep + 0.6 * kpcaSep;
    }

    const currentConfigStr = kValues.join(',');
    const entry = state.globalHistory[currentConfigStr];
    const trueCost = Number.isFinite(entry?.truePiTrajectoryCost) ? entry.truePiTrajectoryCost : null;

    // Combine objectives:
    // - always include inertia
    // - include true `pis` cost if present
    // - include "visual" separation (higher is better) as a bonus
    // We keep weights mild since magnitudes vary by dataset.
    const inertiaTerm = sumInertia;
    const trueTerm = (Array.isArray(state.truePis) && state.truePis.length > 0 && Number.isFinite(trueCost)) ? trueCost : null;

    let combined =
        (trueTerm !== null && allowTrueCost
            ? weights.wInertia * inertiaTerm + weights.wTrue * trueTerm
            : inertiaTerm)
        - weights.wSep * sepScore;

    if (plan) combined += promptConstraintPenalty(kValues, plan);

    return { combined, sumInertia, trueCost: trueTerm, sepScore, clusters };
}

function tryCandidate(kValues, best, scoreOpts, scoreFn = scoreConfig) {
    const { weights, allowTrueCost, plan } = scoreOpts;
    const cand = applyPromptConstraints(kValues, plan);
    const s = scoreFn(cand, scoreOpts);
    if (s.combined < best.combined) {
        return { kValues: cand.slice(), combined: s.combined, details: s, score: s };
    }
    return { ...best, score: s };
}

function updateExploreTimerUI(start, budgetMs, trials, extra = '') {
    const elapsed = performance.now() - start;
    const remain = Math.max(0, budgetMs - elapsed);
    const pct = Math.min(1, elapsed / budgetMs);
    if (aiExploreProgressEl) aiExploreProgressEl.style.width = `${(pct * 100).toFixed(1)}%`;
    if (aiExploreTimerLabelEl) {
        const extraTxt = extra ? ` · ${extra}` : '';
        aiExploreTimerLabelEl.textContent = `${(remain / 1000).toFixed(1)}s left · ${trials} trials${extraTxt}`;
    }
}

function scrollExploreTargetsIntoView() {
    elements.controls?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function runAiExplore({ durationMs = 10_000, prompt = '' } = {}) {
    if (!rawSequences.length) return;
    if (aiExploreRunning) return;

    aiExploreRunning = true;
    const myToken = ++aiExploreCancelToken;
    aiExploreAbortController = new AbortController();
    const signal = aiExploreAbortController.signal;

    const btn = aiExploreBtnEl || document.querySelector('[data-action="ai-explore"]');
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Stop explore';
    }

    const budgetMs = Math.max(1_000, Math.min(120_000, durationMs));
    const T = rawSequences.length;
    const allowTrueCost = Array.isArray(state.truePis) && state.truePis.length > 0;
    const llmCfg = getLlmSettings();

    const plotFeedbackSnapshot = capturePlotFeedbackSnapshot();
    const annotationSnapshot = captureAnnotationSnapshot();
    clearPlotFeedback();

    const feedbackBlock = formatFeedbackPromptBlock(plotFeedbackSnapshot);
    const annotationBlock = formatAnnotationPromptBlock(annotationSnapshot);
    const locksBlock = formatKLocksPromptBlock(T, state.kValues);
    const userPrompt = (prompt || '').trim();
    const promptTrim = [userPrompt, locksBlock, feedbackBlock, annotationBlock].filter(Boolean).join('\n\n');

    let plan = mergePlotFeedbackIntoPlan(fallbackPlan(userPrompt, T), plotFeedbackSnapshot);
    plan = mergeAnnotationsIntoPlan(plan, annotationSnapshot, T);
    plan = applyKLocksToExplorePlan(plan, T, state.kValues);
    let useLlm = llmCfg.enabled;
    const nAnnTotal = Object.values(annotationSnapshot.byPlotId || {}).reduce((s, a) => s + (a?.length || 0), 0);
    const hasSketchAnnotations = nAnnTotal > 0;
    const useVision = !!llmCfg.visionEnabled && !hasSketchAnnotations;
    let candidateQueue = [];
    let lastLlmReasoning = '';
    let startingK = applyPromptConstraints(state.kValues.slice(), plan);
    const triedConfigs = new Set();
    let best = null;
    let trials = 0;
    let stoppedEarly = false;
    let exploreStart = 0;

    setExploreVisualActive(true);
    elements.status.textContent = 'AI exploring hyperparameters…';

    try {
        const durationSec = Math.round(budgetMs / 1000);
        if (promptTrim) {
            appendAiExploreMessage('user', `${promptTrim} (${durationSec}s run)`);
        } else {
            appendAiExploreMessage('user', `Explore for ${durationSec}s.`);
        }
        const nUp = plotFeedbackSnapshot.ups.length;
        const nDown = plotFeedbackSnapshot.downs.length;
        if (nUp || nDown) {
            appendAiExploreMessage(
                'status',
                `Using your feedback: ${nUp} 👍, ${nDown} 👎 (buttons reset until you rate again).`
            );
        }
        if (hasSketchAnnotations) {
            appendAiExploreMessage(
                'status',
                `Using ${nAnnTotal} drawn cluster region(s) — TOP PRIORITY: each circle = one solid color inside; different circles = different colors. ` +
                'Fitting uses classical geometry on PCA coords (no vision model).'
            );
            if (llmCfg.visionEnabled) {
                appendAiExploreMessage(
                    'status',
                    'Vision model skipped for this run — circle constraints are scored directly from point coordinates.'
                );
            }
        }
        const nLock = state.kLocks?.filter(Boolean).length ?? 0;
        if (nLock) {
            appendAiExploreMessage('status', `${nLock} step(s) have K locked for explore.`);
        }

        setAiExploreMinimized(false);
        setAiExploreWidgetVisible(true);
        scrollExploreTargetsIntoView();

        let pendingLlmInitial = null;
        if (useLlm && !isExploreCancelled(myToken, signal)) {
            setAiExploreLiveStatus(`Calling ${llmCfg.model} in background… (local search runs now)`);
            pendingLlmInitial = llmInitialPlan({
                prompt: promptTrim,
                T,
                currentK: state.kValues,
                allowTrueCost,
                signal
            })
                .then(initial => {
                    if (isExploreCancelled(myToken, signal)) return;
                    lastLlmReasoning = initial.reasoning || '';
                    if (lastLlmReasoning) appendAiExploreMessage('assistant', lastLlmReasoning);
                    mergeLlmWeights(plan, initial.weights);
                    const queued = filterUntriedConfigs(initial.candidates, triedConfigs, plan);
                    candidateQueue.push(...rankCandidatesByHeuristic(queued, plan, best));
                    appendAiExploreMessage('status', `LLM proposed ${queued.length} ranked candidate K settings (elbow-bounded).`);
                })
                .catch(err => {
                    if (isExploreCancelled(myToken, signal)) {
                        appendAiExploreMessage('status', 'Stopped during LLM call.');
                        return;
                    }
                    console.warn('LLM plan failed', err);
                    appendAiExploreMessage('status', `LLM unavailable (${err.message}). Using local parser + random search.`);
                    const msg = String(err?.message || '');
                    const looksLikeQuota =
                        /HTTP\s*(402|403|429|503)/i.test(msg)
                        || /\bquota\b|\brate limit\b|\bneurons?\b|\bexceeded\b/i.test(msg);
                    if (llmCfg?.preset === 'cloudflare_workers_ai' && looksLikeQuota) {
                        saveLlmSettings({ enabled: false });
                        appendAiExploreMessage(
                            'status',
                            'Cloudflare free quota/rate-limit hit — disabling hosted LLM and continuing with classical explore.'
                        );
                    }
                    useLlm = false;
                });
        }

        const planSummary = formatPlanSummary(plan);
        if (planSummary) appendAiExploreMessage('status', `Constraints: ${planSummary}`);

        const weights = plan.weights;
        let lastVisionAssessment = null;
        const scoreOpts = {
            weights,
            allowTrueCost,
            plan,
            visionAssessment: null,
            plotFeedback: plotFeedbackSnapshot,
            annotationSnapshot
        };
        startingK = applyPromptConstraints(state.kValues.slice(), plan);
        warmKMeansCacheForExplore(T, plan, [startingK]);
        candidateQueue.push(...seedExploreCandidates(plan, T, startingK, triedConfigs, null));

        applyKValuesToUI(startingK, plan);
        const startScore = scoreConfigFast(startingK, scoreOpts);
        best = { kValues: startingK.slice(), combined: startScore.combined, details: startScore, score: startScore };
        refreshExploreVisualization(startScore.clusters);

        exploreStart = performance.now();
        updateExploreTimerUI(exploreStart, budgetMs, 0, 'starting…');

        let lastVizAt = 0;
        let lastLlmAt = 0;
        let lastVisionAt = 0;
        let pendingLlmNext = null;
        let pendingVision = null;
        let skippedDuplicates = 0;
        const recentTrials = [];

        while (performance.now() - exploreStart < budgetMs) {
            if (isExploreCancelled(myToken, signal)) {
                stoppedEarly = true;
                break;
            }

            const now = performance.now();
            const remainSec = Math.max(0, (budgetMs - (now - exploreStart)) / 1000);

            if (
                useLlm &&
                remainSec > 1.5 &&
                now - lastLlmAt > 2800 &&
                !pendingLlmNext &&
                !isExploreCancelled(myToken, signal)
            ) {
                lastLlmAt = now;
                setAiExploreLiveStatus(`Asking ${llmCfg.model} in background…`);
                pendingLlmNext = llmNextProposal({
                    prompt: promptTrim,
                    T,
                    currentK: state.kValues,
                    allowTrueCost,
                    best,
                    trials,
                    secondsRemaining: remainSec,
                    recentTrials,
                    signal
                })
                    .then(next => {
                        if (isExploreCancelled(myToken, signal)) return;
                        if (next.reasoning) {
                            appendAiExploreMessage('assistant', next.reasoning);
                            lastLlmReasoning = next.reasoning;
                        }
                        const nextK = applyPromptConstraints(next.kValues, plan);
                        if (!hasTriedConfig(triedConfigs, nextK)) {
                            rememberTriedConfig(triedConfigs, nextK);
                            candidateQueue.unshift(nextK);
                        }
                    })
                    .catch(err => {
                        if (isExploreCancelled(myToken, signal)) return;
                        console.warn('LLM next proposal failed', err);
                        const msg = String(err?.message || '');
                        const looksLikeQuota =
                            /HTTP\s*(402|403|429|503)/i.test(msg)
                            || /\bquota\b|\brate limit\b|\bneurons?\b|\bexceeded\b/i.test(msg);
                        if (useLlm && llmCfg?.preset === 'cloudflare_workers_ai' && looksLikeQuota) {
                            saveLlmSettings({ enabled: false });
                            appendAiExploreMessage(
                                'status',
                                'Cloudflare free quota/rate-limit hit — disabling hosted LLM and continuing with classical explore.'
                            );
                            useLlm = false;
                        }
                    })
                    .finally(() => {
                        pendingLlmNext = null;
                    });
            }

            let cand = null;
            let pickAttempts = 0;
            while (pickAttempts < 16) {
                pickAttempts++;
                if (candidateQueue.length) {
                    cand = dequeueBestCandidate(candidateQueue, plan, best);
                } else {
                    cand = proposeExploreCandidate({
                        plan,
                        best,
                        startingK,
                        triedSet: triedConfigs,
                        T,
                        trials,
                        budgetMs,
                        exploreStart
                    });
                }
                if (!cand) continue;
                if (!hasTriedConfig(triedConfigs, cand)) break;
                skippedDuplicates++;
            }
            if (!cand || hasTriedConfig(triedConfigs, cand)) {
                await new Promise(r => setTimeout(r, 0));
                continue;
            }
            rememberTriedConfig(triedConfigs, cand);

            const prevCombined = best.combined;
            best = tryCandidate(cand, best, scoreOpts, scoreConfigFast);
            warmKMeansCacheForExplore(T, plan, [best.kValues, cand]);
            trials++;
            const s = best.score;
            recentTrials.push({
                k: cand.slice(),
                kKey: kConfigKey(cand),
                combined: s.combined,
                inertia: s.sumInertia,
                sumK: cand.reduce((a, b) => a + b, 0),
                improved: best.combined < prevCombined
            });
            if (recentTrials.length > 12) recentTrials.shift();

            scoreOpts.visionAssessment = lastVisionAssessment;

            const mode = [useLlm && 'LLM', useVision && 'vision', 'search'].filter(Boolean).join('+');
            const statusExtra = `best K=${formatKVector(best.kValues)}`;
            updateExploreTimerUI(exploreStart, budgetMs, trials, statusExtra);
            setAiExploreLiveStatus(`${mode}: trying K=${formatKVector(cand)} · ${statusExtra}`);
            applyKValuesToUI(cand, plan);

            await new Promise(r => requestAnimationFrame(() => r()));

            if (now - lastVizAt > 80) {
                refreshExploreVisualization(s.clusters);
                lastVizAt = performance.now();

                if (
                    useVision &&
                    !pendingVision &&
                    !isExploreCancelled(myToken, signal) &&
                    performance.now() - lastVisionAt > 5000
                ) {
                    const pcaFilter = getPositivePcaSteps(plotFeedbackSnapshot);
                    const clusterDiagnostics = buildTrajectoryClusterDiagnostics(
                        state.pcaPoints,
                        s.clusters,
                        state.kValues
                    );
                    const montage = buildClusterVisionMontage({
                        pcaPointsByStep: state.pcaPoints,
                        clusters: s.clusters,
                        kValues: state.kValues,
                        stepsFilter: pcaFilter.length ? pcaFilter : null,
                        annotationSnapshot
                    });
                    if (montage) {
                        lastVisionAt = performance.now();
                        setAiExploreLiveStatus(`Vision (${llmCfg.visionModel || 'llava'}) rating shapes…`);
                        pendingVision = analyzeClusterMontage({
                            montageDataUrl: montage,
                            T,
                            kValues: state.kValues.slice(),
                            prompt: promptTrim,
                            clusterDiagnostics,
                            annotationSnapshot,
                            settings: llmCfg,
                            signal
                        })
                            .then(assessment => {
                                if (isExploreCancelled(myToken, signal)) return;
                                lastVisionAssessment = assessment;
                                scoreOpts.visionAssessment = assessment;
                                const sil = clusterDiagnostics?.meanSilhouette;
                                const summary = assessment.summary
                                    || `Overall shape ${assessment.overall}/10`;
                                const silTxt = Number.isFinite(sil) ? ` (measured silhouette ${sil.toFixed(2)})` : '';
                                appendAiExploreMessage('assistant', `Vision: ${summary}${silTxt}`);
                            })
                            .catch(err => {
                                if (isExploreCancelled(myToken, signal)) return;
                                console.warn('Vision analysis failed', err);
                                appendAiExploreMessage(
                                    'status',
                                    `Vision unavailable (${err.message}). Using geometry-only shape scores. ` +
                                    'Try: ollama pull llava'
                                );
                            })
                            .finally(() => {
                                pendingVision = null;
                            });
                    }
                }
            }

            await new Promise(r => setTimeout(r, 0));
        }

        if (pendingLlmInitial) {
            try {
                await pendingLlmInitial;
            } catch {
                /* logged in promise */
            }
        }
        if (pendingLlmNext) {
            try {
                await pendingLlmNext;
            } catch {
                /* logged in promise */
            }
        }
        if (pendingVision) {
            try {
                await pendingVision;
            } catch {
                /* logged in promise */
            }
        }

        finalizeAiExploreLiveStatus();

        const cancelled = stoppedEarly || isExploreCancelled(myToken, signal);
        if (best?.kValues && best.details) {
            scoreOpts.visionAssessment = lastVisionAssessment;
            best.details = scoreConfigFast(best.kValues, scoreOpts);
            best.combined = best.details.combined;
            applyKValuesToUI(best.kValues, plan);
            updateVisualization();
            let summary = buildAiExploreSummary({
                kValues: best.kValues,
                details: best.details,
                prompt,
                startingK,
                trials,
                cancelled
            });
            if (lastLlmReasoning && useLlm && !cancelled) {
                summary += `\n\nLLM: ${lastLlmReasoning}`;
            }
            const annReport = formatAnnotationFulfillmentReport(best.details?.annotationFidelity);
            if (annReport) {
                summary += `\n\n${annReport}`;
            }
            appendAiExploreMessage('assistant', summary);
            elements.status.textContent = summary.split('\n')[0];
        } else if (cancelled) {
            appendAiExploreMessage('status', 'Explore stopped.');
            elements.status.textContent = 'Explore stopped.';
        }
    } catch (err) {
        console.error('AI explore failed', err);
        appendAiExploreMessage('status', `Explore failed: ${err.message}`);
        elements.status.textContent = `Explore failed: ${err.message}`;
    } finally {
        setExploreVisualActive(false);
        aiExploreRunning = false;
        aiExploreAbortController = null;
        if (aiExploreProgressEl) aiExploreProgressEl.style.width = '0%';
        if (aiExploreTimerLabelEl) aiExploreTimerLabelEl.textContent = '';
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'AI explore';
        }
    }
}

function getKernelPcaPerplexity() {
    const v = parseFloat(elements.kpcaPerplexitySlider?.value);
    return Number.isFinite(v) ? v : 30;
}

function getDendrogramLinkage() {
    return elements.linkageMethodSelect?.value || 'average';
}

function fitCanvasToDisplay(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return false;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (canvas.width === w && canvas.height === h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
}

function fitMainCanvases() {
    let changed = 0;
    if (elements.canvas) changed |= fitCanvasToDisplay(elements.canvas);
    if (elements.globalInertiaCanvas) changed |= fitCanvasToDisplay(elements.globalInertiaCanvas);
    if (elements.globalTruePiCanvas) changed |= fitCanvasToDisplay(elements.globalTruePiCanvas);
    return changed;
}

let resizeRedrawTimer = null;
function scheduleResizeRedraw() {
    if (!rawSequences.length) return;
    clearTimeout(resizeRedrawTimer);
    resizeRedrawTimer = setTimeout(() => {
        if (fitMainCanvases()) updateVisualization();
    }, 100);
}
window.addEventListener('resize', scheduleResizeRedraw);
new ResizeObserver(scheduleResizeRedraw).observe(elements.vizContainer);

function isDataFileName(name) {
    const lower = (name || '').toLowerCase();
    return lower.endsWith('.npz') || lower.endsWith('.pkl');
}

async function finishLoadingDataset() {
    state.pcaPoints = rawSequences.map(seq => runPCARaw(seq.data, seq.shape[0], seq.shape[1]));
    if (rawSequences.length === 0) throw new Error("Could not find valid sequence arrays.");

    state.cache = rawSequences.map(() => ({}));
    state.otCache = Array(rawSequences.length - 1).fill().map(() => ({}));
    state.globalHistory = {};
    state.pointOtGammaCache = Array(rawSequences.length - 1).fill().map(() => ({}));
    state.pointOtAggCache = Array(rawSequences.length - 1).fill().map(() => ({}));
    analysisJobId++;
    kernelPcaStepsRequested.clear();
    kernelPcaComputingSteps.clear();
    dendrogramStepsRequested.clear();
    dendrogramComputingSteps.clear();
    pca3dStepsRequested.clear();
    pca3dComputingSteps.clear();
    state.kernelPcaPoints = [];
    state.pca3dPoints = [];
    state.dendrograms = [];
    clearPlotAnnotations();
    state.kernelPcaPerplexity = getKernelPcaPerplexity();
    state.dendrogramLinkage = getDendrogramLinkage();
    if (elements.globalTruePiPanel) {
        elements.globalTruePiPanel.style.display = (Array.isArray(state.truePis) && state.truePis.length > 0) ? '' : 'none';
    }

    elements.status.textContent = `Loaded ${rawSequences.length} timesteps. Ready.`;
    setAiExploreWidgetVisible(true);
    initUI();
    fitMainCanvases();
    updateVisualization();

    // Kernel PCA and dendrograms are computed when their section is expanded.

    // Point-level OT only applies to standard OT (non-GW) and epsilon>0.
    // We start it lazily when epsilon becomes > 0.
}

async function loadDataFile(file) {
    if (!file?.name || !isDataFileName(file.name)) {
        elements.status.textContent = "Error: Expected a .npz or .pkl file.";
        return;
    }

    lastLoadedFileName = file.name;
    const lower = file.name.toLowerCase();
    try {
        if (lower.endsWith('.npz')) {
            elements.status.textContent = "Parsing NPY / NPZ data...";
            const zip = await JSZip.loadAsync(file);
            const { sequences, pis } = await extractSequencesFromZip(zip);
            setRawSequences(sequences);
            state.truePis = Array.isArray(pis) ? pis : null;
        } else {
            if (file.size > MAX_PKL_BYTES) {
                throw new Error(
                    `Pickle is ${(file.size / 1e9).toFixed(1)} GB — too large for in-browser loading. ` +
                    'Export a smaller .npz with data/processing.py.'
                );
            }
            elements.status.textContent = "Loading Python runtime & parsing pickle (sparse OK)...";
            setRawSequences(await extractSequencesFromPkl(await file.arrayBuffer()));
            state.truePis = null;
        }
        await finishLoadingDataset();
    } catch (err) {
        elements.status.textContent = "Error: " + err.message;
        console.error(err);
    }
}

function setDataPickerState({ visible, hint, files }) {
    setDataPickerAvailable(visible);
    if (!visible) {
        elements.dataPicker.style.display = 'none';
        return;
    }
    elements.dataPicker.style.display = 'block';
    if (typeof hint === 'string') elements.dataPickerHint.textContent = hint;
    if (Array.isArray(files)) {
        elements.dataFileList.innerHTML = '';
        if (files.length === 0) {
            const empty = document.createElement('div');
            empty.style.color = '#9ca3af';
            empty.style.fontSize = '0.85rem';
            empty.textContent = 'No `.npz` or `.pkl` files found in `./data/`.';
            elements.dataFileList.appendChild(empty);
        } else {
            for (const name of files) {
                const btn = document.createElement('button');
                btn.className = 'data-file-btn';
                btn.type = 'button';
                btn.title = name;
                btn.textContent = name;
                btn.onclick = async () => {
                    try {
                        elements.status.textContent = `Loading data/${name}…`;
                        const resp = await fetch(`data/${encodeURIComponent(name)}`, { cache: 'no-store' });
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        const blob = await resp.blob();
                        const f = new File([blob], name, { type: blob.type || 'application/octet-stream' });
                        await loadDataFile(f);
                    } catch (e) {
                        elements.status.textContent = `Error loading data/${name}: ${e.message}`;
                        console.error(e);
                    }
                };
                elements.dataFileList.appendChild(btn);
            }
        }
    }
}

function dataPickerFilesFromNames(names) {
    return Array.from(new Set(names))
        .filter(name => name && (name.toLowerCase().endsWith('.npz') || name.toLowerCase().endsWith('.pkl')))
        .sort((a, b) => a.localeCompare(b));
}

async function tryPopulateDataPicker() {
    const showFiles = (files, hint = 'Pick a file, or drag & drop any `.npz` / `.pkl`.') => {
        setDataPickerState({ visible: true, hint, files });
    };

    try {
        const resp = await fetch('data/', { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const text = await resp.text();
        const doc = new DOMParser().parseFromString(text, 'text/html');
        const links = Array.from(doc.querySelectorAll('a'))
            .map(a => a.getAttribute('href') || '')
            .filter(href => href && !href.startsWith('?') && !href.startsWith('#'))
            .map(href => href.split('/').filter(Boolean).pop());
        showFiles(dataPickerFilesFromNames(links));
        return;
    } catch { /* no directory listing (e.g. GitHub Pages) */ }

    try {
        const resp = await fetch('data/manifest.json', { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const manifest = await resp.json();
        const names = Array.isArray(manifest.files) ? manifest.files : [];
        showFiles(dataPickerFilesFromNames(names));
    } catch {
        setDataPickerState({ visible: false });
    }
}

// --- UI Logic & Interactions ---
function initUI() {
    elements.controls.style.display = 'flex';
    elements.vizContainer.style.display = 'flex';
    elements.dropZone.style.display = 'none';
    elements.dataPicker.style.display = 'none';

    ensureResizeControl(updateVisualization);
    wireCombinedPlotUi();

    const T = rawSequences.length;
    state.kValues = Array(T).fill(5);
    ensureKLocks(T);
    elements.kContainer.innerHTML = '';
    elements.elbowCanvases = [];

    // Controls specific to clustering count changes.
    const kSliders = [];
    const kLabels = [];

    for (let t = 0; t < T; t++) {
        const group = document.createElement('div');
        group.className = 'control-group';
        const label = document.createElement('label');
        label.dataset.step = String(t);
        label.textContent = `Step ${t + 1} Clusters: 5`;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.dataset.step = String(t);
        slider.min = '1';
        slider.max = '15';
        slider.step = '1';
        slider.value = '5';

        const lockBtn = document.createElement('button');
        lockBtn.type = 'button';
        lockBtn.className = 'k-lock-btn';
        lockBtn.dataset.step = String(t);
        lockBtn.onclick = () => {
            toggleKLock(t, T);
            syncKLockButton(lockBtn, t);
            syncKLockSlider(slider, t);
            if (isStepKLocked(t)) {
                label.textContent = `Step ${t + 1} Clusters: ${state.kValues[t]} (locked)`;
            } else {
                label.textContent = `Step ${t + 1} Clusters: ${state.kValues[t]}`;
            }
        };

        const sweepBtn = document.createElement('button');
        sweepBtn.className = 'sweep-btn';
        sweepBtn.textContent = 'Sweep K';
        sweepBtn.onclick = async () => {
            if (isStepKLocked(t)) return;
            for (let k = 1; k <= 15; k++) {
                slider.value = k;
                state.kValues[t] = k;
                label.textContent = `Step ${t + 1} Clusters: ${k}`;
                updateVisualization();
                await new Promise(r => setTimeout(r, 200));
            }
        };

        const elbowBtn = document.createElement('button');
        elbowBtn.className = 'sweep-btn';
        elbowBtn.textContent = 'Auto-Elbow';
        elbowBtn.onclick = () => {
            if (isStepKLocked(t)) return;
            const elbowK = findElbow(state.cache[t]);
            slider.value = elbowK;
            state.kValues[t] = elbowK;
            label.textContent = `Step ${t + 1} Clusters: ${elbowK}`;
            updateVisualization();
        };

        const elbowCanvas = document.createElement('canvas');
        elbowCanvas.className = 'elbow-canvas';
        elbowCanvas.width = 400;
        elbowCanvas.height = 160;
        elements.elbowCanvases.push(elbowCanvas);

        slider.addEventListener('input', (e) => {
            if (isStepKLocked(t)) {
                e.target.value = String(state.kValues[t]);
                return;
            }
            state.kValues[t] = parseInt(e.target.value, 10);
            label.textContent = `Step ${t + 1} Clusters: ${state.kValues[t]}`;
            requestAnimationFrame(updateVisualization);
        });

        kSliders.push(slider);
        kLabels.push(label);
        syncKLockButton(lockBtn, t);
        syncKLockSlider(slider, t);

        group.appendChild(lockBtn);
        group.appendChild(label);
        group.appendChild(slider);
        group.appendChild(sweepBtn);
        group.appendChild(elbowBtn);
        group.appendChild(wrapPlotCanvas(elbowCanvas, plotIdStep('elbow-local', t + 1)));
        elements.kContainer.appendChild(group);
    }

    ensureCanvasEmphasisWrap(elements.globalInertiaCanvas, 'elbow-global:inertia');
    ensureCanvasEmphasisWrap(elements.globalTruePiCanvas, 'elbow-global:truepi');

    elements.epsSlider.addEventListener('input', (e) => {
        state.epsilon = parseFloat(e.target.value);
        elements.epsLabel.textContent = state.epsilon === 0 ? `Regularization (ε): Exact` : `Regularization (ε): ${state.epsilon.toFixed(3)}`;

        // Reset point-level OT caches whenever epsilon changes.
        const T = rawSequences.length;
        state.pointOtGammaCache = Array(T - 1).fill().map(() => ({}));
        state.pointOtAggCache = Array(T - 1).fill().map(() => ({}));
        state.pointOtEpsilon = state.epsilon;
        if (state.epsilon > 0) {
            startProgressivePointOTGammas().catch(err => console.error(err));
        } else {
            // Cancel any in-flight point OT computation.
            pointOtJobId++;
        }

        requestAnimationFrame(updateVisualization);
    });

    // Kernel PCA + dendrogram controls.
    elements.kpcaPerplexityLabel.textContent = `Kernel PCA perplexity: ${getKernelPcaPerplexity().toFixed(0)}`;
    let kpcaTimer = null;
    let dendroTimer = null;

    elements.kpcaPerplexitySlider.oninput = () => {
        const perp = getKernelPcaPerplexity();
        elements.kpcaPerplexityLabel.textContent = `Kernel PCA perplexity: ${perp.toFixed(0)}`;
        state.kernelPcaPerplexity = perp;
        clearTimeout(kpcaTimer);
        kpcaTimer = setTimeout(() => {
            if (kernelPcaStepsRequested.size === 0) return;
            recomputeKernelPcaRequested().catch(err => console.error(err));
        }, 500);
    };

    elements.linkageMethodSelect.onchange = () => {
        state.dendrogramLinkage = getDendrogramLinkage();
        clearTimeout(dendroTimer);
        dendroTimer = setTimeout(() => {
            if (dendrogramStepsRequested.size === 0) return;
            recomputeDendrogramsRequested().catch(err => console.error(err));
        }, 150);
    };

    // Hover tooltip over edges
    elements.canvas.addEventListener('mousemove', (e) => {
        const rect = elements.canvas.getBoundingClientRect();
        const scaleX = elements.canvas.width / rect.width;
        const scaleY = elements.canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let hoveredEdge = null;
        for (let i = drawnEdgePaths.length - 1; i >= 0; i--) {
            const edge = drawnEdgePaths[i];
            elements.ctx.lineWidth = Math.max(8, edge.renderedWidth);
            if (elements.ctx.isPointInStroke(edge.path, mouseX, mouseY)) {
                hoveredEdge = edge;
                const sourceStep = edge.stepIdx;
                const sourceClusterIdx = edge.sourceIdx;
                hoveredEdge.sourceCount = rawSequences[sourceStep].shape[0];
                hoveredEdge.sourceClusterCount = state.cache[sourceStep][state.kValues[sourceStep]].counts[sourceClusterIdx];
                break;
            }
        }

        if (hoveredEdge) {
            elements.tooltip.style.left = (e.clientX + 15) + 'px';
            elements.tooltip.style.top = (e.clientY + 15) + 'px';
            const transportedCount = Math.round(hoveredEdge.mass * hoveredEdge.sourceClusterCount);
            elements.tooltip.innerHTML = `Transported: <b>${transportedCount}</b> / ${hoveredEdge.sourceCount} points`;
            elements.tooltip.style.display = 'block';
            elements.canvas.style.cursor = 'pointer';
        } else {
            elements.tooltip.style.display = 'none';
            elements.canvas.style.cursor = 'crosshair';
        }
    });

    // Global frontier click/hover (inertia and true-pi plots)
    const wireGlobalPlotInteractions = (canvas, points, { label }) => {
        if (!canvas) return;
        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
            const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
            let nearest = null; let minDist = 20;
            points.forEach(p => {
                const dist = Math.hypot(p.cx - mouseX, p.cy - mouseY);
                if (dist < minDist) { minDist = dist; nearest = p; }
            });
            if (nearest) {
                elements.tooltip.style.display = 'none';
                state.kValues = nearest.configStr.split(',').map(Number);
                const inputs = elements.kContainer.querySelectorAll('input[type=\"range\"]');
                const labels = elements.kContainer.querySelectorAll('label');
                state.kValues.forEach((k, i) => {
                    if (inputs[i]) inputs[i].value = k;
                    if (labels[i]) labels[i].textContent = `Step ${i + 1} Clusters: ${k}`;
                });
                updateVisualization();
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
            const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
            let nearest = null; let minDist = 20;
            points.forEach(p => {
                const dist = Math.hypot(p.cx - mouseX, p.cy - mouseY);
                if (dist < minDist) { minDist = dist; nearest = p; }
            });
            if (nearest) {
                elements.tooltip.style.left = e.pageX + 'px';
                elements.tooltip.style.top = (e.pageY - 20) + 'px';
                elements.tooltip.innerHTML = `Sum K: ${nearest.sumK}<br>${label}: ${nearest.cost.toFixed(0)}<br>Click to set K=[${nearest.configStr}]`;
                elements.tooltip.style.display = 'block';
                canvas.style.cursor = 'pointer';
            } else {
                elements.tooltip.style.display = 'none';
                canvas.style.cursor = 'crosshair';
            }
        });
    };

    wireGlobalPlotInteractions(elements.globalInertiaCanvas, drawnGlobalInertiaPoints, { label: 'Inertia' });
    wireGlobalPlotInteractions(elements.globalTruePiCanvas, drawnGlobalTruePiPoints, { label: 'True OT cost' });
}

function plotIdToSnapshotTitle(plotId) {
    const m = plotId?.match?.(/^(pca|kpca|dendro|elbow-local):(\d+)$/);
    if (!m) return plotId || 'Plot';
    const kind = { pca: 'PCA', kpca: 'Kernel PCA', dendro: 'Dendrogram', 'elbow-local': 'Local elbow' }[m[1]] || m[1];
    return `Step ${m[2]} — ${kind}`;
}

function collectSnapshotCanvases() {
    const items = [];
    const push = (canvas, title) => {
        if (canvas && canvas.width > 1 && canvas.height > 1 && typeof canvas.toDataURL === 'function') {
            items.push({ canvas, title });
        }
    };

    push(document.getElementById('t-partite-canvas'), 'T-Partite Optimal Transport');
    push(document.getElementById('global-inertia-canvas'), 'Global cost frontier (sum of inertias)');

    const truePi = document.getElementById('global-truepi-canvas');
    const truePiPanel = document.getElementById('global-truepi-panel');
    if (truePi && truePiPanel?.offsetParent !== null) {
        push(truePi, 'Global cost frontier (true OT pis cost)');
    }

    document.querySelectorAll('.plot-feedback-wrap[data-plot-id]').forEach(wrap => {
        const canvas = wrap.querySelector('canvas.plot-data-canvas')
            || wrap.querySelector('canvas:not(.plot-annotation-layer)');
        push(canvas, plotIdToSnapshotTitle(wrap.dataset.plotId));
    });

    return items;
}

function addPdfCanvasPage(pdf, { canvas, title }, { margin = 32, header = 18 } = {}) {
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(11);
    pdf.text(title, margin, margin + 4);
    pdf.setFont('helvetica', 'normal');

    const imgData = canvas.toDataURL('image/png');
    const maxW = pageW - margin * 2;
    const maxH = pageH - margin * 2 - header;
    const aspect = canvas.height / canvas.width;
    let w = maxW;
    let h = w * aspect;
    if (h > maxH) {
        h = maxH;
        w = h / aspect;
    }
    const x = margin + (maxW - w) / 2;
    const y = margin + header;
    pdf.addImage(imgData, 'PNG', x, y, w, h);
}

function getJsPDFConstructor() {
    return window.jspdf?.jsPDF
        || window.jsPDF
        || (typeof jspdf !== 'undefined' && jspdf.jsPDF);
}

async function saveSnapshotPdf() {
    const JsPDF = getJsPDFConstructor();
    if (!JsPDF) {
        elements.status.textContent = 'PDF library failed to load — hard refresh the page (Ctrl+Shift+R).';
        return;
    }

    const timestamp = new Date().toISOString();
    const configStr = state.kValues.join(',');
    const gwEdges = [];
    for (let t = 0; t < rawSequences.length - 1; t++) {
        if (rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1]) gwEdges.push(t);
    }

    const meta = {
        version: 'interactive-trajectory-inference',
        timestamp,
        sourceFile: lastLoadedFileName,
        epsilon: state.epsilon,
        plotSize: state.plotSize,
        kValues: state.kValues,
        gwEdges,
        configStr
    };

    const plotItems = collectSnapshotCanvases();
    if (!plotItems.length) {
        elements.status.textContent = 'Nothing to save — load a dataset and wait for plots to render.';
        return;
    }

    const pdf = new JsPDF({ unit: 'pt', format: 'letter', orientation: 'portrait' });
    const margin = 48;
    let y = margin;

    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(17);
    pdf.text('Interactive Trajectory Inference', margin, y);

    // Accent rule
    y += 10;
    pdf.setDrawColor(245, 158, 11); // amber
    pdf.setLineWidth(2);
    pdf.line(margin, y, margin + 170, y);
    y += 18;

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(10);

    // Clean key/value summary (more readable than raw JSON)
    const kv = [
        ['Saved', timestamp],
        ['Dataset', lastLoadedFileName || '(unknown)'],
        ['K per step', `[${configStr}]`],
        ['ε', state.epsilon === 0 ? 'exact OT' : String(state.epsilon)],
        ['GW edges', gwEdges.length ? gwEdges.map(t => `${t + 1}→${t + 2}`).join(', ') : 'none'],
        ['Plot size', `${state.plotSize}px`]
    ];
    pdf.setTextColor(229, 231, 235);
    const keyX = margin;
    const valX = margin + 110;
    for (const [k, v] of kv) {
        pdf.setFont('helvetica', 'bold');
        pdf.text(`${k}:`, keyX, y);
        pdf.setFont('helvetica', 'normal');
        pdf.text(String(v), valX, y);
        y += 14;
    }

    // Add a small first thumbnail (transport map) on cover page
    y += 8;
    const coverThumb = plotItems[0]?.canvas;
    if (coverThumb) {
        const pageW = pdf.internal.pageSize.getWidth();
        const maxW = pageW - margin * 2;
        const thumbW = Math.min(maxW, 520);
        const aspect = coverThumb.height / coverThumb.width;
        const thumbH = thumbW * aspect;
        const x = margin;
        const imgData = coverThumb.toDataURL('image/png');
        pdf.addImage(imgData, 'PNG', x, y, thumbW, thumbH);
        y += thumbH + 10;
    }

    for (const item of plotItems) {
        pdf.addPage('letter', 'landscape');
        addPdfCanvasPage(pdf, item);
    }

    const base = lastLoadedFileName ? lastLoadedFileName.replace(/\.[^.]+$/, '') : 'dataset';
    const fileStamp = timestamp.replace(/[:.]/g, '-');
    pdf.save(`trajectory_saved_${base}_${fileStamp}.pdf`);
    elements.status.textContent = `Saved PDF (${plotItems.length} plot page${plotItems.length === 1 ? '' : 's'}).`;
}

function aggregatePointGammaToClusters(gamma, assignmentsA, assignmentsB, k1, k2) {
    const Pagg = Array.from({ length: k1 }, () => new Float64Array(k2));
    const nA = assignmentsA.length;
    const nB = assignmentsB.length;

    for (let i = 0; i < nA; i++) {
        const a = assignmentsA[i];
        const row = gamma[i];
        const Pa = Pagg[a];
        for (let j = 0; j < nB; j++) {
            const b = assignmentsB[j];
            Pa[b] += row[j];
        }
    }
    return Pagg;
}

function computeCentroidDiscretizationCostFromTruePis(clusters) {
    if (!Array.isArray(state.truePis) || state.truePis.length === 0) return null;
    if (clusters.length < 2) return null;

    let cost = 0;
    const steps = Math.min(state.truePis.length, clusters.length - 1);
    for (let t = 0; t < steps; t++) {
        const pi = state.truePis[t];
        if (!pi?.shape || !pi?.data) continue;

        const [nA, nB] = pi.shape;
        const cA = clusters[t];
        const cB = clusters[t + 1];
        if (!cA?.assignments || !cB?.assignments) continue;
        if (cA.assignments.length !== nA || cB.assignments.length !== nB) continue;

        const k1 = cA.centers.length;
        const k2 = cB.centers.length;
        const Pagg = aggregatePointGammaToClustersAsFlat(pi.data, nA, nB, cA.assignments, cB.assignments, k1, k2);

        // cost_t = sum_{i,j} Pagg_ij * ||center_i - center_j||^2
        const d = cA.centers[0]?.length ?? 0;
        if (d !== (cB.centers[0]?.length ?? -1)) continue;

        for (let i = 0; i < k1; i++) {
            const ci = cA.centers[i];
            for (let j = 0; j < k2; j++) {
                const mass = Pagg[i][j];
                if (mass <= 0) continue;
                const cj = cB.centers[j];
                let dist = 0;
                for (let dim = 0; dim < d; dim++) {
                    const diff = ci[dim] - cj[dim];
                    dist += diff * diff;
                }
                cost += mass * dist;
            }
        }
    }
    return cost;
}

function aggregatePointGammaToClustersAsFlat(gammaFlat, nA, nB, assignmentsA, assignmentsB, k1, k2) {
    const Pagg = Array.from({ length: k1 }, () => new Float64Array(k2));
    for (let i = 0; i < nA; i++) {
        const a = assignmentsA[i];
        const rowOff = i * nB;
        const Pa = Pagg[a];
        for (let j = 0; j < nB; j++) {
            const b = assignmentsB[j];
            Pa[b] += gammaFlat[rowOff + j];
        }
    }
    return Pagg;
}

function updateVisualization() {
    fitMainCanvases();
    state.plotSize = getResizePlotSize(state.plotSize);

    const clusters = rawSequences.map((seq, t) => {
        const k = state.kValues[t];
        if (!state.cache[t][k]) state.cache[t][k] = kMeans(seq, k);
        return state.cache[t][k];
    });

    elements.elbowCanvases.forEach((canvas, t) => drawLocalElbowPlot(canvas, state.cache[t], state.kValues[t]));

    const couplings = [];
    const gwEdges = [];

    for (let t = 0; t < clusters.length - 1; t++) {
        const k1 = state.kValues[t];
        const k2 = state.kValues[t + 1];
        const useGW = rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1];
        if (useGW) gwEdges.push(t);

        if (!useGW && state.epsilon > 0) {
            const gammaKey = `point-sinkhorn-eps-${state.epsilon.toFixed(3)}`;
            const aggKey = `${k1}-${k2}-${gammaKey}`;
            const gamma = state.pointOtGammaCache?.[t]?.[gammaKey];

            if (gamma) {
                if (state.pointOtAggCache?.[t]?.[aggKey] === undefined) {
                    const assignmentsA = clusters[t].assignments;
                    const assignmentsB = clusters[t + 1].assignments;
                    state.pointOtAggCache[t][aggKey] = aggregatePointGammaToClusters(gamma, assignmentsA, assignmentsB, k1, k2);
                }
                couplings.push(state.pointOtAggCache[t][aggKey]);
                continue;
            }
        }

        const pairKey = useGW ? `${k1}-${k2}-gw-${state.epsilon}` : `${k1}-${k2}-ot`;
        if (state.otCache[t][pairKey] === undefined) {
            state.otCache[t][pairKey] = useGW
                ? gromovWasserstein(clusters[t], clusters[t + 1], state.epsilon)
                : exactOT(clusters[t], clusters[t + 1]);
        }

        const result = state.otCache[t][pairKey];
        if (useGW) {
            couplings.push(result.P);
        } else {
            couplings.push(state.epsilon === 0 ? result.P : sinkhorn(clusters[t], clusters[t + 1], state.epsilon));
        }
    }

    const transportTitle = document.getElementById('transport-title');
    if (transportTitle) {
        transportTitle.textContent = gwEdges.length
            ? `T-Partite Optimal Transport (GW for step${gwEdges.length > 1 ? 's' : ''} ${gwEdges.map(t => `${t + 1}→${t + 2}`).join(', ')})`
            : 'T-Partite Optimal Transport';
    }

    if (elements.transportNote) {
        if (gwEdges.length) {
            elements.transportNote.style.display = 'block';
            elements.transportNote.innerHTML =
                `<strong>Gromov–Wasserstein mode.</strong> Using GW for ${gwEdges.map(t => `${t + 1}→${t + 2}`).join(', ')} ` +
                `because adjacent marginals have different feature dimensions. ` +
                `In this mode, each PCA panel is scaled independently.`;
        } else {
            elements.transportNote.style.display = 'none';
        }
    }

    lastClusters = clusters;
    lastIndependentScale = gwEdges.length > 0;

    drawScatterPlots(elements.scatterContainer, state.pcaPoints, clusters, state.plotSize, scatterPlotOptions());
    drawGraph(elements, clusters, couplings, drawnEdgePaths);
    updateGlobalCostFrontiers(clusters);
}

function scatterPlotOptions(overrides = {}) {
    return {
        independentScale: lastIndependentScale,
        kernelPcaPointsByStep: state.kernelPcaPoints,
        kernelPcaRequestedSteps: kernelPcaStepsRequested,
        dendrogramsByStep: state.dendrograms,
        dendrogramRequestedSteps: dendrogramStepsRequested,
        pca3dPointsByStep: state.pca3dPoints,
        pca3dRequestedSteps: pca3dStepsRequested,
        ...overrides
    };
}

function redrawScatterIfReady() {
    if (!lastClusters?.length) return;
    drawScatterPlots(
        elements.scatterContainer,
        state.pcaPoints,
        lastClusters,
        state.plotSize,
        scatterPlotOptions()
    );
    requestAnimationFrame(() => refreshAnnotationLayers(elements.scatterContainer));
}

function ensureKernelPcaArray() {
    const T = rawSequences.length;
    if (!Array.isArray(state.kernelPcaPoints) || state.kernelPcaPoints.length !== T) {
        state.kernelPcaPoints = Array(T).fill(null);
    }
}

function ensurePca3dArray() {
    const T = rawSequences.length;
    if (!Array.isArray(state.pca3dPoints) || state.pca3dPoints.length !== T) {
        state.pca3dPoints = Array(T).fill(null);
    }
}

async function ensurePca3dStep(t) {
    const T = rawSequences.length;
    if (!T || t < 0 || t >= T) return;

    ensurePca3dArray();
    pca3dStepsRequested.add(t);
    if (state.pca3dPoints[t]?.length || pca3dComputingSteps.has(t)) return;

    pca3dComputingSteps.add(t);
    state.computingPca3d = true;
    redrawScatterIfReady();

    try {
        elements.status.textContent = `Computing 3D PCA… step ${t + 1}/${T}`;
        const seq = rawSequences[t];
        state.pca3dPoints[t] = runPCARaw3D(seq.data, seq.shape[0], seq.shape[1]);
        redrawScatterIfReady();
    } finally {
        pca3dComputingSteps.delete(t);
        state.computingPca3d = pca3dComputingSteps.size > 0;
    }
}

async function ensureKernelPcaStep(t) {
    const T = rawSequences.length;
    if (!T || t < 0 || t >= T) return;

    ensureKernelPcaArray();
    kernelPcaStepsRequested.add(t);
    if (state.kernelPcaPoints[t] || kernelPcaComputingSteps.has(t)) return;

    kernelPcaComputingSteps.add(t);
    state.computingKernelPca = true;
    redrawScatterIfReady();

    try {
        elements.status.textContent =
            `Computing kernel PCA… step ${t + 1}/${T} (perplexity=${getKernelPcaPerplexity().toFixed(0)})`;
        const seq = rawSequences[t];
        state.kernelPcaPoints[t] = await computeKernelPcaProjection(
            seq.data,
            seq.shape[0],
            seq.shape[1],
            state.kernelPcaPerplexity,
            { yieldEvery: 1, maxPoints: KERNEL_PCA_MAX_POINTS }
        );
        redrawScatterIfReady();
    } finally {
        kernelPcaComputingSteps.delete(t);
        state.computingKernelPca = kernelPcaComputingSteps.size > 0;
    }
}

async function recomputeKernelPcaRequested() {
    const steps = [...kernelPcaStepsRequested].sort((a, b) => a - b);
    if (!steps.length) return;

    const jobId = ++analysisJobId;
    clearKernelPcaPlotAnnotations();
    ensureKernelPcaArray();
    for (const t of steps) state.kernelPcaPoints[t] = null;

    state.computingKernelPca = true;
    redrawScatterIfReady();
    elements.status.textContent =
        `Computing kernel PCA (perplexity=${getKernelPcaPerplexity().toFixed(0)})… kernel PCA circles cleared.`;

    try {
        for (const t of steps) {
            if (jobId !== analysisJobId) return;
            kernelPcaComputingSteps.add(t);
            elements.status.textContent = `Computing kernel PCA… step ${t + 1}/${rawSequences.length}`;
            const seq = rawSequences[t];
            state.kernelPcaPoints[t] = await computeKernelPcaProjection(
                seq.data,
                seq.shape[0],
                seq.shape[1],
                state.kernelPcaPerplexity,
                { yieldEvery: 1, maxPoints: KERNEL_PCA_MAX_POINTS }
            );
            kernelPcaComputingSteps.delete(t);
            redrawScatterIfReady();
        }
    } finally {
        kernelPcaComputingSteps.clear();
        state.computingKernelPca = false;
    }
}

function ensureDendrogramArray() {
    const T = rawSequences.length;
    if (!Array.isArray(state.dendrograms) || state.dendrograms.length !== T) {
        state.dendrograms = Array(T).fill(null);
    }
}

async function ensureDendrogramStep(t) {
    const T = rawSequences.length;
    if (!T || t < 0 || t >= T) return;

    ensureDendrogramArray();
    dendrogramStepsRequested.add(t);
    const existing = state.dendrograms[t];
    if (existing?.nLeaves || dendrogramComputingSteps.has(t)) return;

    dendrogramComputingSteps.add(t);
    state.computingDendrograms = true;
    redrawScatterIfReady();

    try {
        elements.status.textContent =
            `Computing dendrogram… step ${t + 1}/${T} (${getDendrogramLinkage()})`;
        const seq = rawSequences[t];
        state.dendrograms[t] = await computeDendrogram(
            seq.data,
            seq.shape[0],
            seq.shape[1],
            state.dendrogramLinkage,
            { yieldEvery: 1, maxPoints: Infinity }
        );
        redrawScatterIfReady();
    } finally {
        dendrogramComputingSteps.delete(t);
        state.computingDendrograms = dendrogramComputingSteps.size > 0;
    }
}

async function recomputeDendrogramsRequested() {
    const steps = [...dendrogramStepsRequested].sort((a, b) => a - b);
    if (!steps.length) return;

    const jobId = ++analysisJobId;
    ensureDendrogramArray();
    for (const t of steps) state.dendrograms[t] = null;

    state.computingDendrograms = true;
    redrawScatterIfReady();
    elements.status.textContent = `Computing dendrograms (${getDendrogramLinkage()})…`;

    try {
        for (const t of steps) {
            if (jobId !== analysisJobId) return;
            dendrogramComputingSteps.add(t);
            elements.status.textContent =
                `Computing dendrograms (${getDendrogramLinkage()})… step ${t + 1}/${rawSequences.length}`;
            const seq = rawSequences[t];
            state.dendrograms[t] = await computeDendrogram(
                seq.data,
                seq.shape[0],
                seq.shape[1],
                state.dendrogramLinkage,
                { yieldEvery: 1, maxPoints: Infinity }
            );
            dendrogramComputingSteps.delete(t);
            redrawScatterIfReady();
        }
    } finally {
        dendrogramComputingSteps.clear();
        state.computingDendrograms = false;
    }
}

setPlotSectionExpandHandler((sectionKey, stepIndex) => {
    if (stepIndex == null) return;
    if (sectionKey === 'pca3d') {
        ensurePca3dStep(stepIndex).catch(err => {
            console.error(err);
            elements.status.textContent = `3D PCA error: ${err.message}`;
        });
    } else if (sectionKey === 'kpca') {
        ensureKernelPcaStep(stepIndex).catch(err => {
            console.error(err);
            elements.status.textContent = `Kernel PCA error: ${err.message}`;
        });
    } else if (sectionKey === 'dendro') {
        ensureDendrogramStep(stepIndex).catch(err => {
            console.error(err);
            elements.status.textContent = `Dendrogram error: ${err.message}`;
        });
    }
});

function pointGammaKeyForEpsilon(eps) {
    return `point-sinkhorn-eps-${eps.toFixed(3)}`;
}

async function startProgressivePointOTGammas() {
    const T = rawSequences.length;
    if (!T) return;
    if (!rawSequences.length) return;
    if (!(state.epsilon > 0)) return; // only entropic Sinkhorn

    const jobId = ++pointOtJobId;
    state.computingPointOt = true;
    elements.status.textContent = `Computing point-level Sinkhorn OT… (ε=${state.epsilon.toFixed(3)})`;

    for (let t = 0; t < T - 1; t++) {
        if (jobId !== pointOtJobId) return; // superseded

        const seqA = rawSequences[t];
        const seqB = rawSequences[t + 1];
        const useGW = seqA.shape[1] !== seqB.shape[1];
        if (useGW) continue; // point-level sinkhorn not defined across mismatched feature dims

        const gammaKey = pointGammaKeyForEpsilon(state.epsilon);
        if (state.pointOtGammaCache?.[t]?.[gammaKey]) continue;

        elements.status.textContent =
            `Computing point-level Sinkhorn OT… step ${t + 1}/${T - 1} (ε=${state.epsilon.toFixed(3)})`;

        try {
            const { P } = sinkhornPointwise(
                seqA.data,
                seqA.shape[0],
                seqA.shape[1],
                seqB.data,
                seqB.shape[0],
                seqB.shape[1],
                state.epsilon,
                30,
                { maxEntries: 160000 }
            );
            state.pointOtGammaCache[t][gammaKey] = P;
        } catch (err) {
            console.warn('Point-level OT skipped for step pair', t, err);
            elements.status.textContent = `Point-level OT skipped for step ${t + 1} (too large). Using cluster OT.`;
            continue;
        }

        requestAnimationFrame(updateVisualization);
        await new Promise(r => setTimeout(r, 0));
    }

    state.computingPointOt = false;
}

// Wire "Save snapshot" button once at startup.
(function wireSaveButtonOnce() {
    const existing = elements.controls.querySelector('[data-action="save-snapshot"]');
    if (existing) {
        existing.textContent = 'Save snapshot (PDF)';
        existing.onclick = saveSnapshotPdf;
        return;
    }
    const btn = document.createElement('button');
    btn.className = 'sweep-btn save-snapshot-btn';
    btn.type = 'button';
    btn.textContent = 'Save PDF';
    btn.dataset.action = 'save-snapshot';
    btn.onclick = saveSnapshotPdf;
    elements.controls.appendChild(btn);
})();

// --- Drag & Drop + Home ---
let dragCounter = 0;
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
    elements.dropZone.addEventListener(e, ev => ev.preventDefault(), false);
    document.body.addEventListener(e, ev => ev.preventDefault(), false);
});
elements.dropZone.addEventListener('dragenter', () => { dragCounter++; elements.dropZone.classList.add('dragover'); });
elements.dropZone.addEventListener('dragleave', () => { dragCounter--; if (dragCounter === 0) elements.dropZone.classList.remove('dragover'); });
elements.dropZone.addEventListener('drop', async (e) => {
    dragCounter = 0; elements.dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    await loadDataFile(file);
});

elements.backHomeBtn.addEventListener('click', () => {
    requestStopAiExplore();
    aiExploreRunning = false;
    setAiExploreWidgetVisible(false);
    clearAiExploreTranscript();
    setAiExploreMinimized(false);
    showHomePage(dataPickerAvailable);
});

tryPopulateDataPicker();

function bindAiExploreWidgetRefs(widget) {
    aiExploreWidgetEl = widget;
    aiExploreBtnEl = widget.querySelector('[data-action="ai-explore"]');
    aiExploreDurationEl = widget.querySelector('.ai-explore-duration');
    aiExplorePromptEl = widget.querySelector('.ai-explore-prompt');
    aiExploreTranscriptEl = widget.querySelector('.ai-explore-transcript');
    aiExploreTimerLabelEl = widget.querySelector('.ai-explore-timer-label');
    aiExploreProgressEl = widget.querySelector('.ai-explore-progress > div');
    aiExploreMinimizeBtnEl = widget.querySelector('.ai-explore-minimize-btn');
}

// --- AI explore widget (idempotent mount) ---
(function mountAiExploreWidget() {
    let widget = document.getElementById('ai-explore-widget');
    if (widget) {
        bindAiExploreWidgetRefs(widget);
        return;
    }

    widget = document.createElement('div');
    widget.className = 'ai-explore-widget';
    widget.id = 'ai-explore-widget';

    const header = document.createElement('div');
    header.className = 'ai-explore-header';

    const headerTitle = document.createElement('span');
    headerTitle.textContent = 'AI explore';
    headerTitle.style.fontSize = '0.8rem';
    headerTitle.style.color = '#fcd34d';
    headerTitle.style.fontWeight = '600';

    const minimizeBtn = document.createElement('button');
    minimizeBtn.type = 'button';
    minimizeBtn.className = 'ai-explore-minimize-btn';
    minimizeBtn.textContent = '−';
    minimizeBtn.title = 'Minimize AI explore panel';
    minimizeBtn.onclick = () => setAiExploreMinimized(!aiExploreMinimized);

    header.appendChild(headerTitle);
    header.appendChild(minimizeBtn);

    const body = document.createElement('div');
    body.className = 'ai-explore-body';

    const row = document.createElement('div');
    row.className = 'ai-explore-row';

    const duration = document.createElement('select');
    duration.className = 'ai-explore-duration';
    duration.title = 'How long to explore';
    const opts = [
        { s: 5, t: '5s' },
        { s: 10, t: '10s' },
        { s: 20, t: '20s' },
        { s: 30, t: '30s' }
    ];
    for (const o of opts) {
        const opt = document.createElement('option');
        opt.value = String(o.s);
        opt.textContent = o.t;
        if (o.s === 10) opt.selected = true;
        duration.appendChild(opt);
    }

    const btn = document.createElement('button');
    btn.className = 'ai-explore-btn';
    btn.type = 'button';
    btn.dataset.action = 'ai-explore';
    btn.textContent = 'AI explore';
    btn.title = 'Try many K settings for the chosen time and apply the best.';

    row.appendChild(duration);
    row.appendChild(btn);

    const prompt = document.createElement('textarea');
    prompt.className = 'ai-explore-prompt';
    prompt.placeholder = '👍/👎 plots; drag circles on PCA/kernel PCA; click ! for must-match; 🔒 locks K per step. Explore consumes drawings & thumbs, then resets them.';

    const llmDetails = document.createElement('details');
    llmDetails.className = 'ai-explore-llm-settings';
    llmDetails.open = false;
    const llmSummary = document.createElement('summary');
    llmSummary.textContent = 'LLM (Ollama / Qwen / Llama)';
    llmDetails.appendChild(llmSummary);

    const cfg = getLlmSettings();
    const llmEnabled = document.createElement('input');
    llmEnabled.type = 'checkbox';
    llmEnabled.checked = cfg.enabled;
    llmEnabled.id = 'ai-explore-llm-enabled';
    const llmEnabledLabel = document.createElement('label');
    llmEnabledLabel.htmlFor = 'ai-explore-llm-enabled';
    llmEnabledLabel.textContent = 'Use text LLM';
    llmEnabledLabel.style.fontSize = '0.78rem';
    llmEnabledLabel.style.color = '#d1d5db';

    const visionEnabled = document.createElement('input');
    visionEnabled.type = 'checkbox';
    visionEnabled.checked = !!cfg.visionEnabled;
    visionEnabled.id = 'ai-explore-vision-enabled';
    const visionEnabledLabel = document.createElement('label');
    visionEnabledLabel.htmlFor = 'ai-explore-vision-enabled';
    visionEnabledLabel.textContent = 'Use vision on PCA plots (off when circles drawn)';
    visionEnabledLabel.style.fontSize = '0.78rem';
    visionEnabledLabel.style.color = '#d1d5db';

    const visionModelIn = document.createElement('input');
    visionModelIn.className = 'ai-explore-llm-input';
    visionModelIn.type = 'text';
    visionModelIn.placeholder = 'Vision model (ollama pull llava)';
    visionModelIn.value = cfg.visionModel || 'llava';

    const presetSel = document.createElement('select');
    presetSel.className = 'ai-explore-duration';
    presetSel.style.width = '100%';
    listPresets().forEach(p => {
        const o = document.createElement('option');
        o.value = p.id;
        o.textContent = p.label;
        if (p.id === cfg.preset) o.selected = true;
        presetSel.appendChild(o);
    });

    const baseUrl = document.createElement('input');
    baseUrl.className = 'ai-explore-llm-input';
    baseUrl.type = 'text';
    baseUrl.placeholder = 'API base URL';
    baseUrl.value = cfg.baseUrl;

    const modelIn = document.createElement('input');
    modelIn.className = 'ai-explore-llm-input';
    modelIn.type = 'text';
    modelIn.placeholder = 'Model name';
    modelIn.value = cfg.model;

    const apiKey = document.createElement('input');
    apiKey.className = 'ai-explore-llm-input';
    apiKey.type = 'password';
    apiKey.placeholder = 'API key (optional, for cloud)';
    apiKey.value = cfg.apiKey || '';

    const testBtn = document.createElement('button');
    testBtn.type = 'button';
    testBtn.className = 'sweep-btn';
    testBtn.textContent = 'Test connection';
    testBtn.onclick = async () => {
        saveLlmSettings({
            enabled: llmEnabled.checked,
            visionEnabled: visionEnabled.checked,
            preset: presetSel.value,
            baseUrl: baseUrl.value.trim(),
            model: modelIn.value.trim(),
            visionModel: visionModelIn.value.trim(),
            apiKey: apiKey.value.trim()
        });
        testBtn.disabled = true;
        try {
            const ok = await testLlmConnection(getLlmSettings());
            appendAiExploreMessage('status', ok ? `LLM OK (${modelIn.value}).` : 'LLM responded but ping format unexpected.');
        } catch (e) {
            appendAiExploreMessage('status', `LLM test failed: ${e.message}`);
        }
        testBtn.disabled = false;
    };

    presetSel.onchange = () => {
        const applied = applyPreset(presetSel.value);
        baseUrl.value = applied.baseUrl;
        modelIn.value = applied.model;
        if (applied.visionModel) visionModelIn.value = applied.visionModel;
    };

    const persistLlm = () => saveLlmSettings({
        enabled: llmEnabled.checked,
        visionEnabled: visionEnabled.checked,
        preset: presetSel.value,
        baseUrl: baseUrl.value.trim(),
        model: modelIn.value.trim(),
        visionModel: visionModelIn.value.trim(),
        apiKey: apiKey.value.trim()
    });
    llmEnabled.onchange = persistLlm;
    visionEnabled.onchange = persistLlm;
    baseUrl.onchange = persistLlm;
    modelIn.onchange = persistLlm;
    visionModelIn.onchange = persistLlm;
    apiKey.onchange = persistLlm;

    const llmRow = document.createElement('div');
    llmRow.className = 'ai-explore-row';
    llmRow.appendChild(llmEnabled);
    llmRow.appendChild(llmEnabledLabel);

    const visionRow = document.createElement('div');
    visionRow.className = 'ai-explore-row';
    visionRow.appendChild(visionEnabled);
    visionRow.appendChild(visionEnabledLabel);

    llmDetails.appendChild(llmRow);
    llmDetails.appendChild(visionRow);
    llmDetails.appendChild(visionModelIn);
    llmDetails.appendChild(presetSel);
    llmDetails.appendChild(baseUrl);
    llmDetails.appendChild(modelIn);
    llmDetails.appendChild(apiKey);
    llmDetails.appendChild(testBtn);

    const timerLabel = document.createElement('div');
    timerLabel.className = 'ai-explore-timer-label';
    timerLabel.textContent = '';

    const progWrap = document.createElement('div');
    progWrap.className = 'ai-explore-progress';
    const prog = document.createElement('div');
    progWrap.appendChild(prog);

    const transcript = document.createElement('div');
    transcript.className = 'ai-explore-transcript';

    body.appendChild(row);
    body.appendChild(timerLabel);
    body.appendChild(progWrap);
    body.appendChild(prompt);
    body.appendChild(llmDetails);
    body.appendChild(transcript);

    widget.appendChild(header);
    widget.appendChild(body);

    btn.onclick = async () => {
        if (aiExploreRunning) {
            requestStopAiExplore();
            return;
        }
        if (!rawSequences.length) {
            appendAiExploreMessage('status', 'Load a dataset first.');
            return;
        }
        saveLlmSettings({
            enabled: llmEnabled.checked,
            visionEnabled: visionEnabled.checked,
            preset: presetSel.value,
            baseUrl: baseUrl.value.trim(),
            model: modelIn.value.trim(),
            visionModel: visionModelIn.value.trim(),
            apiKey: apiKey.value.trim()
        });
        const seconds = parseFloat(duration.value);
        const durationMs = (Number.isFinite(seconds) ? seconds : 10) * 1000;
        elements.status.textContent = 'AI exploring hyperparameters…';
        await runAiExplore({ durationMs, prompt: prompt.value });
    };

    document.body.appendChild(widget);
    bindAiExploreWidgetRefs(widget);
})();

