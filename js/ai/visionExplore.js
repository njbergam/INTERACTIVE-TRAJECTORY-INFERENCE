import { visionCompletion, parseJsonFromLlm } from './llmClient.js';
import { drawPcaClusterVisionPlot } from '../viz/clusterVisionPlot.js';
import { formatClusterDiagnosticsForPrompt } from './shapeStats.js';
import { getAnnotationsForStep, formatAnnotationPromptBlock, drawAnnotationsOnCtx } from '../ui/plotAnnotations.js';

const PANEL = 240;
const LABEL_H = 22;

/**
 * Build a high-contrast montage from PCA coords + cluster assignments (not a tiny UI screenshot).
 */
export function buildClusterVisionMontage({
    pcaPointsByStep,
    clusters,
    kValues,
    stepsFilter = null,
    annotationSnapshot = null
}) {
    if (!pcaPointsByStep?.length || !clusters?.length) return null;

    const slices = [];
    for (let t = 0; t < clusters.length; t++) {
        const step = t + 1;
        if (Array.isArray(stepsFilter) && stepsFilter.length > 0 && !stepsFilter.includes(step)) continue;
        if (!pcaPointsByStep[t]?.length) continue;
        slices.push({ step, t });
    }
    if (!slices.length) return null;

    const cols = slices.length;
    const off = document.createElement('canvas');
    off.width = PANEL * cols;
    off.height = PANEL + LABEL_H;
    const ctx = off.getContext('2d');
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, off.width, off.height);

    for (let i = 0; i < slices.length; i++) {
        const { step, t } = slices[i];
        const tile = document.createElement('canvas');
        tile.width = PANEL;
        tile.height = PANEL;
        const tctx = tile.getContext('2d');
        drawPcaClusterVisionPlot(tctx, PANEL, PANEL, pcaPointsByStep[t], clusters[t]);
        const userEllipses = getAnnotationsForStep(annotationSnapshot, step);
        drawAnnotationsOnCtx(tctx, PANEL, PANEL, userEllipses);

        const x0 = i * PANEL;
        ctx.drawImage(tile, x0, 0);
        const k = kValues[t] ?? clusters[t].centers?.length ?? '?';
        ctx.fillStyle = '#fde68a';
        ctx.font = 'bold 12px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Step ${step}  K=${k}`, x0 + 8, PANEL + 16);
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('numbered centers = cluster id', x0 + 8, PANEL + 6);
    }

    return off.toDataURL('image/png');
}

/** Legacy: screenshot existing UI canvases (lower quality for vision). */
export function capturePcaMontage(scatterContainer, { stepsFilter = null } = {}) {
    if (!scatterContainer) return null;

    const panels = [...scatterContainer.querySelectorAll('.scatter-panel')];
    const slices = panels
        .map((panel, idx) => {
            const step = idx + 1;
            if (Array.isArray(stepsFilter) && stepsFilter.length > 0 && !stepsFilter.includes(step)) {
                return null;
            }
            const wrap = panel.querySelector('.plot-feedback-wrap[data-plot-id^="pca:"]');
            const canvas = wrap?.querySelector('canvas') || panel.querySelector('canvas');
            if (!canvas?.width || canvas.width < 2) return null;
            return { canvas, step };
        })
        .filter(Boolean);

    if (!slices.length) return null;

    const cols = slices.length;
    const off = document.createElement('canvas');
    off.width = PANEL * cols;
    off.height = PANEL + LABEL_H;
    const ctx = off.getContext('2d');
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, off.width, off.height);

    for (let i = 0; i < slices.length; i++) {
        const { canvas, step } = slices[i];
        const scale = Math.min(PANEL / canvas.width, (PANEL - 4) / canvas.height);
        const w = canvas.width * scale;
        const h = canvas.height * scale;
        const x = i * PANEL + (PANEL - w) / 2;
        const y = 2 + (PANEL - 4 - h) / 2;
        ctx.drawImage(canvas, x, y, w, h);
        ctx.fillStyle = '#fde68a';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.fillText(`Step ${step}`, i * PANEL + 6, PANEL + 16);
    }

    return off.toDataURL('image/png');
}

function normalizeAssessment(json, T, diagnostics = null) {
    if (!json || typeof json !== 'object') return null;
    const overall = clampNum(json.overall ?? json.overall_score ?? json.quality, 1, 10);
    const stepsRaw = json.steps || json.timesteps || [];
    const steps = [];
    for (let i = 0; i < T; i++) {
        const row = stepsRaw.find(s => s.step === i + 1) || stepsRaw[i] || {};
        const diag = diagnostics?.steps?.[i];
        const blobs = row.blobs ?? row.k_visible ?? row.clusters ?? diag?.nonemptyClusters ?? null;
        steps.push({
            step: i + 1,
            blobs,
            K: diag?.K ?? null,
            silhouette: diag?.silhouette ?? null,
            separated: !!row.separated,
            compact: !!(row.compact ?? row.compactness),
            quality: clampNum(row.quality ?? row.shape_quality ?? row.score, 1, 10),
            note: String(row.note || row.notes || '').slice(0, 120)
        });
    }
    const stepMean = steps.length
        ? steps.reduce((s, x) => s + (Number.isFinite(x.quality) ? x.quality : overall), 0) / steps.length
        : overall;
    return {
        overall: Number.isFinite(overall) ? overall : stepMean,
        steps,
        summary: String(json.summary || json.reasoning || '').slice(0, 280)
    };
}

function clampNum(v, lo, hi) {
    const x = Number(v);
    return Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : NaN;
}

/**
 * Vision + measured PCA cluster diagnostics (hybrid — works even with weak vision models).
 */
export async function analyzeClusterMontage({
    montageDataUrl,
    T,
    kValues,
    prompt = '',
    clusterDiagnostics = null,
    annotationSnapshot = null,
    settings = null,
    signal = null
}) {
    const kStr = JSON.stringify(kValues);
    const diagBlock = formatClusterDiagnosticsForPrompt(clusterDiagnostics);
    const annBlock = formatAnnotationPromptBlock(annotationSnapshot);

    const userText = `You are rating trajectory clustering quality from PCA plots (left→right = time).

IMAGE KEY:
- Each colored region is one cluster; white circles with numbers are cluster centroids (0,1,2,...).
- Convex hull outlines show cluster extent.
- YELLOW DASHED circles = MANDATORY user sketches: ≥85% of points inside each circle must be ONE cluster color; each circle must use a DIFFERENT color than other circles on that plot. Rate harshly if violated.
- Legend shows cluster_id:point_count.

Algorithm K per step: ${kStr}
${diagBlock ? `\n${diagBlock}\n` : ''}
${annBlock ? `\n${annBlock}\n` : ''}
User goal: ${prompt || 'clusters should match visible separated blobs; K should match number of distinct groups'}

Count DISTINCT visual groups per panel. Honor user yellow circles. Compare blob count to K and silhouette.

Return JSON only:
{"overall":1-10,"steps":[{"step":1,"blobs":<int visible groups>,"matches_K":true/false,"separated":true/false,"compact":true/false,"quality":1-10,"note":"short"}],"summary":"one sentence"}

10 = clear blobs, blob count matches K, high silhouette; 1 = overlap or wrong K.`;

    const content = await visionCompletion({
        text: userText,
        imageDataUrl: montageDataUrl,
        settings,
        signal,
        timeoutMs: 120_000
    });

    const json = parseJsonFromLlm(content);
    const assessment = normalizeAssessment(json, T, clusterDiagnostics);
    if (!assessment) {
        return {
            overall: 5,
            steps: [],
            summary: content.slice(0, 200),
            raw: content,
            diagnostics: clusterDiagnostics
        };
    }
    assessment.raw = content;
    assessment.diagnostics = clusterDiagnostics;
    return assessment;
}

/** Blend vision rating with measured silhouettes and K/blob agreement. */
export function visionQualityToScore(assessment, diagnostics = null) {
    if (!assessment) return 0;

    const diag = diagnostics || assessment.diagnostics;
    let visionPart = Number.isFinite(assessment.overall) ? assessment.overall / 10 : 0.5;

    if (!diag?.steps?.length) return visionPart;

    let silSum = 0;
    let agree = 0;
    for (let i = 0; i < diag.steps.length; i++) {
        const d = diag.steps[i];
        silSum += Math.max(0, Math.min(1, (d.silhouette + 0.2) / 1.2));

        const row = assessment.steps?.[i];
        const blobs = row?.blobs;
        if (Number.isFinite(blobs) && Number.isFinite(d.K)) {
            if (Math.abs(blobs - d.nonemptyClusters) <= 1 || Math.abs(blobs - d.K) <= 1) agree++;
        }
    }
    const silPart = silSum / diag.steps.length;
    const agreePart = agree / diag.steps.length;

    return 0.35 * visionPart + 0.45 * silPart + 0.2 * agreePart;
}
