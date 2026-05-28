import { clusterColorHex } from './plotColors.js';
import { wrapPlotCanvas, plotIdStep } from '../ui/plotFeedback.js';
import { refreshAnnotationLayers } from '../ui/plotAnnotations.js';
import {
    buildPca3DSectionBody,
    disposePca3DInContainer,
    syncPca3DPlotsInContainer
} from './pca3d.js';

/** Default collapsed state when no per-step override exists. */
const plotSectionDefaultCollapsed = {
    pca: false,
    pca3d: true,
    kpca: true,
    dendro: true
};

/** Per-step collapse state: keys like `kpca:3`. Persists across scatter re-draws. */
export const plotSectionCollapsed = {};

function plotSectionStateKey(sectionKey, stepIndex) {
    return stepIndex != null ? `${sectionKey}:${stepIndex}` : sectionKey;
}

function isPlotSectionCollapsed(sectionKey, stepIndex) {
    const key = plotSectionStateKey(sectionKey, stepIndex);
    if (Object.prototype.hasOwnProperty.call(plotSectionCollapsed, key)) {
        return !!plotSectionCollapsed[key];
    }
    return !!plotSectionDefaultCollapsed[sectionKey];
}

/** Called when a collapsible plot section is expanded: `(sectionKey, stepIndex) => void`. */
let onPlotSectionExpand = null;

export function setPlotSectionExpandHandler(handler) {
    onPlotSectionExpand = typeof handler === 'function' ? handler : null;
}

function appendCollapsibleSection(panel, sectionKey, titleText, buildContent, stepIndex = null) {
    const collapsed = isPlotSectionCollapsed(sectionKey, stepIndex);
    const section = document.createElement('div');
    section.className = `plot-section plot-section--${sectionKey}`;
    if (collapsed) section.classList.add('plot-section--collapsed');

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'plot-section-toggle';
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');

    const chevron = document.createElement('span');
    chevron.className = 'plot-section-chevron';
    chevron.textContent = collapsed ? '▶' : '▼';
    const title = document.createElement('span');
    title.className = 'plot-section-title';
    title.textContent = titleText;
    toggle.append(chevron, title);

    const body = document.createElement('div');
    body.className = 'plot-section-body';
    if (collapsed) body.hidden = true;

    toggle.addEventListener('click', () => {
        const wasCollapsed = body.hidden;
        const nowCollapsed = !wasCollapsed;
        body.hidden = nowCollapsed;
        plotSectionCollapsed[plotSectionStateKey(sectionKey, stepIndex)] = nowCollapsed;
        section.classList.toggle('plot-section--collapsed', nowCollapsed);
        chevron.textContent = nowCollapsed ? '▶' : '▼';
        toggle.setAttribute('aria-expanded', nowCollapsed ? 'false' : 'true');
        if (!nowCollapsed && wasCollapsed) {
            onPlotSectionExpand?.(sectionKey, stepIndex);
        }
    });

    buildContent(body);
    section.append(toggle, body);
    panel.appendChild(section);
}

export function runPCARaw(data, n, d) {
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
    }
    const centered = new Float64Array(n * d);
    for (let i = 0; i < n * d; i++) centered[i] = data[i] - mean[i % d];

    const multiply = (v) => {
        let res = new Float64Array(d);
        for (let i = 0; i < n; i++) {
            let dot = 0;
            for (let j = 0; j < d; j++) dot += centered[i * d + j] * v[j];
            for (let j = 0; j < d; j++) res[j] += dot * centered[i * d + j];
        }
        return res;
    };

    const getEigenvector = () => {
        let v = new Float64Array(d).map(() => Math.random() - 0.5);
        for (let iter = 0; iter < 20; iter++) {
            v = multiply(v);
            let norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
            v = v.map(x => x / norm);
        }
        return v;
    };

    const e1 = getEigenvector();
    const multiply2 = (v) => {
        let res = multiply(v);
        let dot = res.reduce((a, b, i) => a + b * e1[i], 0);
        return res.map((val, i) => val - dot * e1[i]);
    };

    let v2 = new Float64Array(d).map(() => Math.random() - 0.5);
    for (let i = 0; i < 20; i++) {
        v2 = multiply2(v2);
        let norm = Math.sqrt(v2.reduce((a, b) => a + b * b, 0));
        v2 = v2.map(x => x / norm);
    }
    const e2 = v2;

    let projection = [];
    for (let i = 0; i < n; i++) {
        let px = 0, py = 0;
        for (let j = 0; j < d; j++) {
            px += centered[i * d + j] * e1[j];
            py += centered[i * d + j] * e2[j];
        }
        projection.push([px, py]);
    }
    return projection;
}

export function drawScatterPlots(
    container,
    pcaPointsByStep,
    clusters,
    plotSize,
    {
        independentScale = false,
        kernelPcaPointsByStep = null,
        kernelPcaRequestedSteps = null,
        dendrogramsByStep = null,
        dendrogramRequestedSteps = null,
        pca3dPointsByStep = null,
        pca3dRequestedSteps = null
    } = {}
) {
    disposePca3DInContainer(container);
    container.innerHTML = '';

    let globalMinX = Infinity, globalMaxX = -Infinity, globalMinY = Infinity, globalMaxY = -Infinity;
    if (!independentScale) {
        pcaPointsByStep.forEach(proj => {
            proj.forEach(p => {
                globalMinX = Math.min(globalMinX, p[0]); globalMaxX = Math.max(globalMaxX, p[0]);
                globalMinY = Math.min(globalMinY, p[1]); globalMaxY = Math.max(globalMaxY, p[1]);
            });
        });
    }
    const globalRange = Math.max(globalMaxX - globalMinX, globalMaxY - globalMinY) || 1;

    let kernelGlobalMinX = Infinity, kernelGlobalMaxX = -Infinity, kernelGlobalMinY = Infinity, kernelGlobalMaxY = -Infinity;
    const hasKernel = Array.isArray(kernelPcaPointsByStep) && kernelPcaPointsByStep.length > 0;
    if (hasKernel && !independentScale) {
        kernelPcaPointsByStep.forEach(proj => {
            if (!proj) return;
            const pts = Array.isArray(proj) ? proj : (Array.isArray(proj.coords) ? proj.coords : null);
            if (!pts) return;
            pts.forEach(p => {
                kernelGlobalMinX = Math.min(kernelGlobalMinX, p[0]); kernelGlobalMaxX = Math.max(kernelGlobalMaxX, p[0]);
                kernelGlobalMinY = Math.min(kernelGlobalMinY, p[1]); kernelGlobalMaxY = Math.max(kernelGlobalMaxY, p[1]);
            });
        });
    }
    let kernelGlobalRange = Math.max(kernelGlobalMaxX - kernelGlobalMinX, kernelGlobalMaxY - kernelGlobalMinY);
    if (!Number.isFinite(kernelGlobalRange) || kernelGlobalRange <= 0) kernelGlobalRange = 1;

    const dendroHeight = Math.max(180, Math.min(280, Math.round(plotSize * 0.95)));

    clusters.forEach((c, t) => {
        const panel = document.createElement('div');
        panel.className = 'scatter-panel';
        panel.style.cssText = `text-align: center; margin-bottom: 20px; width: ${plotSize}px;`;

        appendCollapsibleSection(panel, 'pca', `Step ${t + 1} (PCA)`, (body) => {
            const pcaCanvas = document.createElement('canvas');
            pcaCanvas.width = plotSize;
            pcaCanvas.height = plotSize;
            pcaCanvas.style.backgroundColor = '#111827';
            pcaCanvas.style.border = '1px solid #374151';
            body.appendChild(wrapPlotCanvas(pcaCanvas, plotIdStep('pca', t + 1)));

            const pcaCtx = pcaCanvas.getContext('2d');
            const pcaPoints = pcaPointsByStep[t];
            let minX = globalMinX, minY = globalMinY, range = globalRange;
            if (independentScale) {
                let localMinX = Infinity, localMaxX = -Infinity, localMinY = Infinity, localMaxY = -Infinity;
                pcaPoints.forEach(p => {
                    localMinX = Math.min(localMinX, p[0]); localMaxX = Math.max(localMaxX, p[0]);
                    localMinY = Math.min(localMinY, p[1]); localMaxY = Math.max(localMaxY, p[1]);
                });
                minX = localMinX;
                minY = localMinY;
                range = Math.max(localMaxX - localMinX, localMaxY - localMinY) || 1;
            }
            pcaPoints.forEach((p, i) => {
                pcaCtx.fillStyle = clusterColorHex(c.assignments[i]);
                const x = ((p[0] - minX) / range) * pcaCanvas.width;
                const y = ((p[1] - minY) / range) * pcaCanvas.height;
                pcaCtx.beginPath();
                pcaCtx.arc(x, y, 2, 0, Math.PI * 2);
                pcaCtx.fill();
            });
        }, t);

        appendCollapsibleSection(panel, 'pca3d', '3D PCA', (body) => {
            buildPca3DSectionBody(body, t, plotSize, plotIdStep('pca3d', t + 1));
        }, t);

        appendCollapsibleSection(panel, 'kpca', 'Kernel PCA', (body) => {
            const kPcaCanvas = document.createElement('canvas');
            kPcaCanvas.width = plotSize;
            kPcaCanvas.height = plotSize;
            kPcaCanvas.style.backgroundColor = '#111827';
            kPcaCanvas.style.border = '1px solid #374151';
            body.appendChild(wrapPlotCanvas(kPcaCanvas, plotIdStep('kpca', t + 1)));

            const kPcaCtx = kPcaCanvas.getContext('2d');
            if (hasKernel && kernelPcaPointsByStep[t]) {
                const kPca = kernelPcaPointsByStep[t];
                const kPcaPoints = Array.isArray(kPca.coords) ? kPca.coords : kPca;
                const kPcaIndices = Array.isArray(kPca.sampledIndices) ? kPca.sampledIndices : null;
                let kMinX = kernelGlobalMinX, kMinY = kernelGlobalMinY, kRange = kernelGlobalRange;
                if (independentScale) {
                    let localMinX = Infinity, localMaxX = -Infinity, localMinY = Infinity, localMaxY = -Infinity;
                    kPcaPoints.forEach(p => {
                        localMinX = Math.min(localMinX, p[0]); localMaxX = Math.max(localMaxX, p[0]);
                        localMinY = Math.min(localMinY, p[1]); localMaxY = Math.max(localMaxY, p[1]);
                    });
                    kMinX = localMinX;
                    kMinY = localMinY;
                    kRange = Math.max(localMaxX - localMinX, localMaxY - localMinY) || 1;
                }
                const drawIdxs = kPcaIndices || kPcaPoints.map((_, i) => i);
                kPcaPoints.forEach((p, localI) => {
                    if (!p) return;
                    const globalI = drawIdxs[localI];
                    kPcaCtx.fillStyle = clusterColorHex(c.assignments[globalI]);
                    const x = ((p[0] - kMinX) / kRange) * kPcaCanvas.width;
                    const y = ((p[1] - kMinY) / kRange) * kPcaCanvas.height;
                    kPcaCtx.beginPath();
                    kPcaCtx.arc(x, y, 2, 0, Math.PI * 2);
                    kPcaCtx.fill();
                });
            } else {
                kPcaCtx.fillStyle = '#9ca3af';
                kPcaCtx.font = '12px sans-serif';
                kPcaCtx.textAlign = 'center';
                const pending = kernelPcaRequestedSteps?.has?.(t) && !kernelPcaPointsByStep?.[t];
                kPcaCtx.fillText(
                    pending ? 'Computing…' : 'Expand to compute',
                    kPcaCanvas.width / 2,
                    kPcaCanvas.height / 2
                );
            }
        }, t);

        appendCollapsibleSection(panel, 'dendro', `Dendrogram (${t + 1})`, (body) => {
            const dendroCanvas = document.createElement('canvas');
            dendroCanvas.width = plotSize;
            dendroCanvas.height = dendroHeight;
            dendroCanvas.className = 'dendrogram-canvas';
            dendroCanvas.style.backgroundColor = '#111827';
            dendroCanvas.style.border = '1px solid #374151';
            dendroCanvas.style.aspectRatio = 'auto';
            body.appendChild(wrapPlotCanvas(dendroCanvas, plotIdStep('dendro', t + 1)));

            const dendroCtx = dendroCanvas.getContext('2d');
            const dendroData = Array.isArray(dendrogramsByStep) ? dendrogramsByStep[t] : null;
            if (dendroData && dendroData.nLeaves) {
                drawDendrogram(dendroCtx, dendroData, dendroCanvas.width, dendroCanvas.height);
            } else {
                dendroCtx.fillStyle = '#9ca3af';
                dendroCtx.font = '12px sans-serif';
                dendroCtx.textAlign = 'center';
                const dendroPending = dendrogramRequestedSteps?.has?.(t) && !dendroData?.nLeaves;
                dendroCtx.fillText(
                    dendroPending ? 'Computing…' : 'Expand to compute',
                    dendroCanvas.width / 2,
                    dendroCanvas.height / 2
                );
            }
        }, t);

        container.appendChild(panel);
    });

    requestAnimationFrame(() => {
        refreshAnnotationLayers(container);
        syncPca3DPlotsInContainer(container, {
            pca3dPointsByStep,
            clusters,
            plotSize,
            pca3dRequestedSteps
        }).catch(err => console.error('syncPca3DPlotsInContainer', err));
    });
}

function drawDendrogram(ctx, dendro, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#9ca3af';

    const { nLeaves, nodeLeft, nodeRight, nodeHeight } = dendro;
    const padLeft = 8;
    const padTop = 10;
    const padBottom = 10;

    const maxH = Math.max(...nodeHeight.slice(nLeaves).filter(x => Number.isFinite(x)), 1e-9);
    const bottomY = height - padBottom;
    const topY = padTop;

    const mapY = (h) => bottomY - (h / maxH) * (bottomY - topY);

    const leafX = (i) => {
        if (nLeaves === 1) return width / 2;
        return padLeft + (i * (width - padLeft * 2)) / (nLeaves - 1);
    };

    const xCache = new Float64Array(nodeHeight.length).fill(NaN);

    const computeX = (node) => {
        if (node < nLeaves) return leafX(node);
        if (!Number.isNaN(xCache[node])) return xCache[node];
        const left = nodeLeft[node];
        const right = nodeRight[node];
        const xl = computeX(left);
        const xr = computeX(right);
        xCache[node] = (xl + xr) / 2;
        return xCache[node];
    };

    const drawNode = (node) => {
        const nodeY = mapY(nodeHeight[node] || 0);
        if (node < nLeaves) return;

        const left = nodeLeft[node];
        const right = nodeRight[node];
        drawNode(left);
        drawNode(right);

        const xl = computeX(left);
        const xr = computeX(right);

        const leftY = node < nLeaves ? bottomY : mapY(nodeHeight[left] || 0);
        const rightY = mapY(nodeHeight[right] || 0);

        // vertical lines
        ctx.beginPath();
        ctx.moveTo(xl, leftY);
        ctx.lineTo(xl, nodeY);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(xr, rightY);
        ctx.lineTo(xr, nodeY);
        ctx.stroke();

        // horizontal line
        ctx.beginPath();
        ctx.moveTo(xl, nodeY);
        ctx.lineTo(xr, nodeY);
        ctx.stroke();
    };

    const root = Number.isFinite(dendro.root) ? dendro.root : nodeLeft.length - 1;
    drawNode(root);
}

/** Max points for kernel PCA (O(n²) memory/time). Larger clouds are subsampled. */
export const KERNEL_PCA_MAX_POINTS = 1500;

function uniformConditionalRow(n, i) {
    const row = new Float64Array(n);
    if (n <= 1) return row;
    const val = 1 / (n - 1);
    for (let j = 0; j < n; j++) {
        if (j !== i) row[j] = val;
    }
    return row;
}

export function computeKernelPcaProjection(
    data,
    n,
    d,
    perplexity = 30,
    { yieldEvery = 5, maxPoints = KERNEL_PCA_MAX_POINTS } = {}
) {
    return (async () => {
        let idx = Array.from({ length: n }, (_, i) => i);
        if (Number.isFinite(maxPoints) && n > maxPoints) {
            // For very large point clouds, full kernel PCA will freeze the browser.
            idx = idx.sort(() => Math.random() - 0.5).slice(0, maxPoints);
            n = idx.length;
        }

        if (n < 1) {
            return { coords: [], sampledIndices: idx };
        }
        if (n === 1) {
            return { coords: [[0, 0]], sampledIndices: idx.slice() };
        }

        // Build sampled data matrix view (still dense).
        const X = new Float64Array(n * d);
        for (let i = 0; i < n; i++) {
            const srcI = idx[i];
            for (let j = 0; j < d; j++) {
                const v = data[srcI * d + j];
                X[i * d + j] = Number.isFinite(v) ? v : 0;
            }
        }

        // Squared distances (clamp non-finite to avoid exp underflow leaving null P rows).
        const dist2 = Array.from({ length: n }, () => new Float64Array(n));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let s = 0;
                for (let dim = 0; dim < d; dim++) {
                    const diff = X[i * d + dim] - X[j * d + dim];
                    s += diff * diff;
                }
                if (!Number.isFinite(s)) s = 1e12;
                dist2[i][j] = s;
                dist2[j][i] = s;
            }
        }

        // Compute conditional probabilities P_{j|i} (t-SNE style) and then symmetrize.
        const targetPerplexity = Math.min(
            Math.max(1, perplexity),
            Math.max(1, (n - 1) / 3)
        );
        const Pcond = Array.from({ length: n }, () => new Float64Array(n));

        const HYPER_ITERS = 25;
        const tol = 1e-5;

        for (let i = 0; i < n; i++) {
            let beta = 1.0;
            let betamin = -Infinity;
            let betamax = Infinity;

            // binary search for beta that matches target perplexity
            let finalP = null;
            for (let iter = 0; iter < HYPER_ITERS; iter++) {
                const Prow = new Float64Array(n);
                let sumP = 0;
                for (let j = 0; j < n; j++) {
                    if (j === i) continue;
                    const dij = dist2[i][j];
                    const v = Number.isFinite(dij) ? Math.exp(-beta * dij) : 0;
                    Prow[j] = v;
                    sumP += v;
                }
                if (sumP === 0 || !Number.isFinite(sumP)) {
                    beta *= 2;
                    if (iter === HYPER_ITERS - 1) finalP = uniformConditionalRow(n, i);
                    continue;
                }
                // normalize
                for (let j = 0; j < n; j++) {
                    if (j === i) continue;
                    Prow[j] /= sumP;
                }

                // entropy (natural log)
                let H = 0;
                for (let j = 0; j < n; j++) {
                    if (j === i) continue;
                    const pij = Prow[j];
                    if (pij > 1e-300) H -= pij * Math.log(pij);
                }
                const perp = Math.exp(H);
                if (Math.abs(perp - targetPerplexity) / targetPerplexity < tol) {
                    finalP = Prow;
                    break;
                }
                if (perp > targetPerplexity) {
                    // entropy too high -> increase beta
                    betamin = beta;
                    if (betamax === Infinity) beta *= 2;
                    else beta = (beta + betamax) / 2;
                } else {
                    betamax = beta;
                    if (betamin === -Infinity) beta /= 2;
                    else beta = (beta + betamin) / 2;
                }
                finalP = Prow;
            }

            Pcond[i] = finalP ?? uniformConditionalRow(n, i);
            if (i % yieldEvery === 0) await new Promise(r => setTimeout(r, 0));
        }

        // Joint symmetric probabilities: P_ij = (P_{j|i} + P_{i|j}) / (2N)
        const K = Array.from({ length: n }, () => new Float64Array(n));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const p = (Pcond[i][j] + Pcond[j][i]) / (2 * n);
                K[i][j] = p;
                K[j][i] = p;
            }
        }

        // Center kernel: Kc = K - rowMean - colMean + overallMean
        const rowMean = new Float64Array(n);
        let overall = 0;
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += K[i][j];
            rowMean[i] = s / n;
            overall += s;
        }
        overall = overall / (n * n);

        const mulKc = (v) => {
            let sumV = 0;
            for (let i = 0; i < n; i++) sumV += v[i];
            let t = 0; // c^T v, with c=rowMean (since K is symmetric)
            for (let i = 0; i < n; i++) t += rowMean[i] * v[i];

            const out = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < n; j++) s += K[i][j] * v[j];
                out[i] = s - rowMean[i] * sumV - t + overall * sumV;
            }
            return out;
        };

        const dot = (a, b) => {
            let s = 0;
            for (let i = 0; i < n; i++) s += a[i] * b[i];
            return s;
        };

        const norm = (v) => Math.sqrt(dot(v, v)) || 1;

        // Top eigenvector via power iteration
        let v1 = new Float64Array(n).map(() => Math.random() - 0.5);
        let v1n = norm(v1);
        for (let i = 0; i < n; i++) v1[i] /= v1n;

        for (let iter = 0; iter < 40; iter++) {
            const w = mulKc(v1);
            const nw = norm(w);
            for (let i = 0; i < n; i++) v1[i] = w[i] / nw;
            if (iter % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }
        const Kc_v1 = mulKc(v1);
        const lambda1 = Math.max(dot(v1, Kc_v1), 0);

        // Second eigenvector with orthogonalization
        let v2 = new Float64Array(n).map(() => Math.random() - 0.5);
        // orthogonalize init
        const proj = dot(v2, v1);
        for (let i = 0; i < n; i++) v2[i] -= proj * v1[i];
        let v2n = norm(v2);
        for (let i = 0; i < n; i++) v2[i] /= v2n;

        for (let iter = 0; iter < 40; iter++) {
            const w = mulKc(v2);
            // subtract projection onto v1
            const pw = dot(w, v1);
            for (let i = 0; i < n; i++) w[i] -= pw * v1[i];
            const nw = norm(w);
            for (let i = 0; i < n; i++) v2[i] = w[i] / nw;
            if (iter % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }
        const Kc_v2 = mulKc(v2);
        const lambda2 = Math.max(dot(v2, Kc_v2), 0);

        // Embed: eigenvectors scaled by sqrt(eigenvalues)
        const coords = new Array(n);
        const s1 = Math.sqrt(lambda1) || 1;
        const s2 = Math.sqrt(lambda2) || 1;
        for (let i = 0; i < n; i++) coords[i] = [v1[i] * s1, v2[i] * s2];

        return { coords, sampledIndices: idx };
    })();
}

export async function computeDendrogram(data, n, d, linkage = 'average', { yieldEvery = 1, maxPoints = Infinity } = {}) {
    // For very large n, dendrogram O(n^3) gets slow quickly.
    let idx = Array.from({ length: n }, (_, i) => i);
    if (Number.isFinite(maxPoints) && n > maxPoints) {
        idx = idx.sort(() => Math.random() - 0.5).slice(0, maxPoints);
        n = idx.length;
    }

    const X = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
        const srcI = idx[i];
        for (let j = 0; j < d; j++) X[i * d + j] = data[srcI * d + j];
    }

    // Distance matrix (squared Euclidean)
    const maxNodes = 2 * n - 1;
    const nodeLeft = new Int32Array(maxNodes).fill(-1);
    const nodeRight = new Int32Array(maxNodes).fill(-1);
    const nodeHeight = new Float64Array(maxNodes).fill(0);

    const dist = Array.from({ length: maxNodes }, () => new Float64Array(maxNodes).fill(Infinity));
    const size = new Float64Array(maxNodes);

    for (let i = 0; i < n; i++) size[i] = 1;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let s = 0;
            for (let dim = 0; dim < d; dim++) {
                const diff = X[i * d + dim] - X[j * d + dim];
                s += diff * diff;
            }
            dist[i][j] = s;
            dist[j][i] = s;
        }
    }

    let active = new Set(Array.from({ length: n }, (_, i) => i));
    let nextNode = n;

    while (active.size > 1) {
        const ids = Array.from(active);
        let bestA = ids[0], bestB = ids[1];
        let bestD = Infinity;
        for (let ai = 0; ai < ids.length; ai++) {
            for (let bi = ai + 1; bi < ids.length; bi++) {
                const a = ids[ai], b = ids[bi];
                const dval = dist[a][b];
                if (dval < bestD) { bestD = dval; bestA = a; bestB = b; }
            }
        }

        const a = bestA, b = bestB;
        const c = nextNode++;
        nodeLeft[c] = a;
        nodeRight[c] = b;
        nodeHeight[c] = bestD;
        size[c] = size[a] + size[b];

        active.delete(a);
        active.delete(b);
        active.add(c);

        // update distances to remaining clusters
        for (const k of active) {
            if (k === c) continue;
            if (linkage === 'single') {
                const v = Math.min(dist[a][k], dist[b][k]);
                dist[c][k] = v; dist[k][c] = v;
            } else if (linkage === 'complete') {
                const v = Math.max(dist[a][k], dist[b][k]);
                dist[c][k] = v; dist[k][c] = v;
            } else {
                // average / UPGMA
                const v = (size[a] * dist[a][k] + size[b] * dist[b][k]) / (size[a] + size[b]);
                dist[c][k] = v; dist[k][c] = v;
            }
        }

        if ((c - n) % yieldEvery === 0) await new Promise(r => setTimeout(r, 0));
    }

    const root = Array.from(active)[0];
    // Root might not be last created node if maxNodes truncated; handle by slicing arrays.
    return {
        nLeaves: n,
        sampledIndices: idx,
        nodeLeft,
        nodeRight,
        nodeHeight,
        root
    };
}

