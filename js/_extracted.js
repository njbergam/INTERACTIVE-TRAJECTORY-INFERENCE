const COLORS = [
            '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#6366f1',
            '#84cc16', '#eab308', '#d946ef', '#0ea5e9', '#fca5a5'
        ];

        let rawSequences = [];
        let state = { kValues: [], epsilon: 0.0, cache: [], otCache: [], globalHistory: {}, pcaPoints: [], plotSize: 400 };
        let drawnEdgePaths = [];
        let drawnGlobalPoints = [];
        let dataPickerAvailable = false;
        let pyodidePromise = null;
        const MAX_PKL_BYTES = 512 * 1024 * 1024;

        const elements = {
            dropZone: document.getElementById('drop-zone'),
            status: document.getElementById('status'),
            backHomeBtn: document.getElementById('back-home-btn'),
            dataPicker: document.getElementById('data-picker'),
            dataPickerHint: document.getElementById('data-picker-hint'),
            dataFileList: document.getElementById('data-file-list'),
            controls: document.getElementById('controls'),
            kContainer: document.getElementById('k-sliders-container'),
            epsSlider: document.getElementById('epsilon-slider'),
            epsLabel: document.getElementById('epsilon-label'),
            vizContainer: document.getElementById('viz-container'),
            scatterContainer: document.getElementById('scatter-container'),
            canvas: document.getElementById('t-partite-canvas'),
            ctx: document.getElementById('t-partite-canvas').getContext('2d'),
            globalCanvas: document.getElementById('global-elbow-canvas'),
            globalCtx: document.getElementById('global-elbow-canvas').getContext('2d'),
            tooltip: document.getElementById('main-tooltip'),
            transportNote: document.getElementById('transport-note'),
            elbowCanvases: []
        };

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
            return fitCanvasToDisplay(elements.canvas) | fitCanvasToDisplay(elements.globalCanvas);
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

            elements.status.textContent = `Loaded ${rawSequences.length} timesteps. Ready.`;
            initUI();
            fitMainCanvases();
            updateVisualization();
        }

        function loadPyodideScript() {
            if (window.loadPyodide) return Promise.resolve();
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                s.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
                s.onload = resolve;
                s.onerror = () => reject(new Error('Failed to load Pyodide runtime.'));
                document.head.appendChild(s);
            });
        }

        async function getPyodide() {
            if (!pyodidePromise) {
                pyodidePromise = (async () => {
                    await loadPyodideScript();
                    const pyodide = await loadPyodide();
                    await pyodide.loadPackage(['numpy', 'scipy']);
                    await pyodide.runPythonAsync(`
def _sort_keys(keys):
    def keyfn(k):
        try:
            return int(k)
        except (TypeError, ValueError):
            return str(k)
    return sorted(keys, key=keyfn)

def _to_dense_2d(a):
    import numpy as np
    if hasattr(a, "todense"):
        m = np.asarray(a.todense(), dtype=np.float64)
    elif hasattr(a, "toarray"):
        m = np.asarray(a.toarray(), dtype=np.float64)
    else:
        m = np.asarray(a, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D per timestep, got shape {m.shape}")
    return np.ascontiguousarray(m)

def extract_pkl_sequences(path):
    import pickle
    import numpy as np

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at top level, got {type(data)}")

    Xs = data.get("Xs")
    if Xs is None:
        for alt in ("marginals", "X", "Ms", "data"):
            if alt in data:
                Xs = data[alt]
                break
    if Xs is None:
        raise KeyError("No sequence data found (expected key 'Xs' or 'marginals')")

    if isinstance(Xs, dict):
        return [_to_dense_2d(Xs[k]) for k in _sort_keys(Xs.keys())]
    if isinstance(Xs, (list, tuple)):
        return [_to_dense_2d(x) for x in Xs]

    arr = _to_dense_2d(Xs)
    if arr.ndim == 3:
        return [np.ascontiguousarray(arr[t], dtype=np.float64) for t in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"Unsupported Xs shape {arr.shape}")
`);
                    return pyodide;
                })();
            }
            return pyodidePromise;
        }

        function pyodideArraysToRawSequences(seqProxies) {
            const raw = [];
            for (const m of seqProxies) {
                const n = m.shape.get(0);
                const d = m.shape.get(1);
                const buf = m.getBuffer('f64');
                raw.push({ shape: [n, d], data: new Float64Array(buf.data.slice()) });
                buf.release();
                m.destroy();
            }
            seqProxies.destroy();
            return raw;
        }

        async function pyodideNpyToSequences(arrayBuffer) {
            const pyodide = await getPyodide();
            pyodide.globals.set('npy_bytes', new Uint8Array(arrayBuffer));
            const seqProxies = pyodide.runPython(`
import numpy as np
from io import BytesIO

arr = np.load(BytesIO(bytes(npy_bytes)), allow_pickle=True)

def to_seq(m):
    if hasattr(m, "todense"):
        m = np.asarray(m.todense(), dtype=np.float64)
    elif hasattr(m, "toarray"):
        m = np.asarray(m.toarray(), dtype=np.float64)
    else:
        m = np.ascontiguousarray(np.asarray(m, dtype=np.float64))
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D per timestep, got shape {m.shape}")
    return m

seqs = []
if arr.dtype == object:
    for x in arr.ravel():
        seqs.append(to_seq(x))
elif arr.ndim == 3:
    for t in range(arr.shape[0]):
        seqs.append(to_seq(arr[t]))
elif arr.ndim == 2:
    seqs.append(to_seq(arr))
else:
    raise ValueError(f"Unsupported Xs shape {arr.shape}, dtype {arr.dtype}")
seqs
`);
            return pyodideArraysToRawSequences(seqProxies);
        }

        async function extractSequencesFromPkl(arrayBuffer) {
            const pyodide = await getPyodide();
            pyodide.FS.writeFile('/input.pkl', new Uint8Array(arrayBuffer));
            let seqProxies;
            try {
                seqProxies = pyodide.runPython('extract_pkl_sequences("/input.pkl")');
            } catch (err) {
                const msg = String(err?.message || err);
                if (/anndata|AnnData|No module named/i.test(msg)) {
                    throw new Error(
                        'This pickle uses Python types not available in the browser (e.g. AnnData). ' +
                        'Convert to .npz with data/processing.py first.'
                    );
                }
                throw err;
            } finally {
                try { pyodide.FS.unlink('/input.pkl'); } catch { /* ignore */ }
            }

            return pyodideArraysToRawSequences(seqProxies);
        }

        async function loadDataFile(file) {
            if (!file?.name || !isDataFileName(file.name)) {
                elements.status.textContent = "Error: Expected a .npz or .pkl file.";
                return;
            }

            const lower = file.name.toLowerCase();
            try {
                if (lower.endsWith('.npz')) {
                    elements.status.textContent = "Parsing NPY / NPZ data...";
                    const zip = await JSZip.loadAsync(file);
                    rawSequences = await extractSequences(zip);
                } else {
                    if (file.size > MAX_PKL_BYTES) {
                        throw new Error(
                            `Pickle is ${(file.size / 1e9).toFixed(1)} GB — too large for in-browser loading. ` +
                            'Export a smaller .npz with data/processing.py.'
                        );
                    }
                    elements.status.textContent = "Loading Python runtime & parsing pickle (sparse OK)...";
                    rawSequences = await extractSequencesFromPkl(await file.arrayBuffer());
                }
                await finishLoadingDataset();
            } catch (err) {
                elements.status.textContent = "Error: " + err.message;
                console.error(err);
            }
        }

        function setDataPickerState({ visible, hint, files }) {
            dataPickerAvailable = visible;
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

        async function tryPopulateDataPicker() {
            // Note: browsers cannot list local directories directly. This works when the page is served
            // by a server that exposes a directory listing at `./data/` (e.g. `python -m http.server`).
            try {
                const resp = await fetch('data/', { cache: 'no-store' });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const text = await resp.text();
                const doc = new DOMParser().parseFromString(text, 'text/html');
                const links = Array.from(doc.querySelectorAll('a'))
                    .map(a => a.getAttribute('href') || '')
                    .filter(href => href && !href.startsWith('?') && !href.startsWith('#'))
                    .map(href => href.split('/').filter(Boolean).pop())
                    .filter(name => name && (name.toLowerCase().endsWith('.npz') || name.toLowerCase().endsWith('.pkl')));
                const files = Array.from(new Set(links)).sort((a, b) => a.localeCompare(b));
                setDataPickerState({
                    visible: true,
                    hint: 'Pick a file, or drag & drop any `.npz` / `.pkl`.',
                    files
                });
            } catch {
                // Silent fallback: keep drag & drop only.
                setDataPickerState({ visible: false });
            }
        }

        // --- Drag & Drop ---
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

        elements.backHomeBtn.addEventListener('click', showHomePage);

        // Try to show `.npz` files from `./data/` (if served by a server with directory listing enabled).
        tryPopulateDataPicker();

        function showHomePage() {
            rawSequences = [];
            state = { kValues: [], epsilon: 0.0, cache: [], otCache: [], globalHistory: {}, pcaPoints: [], plotSize: 400 };
            drawnEdgePaths = [];
            drawnGlobalPoints = [];
            elements.elbowCanvases = [];

            elements.controls.style.display = 'none';
            elements.vizContainer.style.display = 'none';
            elements.dropZone.style.display = 'block';
            elements.dataPicker.style.display = dataPickerAvailable ? 'block' : 'none';

            elements.status.textContent = 'Waiting for file...';
            elements.tooltip.style.display = 'none';
            if (elements.transportNote) elements.transportNote.style.display = 'none';

            elements.epsSlider.value = '0';
            elements.epsLabel.textContent = 'Regularization (ε): Exact ';

            elements.kContainer.innerHTML = '';
            elements.scatterContainer.innerHTML = '';
            document.getElementById('resize-plots-control')?.remove();
            elements.ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
            elements.globalCtx.clearRect(0, 0, elements.globalCanvas.width, elements.globalCanvas.height);
        }

        function ensureResizeControl() {
            if (document.getElementById('resize-plots-control')) return;

            const resizeContainer = document.createElement('div');
            resizeContainer.id = 'resize-plots-control';
            resizeContainer.style.marginTop = '10px';

            const resizeLabel = document.createElement('label');
            resizeLabel.textContent = 'Resize Plots: ';

            const resizeSlider = document.createElement('input');
            resizeSlider.type = 'range';
            resizeSlider.min = '200';
            resizeSlider.max = '800';
            resizeSlider.value = String(state.plotSize);
            resizeSlider.addEventListener('input', (e) => {
                state.plotSize = parseInt(e.target.value, 10);
                updateVisualization();
            });

            resizeContainer.appendChild(resizeLabel);
            resizeContainer.appendChild(resizeSlider);
            elements.vizContainer.prepend(resizeContainer);
        }

        // --- UI Logic & Interactions ---
        function initUI() {
            elements.controls.style.display = 'flex';
            elements.vizContainer.style.display = 'flex';
            elements.dropZone.style.display = 'none';
            elements.dataPicker.style.display = 'none';

            ensureResizeControl();

            const T = rawSequences.length;
            state.kValues = Array(T).fill(5);
            elements.kContainer.innerHTML = '';
            elements.elbowCanvases = [];

            for (let t = 0; t < T; t++) {
                const group = document.createElement('div'); group.className = 'control-group';
                const label = document.createElement('label'); label.textContent = `Step ${t+1} Clusters: 5`;

                const slider = document.createElement('input');
                slider.type = 'range'; slider.min = '1'; slider.max = '15'; slider.step = '1'; slider.value = '5';

                const sweepBtn = document.createElement('button');
                sweepBtn.className = 'sweep-btn';
                sweepBtn.textContent = 'Sweep K';



                // Sweep Logic
                sweepBtn.onclick = async () => {
                    for(let k=1; k<=15; k++){
                        slider.value = k;
                        state.kValues[t] = k;
                        label.textContent = `Step ${t+1} Clusters: ${k}`;
                        updateVisualization();
                        await new Promise(r => setTimeout(r, 200));
                    }
                };

                // 2. CREATE THE AUTO-ELBOW BUTTON HERE
                const elbowBtn = document.createElement('button');
                elbowBtn.className = 'sweep-btn';
                elbowBtn.textContent = 'Auto-Elbow';
                elbowBtn.onclick = () => {
                    const elbowK = findElbow(state.cache[t]);
                    slider.value = elbowK;
                    state.kValues[t] = elbowK;
                    label.textContent = `Step ${t+1} Clusters: ${elbowK}`;
                    updateVisualization();
                };

                const elbowCanvas = document.createElement('canvas');
                elbowCanvas.className = 'elbow-canvas';
                elbowCanvas.width = 400; elbowCanvas.height = 160;
                elements.elbowCanvases.push(elbowCanvas);

                slider.addEventListener('input', (e) => {
                    state.kValues[t] = parseInt(e.target.value);
                    label.textContent = `Step ${t+1} Clusters: ${state.kValues[t]}`;
                    requestAnimationFrame(updateVisualization);
                });

                group.appendChild(label); group.appendChild(slider); group.appendChild(sweepBtn); group.appendChild(elbowBtn); group.appendChild(elbowCanvas);
                elements.kContainer.appendChild(group);
            }

            elements.epsSlider.addEventListener('input', (e) => {
                state.epsilon = parseFloat(e.target.value);
                elements.epsLabel.textContent = state.epsilon === 0 ? `Regularization (ε): Exact` : `Regularization (ε): ${state.epsilon.toFixed(3)}`;
                requestAnimationFrame(updateVisualization);
            });

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
            // Get the source cluster count to calculate the fraction
            const sourceStep = edge.stepIdx;
            const sourceClusterIdx = edge.sourceIdx;
            const count = rawSequences[sourceStep].shape[0];
            hoveredEdge.sourceCount = rawSequences[sourceStep].shape[0];
            hoveredEdge.sourceClusterCount = state.cache[sourceStep][state.kValues[sourceStep]].counts[sourceClusterIdx];
            break;
        }
    }

    if (hoveredEdge) {
        // Precise positioning near the cursor
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

            elements.globalCanvas.addEventListener('click', (e) => {
                const rect = elements.globalCanvas.getBoundingClientRect();
                const mouseX = (e.clientX - rect.left) * (elements.globalCanvas.width / rect.width);
                const mouseY = (e.clientY - rect.top) * (elements.globalCanvas.height / rect.height);
                let nearest = null; let minDist = 20;
                drawnGlobalPoints.forEach(p => {
                    const dist = Math.hypot(p.cx - mouseX, p.cy - mouseY);
                    if (dist < minDist) { minDist = dist; nearest = p; }
                });
                if (nearest) {
                    elements.tooltip.style.display = 'none';
                    state.kValues = nearest.configStr.split(',').map(Number);
                    const inputs = elements.kContainer.querySelectorAll('input[type="range"]');
                    const labels = elements.kContainer.querySelectorAll('label');
                    state.kValues.forEach((k, i) => {
                        if(inputs[i]) inputs[i].value = k;
                        if(labels[i]) labels[i].textContent = `Step ${i+1} Clusters: ${k}`;
                    });
                    updateVisualization();
                }
            });

            elements.globalCanvas.addEventListener('mousemove', (e) => {
                const rect = elements.globalCanvas.getBoundingClientRect();
                const mouseX = (e.clientX - rect.left) * (elements.globalCanvas.width / rect.width);
                const mouseY = (e.clientY - rect.top) * (elements.globalCanvas.height / rect.height);
                let nearest = null; let minDist = 20;
                drawnGlobalPoints.forEach(p => {
                    const dist = Math.hypot(p.cx - mouseX, p.cy - mouseY);
                    if (dist < minDist) { minDist = dist; nearest = p; }
                });
                if (nearest) {
                    elements.tooltip.style.left = e.pageX + 'px';
                    elements.tooltip.style.top = (e.pageY - 20) + 'px';
                    elements.tooltip.innerHTML = `Sum K: ${nearest.sumK}<br>Inertia: ${nearest.cost.toFixed(0)}<br>Click to set K=[${nearest.configStr}]`;
                    elements.tooltip.style.display = 'block';
                    elements.globalCanvas.style.cursor = 'pointer';
                } else {
                    elements.tooltip.style.display = 'none';
                    elements.globalCanvas.style.cursor = 'crosshair';
                }
            });
        }

        // ... (K-Means, ExactOT, Sinkhorn, Drawing functions are retained from the previous version)

        // Helper: K-Means with cache (retained from last)
        function updateVisualization() {
            fitMainCanvases();
            const clusters = rawSequences.map((seq, t) => {
                const k = state.kValues[t];
                if (!state.cache[t][k]) state.cache[t][k] = kMeans(seq, k);
                return state.cache[t][k];
            });

            elements.elbowCanvases.forEach((canvas, t) => drawLocalElbowPlot(canvas, state.cache[t], state.kValues[t]));

            const sumInertia = clusters.reduce((sum, c) => sum + c.inertia, 0);
            const couplings = [];
            const gwEdges = [];

            for (let t = 0; t < clusters.length - 1; t++) {
                const k1 = state.kValues[t];
                const k2 = state.kValues[t + 1];
                const useGW = rawSequences[t].shape[1] !== rawSequences[t + 1].shape[1];
                if (useGW) gwEdges.push(t);

                const pairKey = useGW
                    ? `${k1}-${k2}-gw-${state.epsilon}`
                    : `${k1}-${k2}-ot`;

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

            const currentConfigStr = state.kValues.join(',');
            const sumK = state.kValues.reduce((a, b) => a + b, 0);
            if (!state.globalHistory[currentConfigStr]) state.globalHistory[currentConfigStr] = { sumK, sumInertia };

            drawScatterPlots(rawSequences, clusters, { independentScale: gwEdges.length > 0 });
            drawGraph(clusters, couplings);
            drawGlobalElbowPlot(sumK, sumInertia, currentConfigStr);
        }

        // ... [Insert kMeans, exactOT, sinkhorn, drawScatterPlots, drawGraph, drawLocalElbowPlot, drawGlobalElbowPlot, extractSequences, parseNPY functions from previous block here]
        function kMeans(matrix, k, iters=30, n_init=3) {
            const [n, d] = matrix.shape;
            const data = matrix.data;
            if (k > n) k = n;

            let bestInertia = Infinity;
            let bestResult = null;

            for (let init = 0; init < n_init; init++) {
                let centers = [];
                const firstIdx = Math.floor(Math.random() * n);
                centers.push(Array.from(data.slice(firstIdx * d, firstIdx * d + d)));

                for (let c = 1; c < k; c++) {
                    let distances = new Float64Array(n); let sumSqDist = 0;
                    for (let i = 0; i < n; i++) {
                        let minDist = Infinity;
                        for (let j = 0; j < c; j++) {
                            let dist = 0;
                            for (let dim = 0; dim < d; dim++) dist += Math.pow(data[i * d + dim] - centers[j][dim], 2);
                            if (dist < minDist) minDist = dist;
                        }
                        distances[i] = minDist; sumSqDist += minDist;
                    }
                    let target = Math.random() * sumSqDist, cumulative = 0, selectedIdx = n - 1;
                    for (let i = 0; i < n; i++) {
                        cumulative += distances[i];
                        if (cumulative >= target) { selectedIdx = i; break; }
                    }
                    centers.push(Array.from(data.slice(selectedIdx * d, selectedIdx * d + d)));
                }

                let counts = new Int32Array(k), assignments = new Int32Array(n);
                for (let iter = 0; iter < iters; iter++) {
                    counts.fill(0); let newCenters = Array.from({length: k}, () => new Float64Array(d));
                    for (let i = 0; i < n; i++) {
                        let minDist = Infinity, minIdx = 0;
                        for (let c = 0; c < k; c++) {
                            let dist = 0;
                            for (let j = 0; j < d; j++) dist += Math.pow(data[i * d + j] - centers[c][j], 2);
                            if (dist < minDist) { minDist = dist; minIdx = c; }
                        }
                        assignments[i] = minIdx; counts[minIdx]++;
                        for (let j = 0; j < d; j++) newCenters[minIdx][j] += data[i * d + j];
                    }
                    for (let c = 0; c < k; c++) if (counts[c] > 0) for (let j = 0; j < d; j++) centers[c][j] = newCenters[c][j] / counts[c];
                }

                let inertia = 0;
                for (let i = 0; i < n; i++) {
                    let cIdx = assignments[i];
                    for (let j = 0; j < d; j++) inertia += Math.pow(data[i * d + j] - centers[cIdx][j], 2);
                }

                if (inertia < bestInertia) {
                    bestInertia = inertia;
                    bestResult = { centers, counts: new Int32Array(counts), marginals: Array.from(counts).map(c => c / n), assignments: new Int32Array(assignments), inertia };
                }
            }

            let order = Array.from({length: k}, (_, i) => i).sort((a, b) => bestResult.centers[a][1] - bestResult.centers[b][1]);
            let orderedCenters = order.map(i => bestResult.centers[i]);
            let orderedCounts = new Int32Array(k).map((_, i) => bestResult.counts[order[i]]);
            let orderedMarginals = order.map(i => bestResult.marginals[i]);
            let orderedAssignments = new Int32Array(n);
            let revOrder = {}; order.forEach((oldIdx, newIdx) => revOrder[oldIdx] = newIdx);
            for(let i=0; i<n; i++) orderedAssignments[i] = revOrder[bestResult.assignments[i]];

            return { centers: orderedCenters, counts: orderedCounts, marginals: orderedMarginals, assignments: orderedAssignments, inertia: bestResult.inertia };
        }

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

        function exactOT(c1, c2) {
            const a = Array.from(c1.marginals), b = Array.from(c2.marginals);
            return exactOTFromCost(wassersteinCostMatrix(c1, c2), a, b);
        }

        function sinkhorn(c1, c2, epsilon, iters = 50) {
            const a = Array.from(c1.marginals), b = Array.from(c2.marginals);
            return sinkhornFromCost(wassersteinCostMatrix(c1, c2), a, b, epsilon, iters);
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

        function gromovWasserstein(c1, c2, epsilon = 0, maxIter = 80, tol = 1e-7) {
            const C1 = clusterDistMatrix(c1.centers);
            const C2 = clusterDistMatrix(c2.centers);
            const p = Array.from(c1.marginals);
            const q = Array.from(c2.marginals);

            let T = outerProductPlan(p, q);
            for (let iter = 0; iter < maxIter; iter++) {
                const M = gromovLinearizedCost(C1, C2, p, q, T);
                const Tnew = epsilon === 0
                    ? exactOTFromCost(M, p, q).P
                    : sinkhornFromCost(M, p, q, epsilon);
                const change = planFrobeniusNormDiff(T, Tnew);
                T = Tnew;
                if (change < tol) break;
            }
            return { P: T, cost: gwObjective(C1, C2, T), method: 'gw' };
        }

        // --- Visualizations ---
        function drawLocalElbowPlot(canvas, cacheData, currentK) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const knownKs = Object.keys(cacheData).map(Number).sort((a,b) => a-b);
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
                ctx.beginPath(); ctx.arc(x, y, isActive ? 8 : 4, 0, Math.PI*2);
                ctx.fillStyle = isActive ? '#3b82f6' : '#d1d5db'; ctx.fill();
            });
        }

        function drawGlobalElbowPlot(currentSumK, currentSumInertia, currentConfigStr) {
    const cvs = elements.globalCanvas;
    const ctx = elements.globalCtx;
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    drawnGlobalPoints = [];

    let bestByX = {};
    for (let key in state.globalHistory) {
        const entry = state.globalHistory[key];
        if (bestByX[entry.sumK] === undefined || entry.sumInertia < bestByX[entry.sumK].cost) {
            bestByX[entry.sumK] = { cost: entry.sumInertia, configStr: key };
        }
    }

    const points = Object.keys(bestByX).map(Number).sort((a,b) => a-b).map(x => ({
        x, cost: bestByX[x].cost, configStr: bestByX[x].configStr
    }));

    if (points.length === 0) return;

    const padX = 70, padY = 40;
    const w = cvs.width - padX - 30;
    const h = cvs.height - padY - 30;

    const minX = rawSequences.length;
    const maxX = rawSequences.length * 15;
    const minY = Math.min(...points.map(p => p.cost), currentSumInertia) * 0.9;
    const maxY = Math.max(...points.map(p => p.cost), currentSumInertia) * 1.1;

    const getX = x => padX + ((x - minX) / (maxX - minX)) * w;
    const getY = y => padY + h - ((y - minY) / (maxY - minY || 1)) * h;

    // --- Draw Grid/Ticks ---
    ctx.font = "10px sans-serif"; ctx.fillStyle = "#6b7280"; ctx.textAlign = "center";

    // X-Ticks
    for(let i=0; i<=5; i++) {
        const val = minX + (i/5)*(maxX - minX);
        const x = getX(val);
        ctx.beginPath(); ctx.moveTo(x, padY + h); ctx.lineTo(x, padY + h + 5); ctx.stroke();
        ctx.fillText(Math.round(val), x, padY + h + 15);
    }

    // Y-Ticks
    ctx.textAlign = "right";
    for(let i=0; i<=5; i++) {
        const val = minY + (i/5)*(maxY - minY);
        const y = getY(val);
        ctx.beginPath(); ctx.moveTo(padX, y); ctx.lineTo(padX - 5, y); ctx.stroke();
        ctx.fillText(Math.round(val), padX - 8, y + 4);
    }

    // Axis Labels
    ctx.fillStyle = '#9ca3af'; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Sum of Clusters (K_total)", padX + w/2, cvs.height - 5);
    ctx.save();
    ctx.translate(20, cvs.height / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Sum of Inertias", 0, 0);
    ctx.restore();

    // Optimal Frontier Line
    ctx.beginPath(); ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 3;
    points.forEach((p, i) => {
        const x = getX(p.x), y = getY(p.cost);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Points
    points.forEach(p => {
        const cx = getX(p.x), cy = getY(p.cost);
        drawnGlobalPoints.push({ cx, cy, cost: p.cost, sumK: p.x, configStr: p.configStr });
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI*2);
        ctx.fillStyle = (p.configStr === currentConfigStr) ? '#3b82f6' : '#d1d5db';
        ctx.fill();
    });

    // Current Highlight
    const cx = getX(currentSumK);
    const cy = getY(currentSumInertia);
    ctx.beginPath(); ctx.setLineDash([5, 5]); ctx.moveTo(cx, cy); ctx.lineTo(cx, padY + h);
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI*2);
    ctx.fillStyle = '#3b82f6'; ctx.fill();
}


function getGlobalPCAProjection(allSequences) {
    // 1. Run PCA on all, but collect the raw values first
    let allProjections = allSequences.map(seq => runPCARaw(seq.data, seq.shape[0], seq.shape[1]));

    // 2. Find min/max across all projections
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    allProjections.forEach(proj => {
        proj.forEach(p => {
            minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
        });
    });

    // 3. Define a shared range (square aspect ratio for consistency)
    const range = Math.max(maxX - minX, maxY - minY);
    return { minX, minY, range };
}

function runPCARaw(data, n, d) {
  // 1. Center the data (subtract mean)
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
  }
  const centered = new Float64Array(n * d);
  for (let i = 0; i < n * d; i++) centered[i] = data[i] - mean[i % d];

  // 2. Compute Covariance Matrix (simplified: X^T * X / (n-1))
  // We don't actually build the full d x d matrix to save memory.
  // We define a function for matrix-vector multiplication (Av = X^T * (X * v))
  const multiply = (v) => {
      let res = new Float64Array(d);
      for (let i = 0; i < n; i++) {
          let dot = 0;
          for (let j = 0; j < d; j++) dot += centered[i * d + j] * v[j];
          for (let j = 0; j < d; j++) res[j] += dot * centered[i * d + j];
      }
      return res;
  };

  // 3. Power Iteration for top 2 eigenvectors
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
  // For e2, we make the multiplication orthogonal to e1
  const multiply2 = (v) => {
      let res = multiply(v);
      let dot = res.reduce((a, b, i) => a + b * e1[i], 0);
      return res.map((val, i) => val - dot * e1[i]);
  };

  let v2 = new Float64Array(d).map(() => Math.random() - 0.5);
  for(let i=0; i<20; i++) { v2 = multiply2(v2); let norm = Math.sqrt(v2.reduce((a,b)=>a+b*b,0)); v2 = v2.map(x=>x/norm); }
  const e2 = v2;

    // ... [keep the same centering and eigenvector logic] ...

    let projection = [];
    for (let i = 0; i < n; i++) {
        let px = 0, py = 0;
        for (let j = 0; j < d; j++) {
            px += centered[i * d + j] * e1[j];
            py += centered[i * d + j] * e2[j];
        }
        projection.push([px, py]);
    }
    return projection; // Return raw coordinates, not normalized [0, 1]
}
function drawScatterPlots(seqs, clusters, { independentScale = false } = {}) {
    const container = document.getElementById('scatter-container');
    container.innerHTML = '';

    // In standard OT regime, keep a shared/global PCA scale across timesteps.
    // In GW regime (dimension mismatch), scale each PCA panel independently.
    let globalMinX = Infinity, globalMaxX = -Infinity, globalMinY = Infinity, globalMaxY = -Infinity;
    if (!independentScale) {
        state.pcaPoints.forEach(proj => {
            proj.forEach(p => {
                globalMinX = Math.min(globalMinX, p[0]); globalMaxX = Math.max(globalMaxX, p[0]);
                globalMinY = Math.min(globalMinY, p[1]); globalMaxY = Math.max(globalMaxY, p[1]);
            });
        });
    }
    const globalRange = Math.max(globalMaxX - globalMinX, globalMaxY - globalMinY) || 1;

    // 2. Render plots
    seqs.forEach((seq, t) => {
        const panel = document.createElement('div');
        panel.className = 'scatter-panel';
        panel.style.cssText = `text-align: center; margin-bottom: 20px; width: ${state.plotSize}px;`;

        const label = document.createElement('div');
        label.textContent = `Step ${t + 1} (PCA)`;
        label.style.color = '#9ca3af'; label.style.marginBottom = '5px';

        const canvas = document.createElement('canvas');
        canvas.width = state.plotSize; canvas.height = state.plotSize;
        canvas.style.backgroundColor = '#111827';
        canvas.style.border = '1px solid #374151';

        const ctx = canvas.getContext('2d');
        const points = state.pcaPoints[t]; // Use the anchored points

        let minX = globalMinX, minY = globalMinY, range = globalRange;
        if (independentScale) {
            let localMinX = Infinity, localMaxX = -Infinity, localMinY = Infinity, localMaxY = -Infinity;
            points.forEach(p => {
                localMinX = Math.min(localMinX, p[0]); localMaxX = Math.max(localMaxX, p[0]);
                localMinY = Math.min(localMinY, p[1]); localMaxY = Math.max(localMaxY, p[1]);
            });
            minX = localMinX;
            minY = localMinY;
            range = Math.max(localMaxX - localMinX, localMaxY - localMinY) || 1;
        }

        points.forEach((p, i) => {
            const clusterIdx = clusters[t].assignments[i];
            ctx.fillStyle = COLORS[clusterIdx % COLORS.length];

            // Map using either global or per-panel scale
            const x = ((p[0] - minX) / range) * canvas.width;
            const y = ((p[1] - minY) / range) * canvas.height;

            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();
        });

        panel.appendChild(label);
        panel.appendChild(canvas);
        container.appendChild(panel);
    });
}

        function drawGraph(clusters, couplings) {
            const { canvas, ctx } = elements;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawnEdgePaths = [];

            const T = clusters.length;
            const paddingX = 80; const paddingY = 60;
            const stepX = (canvas.width - 2 * paddingX) / (T > 1 ? T - 1 : 1);
            const getY = (t, i, k) => paddingY + (k > 1 ? i * ((canvas.height - 2 * paddingY) / (k - 1)) : (canvas.height / 2 - paddingY));

            for (let t = 0; t < T - 1; t++) {
                const P = couplings[t];
                const k1 = clusters[t].centers.length;
                const k2 = clusters[t+1].centers.length;

                let maxP = 0;
                for(let i=0; i<k1; i++) for(let j=0; j<k2; j++) if(P[i][j] > maxP) maxP = P[i][j];

                for (let i = 0; i < k1; i++) {
                    for (let j = 0; j < k2; j++) {
                        const mass = P[i][j];
                        if (mass < 1e-6) continue;

                        const x1 = paddingX + t * stepX, y1 = getY(t, i, k1);
                        const x2 = paddingX + (t + 1) * stepX, y2 = getY(t+1, j, k2);

                        const intensity = mass / (maxP || mass || 1);

                        const path = new Path2D();
                        path.moveTo(x1, y1);
                        path.bezierCurveTo(x1 + stepX/2, y1, x2 - stepX/2, y2, x2, y2);

                        const hex = COLORS[i % COLORS.length];
                        const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);

                        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${intensity * 0.8})`;
                        const lineWidth = Math.max(1, intensity * 25);
                        ctx.lineWidth = lineWidth;
                        ctx.stroke(path);

                        // --- PASTE THE STORAGE LINE HERE ---
                        drawnEdgePaths.push({
                            path: path,
                            mass: mass,
                            renderedWidth: lineWidth,
                            stepIdx: t,
                            sourceIdx: i
                        });
                        // ------------------------------------
                    }
                }
            }

            ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = "bold 13px sans-serif";

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

                ctx.fillStyle = '#9ca3af'; ctx.fillText(`Abstract State ${t+1}`, paddingX + t * stepX, 20);
            }
        }

        // --- NPY Parsing ---
        function npyHeader(buffer) {
            const view = new DataView(buffer);
            const headerLen = view.getUint16(8, true);
            const header = new TextDecoder().decode(new Uint8Array(buffer, 10, headerLen));
            const shape = (header.match(/'shape':\s*\(([^)]*)\)/) || [])[1]
                .split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const descr = (header.match(/'descr':\s*'([^']*)'/) || [])[1];
            return { shape, descr, offset: 10 + headerLen };
        }

        function isObjectDescr(descr) {
            return /O/i.test(descr);
        }

        function toFloat64Array(data) {
            return data instanceof Float64Array ? data : Float64Array.from(data);
        }

        function parsedMatrixToSequence(parsed) {
            const [n, d] = parsed.shape;
            return { shape: [n, d], data: toFloat64Array(parsed.data) };
        }

        async function parseSequenceNpyBuffer(buffer) {
            const { descr } = npyHeader(buffer);
            if (isObjectDescr(descr)) return pyodideNpyToSequences(buffer);

            const parsed = parseNPY(buffer);
            if (parsed.shape.length === 2) return [parsedMatrixToSequence(parsed)];
            if (parsed.shape.length === 3) {
                const [T, n, d] = parsed.shape;
                const seqs = [];
                for (let t = 0; t < T; t++) {
                    seqs.push({
                        shape: [n, d],
                        data: toFloat64Array(parsed.data.slice(t * n * d, (t + 1) * n * d))
                    });
                }
                return seqs;
            }
            throw new Error(`Unsupported Xs shape: (${parsed.shape.join(', ')})`);
        }

        async function extractSequences(zip) {
            let files = Object.keys(zip.files).filter(k => k.startsWith('Xs') && k.endsWith('.npy'));
            if (files.length === 0) return [];
            if (files.length > 1 || files[0] !== 'Xs.npy') {
                files.sort((a, b) => parseInt(a.match(/\d+/)?.[0] || 0) - parseInt(b.match(/\d+/)?.[0] || 0));
                const seqs = [];
                for (const file of files) {
                    const buffer = await zip.files[file].async('arraybuffer');
                    seqs.push(...await parseSequenceNpyBuffer(buffer));
                }
                return seqs;
            }
            const buffer = await zip.files['Xs.npy'].async('arraybuffer');
            return parseSequenceNpyBuffer(buffer);
        }

        function parseNPY(buffer) {
            const { shape, descr, offset } = npyHeader(buffer);
            if (isObjectDescr(descr)) {
                throw new Error('Object-dtype .npy requires Pyodide parsing.');
            }
            if (descr.includes('f8')) return { shape, data: new Float64Array(buffer.slice(offset)) };
            if (descr.includes('f4')) return { shape, data: Float64Array.from(new Float32Array(buffer.slice(offset))) };
            throw new Error(`Unsupported NPY dtype "${descr}" (expected float32/float64 or object array via .npz)`);
        }

        // Add this helper function
        function findElbow(inertiaCache) {
            const ks = Object.keys(inertiaCache).map(Number).sort((a, b) => a - b);
            if (ks.length < 3) return ks[0];

            // Points (x, y)
            const points = ks.map(k => ({ x: k, y: inertiaCache[k].inertia }));
            const first = points[0];
            const last = points[points.length - 1];

            // Find line equation: y = mx + c
            const m = (last.y - first.y) / (last.x - first.x);
            const c = first.y - m * first.x;

            let maxDist = -1;
            let elbowK = ks[0];

            // Maximize perpendicular distance
            points.forEach(p => {
                // Distance from point to line: |mx - y + c| / sqrt(m^2 + 1)
                const dist = Math.abs(m * p.x - p.y + c) / Math.sqrt(m * m + 1);
                if (dist > maxDist) {
                    maxDist = dist;
                    elbowK = p.x;
                }
            });
            return elbowK;
        }


        // --- Scalable Scatter Interaction ---
        function enableZoom(canvas) {
            let scale = 1.0;
            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                scale = Math.min(Math.max(scale * delta, 1.0), 5.0); // Limit zoom 1x to 5x
                canvas.style.transform = `scale(${scale})`;
                canvas.style.transformOrigin = 'center';
            });
        }

        // A. Linear PCA Projection
        function runPCA(data, n, d) {
      // 1. Center the data (subtract mean)
      const mean = new Float64Array(d);
      for (let i = 0; i < n; i++) {
          for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
      }
      const centered = new Float64Array(n * d);
      for (let i = 0; i < n * d; i++) centered[i] = data[i] - mean[i % d];

      // 2. Compute Covariance Matrix (simplified: X^T * X / (n-1))
      // We don't actually build the full d x d matrix to save memory.
      // We define a function for matrix-vector multiplication (Av = X^T * (X * v))
      const multiply = (v) => {
          let res = new Float64Array(d);
          for (let i = 0; i < n; i++) {
              let dot = 0;
              for (let j = 0; j < d; j++) dot += centered[i * d + j] * v[j];
              for (let j = 0; j < d; j++) res[j] += dot * centered[i * d + j];
          }
          return res;
      };

      // 3. Power Iteration for top 2 eigenvectors
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
      // For e2, we make the multiplication orthogonal to e1
      const multiply2 = (v) => {
          let res = multiply(v);
          let dot = res.reduce((a, b, i) => a + b * e1[i], 0);
          return res.map((val, i) => val - dot * e1[i]);
      };

      let v2 = new Float64Array(d).map(() => Math.random() - 0.5);
      for(let i=0; i<20; i++) { v2 = multiply2(v2); let norm = Math.sqrt(v2.reduce((a,b)=>a+b*b,0)); v2 = v2.map(x=>x/norm); }
      const e2 = v2;

      // 4. Project points into 2D and Normalize to [0, 1]
      let projection = [];
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

      for (let i = 0; i < n; i++) {
          let px = 0, py = 0;
          for (let j = 0; j < d; j++) {
              px += centered[i * d + j] * e1[j];
              py += centered[i * d + j] * e2[j];
          }
          projection.push([px, py]);
          minX = Math.min(minX, px); maxX = Math.max(maxX, px);
          minY = Math.min(minY, py); maxY = Math.max(maxY, py);
      }

      return projection.map(p => [
          (p[0] - minX) / (maxX - minX || 1),
          (p[1] - minY) / (maxY - minY || 1)
      ]);
  }