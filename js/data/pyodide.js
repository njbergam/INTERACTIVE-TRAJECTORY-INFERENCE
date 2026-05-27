let pyodidePromise = null;

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

export async function getPyodide() {
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

export async function pyodideNpyToSequences(arrayBuffer) {
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

export async function extractSequencesFromPkl(arrayBuffer) {
    const pyodide = await getPyodide();
    pyodide.FS.writeFile('/input.pkl', new Uint8Array(arrayBuffer));
    let seqProxies;
    try {
        seqProxies = pyodide.runPython('extract_pkl_sequences(\"/input.pkl\")');
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

