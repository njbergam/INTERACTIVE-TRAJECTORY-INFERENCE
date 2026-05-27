import { pyodideNpyToSequences } from './pyodide.js';

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

function parseNPY(buffer) {
    const { shape, descr, offset } = npyHeader(buffer);
    if (isObjectDescr(descr)) {
        throw new Error('Object-dtype .npy requires Pyodide parsing.');
    }
    if (descr.includes('f8')) return { shape, data: new Float64Array(buffer.slice(offset)) };
    if (descr.includes('f4')) return { shape, data: Float64Array.from(new Float32Array(buffer.slice(offset))) };
    throw new Error(`Unsupported NPY dtype "${descr}" (expected float32/float64 or object array via .npz)`);
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

export async function extractSequencesFromZip(zip) {
    let files = Object.keys(zip.files).filter(k => k.startsWith('Xs') && k.endsWith('.npy'));
    if (files.length === 0) return [];

    if (files.length > 1 || files[0] !== 'Xs.npy') {
        files.sort((a, b) => parseInt(a.match(/\d+/)?.[0] || 0) - parseInt(b.match(/\d+/)?.[0] || 0));
        const seqs = [];
        for (const file of files) {
            const buffer = await zip.files[file].async('arraybuffer');
            seqs.push(...await parseSequenceNpyBuffer(buffer));
        }
        // Optional `pis` (true OT marginals/couplings between successive timesteps)
        const pis = await extractPisFromZip(zip);
        return { sequences: seqs, pis };
    }

    const buffer = await zip.files['Xs.npy'].async('arraybuffer');
    const seqs = await parseSequenceNpyBuffer(buffer);
    const pis = await extractPisFromZip(zip);
    return { sequences: seqs, pis };
}

async function extractPisFromZip(zip) {
    // Supported conventions:
    // - `pis.npy` containing object array of (n_t, n_{t+1}) arrays or sparse matrices
    // - `pis_0.npy`, `pis_1.npy`, ... (each a single (n_t, n_{t+1}) array)
    const keys = Object.keys(zip.files);
    let piFiles = keys.filter(k => (k === 'pis.npy' || k.startsWith('pis_')) && k.endsWith('.npy'));
    if (piFiles.length === 0) return null;

    if (piFiles.length === 1 && piFiles[0] === 'pis.npy') {
        const buffer = await zip.files['pis.npy'].async('arraybuffer');
        const mats = await parseSequenceNpyBuffer(buffer);
        return mats.map(m => ({ shape: m.shape, data: m.data }));
    }

    piFiles.sort((a, b) => parseInt(a.match(/\d+/)?.[0] || 0) - parseInt(b.match(/\d+/)?.[0] || 0));
    const out = [];
    for (const file of piFiles) {
        const buffer = await zip.files[file].async('arraybuffer');
        const mats = await parseSequenceNpyBuffer(buffer);
        out.push(...mats.map(m => ({ shape: m.shape, data: m.data })));
    }
    return out.length ? out : null;
}

