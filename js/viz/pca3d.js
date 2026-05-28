/**
 * Lazy 3D PCA scatter (Three.js) with orbit rotate / zoom.
 */

import { assignmentColorsForThree } from './plotColors.js';
import { wrapPlotHost } from '../ui/plotFeedback.js';

// Resolved via import map in trajectory.html (OrbitControls imports bare "three").
let threeModulePromise = null;
let orbitModulePromise = null;

const viewsByHost = new WeakMap();

function loadThree() {
    if (!threeModulePromise) threeModulePromise = import('three');
    return threeModulePromise;
}

function loadOrbitControls() {
    if (!orbitModulePromise) {
        orbitModulePromise = import('three/addons/controls/OrbitControls.js');
    }
    return orbitModulePromise;
}

/** First three principal components (power iteration + Gram–Schmidt). */
export function runPCARaw3D(data, n, d) {
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
    }
    const centered = new Float64Array(n * d);
    for (let i = 0; i < n * d; i++) centered[i] = data[i] - mean[i % d];

    const multiply = (v) => {
        const res = new Float64Array(d);
        for (let i = 0; i < n; i++) {
            let dot = 0;
            for (let j = 0; j < d; j++) dot += centered[i * d + j] * v[j];
            for (let j = 0; j < d; j++) res[j] += dot * centered[i * d + j];
        }
        return res;
    };

    const powerEigen = (ortho = []) => {
        let v = new Float64Array(d).map(() => Math.random() - 0.5);
        for (const u of ortho) {
            let dot = 0;
            for (let j = 0; j < d; j++) dot += v[j] * u[j];
            for (let j = 0; j < d; j++) v[j] -= dot * u[j];
        }
        let norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
        v = v.map(x => x / norm);

        for (let iter = 0; iter < 24; iter++) {
            let w = multiply(v);
            for (const u of ortho) {
                let dot = 0;
                for (let j = 0; j < d; j++) dot += w[j] * u[j];
                for (let j = 0; j < d; j++) w[j] -= dot * u[j];
            }
            norm = Math.sqrt(w.reduce((a, b) => a + b * b, 0)) || 1;
            v = w.map(x => x / norm);
        }
        return v;
    };

    const e1 = powerEigen([]);
    const e2 = powerEigen([e1]);
    const e3 = powerEigen([e1, e2]);

    const projection = [];
    for (let i = 0; i < n; i++) {
        let px = 0;
        let py = 0;
        let pz = 0;
        for (let j = 0; j < d; j++) {
            const c = centered[i * d + j];
            px += c * e1[j];
            py += c * e2[j];
            pz += c * e3[j];
        }
        projection.push([px, py, pz]);
    }
    return projection;
}

export function disposePca3DInContainer(container) {
    if (!container) return;
    container.querySelectorAll('[data-pca3d-host]').forEach(host => disposePca3DView(host));
}

export function disposePca3DView(host) {
    const view = viewsByHost.get(host);
    if (!view) return;
    view.disposed = true;
    if (view.rafId) cancelAnimationFrame(view.rafId);
    view.resizeObserver?.disconnect();
    view.controls?.dispose();
    view.renderer?.dispose();
    if (view.geometry) view.geometry.dispose();
    if (view.material) view.material.dispose();
    viewsByHost.delete(host);
    host.innerHTML = '';
}

function bounds3(points) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const p of points) {
        if (!p) continue;
        minX = Math.min(minX, p[0]);
        maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]);
        maxY = Math.max(maxY, p[1]);
        minZ = Math.min(minZ, p[2]);
        maxZ = Math.max(maxZ, p[2]);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
    return { cx, cy, cz, span };
}

function setHostPlaceholder(host, text) {
    disposePca3DView(host);
    host.innerHTML = '';
    const msg = document.createElement('div');
    msg.className = 'pca3d-placeholder';
    msg.textContent = text;
    host.appendChild(msg);
}

export async function mountPca3DView({ host, points3d, assignments, plotSize }) {
    if (!host || !points3d?.length) return;

    disposePca3DView(host);
    host.innerHTML = '';

    const [THREE, { OrbitControls }] = await Promise.all([loadThree(), loadOrbitControls()]);
    if (!host.isConnected) return;

    const width = Math.max(200, plotSize);
    const height = Math.max(200, plotSize);
    host.style.width = '100%';
    host.style.height = `${height}px`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);

    const { cx, cy, cz, span } = bounds3(points3d);
    const positions = new Float32Array(points3d.length * 3);
    for (let i = 0; i < points3d.length; i++) {
        positions[i * 3] = (points3d[i][0] - cx) / span;
        positions[i * 3 + 1] = (points3d[i][1] - cy) / span;
        positions[i * 3 + 2] = (points3d[i][2] - cz) / span;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const colorAttr = new THREE.BufferAttribute(assignmentColorsForThree(assignments, THREE), 3);
    geometry.setAttribute('color', colorAttr);

    const material = new THREE.PointsMaterial({
        size: 0.028,
        sizeAttenuation: true,
        vertexColors: true,
        toneMapped: false
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);
    camera.position.set(0.85, 0.65, 1.1);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.65;
    controls.zoomSpeed = 1.1;
    controls.target.set(0, 0, 0);

    const hint = document.createElement('div');
    hint.className = 'pca3d-hint';
    hint.textContent = 'Drag to rotate · scroll to zoom';
    host.appendChild(hint);

    const view = {
        host,
        THREE,
        scene,
        camera,
        renderer,
        controls,
        geometry,
        material,
        colorAttr,
        points,
        pointCount: points3d.length,
        disposed: false,
        rafId: 0
    };
    viewsByHost.set(host, view);

    const tick = () => {
        if (view.disposed) return;
        controls.update();
        renderer.render(scene, camera);
        view.rafId = requestAnimationFrame(tick);
    };
    tick();

    const ro = new ResizeObserver(() => {
        if (view.disposed) return;
        const rect = host.getBoundingClientRect();
        const w = Math.max(120, Math.round(rect.width));
        const h = Math.max(120, Math.round(rect.height));
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
    });
    ro.observe(host);
    view.resizeObserver = ro;
}

export function updatePca3DViewColors(host, assignments) {
    const view = viewsByHost.get(host);
    if (!view?.colorAttr || !assignments || !view.THREE) return;
    const colors = assignmentColorsForThree(assignments, view.THREE);
    view.colorAttr.array.set(colors);
    view.colorAttr.needsUpdate = true;
}

/**
 * Mount 3D views for expanded sections that already have computed PCA.
 */
export async function syncPca3DPlotsInContainer(
    container,
    { pca3dPointsByStep, clusters, plotSize, pca3dRequestedSteps }
) {
    if (!container) return;
    const hosts = container.querySelectorAll('[data-pca3d-host]');
    for (const host of hosts) {
        const t = parseInt(host.dataset.stepIndex, 10);
        if (!Number.isFinite(t)) continue;

        const section = host.closest('.plot-section--pca3d');
        if (section?.classList.contains('plot-section--collapsed')) continue;

        const pts = pca3dPointsByStep?.[t];
        const pending = pca3dRequestedSteps?.has?.(t) && !pts;

        if (pending) {
            setHostPlaceholder(host, 'Computing…');
            continue;
        }
        if (!pts?.length || !clusters?.[t]?.assignments) {
            setHostPlaceholder(host, 'Expand to compute');
            continue;
        }

        const existing = viewsByHost.get(host);
        if (existing?.pointCount === pts.length && !existing.disposed) {
            updatePca3DViewColors(host, clusters[t].assignments);
            continue;
        }

        try {
            await mountPca3DView({
                host,
                points3d: pts,
                assignments: clusters[t].assignments,
                plotSize
            });
        } catch (err) {
            console.error('3D PCA mount failed', err);
            setHostPlaceholder(host, `3D view error: ${err.message}`);
        }
    }
}

/** Build collapsible-section body content for one step. */
export function buildPca3DSectionBody(body, t, plotSize, plotId) {
    const host = document.createElement('div');
    host.className = 'pca3d-viewport';
    host.dataset.pca3dHost = '1';
    host.dataset.stepIndex = String(t);
    body.appendChild(wrapPlotHost(host, plotId));
}
