import { COLORS } from '../constants.js';

/** Same palette index as 2D PCA scatter (`COLORS[clusterIndex % n]`). */
export function clusterColorHex(clusterIndex) {
    const n = COLORS.length;
    const idx = ((Math.round(clusterIndex) % n) + n) % n;
    return COLORS[idx];
}

/** Float32 RGB (linear) for Three.js vertex colors — matches canvas hex when output is sRGB. */
export function assignmentColorsForThree(assignments, THREE) {
    const colors = new Float32Array(assignments.length * 3);
    const tmp = new THREE.Color();
    for (let i = 0; i < assignments.length; i++) {
        tmp.set(clusterColorHex(assignments[i]));
        colors[i * 3] = tmp.r;
        colors[i * 3 + 1] = tmp.g;
        colors[i * 3 + 2] = tmp.b;
    }
    return colors;
}
