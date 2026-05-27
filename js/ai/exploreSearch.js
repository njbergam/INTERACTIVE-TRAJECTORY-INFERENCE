/**
 * Smarter local search for AI explore: deduplication, elbow-aware scoring bias,
 * and diverse K proposals (not only random walks upward from the current best).
 */

import { state } from '../state.js';
import { findElbow } from '../algorithms/kmeans.js';
import { applyPromptConstraints, proposeCandidateWithPlan } from './promptPlan.js';

function clampInt(x, lo, hi) {
    const v = Math.round(x);
    return Math.max(lo, Math.min(hi, Number.isFinite(v) ? v : lo));
}

export function kConfigKey(kValues) {
    return kValues.join(',');
}

export function hasTriedConfig(triedSet, kValues) {
    return triedSet.has(kConfigKey(kValues));
}

export function rememberTriedConfig(triedSet, kValues) {
    triedSet.add(kConfigKey(kValues));
}

/** Per-step elbow K from cached sweeps (falls back to 3 if cache empty). */
export function getElbowKVector(T, plan) {
    const k = [];
    for (let t = 0; t < T; t++) {
        const cache = state.cache[t];
        const keys = cache ? Object.keys(cache) : [];
        k.push(keys.length >= 2 ? findElbow(cache) : 3);
    }
    return applyPromptConstraints(k, plan);
}

/**
 * Penalize K above the local elbow more strongly than below (parsimony).
 */
export function exploreComplexityPenalty(kValues, plan = null) {
    let pen = 0;
    const T = kValues.length;
    for (let t = 0; t < T; t++) {
        const cache = state.cache[t];
        if (!cache || Object.keys(cache).length < 2) continue;
        const elbowK = findElbow(cache);
        const minAnn = plan?.annotationMinK?.[t] ?? 1;
        const effectiveElbow = plan?.prioritizeAnnotations ? Math.max(elbowK, minAnn) : elbowK;
        const excess = kValues[t] - effectiveElbow;
        if (excess > 0) pen += 14 * excess * excess;
        else pen += 3 * excess * excess;
    }
    const sumK = kValues.reduce((a, b) => a + b, 0);
    const avgK = sumK / Math.max(T, 1);
    if (avgK > 6) pen += 8 * (avgK - 6) * (avgK - 6);
    if (plan?.prioritizeAnnotations) pen *= 0.4;
    return pen;
}

function mutateKValues(base, plan, { preferDecrease = false, maxStep = 2 } = {}) {
    const T = base.length;
    const out = applyPromptConstraints(base.slice(), plan);
    const mutable = [];
    for (let t = 0; t < T; t++) {
        if (plan.exploreMask[t] && plan.fixedK[t] === null) mutable.push(t);
    }
    if (!mutable.length) return out;

    const nChange = clampInt(1 + Math.random() * Math.min(2, mutable.length), 1, mutable.length);
    const steps = mutable.sort(() => Math.random() - 0.5).slice(0, nChange);

    for (const t of steps) {
        if (plan.targetK[t] !== null && Math.random() < 0.4) {
            out[t] = plan.targetK[t];
            continue;
        }
        const cache = state.cache[t];
        const elbowK = cache && Object.keys(cache).length >= 2 ? findElbow(cache) : out[t];
        let sign;
        if (out[t] > elbowK) sign = preferDecrease || Math.random() < 0.72 ? -1 : 1;
        else if (out[t] < elbowK) sign = Math.random() < 0.55 ? 1 : -1;
        else sign = Math.random() < 0.5 ? -1 : 1;

        const step = sign * clampInt(1 + Math.random() * maxStep, 1, maxStep);
        out[t] = clampInt(out[t] + step, 1, 15);
    }
    return applyPromptConstraints(out, plan);
}

function randomNeighborhoodSample(plan, T, triedSet, maxAttempts = 48) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const k = [];
        for (let t = 0; t < T; t++) {
            if (plan.fixedK[t] !== null) {
                k.push(plan.fixedK[t]);
                continue;
            }
            const cache = state.cache[t];
            const elbowK = cache && Object.keys(cache).length >= 2 ? findElbow(cache) : 4;
            const lo = Math.max(1, elbowK - 3);
            const hi = Math.min(15, elbowK + 2);
            k.push(lo + Math.floor(Math.random() * (hi - lo + 1)));
        }
        const cand = applyPromptConstraints(k, plan);
        if (!hasTriedConfig(triedSet, cand)) return cand;
    }
    return null;
}

/** Diverse candidate: elbow point, random near-elbow, downward mutate, or small random walk. */
export function proposeExploreCandidate({ plan, best, startingK, triedSet, T }) {
    const r = Math.random();

    if (plan.prioritizeAnnotations && Array.isArray(plan.annotationMinK) && Math.random() < 0.35) {
        const annK = applyPromptConstraints(
            best.kValues.map((k, t) => clampInt(Math.max(k, plan.annotationMinK[t] || 1), 1, 15)),
            plan
        );
        if (!hasTriedConfig(triedSet, annK)) return annK;
    }

    if (r < 0.22) {
        const elbow = getElbowKVector(T, plan);
        if (!hasTriedConfig(triedSet, elbow)) return elbow;
    }

    if (r < 0.44) {
        const novel = randomNeighborhoodSample(plan, T, triedSet);
        if (novel) return novel;
    }

    if (r < 0.62) {
        const from = Math.random() < 0.5 ? best.kValues : startingK;
        const down = mutateKValues(from, plan, { preferDecrease: true, maxStep: 2 });
        if (!hasTriedConfig(triedSet, down)) return down;
    }

    if (r < 0.78) {
        const up = mutateKValues(best.kValues, plan, { preferDecrease: false, maxStep: 1 });
        if (!hasTriedConfig(triedSet, up)) return up;
    }

    const walk = proposeCandidateWithPlan(best.kValues, plan, { preferDecrease: true });
    if (!hasTriedConfig(triedSet, walk)) return walk;

    const novel = randomNeighborhoodSample(plan, T, triedSet, 64);
    if (novel) return novel;

    return getElbowKVector(T, plan);
}

/** Seed queue with elbow + starting ±1 variants (deduped). */
export function seedExploreCandidates(plan, T, startingK, triedSet) {
    const out = [];
    const push = (k) => {
        const c = applyPromptConstraints(k, plan);
        if (!hasTriedConfig(triedSet, c)) {
            rememberTriedConfig(triedSet, c);
            out.push(c);
        }
    };

    push(startingK.slice());
    push(getElbowKVector(T, plan));

    const minus = startingK.map((k, t) =>
        plan.fixedK[t] !== null ? plan.fixedK[t] : clampInt(k - 1, 1, 15)
    );
    const plus = startingK.map((k, t) =>
        plan.fixedK[t] !== null ? plan.fixedK[t] : clampInt(k + 1, 1, 15)
    );
    push(minus);
    push(plus);

    if (Array.isArray(plan.annotationMinK)) {
        const atMin = startingK.map((k, t) =>
            plan.fixedK[t] !== null ? plan.fixedK[t] : clampInt(Math.max(k, plan.annotationMinK[t] || 1), 1, 15)
        );
        push(atMin);
        const aboveMin = atMin.map((k, t) =>
            plan.fixedK[t] !== null ? plan.fixedK[t] : clampInt(k + 1, 1, 15)
        );
        push(aboveMin);
    }

    return out;
}

export function filterUntriedConfigs(candidates, triedSet, plan) {
    const out = [];
    for (const k of candidates) {
        const c = applyPromptConstraints(k, plan);
        if (!hasTriedConfig(triedSet, c)) {
            rememberTriedConfig(triedSet, c);
            out.push(c);
        }
    }
    return out;
}
