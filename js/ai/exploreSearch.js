/**
 * Selective local search for AI explore: elbow-bounded K, heuristic ranking,
 * and phased refinement (broad seed → local polish) instead of random walks.
 */

import { state, rawSequences } from '../state.js';
import { kMeans, findElbow } from '../algorithms/kmeans.js';
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

export function getElbowKForStep(t) {
    const cache = state.cache[t];
    const keys = cache ? Object.keys(cache) : [];
    return keys.length >= 2 ? findElbow(cache) : 3;
}

/** Per-step elbow K from cached sweeps (falls back to 3 if cache empty). */
export function getElbowKVector(T, plan) {
    const k = [];
    for (let t = 0; t < T; t++) k.push(getElbowKForStep(t));
    return applyPromptConstraints(k, plan);
}

/**
 * Allowed K search window per step: tight band around elbow (+ annotation / prompt targets).
 */
export function getAllowedKRange(t, plan) {
    if (plan.fixedK[t] !== null) {
        const fk = clampInt(plan.fixedK[t], 1, 15);
        return [fk, fk];
    }

    const elbowK = getElbowKForStep(t);
    const annMin = plan.annotationMinK?.[t] ?? 1;
    const below = plan.prioritizeAnnotations ? 1 : 2;
    const above = plan.prioritizeAnnotations ? 2 : 1;

    let lo = Math.max(1, elbowK - below, annMin);
    let hi = Math.min(15, elbowK + above);

    if (plan.targetK[t] !== null) {
        const tk = plan.targetK[t];
        lo = Math.min(lo, tk);
        hi = Math.max(hi, tk);
    }

    if (Array.isArray(plan.annotationMinK) && plan.annotationMinK[t] > 1) {
        lo = Math.max(lo, plan.annotationMinK[t]);
        hi = Math.max(hi, plan.annotationMinK[t] + 1);
    }

    return [clampInt(lo, 1, 15), clampInt(hi, 1, 15)];
}

export function clampKInRange(k, t, plan) {
    const [lo, hi] = getAllowedKRange(t, plan);
    return clampInt(k, lo, hi);
}

/** Lower = more promising (try first). Cheap proxy — not a full OT score. */
export function estimateCandidateHeuristic(kValues, plan, best = null) {
    let h = 0;
    const T = kValues.length;

    for (let t = 0; t < T; t++) {
        const elbowK = getElbowKForStep(t);
        const [lo, hi] = getAllowedKRange(t, plan);
        const k = kValues[t];

        if (k < lo || k > hi) h += 120 + 40 * Math.max(lo - k, k - hi, 0);

        const dElbow = k - elbowK;
        if (dElbow > 0) h += 18 * dElbow * dElbow;
        else h += 6 * dElbow * dElbow;

        if (plan.targetK[t] !== null) {
            const d = k - plan.targetK[t];
            h += 25 * d * d;
        }

        if (best?.kValues) {
            const db = k - best.kValues[t];
            h += 9 * db * db;
        }
    }

    const sumK = kValues.reduce((a, b) => a + b, 0);
    const avgK = sumK / Math.max(T, 1);
    if (avgK > 6) h += 12 * (avgK - 6) * (avgK - 6);

    return h + exploreComplexityPenalty(kValues, plan) * 0.15;
}

export function rankCandidatesByHeuristic(candidates, plan, best) {
    return candidates
        .slice()
        .sort((a, b) => estimateCandidateHeuristic(a, plan, best) - estimateCandidateHeuristic(b, plan, best));
}

/** Pop the most promising queued config (not FIFO). */
export function dequeueBestCandidate(queue, plan, best) {
    if (!queue.length) return null;
    if (queue.length === 1) return queue.shift();
    const ranked = rankCandidatesByHeuristic(queue, plan, best);
    const pick = ranked[0];
    const idx = queue.findIndex(c => kConfigKey(c) === kConfigKey(pick));
    if (idx >= 0) queue.splice(idx, 1);
    return pick;
}

/**
 * Precompute k-means only for K values in the elbow window (+ anchors), not all 1–15.
 */
export function warmKMeansCacheForExplore(T, plan, anchorVectors = []) {
    const warmed = [];
    for (let t = 0; t < T; t++) {
        const [lo, hi] = getAllowedKRange(t, plan);
        const ks = new Set();
        for (let k = lo; k <= hi; k++) ks.add(k);

        for (const vec of anchorVectors) {
            if (!vec?.length) continue;
            for (let k = Math.max(1, vec[t] - 1); k <= Math.min(15, vec[t] + 1); k++) ks.add(k);
        }

        const sorted = [...ks].sort((a, b) => a - b);
        warmed.push(sorted);
        for (const k of sorted) {
            if (!state.cache[t][k]) state.cache[t][k] = kMeans(rawSequences[t], k);
        }
    }
    return warmed;
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
        if (excess > 0) pen += 18 * excess * excess;
        else pen += 4 * excess * excess;

        const [lo, hi] = getAllowedKRange(t, plan || { fixedK: [], targetK: [], exploreMask: [], annotationMinK: null });
        if (kValues[t] < lo) pen += 30 * (lo - kValues[t]) ** 2;
        if (kValues[t] > hi) pen += 45 * (kValues[t] - hi) ** 2;
    }
    const sumK = kValues.reduce((a, b) => a + b, 0);
    const avgK = sumK / Math.max(T, 1);
    if (avgK > 6) pen += 10 * (avgK - 6) * (avgK - 6);
    if (plan?.prioritizeAnnotations) pen *= 0.4;
    return pen;
}

function clampVectorToSearchWindow(kValues, plan) {
    const out = kValues.slice();
    for (let t = 0; t < out.length; t++) {
        if (plan.fixedK[t] !== null) out[t] = plan.fixedK[t];
        else out[t] = clampKInRange(out[t], t, plan);
    }
    return applyPromptConstraints(out, plan);
}

function mutateTowardElbow(base, plan, { maxStepsChanged = 1 } = {}) {
    const T = base.length;
    const out = applyPromptConstraints(base.slice(), plan);
    const mutable = [];
    for (let t = 0; t < T; t++) {
        if (plan.exploreMask[t] && plan.fixedK[t] === null) mutable.push(t);
    }
    if (!mutable.length) return out;

    const nChange = clampInt(1 + Math.random() * Math.min(maxStepsChanged, mutable.length), 1, maxStepsChanged);
    const steps = mutable.sort(() => Math.random() - 0.5).slice(0, nChange);

    for (const t of steps) {
        if (plan.targetK[t] !== null && Math.random() < 0.55) {
            out[t] = clampKInRange(plan.targetK[t], t, plan);
            continue;
        }
        const elbowK = getElbowKForStep(t);
        if (out[t] > elbowK) out[t] = clampKInRange(out[t] - 1, t, plan);
        else if (out[t] < elbowK) out[t] = clampKInRange(out[t] + 1, t, plan);
        else out[t] = clampKInRange(out[t] + (Math.random() < 0.5 ? -1 : 1), t, plan);
    }
    return applyPromptConstraints(out, plan);
}

function mutateFromBest(best, plan, { maxStepsChanged = 1 } = {}) {
    return mutateTowardElbow(best.kValues, plan, { maxStepsChanged });
}

function refinedNeighborhoodSample(plan, T, triedSet, best, maxAttempts = 32) {
    const elbow = getElbowKVector(T, plan);
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const k = elbow.slice();
        const mutable = [];
        for (let t = 0; t < T; t++) {
            if (plan.exploreMask[t] && plan.fixedK[t] === null) mutable.push(t);
        }
        if (!mutable.length) break;

        const nTweak = clampInt(1 + Math.random() * Math.min(2, mutable.length), 1, 2);
        const pick = mutable.sort(() => Math.random() - 0.5).slice(0, nTweak);
        for (const t of pick) {
            const [lo, hi] = getAllowedKRange(t, plan);
            const options = [];
            for (let kk = lo; kk <= hi; kk++) options.push(kk);
            if (best?.kValues) options.push(clampKInRange(best.kValues[t], t, plan));
            k[t] = options[(Math.random() * options.length) | 0];
        }

        const cand = applyPromptConstraints(k, plan);
        if (!hasTriedConfig(triedSet, cand)) return cand;
    }
    return null;
}

function proposeExploreCandidateOnce({ plan, best, startingK, triedSet, T, refineLocally = false }) {
    if (plan.prioritizeAnnotations && Array.isArray(plan.annotationMinK) && Math.random() < 0.4) {
        const annK = applyPromptConstraints(
            best.kValues.map((k, t) => clampInt(Math.max(k, plan.annotationMinK[t] || 1), 1, 15)),
            plan
        );
        if (!hasTriedConfig(triedSet, annK)) return clampVectorToSearchWindow(annK, plan);
    }

    const tryList = [];

    const elbow = getElbowKVector(T, plan);
    tryList.push(clampVectorToSearchWindow(elbow, plan));

    if (best?.kValues) {
        tryList.push(mutateFromBest(best, plan, { maxStepsChanged: 1 }));
        tryList.push(mutateFromBest(best, plan, { maxStepsChanged: 2 }));
        tryList.push(clampVectorToSearchWindow(best.kValues, plan));
    }

    tryList.push(mutateTowardElbow(startingK, plan, { maxStepsChanged: refineLocally ? 1 : 2 }));

    const refined = refinedNeighborhoodSample(plan, T, triedSet, best);
    if (refined) tryList.push(refined);

    if (!refineLocally) {
        const novel = refinedNeighborhoodSample(plan, T, triedSet, best, 48);
        if (novel) tryList.push(novel);
    } else {
        tryList.push(proposeCandidateWithPlan(best.kValues, plan, { preferDecrease: true }));
    }

    const untried = tryList
        .map(c => clampVectorToSearchWindow(c, plan))
        .filter(c => !hasTriedConfig(triedSet, c));

    if (untried.length) {
        return rankCandidatesByHeuristic(untried, plan, best)[0];
    }

    return clampVectorToSearchWindow(mutateFromBest(best, plan, { maxStepsChanged: 1 }), plan);
}

/** Selective candidate: rank several local proposals, prefer elbow-neighborhood settings. */
export function proposeExploreCandidate({ plan, best, startingK, triedSet, T, trials = 0, budgetMs = 10_000, exploreStart = 0 }) {
    const elapsed = exploreStart > 0 ? (performance.now() - exploreStart) / Math.max(budgetMs, 1) : 0;
    const refineLocally = trials >= 5 || elapsed > 0.35;

    for (let attempt = 0; attempt < 10; attempt++) {
        const cand = proposeExploreCandidateOnce({
            plan,
            best,
            startingK,
            triedSet,
            T,
            refineLocally
        });
        if (!hasTriedConfig(triedSet, cand)) return cand;
    }

    const elbow = getElbowKVector(T, plan);
    return clampVectorToSearchWindow(elbow, plan);
}

/** Seed queue with elbow + tight neighbors (deduped, ranked). */
export function seedExploreCandidates(plan, T, startingK, triedSet, best = null) {
    const raw = [];
    const push = (k) => {
        raw.push(clampVectorToSearchWindow(k, plan));
    };

    push(startingK.slice());
    push(getElbowKVector(T, plan));

    const towardElbow = startingK.map((k, t) => {
        if (plan.fixedK[t] !== null) return plan.fixedK[t];
        const e = getElbowKForStep(t);
        if (k > e) return clampKInRange(k - 1, t, plan);
        if (k < e) return clampKInRange(k + 1, t, plan);
        return k;
    });
    push(towardElbow);

    if (Array.isArray(plan.annotationMinK)) {
        const atMin = startingK.map((k, t) =>
            plan.fixedK[t] !== null ? plan.fixedK[t] : clampInt(Math.max(k, plan.annotationMinK[t] || 1), 1, 15)
        );
        push(atMin);
    }

    const ranked = rankCandidatesByHeuristic(raw, plan, best);
    const out = [];
    for (const k of ranked) {
        if (!hasTriedConfig(triedSet, k)) {
            rememberTriedConfig(triedSet, k);
            out.push(k);
        }
    }
    return out;
}

export function filterUntriedConfigs(candidates, triedSet, plan) {
    const ranked = rankCandidatesByHeuristic(
        candidates.map(k => clampVectorToSearchWindow(k, plan)),
        plan,
        null
    );
    const out = [];
    for (const k of ranked) {
        if (!hasTriedConfig(triedSet, k)) {
            rememberTriedConfig(triedSet, k);
            out.push(k);
        }
    }
    return out;
}
