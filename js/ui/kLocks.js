import { state } from '../state.js';

function clampK(k) {
    const v = Math.round(k);
    return Math.max(1, Math.min(15, Number.isFinite(v) ? v : 1));
}

export function ensureKLocks(T) {
    if (!Array.isArray(state.kLocks) || state.kLocks.length !== T) {
        const prev = state.kLocks || [];
        state.kLocks = Array(T).fill(false);
        for (let t = 0; t < T && t < prev.length; t++) {
            state.kLocks[t] = !!prev[t];
        }
    }
}

export function isStepKLocked(t) {
    return !!state.kLocks?.[t];
}

export function toggleKLock(stepIndex, T) {
    ensureKLocks(T);
    state.kLocks[stepIndex] = !state.kLocks[stepIndex];
    return state.kLocks[stepIndex];
}

export function syncKLockButton(btn, stepIndex) {
    if (!btn) return;
    const locked = isStepKLocked(stepIndex);
    btn.textContent = locked ? '🔒' : '🔓';
    btn.classList.toggle('k-lock-btn--locked', locked);
    btn.setAttribute('aria-pressed', locked ? 'true' : 'false');
    btn.title = locked
        ? 'K locked — AI explore will not change this step'
        : 'Lock K — AI explore will not change this step';
}

export function syncKLockSlider(slider, stepIndex) {
    if (!slider) return;
    const locked = isStepKLocked(stepIndex);
    slider.classList.toggle('k-slider-locked', locked);
    slider.disabled = locked;
    slider.setAttribute('aria-disabled', locked ? 'true' : 'false');
}

/** Force locked steps to their fixed K (safety net after mutations). */
export function enforceLockedKInVector(kValues, plan) {
    if (!plan?.fixedK || !kValues?.length) return kValues?.slice?.() ?? kValues;
    const out = kValues.slice();
    for (let t = 0; t < out.length; t++) {
        if (plan.fixedK[t] !== null) out[t] = plan.fixedK[t];
    }
    return out;
}

/** Apply locks into explore plan (fixed K at current slider values). */
export function applyKLocksToExplorePlan(plan, T, kValues) {
    ensureKLocks(T);
    const notes = [...(plan.notes || [])];
    const fixedK = plan.fixedK.slice();
    const exploreMask = plan.exploreMask.slice();
    const targetK = plan.targetK.slice();
    const annotationMinK = Array.isArray(plan.annotationMinK) ? plan.annotationMinK.slice() : null;

    for (let t = 0; t < T; t++) {
        if (!state.kLocks[t]) continue;
        const k = clampK(kValues[t]);
        fixedK[t] = k;
        exploreMask[t] = false;
        targetK[t] = null;
        if (annotationMinK) annotationMinK[t] = 1;
        notes.push(`Step ${t + 1} K locked at ${k}.`);
    }

    return {
        ...plan,
        fixedK,
        exploreMask,
        targetK,
        annotationMinK,
        notes: [...new Set(notes)]
    };
}

export function formatKLocksPromptBlock(T, kValues) {
    ensureKLocks(T);
    const locked = [];
    for (let t = 0; t < T; t++) {
        if (state.kLocks[t]) locked.push(`Step ${t + 1}: K=${kValues[t]}`);
    }
    if (!locked.length) return '';
    return `[Locked cluster counts — do NOT change these steps]\n${locked.map(l => `- ${l}`).join('\n')}`;
}
