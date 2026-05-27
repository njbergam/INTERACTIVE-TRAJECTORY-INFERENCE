/**
 * Parse natural-language AI explore prompts into structured constraints.
 * Supports per-step K (fixed / preferred), global K, and objective weight hints.
 */

function clampInt(x, lo, hi) {
    const v = Math.round(x);
    return Math.max(lo, Math.min(hi, Number.isFinite(v) ? v : lo));
}

/** 1-based step index from user text → 0-based array index */
function stepIndexFromUser(n, T) {
    const i = parseInt(n, 10);
    if (!Number.isFinite(i)) return null;
    if (i >= 1 && i <= T) return i - 1;
    if (i >= 0 && i < T) return i;
    return null;
}

function parseObjectiveWeights(lower) {
    let wInertia = 0.55;
    let wTrue = 0.45;
    let wSep = 0.15;

    if (/\binertia\b|\bkmeans\b|\belbow\b/.test(lower)) {
        wInertia += 0.15;
        wSep += 0.05;
    }
    if (/\btrue\b.*\bot\b|\bpis\b|\btransport\b|\btrajectory\b/.test(lower)) {
        wTrue += 0.20;
    }
    if (/\bseparation\b|\bseparate\b|\bclusters?\b.*\bsepar\b|\bclear\b|\bclean\b|\bdendrogram\b|\bkpca\b|\bkernel\b|\bpca\b/.test(lower)) {
        wSep += 0.15;
    }
    if (/\bspeed\b|\bfast\b|\blight\b|\bsimple\b/.test(lower)) {
        wSep *= 0.8;
    }
    if (/\bcompact\b|\btight\b|\bsmall\b/.test(lower)) {
        wSep += 0.05;
    }
    if (/\bovercluster|\btoo many clusters|\bhigh k\b/.test(lower)) {
        wSep += 0.08;
    }

    const s = wInertia + wTrue;
    if (s > 0) {
        wInertia /= s;
        wTrue /= s;
    }
    return { wInertia, wTrue, wSep };
}

/**
 * @param {string} promptRaw
 * @param {number} T number of timesteps
 * @returns plan object used by AI explore
 */
export function parseExplorePrompt(promptRaw, T) {
    const prompt = (promptRaw || '').trim();
    const lower = prompt.toLowerCase();
    const notes = [];

    const fixedK = Array(T).fill(null);
    const targetK = Array(T).fill(null);
    const exploreMask = Array(T).fill(true);
    let globalK = null;
    let globalFixed = false;

    const setFixed = (t, k, note) => {
        if (t === null || t < 0 || t >= T) return;
        fixedK[t] = clampInt(k, 1, 15);
        exploreMask[t] = false;
        if (note) notes.push(note);
    };

    const setTarget = (t, k, note) => {
        if (t === null || t < 0 || t >= T) return;
        if (fixedK[t] !== null) return;
        targetK[t] = clampInt(k, 1, 15);
        if (note) notes.push(note);
    };

    // --- Global K ---
    const globalPatterns = [
        /\b(?:all|every)(?:\s+steps?)?\s+(?:to\s+)?(?:k\s*=?\s*)?(\d{1,2})\b/i,
        /\b(?:k|clusters?)\s*=?\s*(\d{1,2})\s+(?:for\s+)?(?:all|everywhere|each)\b/i,
        /\bset\s+(?:all\s+)?(?:steps?\s+)?(?:to\s+)?(\d{1,2})\s+clusters?\b/i,
    ];
    for (const re of globalPatterns) {
        const m = lower.match(re);
        if (m) {
            globalK = clampInt(m[1], 1, 15);
            if (/\bfix\b|\block\b|\bkeep\b|\bmust\b|\bexactly\b/.test(lower)) {
                globalFixed = true;
                for (let t = 0; t < T; t++) setFixed(t, globalK, `All steps fixed at K=${globalK}.`);
            } else {
                notes.push(`Global target K=${globalK} for all steps.`);
            }
            break;
        }
    }

    // --- Per-step: "step 2 k=5", "k=5 for step 2", "step 2: 5 clusters" ---
    const seen = new Set();
    const recordStepK = (stepNum, kVal, index, matchLen) => {
        const t = stepIndexFromUser(stepNum, T);
        const k = clampInt(kVal, 1, 15);
        if (t === null) return;
        const key = `${t}:${k}`;
        if (seen.has(key)) return;
        seen.add(key);

        const snippet = prompt.slice(Math.max(0, index - 25), index + matchLen + 25).toLowerCase();
        const fix = /\b(fix|fixed|lock|keep|must|exactly|always)\b/.test(snippet);
        const soft = /\b(consider|try|prefer|around|about|evaluate|test|explore|maybe|roughly)\b/.test(snippet);

        if (fix && !soft) setFixed(t, k, `Step ${t + 1} fixed at K=${k}.`);
        else setTarget(t, k, `Step ${t + 1} target K=${k}.`);
    };

    const patterns = [
        { re: /\b(?:step|timestep|timepoint|time\s*point|margin|marginal|t)\s*(\d{1,2})\b[^.\n]{0,45}?\b(?:k\s*=?\s*|clusters?\s*=?\s*|to\s+|at\s+|=\s*|:\s*)(\d{1,2})\b/gi, step: 1, k: 2 },
        { re: /\b(?:k\s*=?\s*)(\d{1,2})\b[^.\n]{0,28}?\b(?:for|on)\s+(?:step|timestep|timepoint|t)\s*(\d{1,2})\b/gi, step: 2, k: 1 },
        { re: /\b(\d{1,2})\s+clusters?\b[^.\n]{0,22}?\b(?:for|on|at)\s+(?:step|timestep|timepoint|t)\s*(\d{1,2})\b/gi, step: 2, k: 1 },
        { re: /\b(?:consider|try|use|prefer|evaluate|test)\b[^.\n]{0,25}?\b(?:step|timestep|t)\s*(\d{1,2})\b[^.\n]{0,35}?\b(?:k\s*=?\s*|at\s+|=\s*)?(\d{1,2})\b/gi, step: 1, k: 2 },
        { re: /\b(?:step|timestep|t)\s*(\d{1,2})\b\s+(?:with\s+)?(?:k\s*=?\s*)?(\d{1,2})\b/gi, step: 1, k: 2 },
    ];

    for (const { re, step, k } of patterns) {
        let m;
        while ((m = re.exec(prompt)) !== null) {
            recordStepK(m[step], m[k], m.index, m[0].length);
        }
    }

    // "only change / explore step 2" or "don't change step 1"
    const onlyChange = lower.match(/\bonly\s+(?:change|explore|tune|adjust|vary)\s+(?:step|timestep|t)\s*(\d{1,2})\b/);
    if (onlyChange) {
        const t = stepIndexFromUser(onlyChange[1], T);
        if (t !== null) {
            for (let i = 0; i < T; i++) exploreMask[i] = i === t;
            notes.push(`Only varying step ${t + 1}.`);
        }
    }

    const onlyThese = [...lower.matchAll(/\bonly\s+(?:change|explore)\s+steps?\s*([\d\s,and\-]+)/g)];
    for (const m of onlyThese) {
        const part = m[1];
        const nums = part.match(/\d{1,2}/g);
        if (nums && nums.length) {
            exploreMask.fill(false);
            for (const n of nums) {
                const t = stepIndexFromUser(n, T);
                if (t !== null) exploreMask[t] = true;
            }
            notes.push(`Only varying steps: ${nums.map(n => stepIndexFromUser(n, T) + 1).join(', ')}.`);
        }
    }

    const dontChange = [...lower.matchAll(/\b(?:don'?t|do not|leave)\s+(?:change|touch|modify)\s+(?:step|timestep|t)\s*(\d{1,2})\b/g)];
    for (const m of dontChange) {
        const t = stepIndexFromUser(m[1], T);
        if (t !== null) {
            exploreMask[t] = false;
            notes.push(`Leaving step ${t + 1} unchanged.`);
        }
    }

    // Apply global soft target to unset steps
    if (globalK !== null && !globalFixed) {
        for (let t = 0; t < T; t++) {
            if (fixedK[t] === null && targetK[t] === null) targetK[t] = globalK;
        }
    }

    const weights = parseObjectiveWeights(lower);

    if (notes.length === 0 && prompt) {
        notes.push('No explicit per-step K found — using prompt for objective weights only.');
    }

    return {
        weights,
        fixedK,
        targetK,
        exploreMask,
        globalK,
        globalFixed,
        notes: [...new Set(notes)]
    };
}

export function applyPromptConstraints(kValues, plan) {
    const out = kValues.slice();
    const T = out.length;
    const minK = plan.annotationMinK;
    for (let t = 0; t < T; t++) {
        if (plan.fixedK[t] !== null) out[t] = plan.fixedK[t];
        else if (Array.isArray(minK) && minK[t] > 1) {
            out[t] = clampInt(Math.max(out[t], minK[t]), 1, 15);
        }
    }
    return out;
}

export function promptConstraintPenalty(kValues, plan) {
    let penalty = 0;
    const T = kValues.length;
    for (let t = 0; t < T; t++) {
        if (plan.fixedK[t] !== null && kValues[t] !== plan.fixedK[t]) {
            penalty += 1e8;
        }
        if (plan.fixedK[t] === null && plan.targetK[t] !== null) {
            const d = Math.abs(kValues[t] - plan.targetK[t]);
            // Strong preference so "consider K=5 for step 2" actually wins when good enough
            penalty += 80 * d * d;
        }
        if (
            plan.fixedK[t] === null
            && Array.isArray(plan.annotationMinK)
            && plan.annotationMinK[t] > 1
            && kValues[t] < plan.annotationMinK[t]
        ) {
            const short = plan.annotationMinK[t] - kValues[t];
            penalty += 220 * short * short;
        }
    }
    return penalty;
}

export function proposeCandidateWithPlan(baseKValues, plan, { preferDecrease = false } = {}) {
    const T = baseKValues.length;
    let out = applyPromptConstraints(baseKValues.slice(), plan);

    const mutableSteps = [];
    for (let t = 0; t < T; t++) {
        if (plan.exploreMask[t] && plan.fixedK[t] === null) mutableSteps.push(t);
    }
    if (mutableSteps.length === 0) return out;

    // Global jump (only on mutable steps)
    if (plan.globalK !== null && !plan.globalFixed && Math.random() < 0.22) {
        for (const t of mutableSteps) out[t] = plan.globalK;
        return applyPromptConstraints(out, plan);
    }

    const numChanges = clampInt(1 + Math.random() * Math.min(3, mutableSteps.length), 1, mutableSteps.length);
    const shuffled = mutableSteps.slice().sort(() => Math.random() - 0.5);

    for (let c = 0; c < numChanges && c < shuffled.length; c++) {
        const t = shuffled[c];

        // Prefer explicit target from prompt
        if (plan.targetK[t] !== null) {
            if (Math.random() < 0.55) {
                out[t] = plan.targetK[t];
                continue;
            }
            const step = (Math.random() < 0.5 ? -1 : 1);
            out[t] = clampInt(plan.targetK[t] + step, 1, 15);
            continue;
        }

        const sign = preferDecrease
            ? (Math.random() < 0.7 ? -1 : 1)
            : (Math.random() < 0.5 ? -1 : 1);
        const step = sign * clampInt(1 + Math.random() * 2, 1, 2);
        out[t] = clampInt(out[t] + step, 1, 15);
    }

    return applyPromptConstraints(out, plan);
}

export function formatPlanSummary(plan) {
    if (!plan.notes.length) return '';
    return plan.notes.join(' ');
}
