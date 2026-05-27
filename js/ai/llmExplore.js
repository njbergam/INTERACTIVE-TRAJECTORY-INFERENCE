import { chatCompletion, parseJsonFromLlm, getLlmSettings } from './llmClient.js';
import { parseExplorePrompt, applyPromptConstraints, formatPlanSummary } from './promptPlan.js';

function clampK(k) {
    return Math.max(1, Math.min(15, Math.round(k)));
}

function normalizeKArray(arr, T) {
    if (!Array.isArray(arr) || arr.length !== T) return null;
    return arr.map(k => clampK(k));
}

function buildSystemPrompt(T, allowTrueCost) {
    return `You are an expert assistant for interactive trajectory inference visualization.
The user has ${T} timesteps. Each timestep t has a cluster count K_t (integer 1–15).
Objectives (all "lower is better" unless noted):
- sum_inertia: sum of k-means inertias across steps
- true_ot_cost: discretized transport cost from precomputed point-level OT couplings "pis" (only if available)
- visual_sep: cluster separation in PCA (HIGHER is better)
- cluster_diagnostics: per-step silhouette in [-1,1], cluster sizes — higher silhouette = clearer blobs matching K

You must respond with valid JSON only (no markdown prose outside JSON).
Schemas:
1) Initial plan: {"reasoning":"...","k_values":[...length ${T}...],"candidates":[[...],[...]],"weights":{"inertia":0.5,"true_ot":0.5,"visual":0.1}}
2) Next proposal: {"reasoning":"one short sentence","k_values":[...length ${T}...]}

Respect user instructions exactly (fix K for specific steps, only explore certain steps, etc.).
Prefer parsimonious K near the elbow of k-means inertia — avoid inflating K unless clearly better separation/OT cost.
If the user drew yellow cluster circles on PCA plots, that overrides parsimony: each circle must become one color-pure cluster and circles must get different colors; use K >= number of circles per step.
Propose diverse candidates; do not repeat the same K vector.
true_ot_cost is ${allowTrueCost ? 'AVAILABLE' : 'NOT available — ignore true OT'}.`;
}

function buildContextMessage(ctx) {
    return JSON.stringify({
        timesteps: ctx.T,
        current_k: ctx.currentK,
        best_k: ctx.bestK,
        best_sum_inertia: ctx.bestInertia,
        best_true_ot_cost: ctx.bestTrueOt,
        best_visual_sep: ctx.bestSep,
        cluster_shape_geom: ctx.shapeGeom,
        cluster_shape_vision: ctx.shapeVision,
        cluster_diagnostics: ctx.clusterDiagnostics,
        trials_so_far: ctx.trials,
        seconds_remaining: ctx.secondsRemaining,
        user_prompt: ctx.prompt,
        recent_candidates: ctx.recentTrials?.slice(-5) || []
    }, null, 0);
}

export async function llmInitialPlan({ prompt, T, currentK, allowTrueCost, signal }) {
    const user = `User prompt:\n${prompt || '(no specific prompt — find a balanced good clustering)'}\n\nCurrent K: ${JSON.stringify(currentK)}\n\nReturn initial plan JSON with your best k_values plus 2-4 diverse candidates to try.`;

    const content = await chatCompletion({
        signal,
        messages: [
            { role: 'system', content: buildSystemPrompt(T, allowTrueCost) },
            { role: 'user', content: user }
        ]
    });

    const json = parseJsonFromLlm(content);
    if (!json) throw new Error('LLM plan was not valid JSON.');

    const kValues = normalizeKArray(json.k_values, T);
    const candidates = (json.candidates || [])
        .map(c => normalizeKArray(c, T))
        .filter(Boolean);

    if (kValues) candidates.unshift(kValues);
    if (!candidates.length) throw new Error('LLM returned no valid K candidates.');

    return {
        reasoning: json.reasoning || '',
        candidates,
        weights: json.weights || null,
        raw: content
    };
}

export async function llmNextProposal({ prompt, T, currentK, allowTrueCost, best, trials, secondsRemaining, recentTrials, signal }) {
    const ctx = {
        T,
        prompt,
        currentK,
        bestK: best?.kValues,
        bestInertia: best?.details?.sumInertia,
        bestTrueOt: best?.details?.trueCost,
        bestSep: best?.details?.sepScore,
        shapeGeom: best?.details?.shapeStats?.score ?? null,
        shapeVision: best?.details?.visionQuality ?? null,
        clusterDiagnostics: best?.details?.shapeStats?.steps?.length
            ? {
                steps: best.details.shapeStats.steps,
                meanSilhouette: best.details.shapeStats.steps.reduce((s, x) => s + (x.silhouette || 0), 0)
                    / best.details.shapeStats.steps.length
            }
            : null,
        trials,
        secondsRemaining,
        recentTrials
    };

    const content = await chatCompletion({
        signal,
        messages: [
            { role: 'system', content: buildSystemPrompt(T, allowTrueCost) },
            { role: 'user', content: `Context:\n${buildContextMessage(ctx)}\n\nPropose the next k_values to evaluate. JSON only.` }
        ]
    });

    const json = parseJsonFromLlm(content);
    const k = json ? normalizeKArray(json.k_values, T) : null;
    if (!k) throw new Error('LLM next proposal was not valid JSON.');
    return { kValues: k, reasoning: json.reasoning || '', raw: content };
}

/** Merge LLM objective weights into rule-based plan */
export function mergeLlmWeights(plan, llmWeights) {
    if (!llmWeights) return plan;
    const w = plan.weights;
    const li = Number(llmWeights.inertia);
    const lt = Number(llmWeights.true_ot ?? llmWeights.trueOt);
    const lv = Number(llmWeights.visual ?? llmWeights.visual_sep);
    if (Number.isFinite(li)) w.wInertia = li;
    if (Number.isFinite(lt)) w.wTrue = lt;
    if (Number.isFinite(lv)) w.wSep = lv;
    const s = w.wInertia + w.wTrue;
    if (s > 0) {
        w.wInertia /= s;
        w.wTrue /= s;
    }
    return plan;
}

export function fallbackPlan(prompt, T) {
    return parseExplorePrompt(prompt, T);
}

export { formatPlanSummary, applyPromptConstraints };
