const STORAGE_KEY = 'iti-llm-settings';

const PRESETS = {
    ollama: {
        label: 'Ollama (local Llama / Qwen)',
        baseUrl: 'http://localhost:11434/v1',
        model: 'qwen2.5:7b'
    },
    ollama_llama: {
        label: 'Ollama — llama3.2',
        baseUrl: 'http://localhost:11434/v1',
        model: 'llama3.2'
    },
    openai_compat: {
        label: 'OpenAI-compatible (custom URL)',
        baseUrl: 'http://localhost:8080/v1',
        model: 'default'
    },
    ollama_vision: {
        label: 'Ollama vision (llava / moondream)',
        baseUrl: 'http://localhost:11434/v1',
        model: 'llava',
        visionModel: 'llava'
    }
};

export function getLlmSettings() {
    try {
        const raw = sessionStorage.getItem(STORAGE_KEY);
        if (raw) return { ...defaultSettings(), ...JSON.parse(raw) };
    } catch { /* ignore */ }
    return defaultSettings();
}

export function saveLlmSettings(partial) {
    const next = { ...getLlmSettings(), ...partial };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    return next;
}

export function defaultSettings() {
    return {
        enabled: false,
        visionEnabled: false,
        preset: 'ollama',
        baseUrl: PRESETS.ollama.baseUrl,
        model: PRESETS.ollama.model,
        visionModel: 'llava',
        apiKey: '',
        temperature: 0.25
    };
}

export function applyPreset(presetId) {
    const p = PRESETS[presetId] || PRESETS.ollama;
    const patch = {
        preset: presetId,
        baseUrl: p.baseUrl,
        model: p.model
    };
    if (p.visionModel) patch.visionModel = p.visionModel;
    return saveLlmSettings(patch);
}

export function listPresets() {
    return Object.entries(PRESETS).map(([id, p]) => ({ id, label: p.label }));
}

function chatCompletionsUrl(baseUrl) {
    const base = (baseUrl || '').replace(/\/+$/, '');
    if (base.endsWith('/chat/completions')) return base;
    if (base.endsWith('/v1')) return `${base}/chat/completions`;
    return `${base}/v1/chat/completions`;
}

export async function chatCompletion({ messages, settings = null, signal = null, timeoutMs = 90_000 }) {
    const cfg = settings || getLlmSettings();
    const url = chatCompletionsUrl(cfg.baseUrl);
    const headers = { 'Content-Type': 'application/json' };
    if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    if (signal) {
        if (signal.aborted) controller.abort();
        else signal.addEventListener('abort', () => controller.abort(), { once: true });
    }

    let res;
    try {
        res = await fetch(url, {
            method: 'POST',
            headers,
            signal: controller.signal,
            body: JSON.stringify({
                model: cfg.model,
                messages,
                temperature: cfg.temperature ?? 0.25,
                stream: false
            })
        });
    } catch (err) {
        if (controller.signal.aborted) {
            throw new Error(signal?.aborted ? 'Request cancelled.' : 'LLM request timed out.');
        }
        throw err;
    } finally {
        clearTimeout(timeoutId);
    }

    if (!res.ok) {
        const errText = await res.text().catch(() => '');
        throw new Error(`LLM HTTP ${res.status}: ${errText.slice(0, 200) || res.statusText}`);
    }

    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    if (!content) throw new Error('LLM returned empty response.');
    return content;
}

/** Pull first JSON object from model text */
export function parseJsonFromLlm(text) {
    if (!text) return null;
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const raw = fenced ? fenced[1].trim() : text.trim();
    try {
        return JSON.parse(raw);
    } catch {
        const start = raw.indexOf('{');
        const end = raw.lastIndexOf('}');
        if (start >= 0 && end > start) {
            try {
                return JSON.parse(raw.slice(start, end + 1));
            } catch { /* fall through */ }
        }
    }
    return null;
}

/** OpenAI-compatible multimodal chat (Ollama llava, moondream, qwen2-vl, GPT-4o, etc.) */
export async function visionCompletion({
    text,
    imageDataUrl,
    settings = null,
    signal = null,
    timeoutMs = 120_000
}) {
    const base = settings || getLlmSettings();
    const cfg = {
        ...base,
        model: base.visionModel || base.model || 'llava',
        temperature: Math.min(0.4, base.temperature ?? 0.25)
    };

    const messages = [{
        role: 'user',
        content: [
            { type: 'text', text },
            { type: 'image_url', image_url: { url: imageDataUrl } }
        ]
    }];

    return chatCompletion({ messages, settings: cfg, signal, timeoutMs });
}

export async function testLlmConnection(settings = null) {
    const content = await chatCompletion({
        settings,
        messages: [
            { role: 'system', content: 'Reply with exactly: {"ok":true}' },
            { role: 'user', content: 'ping' }
        ]
    });
    const j = parseJsonFromLlm(content);
    return j?.ok === true || /ok|ready|pong/i.test(content);
}
