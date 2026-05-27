export interface Env {
  AI: Ai;
}

const ALLOWED_ORIGINS = new Set([
  "https://njbergam.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
]);

function corsHeaders(request: Request) {
  const origin = request.headers.get("Origin") || "";
  const allowOrigin = ALLOWED_ORIGINS.has(origin) ? origin : "";
  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Vary": "Origin",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
  };
}

function json(data: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init.headers || {}),
    },
  });
}

function openAiError(message: string, status = 400) {
  return json(
    { error: { message, type: "invalid_request_error" } },
    { status }
  );
}

function messagesToPrompt(messages: any[]): string {
  if (!Array.isArray(messages)) return "";
  return messages
    .map((m) => {
      const role = typeof m?.role === "string" ? m.role : "user";
      const content = m?.content;
      if (typeof content === "string") return `${role}: ${content}`;
      // If multimodal, keep just text parts.
      if (Array.isArray(content)) {
        const text = content
          .map((p) => (p?.type === "text" ? p?.text : ""))
          .filter(Boolean)
          .join("\n");
        return text ? `${role}: ${text}` : "";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const cors = corsHeaders(request);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    // Basic CORS enforcement (block non-browser/unknown origins).
    const origin = request.headers.get("Origin") || "";
    if (origin && !ALLOWED_ORIGINS.has(origin)) {
      return openAiError("Origin not allowed.", 403);
    }

    if (request.method !== "POST") {
      return openAiError("Use POST.", 405);
    }

    // Support OpenAI-compatible route your frontend already uses.
    if (url.pathname !== "/v1/chat/completions") {
      return openAiError("Not found.", 404);
    }

    let body: any;
    try {
      body = await request.json();
    } catch {
      return openAiError("Invalid JSON body.", 400);
    }

    const messages = body?.messages;
    const prompt = messagesToPrompt(messages);
    if (!prompt) return openAiError("Missing messages.", 400);

    // Keep costs low: small instruct model.
    const model = "@cf/meta/llama-3.2-1b-instruct";

    try {
      // Workers AI supports both {prompt} and {messages} for many LLMs.
      const result: any = await env.AI.run(model, {
        messages,
        max_tokens: Math.min(500, Math.max(32, body?.max_tokens ?? 220)),
      });

      const content =
        typeof result === "string"
          ? result
          : result?.response ?? result?.result ?? result?.output_text ?? null;

      if (!content) return openAiError("Model returned empty response.", 502);

      return json(
        {
          id: `iti_${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: String(content) },
              finish_reason: "stop",
            },
          ],
        },
        { status: 200, headers: cors }
      );
    } catch (e: any) {
      return json(
        { error: { message: e?.message || "Upstream error." } },
        { status: 502, headers: cors }
      );
    }
  },
} satisfies ExportedHandler<Env>;

