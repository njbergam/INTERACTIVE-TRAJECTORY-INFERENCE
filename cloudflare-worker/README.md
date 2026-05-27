# Cloudflare Worker (free-ish LLM for GitHub Pages)

This Worker exposes an **OpenAI-compatible** endpoint:

- `POST /v1/chat/completions`

It runs inference using **Cloudflare Workers AI** (free daily neuron allocation, then paid).

## Prereqs

- A Cloudflare account
- Node installed
- Wrangler installed:

```bash
npm i -g wrangler
wrangler login
```

## Deploy

From the repo root:

```bash
cd cloudflare-worker
wrangler deploy
```

Wrangler will print a URL like `https://iti-llm-proxy.<you>.workers.dev`.

## Connect the frontend

In the web UI, select the preset **"Cloudflare Workers AI (hosted)"** and set:

- **Base URL**: `https://<your-worker>.workers.dev/v1`
- **Model**: `default` (ignored by the Worker; it chooses a cheap model)

## CORS

The Worker only allows requests from:

- `https://njbergam.github.io`
- `http://localhost:8000` (dev)

Edit `ALLOWED_ORIGINS` in `src/index.ts` if your Pages origin changes.

