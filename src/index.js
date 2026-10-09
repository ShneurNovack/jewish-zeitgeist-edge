// Edge helper for the Jewish Zeitgeist Base44 app.
//   POST /embed  { texts: string[] }  -> { vectors: string[] }   multilingual embeddings (Workers AI, bge-m3)
//   GET  /fetch?url=...               -> upstream body            relay for feeds that refuse Base44's servers
//   POST /run                         -> runs one pipeline tick now (same as the cron)
//   GET  /health
//   POST /analyst/pull, /analyst/push -> the daily offline job on GitHub Actions; authenticated by a GitHub OIDC token
// A cron trigger runs the pipeline every 10 minutes: ingest, cluster, probe, analyze, structure, mm, enrich.
// Every route except /health needs the shared key in the x-zg-key header (ZG_KEY secret).

import { verifyGithubToken } from "./oidc.js";

const MODEL = "@cf/baai/bge-m3";
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const MAX_TEXTS = 100;
const MAX_BODY = 3_000_000;

const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

// Unit vector -> int8 -> base64. 1024 dims becomes about 1.4 KB, small enough to store on a database row.
function pack(vec) {
  let max = 1e-9;
  for (let i = 0; i < vec.length; i++) max = Math.max(max, Math.abs(vec[i]));
  let bin = "";
  for (let i = 0; i < vec.length; i++) bin += String.fromCharCode(Math.round((vec[i] / max) * 127) & 0xff);
  return btoa(bin);
}

// Slicing can cut an emoji in half, and the model rejects malformed text. Drop lone surrogates and control characters.
function tidy(t) {
  const s = String(t || "").slice(0, 800)
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "")
    .replace(/[\u0000-\u001F\u007F]/g, " ").trim();
  return s || "untitled";
}

function safeTarget(raw) {
  let u;
  try { u = new URL(raw); } catch (_) { return null; }
  if (u.protocol !== "https:" && u.protocol !== "http:") return null;
  const h = u.hostname;
  if (h === "localhost" || h.endsWith(".local") || h.endsWith(".internal") || /^(\d+\.){3}\d+$/.test(h) || h.includes(":")) return null;
  return u;
}

async function embed(request, env) {
  const body = await request.json().catch(() => null);
  const texts = Array.isArray(body?.texts) ? body.texts.slice(0, MAX_TEXTS).map(tidy) : null;
  if (!texts || !texts.length) return json({ error: "texts[] required" }, 400);
  const vectors = [];
  for (let i = 0; i < texts.length; i += 50) {
    const batch = texts.slice(i, i + 50);
    try {
      const res = await env.AI.run(MODEL, { text: batch });
      for (const v of res.data) vectors.push(pack(v));
    } catch (_) {
      // One bad input should not sink the batch: embed one by one and fall back to a neutral text.
      for (const t of batch) {
        let res;
        try { res = await env.AI.run(MODEL, { text: [t] }); } catch (_e) { res = await env.AI.run(MODEL, { text: [t.replace(/[^\p{L}\p{N}\s.,'-]/gu, " ").trim() || "untitled"] }); }
        vectors.push(pack(res.data[0]));
      }
    }
  }
  return json({ model: MODEL, dims: 1024, vectors });
}

async function relay(url) {
  const target = safeTarget(url.searchParams.get("url") || "");
  if (!target) return json({ error: "bad url" }, 400);
  const res = await fetch(target.toString(), {
    headers: {
      "User-Agent": UA,
      "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, application/json, text/html;q=0.8, */*;q=0.5",
      "Accept-Language": "en-US,en;q=0.9,he;q=0.8",
    },
    redirect: "follow",
    cf: { cacheTtl: 120, cacheEverything: true },
    signal: AbortSignal.timeout(20000),
  });
  const buf = await res.arrayBuffer();
  if (buf.byteLength > MAX_BODY) return json({ error: "upstream body too large" }, 413);
  return new Response(buf, { status: res.status, headers: { "content-type": res.headers.get("content-type") || "text/plain; charset=utf-8", "x-upstream-status": String(res.status) } });
}

// Scheduler. Calls the Base44 pipeline stages in order. Each stage is its own function so a slow one
// cannot starve the others, and the LLM stage throttles itself on the Base44 side.
async function runPipeline(env, opts = {}) {
  const base = (env.APP_URL || "").replace(/\/$/, "");
  const log = [];
  const call = async (name, body = {}) => {
    const t0 = Date.now();
    try {
      const res = await fetch(`${base}/functions/${name}`, { method: "POST", headers: { "content-type": "application/json", "x-zg-key": env.ZG_KEY }, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) });
      const text = await res.text();
      let data = null; try { data = JSON.parse(text); } catch (_) { /* keep raw */ }
      log.push({ stage: name, status: res.status, ms: Date.now() - t0, result: data ?? text.slice(0, 200) });
      return data || {};
    } catch (e) { log.push({ stage: name, error: String(e?.message || e), ms: Date.now() - t0 }); return {}; }
  };
  await call("ingest", { max_seconds: 40 });
  for (let i = 0; i < (opts.passes || 3); i++) { const r = await call("cluster"); if (!r.more) break; }
  // Search check: looks up the hot stories on Google autocomplete, Wikipedia and news search (only the ones that are due).
  await call("probe");
  // Trend analytics, readings, decisions; every half hour also the broad topics, the map and a snapshot.
  await call("analyze");
  // Angles and social objects inside the hottest stories that changed.
  await call("structure");
  // Nearest-neighbor read against Meaningful Minute's own post history (a no-op until history is imported).
  await call("mm", { action: "affinity" });
  // One batched LLM read, throttled on the Base44 side.
  const read = await call("enrich", opts.force_enrich ? { force: true } : {});
  // A read changes MM fit, social fit and relevance, so decide again right away instead of ten minutes later.
  if ((read.enriched || 0) + (read.merged || 0) + (read.hidden || 0) > 0) await call("analyze");
  return log;
}

// ---- the offline analyst (GitHub Actions) ----------------------------------------------------------------------
// The job carries no secret. It presents a GitHub OIDC token, verified in oidc.js.
const ANALYST_ENTITIES = new Set(["Signal", "Story", "MMPost", "Recommendation", "Theme"]);

async function analyst(request, env, url) {
  const auth = request.headers.get("authorization") || "";
  const claims = await verifyGithubToken(auth.startsWith("Bearer ") ? auth.slice(7) : "", env);
  if (!claims) return json({ error: "unauthorized" }, 401);
  const body = await request.json().catch(() => ({}));
  const base = (env.APP_URL || "").replace(/\/$/, "");
  let fn;
  if (url.pathname === "/analyst/pull") { if (!ANALYST_ENTITIES.has(body.entity)) return json({ error: "entity not allowed" }, 400); fn = "export"; }
  else if (url.pathname === "/analyst/push") fn = "analyst";
  else return json({ error: "not found" }, 404);
  const res = await fetch(`${base}/functions/${fn}`, { method: "POST", headers: { "content-type": "application/json", "x-zg-key": env.ZG_KEY }, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) });
  return new Response(res.body, { status: res.status, headers: { "content-type": "application/json" } });
}

export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(runPipeline(env).then((log) => console.log(JSON.stringify(log.map((l) => ({ stage: l.stage, status: l.status, ms: l.ms, error: l.error }))))));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health" || url.pathname === "/") return json({ ok: true, service: "jewish-zeitgeist-edge", model: MODEL, analyst: true });
    if (url.pathname.startsWith("/analyst/") && request.method === "POST") {
      try { return await analyst(request, env, url); } catch (e) { return json({ error: String(e?.message || e).slice(0, 200) }, 502); }
    }
    if (!env.ZG_KEY || request.headers.get("x-zg-key") !== env.ZG_KEY) return json({ error: "unauthorized" }, 401);
    try {
      if (url.pathname === "/embed" && request.method === "POST") return await embed(request, env);
      if (url.pathname === "/fetch" && request.method === "GET") return await relay(url);
      if (url.pathname === "/run" && request.method === "POST") return json({ ok: true, log: await runPipeline(env, { passes: Number(url.searchParams.get("passes")) || 3, force_enrich: url.searchParams.get("enrich") === "1" }) });
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: String(e?.message || e).slice(0, 300) }, 502);
    }
  },
};
