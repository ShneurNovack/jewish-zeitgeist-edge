// Edge helper for the Jewish Zeitgeist Base44 app.
//   POST /embed  { texts: string[] }  -> { vectors: string[] }   multilingual embeddings (Workers AI, bge-m3)
//   GET  /fetch?url=...               -> upstream body            relay for feeds that refuse Base44's servers
//   GET  /health
// Every route except /health needs the shared key in the x-zg-key header (ZG_KEY secret).

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
  const texts = Array.isArray(body?.texts) ? body.texts.slice(0, MAX_TEXTS).map((t) => String(t || "").slice(0, 800) || " ") : null;
  if (!texts || !texts.length) return json({ error: "texts[] required" }, 400);
  const vectors = [];
  for (let i = 0; i < texts.length; i += 50) {
    const res = await env.AI.run(MODEL, { text: texts.slice(i, i + 50) });
    for (const v of res.data) vectors.push(pack(v));
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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health" || url.pathname === "/") return json({ ok: true, service: "jewish-zeitgeist-edge", model: MODEL });
    if (!env.ZG_KEY || request.headers.get("x-zg-key") !== env.ZG_KEY) return json({ error: "unauthorized" }, 401);
    try {
      if (url.pathname === "/embed" && request.method === "POST") return await embed(request, env);
      if (url.pathname === "/fetch" && request.method === "GET") return await relay(url);
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: String(e?.message || e).slice(0, 300) }, 502);
    }
  },
};
