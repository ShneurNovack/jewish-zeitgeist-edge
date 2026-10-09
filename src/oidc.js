// Verification of GitHub Actions OIDC tokens for the offline analyst job.
// The job carries no secret. It presents a GitHub OIDC token, a short-lived JWT that GitHub signs for one workflow
// run. This verifies the signature against GitHub's published keys and then checks that the token was issued to
// this repository's analyst workflow on the main branch, for this audience. A fork, a pull request or any other
// repository gets a different "repository" or "ref" claim and is refused.
const GH_ISSUER = "https://token.actions.githubusercontent.com";
const ANALYST_AUDIENCE = "jewish-zeitgeist-analyst";
let jwks = { at: 0, keys: [] };
const b64url = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "=")), (c) => c.charCodeAt(0));

export async function verifyGithubToken(token, env, fetcher = fetch, nowMs = Date.now()) {
  const parts = String(token || "").split(".");
  if (parts.length !== 3) return null;
  let header, claims;
  try { header = JSON.parse(new TextDecoder().decode(b64url(parts[0]))); claims = JSON.parse(new TextDecoder().decode(b64url(parts[1]))); } catch (_) { return null; }
  if (header.alg !== "RS256" || !header.kid) return null;
  if (nowMs - jwks.at > 3600e3 || !jwks.keys.some((k) => k.kid === header.kid)) {
    const res = await fetcher(`${GH_ISSUER}/.well-known/jwks`);
    if (!res.ok) return null;
    jwks = { at: nowMs, keys: (await res.json()).keys || [] };
  }
  const jwk = jwks.keys.find((k) => k.kid === header.kid);
  if (!jwk) return null;
  const key = await crypto.subtle.importKey("jwk", { kty: "RSA", n: jwk.n, e: jwk.e, alg: "RS256", ext: true }, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, b64url(parts[2]), new TextEncoder().encode(`${parts[0]}.${parts[1]}`));
  if (!ok) return null;
  const now = nowMs / 1000, repo = env.ANALYST_REPO || "ShneurNovack/jewish-zeitgeist-edge";
  if (claims.iss !== GH_ISSUER || claims.aud !== ANALYST_AUDIENCE) return null;
  if (!(claims.exp > now - 30) || (claims.nbf && claims.nbf > now + 60)) return null;
  if (String(claims.repository || "").toLowerCase() !== repo.toLowerCase() || claims.ref !== "refs/heads/main") return null;
  if (!String(claims.job_workflow_ref || "").toLowerCase().startsWith(`${repo.toLowerCase()}/.github/workflows/analyst.yml@`)) return null;
  return claims;
}
