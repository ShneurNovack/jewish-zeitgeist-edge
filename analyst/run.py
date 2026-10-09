"""Offline analyst for the Jewish Zeitgeist engine.

Runs once a day on GitHub Actions. It does the two jobs that are too heavy, or too slow to matter minute by minute,
for the live pipeline:

  1. Audit the live clustering. Every item of the last 60 hours is clustered again from scratch with HDBSCAN (on a
     PCA reduction of the embeddings) and the result is compared with the stories the streaming engine built. The
     report gives agreement scores and names the stories HDBSCAN would merge or split.
  2. Learn from Meaningful Minute's own results. Once enough past posts are imported, fit a ridge regression from a
     caption's embedding to its normalized performance, compare it on held-out recent months with a nearest-neighbor
     baseline and with LightGBM, and export the ridge weights for the live engine to use.

It never prints data. Results go back to the app, which shows them on the Engine page.

Auth: in GitHub Actions the job has no secrets. It asks GitHub for a short-lived OIDC token and the Worker verifies
that the token was issued to this repository's main branch. For local runs set ZG_KEY and APP_URL instead.
"""
import base64
import calendar
import os
import sys
import time
import traceback

import warnings

import numpy as np
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

WORKER = os.environ.get("WORKER_URL", "").rstrip("/")
APP = os.environ.get("APP_URL", "").rstrip("/")
KEY = os.environ.get("ZG_KEY", "")
AUDIENCE = "jewish-zeitgeist-analyst"
MIN_POSTS = 300


class Client:
    def __init__(self):
        self.token = None
        self.direct = bool(KEY and APP)
        if not self.direct:
            url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] + "&audience=" + AUDIENCE
            r = requests.get(url, headers={"Authorization": "bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]}, timeout=30)
            r.raise_for_status()
            self.token = r.json()["value"]

    def _post(self, path, fn, body):
        if self.direct:
            r = requests.post(f"{APP}/functions/{fn}", json=body, headers={"x-zg-key": KEY}, timeout=170)
        else:
            r = requests.post(f"{WORKER}/analyst/{path}", json=body, headers={"Authorization": "Bearer " + self.token}, timeout=170)
        if r.status_code >= 400:
            raise RuntimeError(f"{path} failed with HTTP {r.status_code}")
        return r.json()

    def pull(self, entity, query, fields, cap=6000):
        """Page by created_date, newest first. Skip-based paging is unstable when many rows share a timestamp."""
        out, seen, cursor = [], set(), None
        fields = sorted(set(fields) | {"created_date"})
        while len(out) < cap:
            q = dict(query)
            if cursor:
                q["created_date"] = {"$lte": cursor}
            rows = self._post("pull", "export", {"entity": entity, "query": q, "sort": "-created_date", "limit": 500, "fields": fields}).get("rows", [])
            fresh = [r for r in rows if r.get("id") not in seen]
            for r in fresh:
                seen.add(r["id"])
            out.extend(fresh)
            if len(rows) < 500 or not fresh:
                break
            cursor = rows[-1]["created_date"]
        return out[:cap]

    def push(self, payload):
        return self._post("push", "analyst", payload)


def unpack(b64):
    """Vectors are stored as int8 scaled by their largest component. Direction is all that matters."""
    v = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


# ----------------------------------------------------------------------------------------------- clustering audit
def audit(client):
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_completeness_v_measure

    since = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - 60 * 3600))
    rows = client.pull("Signal", {"state": "clustered", "fetched_at": {"$gte": since}}, ["story_id", "vec", "title", "ind", "dup_of", "family"], cap=5000)
    rows = [r for r in rows if r.get("vec") and r.get("story_id") and not r.get("dup_of") and r.get("family") in ("news", "social", "video")]
    if len(rows) < 150:
        return {"status": "waiting", "items": len(rows), "note": "not enough recent items to audit"}
    X = np.stack([unpack(r["vec"]) for r in rows])
    Z = PCA(n_components=min(50, len(rows) - 1), random_state=7).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    hdb = HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean").fit_predict(Z)

    live_ids = [r["story_id"] for r in rows]
    counts = {}
    for s in live_ids:
        counts[s] = counts.get(s, 0) + 1
    # Agreement is measured where both sides made a claim: the live story has company, and HDBSCAN did not call it noise.
    both = [i for i in range(len(rows)) if hdb[i] >= 0 and counts[live_ids[i]] >= 2]
    report = {
        "status": "ok", "items": len(rows), "live_stories": len(counts), "live_multi": sum(1 for c in counts.values() if c >= 2),
        "hdbscan_clusters": int(hdb.max() + 1), "noise_share": round(float((hdb < 0).mean()), 3), "compared": len(both),
    }
    if len(both) >= 30:
        a = [live_ids[i] for i in both]
        b = [int(hdb[i]) for i in both]
        h, c, v = homogeneity_completeness_v_measure(a, b)
        report.update({"ari": round(float(adjusted_rand_score(a, b)), 3), "ami": round(float(adjusted_mutual_info_score(a, b)), 3), "homogeneity": round(float(h), 3), "completeness": round(float(c), 3)})

    # Where the two disagree, name it. A merge: one HDBSCAN cluster holds a solid share of two live stories.
    by_cluster, by_story = {}, {}
    for i, r in enumerate(rows):
        if hdb[i] >= 0:
            by_cluster.setdefault(int(hdb[i]), {}).setdefault(r["story_id"], []).append(i)
            by_story.setdefault(r["story_id"], {}).setdefault(int(hdb[i]), []).append(i)
    merges, splits = [], []
    for cl, stories in by_cluster.items():
        strong = [(s, idx) for s, idx in stories.items() if len(idx) >= 3 or (len(idx) >= 2 and len(idx) >= 0.6 * counts[s])]
        if len(strong) >= 2:
            strong.sort(key=lambda x: -len(x[1]))
            merges.append({"size": sum(len(i) for _, i in strong), "stories": [{"id": s, "n": len(idx), "title": rows[idx[0]]["title"][:110]} for s, idx in strong[:4]]})
    for s, clusters in by_story.items():
        parts = sorted(clusters.values(), key=len, reverse=True)
        if counts[s] >= 8 and len(parts) >= 2 and len(parts[1]) >= 3 and len(parts[1]) >= 0.25 * len(parts[0]):
            splits.append({"id": s, "n": counts[s], "parts": [{"n": len(p), "title": rows[p[0]]["title"][:110]} for p in parts[:3]]})
    merges.sort(key=lambda m: -m["size"])
    splits.sort(key=lambda m: -m["n"])
    report["merges"], report["splits"] = merges[:12], splits[:8]
    return report


# --------------------------------------------------------------------------------------- learning from MM history
def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return 0.0 if np.isnan(r) else round(float(r), 3)


def train(posts):
    """posts: dicts with vec (unit vector), y in 0..1, format, posted_at (epoch seconds). Returns (report, model)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    posts = sorted(posts, key=lambda p: p["t"])
    n = len(posts)
    X = np.stack([p["vec"] for p in posts])
    y = np.array([p["y"] for p in posts], dtype=np.float32)
    formats = ["reel", "carousel", "image"]
    F = np.array([[1.0 if p["format"] == f else 0.0 for f in formats] for p in posts], dtype=np.float32)
    cut = int(n * 0.8)  # the most recent fifth is held out: the model has to predict the future, not interpolate
    tr, te = slice(0, cut), slice(cut, n)

    # baseline: similarity-weighted average of the 20 nearest earlier posts (what the live engine does without a model)
    S = X[te] @ X[tr].T
    idx = np.argsort(-S, axis=1)[:, :20]
    w = np.take_along_axis(np.clip(S, 0, None), idx, axis=1) ** 4
    knn = (w * y[tr][idx]).sum(1) / (w.sum(1) + 1e-9)

    alphas = np.logspace(-1, 3, 9)
    ridge = RidgeCV(alphas=alphas).fit(np.hstack([X[tr], F[tr]]), y[tr])
    ridge_pred = ridge.predict(np.hstack([X[te], F[te]]))

    scores = {"knn": spearman(knn, y[te]), "ridge": spearman(ridge_pred, y[te])}
    try:
        import lightgbm as lgb
        pca = PCA(n_components=32, random_state=7).fit(X[tr])
        cal = np.array([[time.gmtime(p["t"]).tm_hour, time.gmtime(p["t"]).tm_wday] for p in posts], dtype=np.float32)
        G = np.hstack([pca.transform(X), F, cal])
        gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=7)
        gbm.fit(G[tr], y[tr])
        scores["lightgbm"] = spearman(gbm.predict(G[te]), y[te])
    except Exception as e:  # LightGBM is a comparison, never a dependency
        scores["lightgbm"] = None
        scores["lightgbm_error"] = type(e).__name__

    # Refit ridge on everything for export. LightGBM replaces it only if it wins clearly, and that export is a
    # separate step: until then ridge is what ships, because it is one dot product in the live engine.
    final = RidgeCV(alphas=alphas).fit(np.hstack([X, F]), y)
    coef = final.coef_.astype(np.float32)
    model = {
        "kind": "ridge", "dims": int(X.shape[1]), "w": base64.b64encode(coef[: X.shape[1]].tobytes()).decode(), "b": float(final.intercept_),
        "format_bias": {f: float(coef[X.shape[1] + i]) for i, f in enumerate(formats)}, "alpha": float(final.alpha_),
        "spearman": scores["ridge"], "trained_on": n, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lg = scores.get("lightgbm")
    report = {"status": "trained", "posts": n, "held_out": n - cut, "spearman": scores, "alpha": float(final.alpha_),
              "winner": "lightgbm" if lg is not None and lg >= scores["ridge"] + 0.03 else "ridge",
              "useful": scores["ridge"] >= 0.1 and scores["ridge"] >= scores["knn"] - 0.02}
    return report, (model if report["useful"] else None)


def learn(client):
    rows = client.pull("MMPost", {}, ["vec", "y", "yv", "format", "posted_at"], cap=20000)
    posts = []
    for r in rows:
        if not r.get("vec") or r.get("y") is None:
            continue
        try:
            t = calendar.timegm(time.strptime(r["posted_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            continue
        posts.append({"vec": unpack(r["vec"]), "y": max(float(r["y"]), float(r.get("yv") or 0)), "format": r.get("format") or "image", "t": t})
    if len(posts) < MIN_POSTS:
        return {"status": "waiting", "posts": len(posts), "need": MIN_POSTS}, None
    return train(posts)


def main():
    t0 = time.time()
    client = Client()
    payload = {"status": "ok", "report": {}}
    try:
        payload["report"]["audit"] = audit(client)
        mm, model = learn(client)
        payload["report"]["mm"] = mm
        if model:
            payload["model"] = model
        a = payload["report"]["audit"]
        payload["message"] = (
            (f"Audit of {a['items']} items: agreement {a.get('ami', 'n/a')}, {len(a.get('merges', []))} possible merges, {len(a.get('splits', []))} possible splits. " if a.get("status") == "ok" else "Audit waiting for more items. ")
            + (f"MM model trained on {mm['posts']} posts, rank correlation {mm['spearman']['ridge']} on held-out months." if mm.get("status") == "trained" else f"MM model waiting for history ({mm.get('posts', 0)} of {MIN_POSTS} posts).")
        )
    except Exception as e:
        # The repository is public and so are its logs: report the kind of failure, never the data.
        payload = {"status": "error", "message": f"{type(e).__name__}: {str(e)[:200]}", "report": {"trace": traceback.format_exc()[-1500:]}}
    payload["seconds"] = round(time.time() - t0, 1)
    client.push(payload)
    print("analyst finished:", payload["status"], "in", payload["seconds"], "s")
    if payload["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
