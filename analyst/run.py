"""Offline analyst for the Jewish Zeitgeist engine.

Runs once a day on GitHub Actions. It does the two jobs that are too heavy, or too slow to matter minute by minute,
for the live pipeline:

  1. Audit the live clustering. Every item of the last 60 hours is clustered again from scratch with HDBSCAN (on a
     PCA reduction of the embeddings) and the result is compared with the stories the streaming engine built. The
     report gives agreement scores and names the stories HDBSCAN would merge or split.
  2. Learn from Meaningful Minute's own results. Once enough past posts are imported, test nearest neighbors, a
     ridge regression, their blend and LightGBM on posts none of them has seen (five forward folds), and export the
     ridge weights, the validated settings and the calibration the live engine needs to read a story's prediction.

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


FORMATS = ["reel", "podcast", "carousel", "image"]
KNN = {"k": 40, "power": 8, "half": 365}   # chosen on held-out posts, see docs/ENGINE.md section 12
BLEND = 0.3                                # share of the ridge model in the blend with nearest neighbors


def knn_predict(Xtr, ytr, ttr, Xq, tq, k=KNN["k"], power=KNN["power"], half=KNN["half"]):
    """What the live engine does: similarity-weighted average result of the k nearest earlier posts."""
    S = Xq @ Xtr.T
    k = min(k, Xtr.shape[0])
    idx = np.argsort(-S, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(S, idx, axis=1), 0, None) ** power
    if half:
        w = w * 0.5 ** (np.clip(tq[:, None] - ttr[idx], 0, None) / 86400.0 / half)
    return (w * ytr[idx]).sum(1) / (w.sum(1) + 1e-9)


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def train(posts, cover_vecs=None, stories=None):
    """posts: MM's own scored posts as dicts with vec (unit vector), y in 0..1, format, t (epoch seconds).
    cover_vecs: every post MM put on its grid, collabs included (used only for coverage).
    stories: centroids of the live stories, used to put the model on the scale it will be read on.
    Every number in the report comes from posts the model had not seen: five folds, each predicting the tenth of
    the history that comes right after everything it was trained on."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    posts = sorted(posts, key=lambda p: p["t"])
    n = len(posts)
    X = np.stack([p["vec"] for p in posts])
    y = np.array([p["y"] for p in posts], dtype=np.float32)
    t = np.array([p["t"] for p in posts], dtype=np.float64)
    F = np.array([[1.0 if p["format"] == f else 0.0 for f in FORMATS] for p in posts], dtype=np.float32)
    alphas = np.logspace(-1, 3, 13)
    folds = [(int(n * a), int(n * b)) for a, b in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))]
    oof = {k: np.full(n, np.nan) for k in ("knn", "ridge", "blend", "lightgbm")}
    notes = {}
    for a, b in folds:
        tr, te = slice(0, a), slice(a, b)
        recent = 0.5 ** ((t[tr].max() - t[tr]) / 86400.0 / 365.0)  # the account drifts: recent posts count more
        oof["knn"][te] = knn_predict(X[tr], y[tr], t[tr], X[te], t[te])
        oof["ridge"][te] = RidgeCV(alphas=alphas).fit(X[tr], y[tr], sample_weight=recent).predict(X[te])
        oof["blend"][te] = (1 - BLEND) * zscore(oof["knn"][te]) + BLEND * zscore(oof["ridge"][te])
        try:
            import lightgbm as lgb
            pca = PCA(n_components=min(48, a - 1), random_state=7).fit(X[tr])
            gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=7)
            gbm.fit(np.hstack([pca.transform(X[tr]), F[tr]]), y[tr], sample_weight=recent)
            oof["lightgbm"][te] = gbm.predict(np.hstack([pca.transform(X[te]), F[te]]))
        except Exception as e:  # LightGBM is a comparison, never a dependency
            notes["lightgbm_error"] = type(e).__name__
    m = ~np.isnan(oof["blend"])
    scores = {k: (spearman(v[m], y[m]) if not np.isnan(v[m]).any() else None) for k, v in oof.items()}
    order = np.argsort(-oof["blend"][m])
    ym = y[m][order]
    fifth = max(1, len(ym) // 5)
    by_format = {}
    for f in FORMATS:
        idx = np.array([i for i in range(n) if m[i] and posts[i]["format"] == f], dtype=int)
        if len(idx) >= 30:
            by_format[f] = spearman(oof["blend"][idx], y[idx])
    report = {
        "status": "trained", "posts": n, "held_out": int(m.sum()), "spearman": {**scores, **notes},
        "fold_spearman": [spearman(oof["blend"][a:b], y[a:b]) for a, b in folds],
        "hit_rate": {"top_fifth": round(float((ym[:fifth] >= 0.75).mean()), 3), "bottom_fifth": round(float((ym[-fifth:] >= 0.75).mean()), 3), "all": round(float((ym >= 0.75).mean()), 3)},
        "by_format": by_format,
        "settings": {"knn": KNN, "blend": BLEND},
    }
    report["winner"] = max((k for k in ("knn", "ridge", "blend", "lightgbm") if scores.get(k) is not None), key=lambda k: scores[k])
    report["useful"] = scores["blend"] is not None and scores["blend"] >= 0.1 and scores["blend"] >= scores["knn"] - 0.01
    if not report["useful"]:
        return report, None

    # Final fit on everything.
    recent = 0.5 ** ((t.max() - t) / 86400.0 / 365.0)
    final = RidgeCV(alphas=alphas).fit(X, y, sample_weight=recent)
    w = final.coef_.astype(np.float64)
    b = float(final.intercept_)
    model = {"kind": "ridge", "dims": int(X.shape[1]), "alpha": float(final.alpha_), "knn": KNN, "blend": BLEND,
             "spearman": scores["blend"], "trained_on": n, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Coverage scale. LO: where an ordinary story of today sits against the history (lower quartile of the mean
    # similarity to its eight nearest posts). HI: where MM's own posts sit against each other (median, leave one out).
    C = cover_vecs if cover_vecs is not None and len(cover_vecs) else X
    own8 = np.sort(X @ C.T, axis=1)[:, -9:-1].mean(1)   # the nearest one is the post itself
    model["cover_hi"] = round(float(np.median(own8)), 4)
    if stories is not None and len(stories) >= 150:
        st8 = np.sort(stories @ C.T, axis=1)[:, -8:].mean(1)
        model["cover_lo"] = round(float(np.quantile(st8, 0.25)), 4)
        # The model was trained on captions and will be read on story centroids built from headlines. Put its
        # output on the scale the nearest-neighbor read has on those stories, so the 70/30 blend means the same
        # thing live as it did in the test.
        now = np.full(len(stories), time.time())
        k_s = knn_predict(X, y, t, stories, now)
        r_s = stories @ w + b
        scale = float(k_s.std() / (r_s.std() + 1e-9))
        w = w * scale
        b = float(k_s.mean() + (b - r_s.mean()) * scale)
        blended = (1 - BLEND) * k_s + BLEND * np.clip(stories @ w + b, 0, 1)
        covered = blended[st8 >= model["cover_lo"]]
        # Spread of predictions over today's stories in MM's territory: turns a prediction into a rank.
        model["q"] = [round(float(v), 4) for v in np.quantile(covered if len(covered) >= 100 else blended, np.linspace(0, 1, 21))]
        report["calibrated_on_stories"] = int(len(stories))
    model["w"] = base64.b64encode(w.astype(np.float32).tobytes()).decode()
    model["b"] = b
    report["alpha"] = model["alpha"]
    report["cover"] = {"lo": model.get("cover_lo"), "hi": model["cover_hi"]}
    return report, model


def learn(client):
    rows = client.pull("MMPost", {}, ["vec", "y", "yv", "format", "posted_at", "source"], cap=20000)
    posts, cover = [], []
    for r in rows:
        if not r.get("vec"):
            continue
        v = unpack(r["vec"])
        cover.append(v)
        ys = [float(r[k]) for k in ("y", "yv") if r.get(k) is not None]
        if r.get("source") == "collab" or not ys:
            continue
        try:
            t = calendar.timegm(time.strptime(r["posted_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            continue
        posts.append({"vec": v, "y": sum(ys) / len(ys), "format": r.get("format") or "image", "t": t})
    if len(posts) < MIN_POSTS:
        return {"status": "waiting", "posts": len(posts), "need": MIN_POSTS}, None
    srows = client.pull("Story", {"status": "active"}, ["centroid", "hidden", "merged_into"], cap=1500)
    stories = [unpack(r["centroid"]) for r in srows if r.get("centroid") and not r.get("hidden") and not r.get("merged_into")]
    return train(posts, np.stack(cover), np.stack(stories) if stories else None)


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
            + (f"MM model trained on {mm['posts']} posts, rank correlation {mm['spearman']['blend']} on {mm['held_out']} held-out posts." if mm.get("status") == "trained" else f"MM model waiting for history ({mm.get('posts', 0)} of {MIN_POSTS} posts).")
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
