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


# ------------------------------------------------------------------------------------------ atlas of past posts
def _stat(vals):
    v = np.asarray(vals, dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 3), "hit": round(float((v >= 0.75).mean()), 3), "flop": round(float((v <= 0.25).mean()), 3), "se": round(float(v.std() / max(1.0, np.sqrt(len(v)))), 3)}


def atlas(client):
    """Everything the "Past posts" page shows: a map of MM's own posts by meaning, the topics they fall into and
    how each topic performs, and what posting time, cadence and caption traits are worth. Same embeddings and the
    same normalized results the live engine scores stories with."""
    import json
    import re
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rows = client.pull("MMPost", {}, ["vec", "y", "yv", "r", "rv", "format", "posted_at", "source", "caption", "likes", "comments", "views", "ext_id"], cap=20000)
    P = []
    for r in rows:
        ys = [float(r[k]) for k in ("y", "yv") if r.get(k) is not None]
        if not r.get("vec") or r.get("source") == "collab" or not ys:
            continue
        try:
            t = calendar.timegm(time.strptime(r["posted_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            continue
        P.append({**r, "v": unpack(r["vec"]), "p": sum(ys) / len(ys), "t": t})
    if len(P) < MIN_POSTS:
        return None
    P.sort(key=lambda p: p["t"])
    n = len(P)
    X = np.stack([p["v"] for p in P])
    y = np.array([p["p"] for p in P])
    t = np.array([p["t"] for p in P], dtype=np.float64)
    ny = ZoneInfo("America/New_York")
    dt = [datetime.fromtimestamp(p["t"], ny) for p in P]

    # Map: t-SNE of the caption embeddings (through 50 principal components), fixed seed so it is stable day to day.
    Z = PCA(n_components=min(50, n - 1), random_state=7).fit_transform(X)
    xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=7, metric="cosine").fit_transform(Z)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)

    # Topics: fixed, named centroids (analyst/topics.json). A post belongs to the nearest one.
    here = os.path.dirname(os.path.abspath(__file__))
    named = json.load(open(os.path.join(here, "topics.json")))
    C = np.stack([unpack(tp["c"]) for tp in named])
    lab = np.argmax(X @ C.T, axis=1)
    fmts = FORMATS
    recent_cut = t.max() - 180 * 86400
    topics = []
    for k, tp in enumerate(named):
        idx = np.where(lab == k)[0]
        if not len(idx):
            continue
        order = idx[np.argsort(-y[idx])]
        mult = [np.exp(P[i]["r"]) for i in idx if P[i].get("r") is not None]
        by_fmt = {f: _stat([y[i] for i in idx if P[i].get("format") == f]) for f in fmts}
        best = max((f for f in fmts if f != "podcast" and by_fmt[f]["n"] >= 8), key=lambda f: by_fmt[f]["mean"], default="")
        rec = [y[i] for i in idx if t[i] >= recent_cut]
        topics.append({"id": k, "name": tp["name"], **_stat(y[idx]), "mult": round(float(np.median(mult)), 2) if mult else None,
                       "recent": _stat(rec), "best_format": best, "by_format": {f: by_fmt[f] for f in fmts if by_fmt[f]["n"]},
                       "x": round(float(np.median(xy[idx, 0])), 4), "y": round(float(np.median(xy[idx, 1])), 4),
                       "top": [P[i]["ext_id"] for i in order[:5]], "bottom": [P[i]["ext_id"] for i in order[::-1][:3]]})

    # Timing, in New York time.
    hour = np.array([d.hour for d in dt])
    wd = np.array([d.weekday() for d in dt])
    mon = np.array([d.month for d in dt])
    dom = np.array([d.day for d in dt])
    hb = hour // 3
    fm = np.array([p.get("format") or "image" for p in P])
    def timing(mask):
        return {"weekday": [_stat(y[mask & (wd == k)]) for k in range(7)], "block": [_stat(y[mask & (hb == k)]) for k in range(8)],
                "grid": [[_stat(y[mask & (wd == a) & (hb == b)]) for b in range(8)] for a in range(7)],
                "month": [_stat(y[mask & (mon == k)]) for k in range(1, 13)],
                "third": [_stat(y[mask & (dom <= 10)]), _stat(y[mask & (dom > 10) & (dom <= 20)]), _stat(y[mask & (dom > 20)])]}
    everything = np.ones(n, dtype=bool)
    tim = {"all": timing(everything), **{f: timing(fm == f) for f in fmts if (fm == f).sum() >= 150}}

    # Traits: what else moves the result. Repeat coverage is measured against MM's own posts of the three days before.
    S = X @ X.T
    sat = np.zeros(n)
    for i in range(1, n):
        m = t[:i] >= t[i] - 3 * 86400
        sat[i] = S[i, :i][m].max() if m.any() else 0.0
    cap = [p.get("caption") or "" for p in P]
    clen = np.array([len(c) for c in cap])
    first = [c.split("\n")[0][:160] for c in cap]
    has_num = np.array([bool(re.search(r"\d", f)) for f in first])
    has_q = np.array(["?" in f for f in first])
    days = {}
    for d in dt:
        days[d.date()] = days.get(d.date(), 0) + 1
    perday = np.array([days[d.date()] for d in dt])
    def rows_of(pairs):
        return [{"label": lbl, **_stat(y[m])} for lbl, m in pairs if m.sum() >= 20]
    traits = [
        {"name": "Caption length", "rows": rows_of([("Under 150 characters", clen < 150), ("150 to 400", (clen >= 150) & (clen < 400)), ("Over 400", clen >= 400)])},
        {"name": "First line", "rows": rows_of([("Has a number in it", has_num), ("No number", ~has_num), ("Is a question", has_q), ("Not a question", ~has_q)])},
        {"name": "Closest post of the previous three days", "rows": rows_of([("Nothing similar", sat < 0.6), ("Loosely related", (sat >= 0.6) & (sat < 0.7)), ("Same story or very close", sat >= 0.7)])},
        {"name": "Posts that day", "rows": rows_of([("1 to 2", perday <= 2), ("3 to 4", (perday >= 3) & (perday <= 4)), ("5 to 6", (perday >= 5) & (perday <= 6)), ("7 or more", perday >= 7)])},
    ]

    # Formats, and the account over time.
    formats = []
    for f in fmts:
        m = fm == f
        if not m.sum():
            continue
        likes = [P[i].get("likes") or 0 for i in np.where(m)[0] if (P[i].get("likes") or 0) > 0]
        views = [P[i].get("views") or 0 for i in np.where(m)[0] if (P[i].get("views") or 0) > 0]
        formats.append({"format": f, "n": int(m.sum()), "median_likes": int(np.median(likes)) if likes else 0, "median_views": int(np.median(views)) if views else 0})
    months = {}
    for i, d in enumerate(dt):
        key = d.strftime("%Y-%m")
        mm = months.setdefault(key, {f: [] for f in fmts})
        if (P[i].get("likes") or 0) > 0:
            mm[fm[i]].append(P[i]["likes"])
    monthly = [{"month": k, **{f: {"n": len(v[f]), "likes": int(np.median(v[f])) if v[f] else 0} for f in fmts}} for k, v in sorted(months.items())]

    points = []
    for i, p in enumerate(P):
        mult = np.exp(p["r"]) if p.get("r") is not None else (np.exp(p["rv"]) if p.get("rv") is not None else None)
        points.append([p.get("ext_id") or "", int(round(xy[i, 0] * 1000)), int(round(xy[i, 1] * 1000)), int(lab[i]), int(round(y[i] * 100)),
                       round(float(mult), 2) if mult is not None else None, fmts.index(fm[i]) if fm[i] in fmts else 3, int(p["t"]),
                       int(p.get("likes") or 0), int(p.get("comments") or 0), int(p.get("views") or 0), re.sub(r"[\ud800-\udfff]", "", re.sub(r"\s+", " ", cap[i])[:110])])
    return {"built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "posts": n, "first": int(t.min()), "last": int(t.max()), "formats_order": fmts,
            "overall": _stat(y), "topics": topics, "timing": tim, "traits": traits, "formats": formats, "monthly": monthly, "points": points}


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
        try:
            at = atlas(client)
            if at:
                payload["atlas"] = at
        except Exception as e:  # the atlas is a view, never a reason to lose the audit or the model
            payload["report"]["atlas_error"] = f"{type(e).__name__}: {str(e)[:160]}"
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
