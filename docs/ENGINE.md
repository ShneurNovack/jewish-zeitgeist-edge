# Jewish Zeitgeist engine, second generation

What this system answers: **what should Meaningful Minute post about right now?**

It is a trend and recommendation engine, not a news reader. Everything up to the last step is arithmetic,
statistics and embeddings. A language model names things at the end and gives one opinion among several.

## Shape of the system

```
sources (about 90)                      Cloudflare Worker (this repo)
   |                                      cron every 10 minutes, embeddings, Bluesky sampling
   v
ingest -> cluster -> probe -> analyze -> structure -> mm -> enrich -> analyze (if the read changed anything)
            |                    |           |                 |
            v                    v           v                 v
         Signal, Story       Story scores  angles,          names, summaries,
         (items filed        decisions,    social objects   one more opinion on fit
          into stories)      Theme, map,
                             Snapshot
                                                     GitHub Actions (daily): analyst/
                                                     clustering audit, MM model training
```

Runs on Base44 (database, functions, app) with this Worker as the scheduler. One tick costs about 30 seconds of
function time and, at most every 30 minutes, one batched language-model call.

### The four levels

| Level | Entity | How it is found |
| --- | --- | --- |
| Broad topic ("October 7", "UK Green Party Zionism row") | `Theme` | Louvain communities on the story similarity graph |
| Active story | `Story` | Streaming centroid clustering of incoming items |
| Angle (a thread inside a story) | `Story.angles` | Average-linkage clustering inside one hot story |
| Social object (clip, quote, person, photo, post) | `Story.objects` | Media IDs in pages and feeds, repeated quotes, named people |

A broad topic is never treated as a story. It has its own share-of-conversation series and its own baseline.

## What was selected, and why

Each entry lists the job, the method, what it contributes, and what was evaluated and left out.

### 1. Copy and syndication detection
- **Method:** MinHash (48 hashes on the title, 48 on title plus summary) over five-character shingles. Pure JS.
- **Contributes:** tells a wire copy or a light rewrite from independent reporting. A copy counts for a fraction of an
  original, a repeat from the same outlet for almost nothing. Feeds source diversity and "originality".
- **Over:** SimHash (weaker on texts as short as a headline), embedding-only near-duplicate checks (cannot tell a
  rewrite from a second outlet reporting the same event).

### 2. Filing items into stories
- **Method:** single-pass centroid clustering with a blended similarity:
  `0.55 dense + 0.15 recent items + 0.15 TF-IDF + 0.10 shared names + 0.05 time`, join at 0.6. The story center is a
  time-decayed mean (24 hour decay, capped at 40 items), so a story can evolve. Missing features are imputed as
  "unknown" instead of handing their weight to the dense score.
- **Contributes:** stories that stay coherent as they grow, across English and Hebrew (multilingual embeddings from
  Workers AI, 1,024 dimensions).
- **Why this design:** it is where the news-stream clustering literature converges: Miranda et al. 2018, Staykovski
  et al. 2019, Linger and Hajaiej 2020, Saravanakumar et al. 2021, USTORY 2023 all mix dense, sparse, entity and
  time similarity around centroids.
- **Evaluated and not used on the live path:** BERTopic online mode (mini-batch, needs a fixed vocabulary, labels
  shift between batches), River DBSTREAM / DenStream / CluStream (density micro-clusters do not behave in 1,024
  dimensions without reduction, and reduction has to be refit), Top2Vec (batch only), FISHDBC and incremental
  HDBSCAN variants (maintenance and no clear gain at this volume). HDBSCAN is used offline as an auditor, see 12.

### 3. Content type
- **Method:** rules on URL paths, title patterns, RSS categories and media enclosures give the form (report, opinion,
  video, interview, speech, obituary, reaction, breaking, promotion, fundraiser, listing). Embedding prototypes
  (short descriptions embedded once, items mean-centered across classes, calibrated on live data daily) give tone
  (uplifting, heartwarming, grief, outrage, pride, fear), kind (human interest, politics, war, hostile...) and the
  twelve Meaningful Minute content pillars.
- **Contributes:** promotions and fundraisers are discounted to near zero; "WATCH", clip and footage signals across
  several independent items mark a video-driven story; tone and human interest feed Social and MM fit.
- **Over:** a trained classifier (no labels yet; the prototypes become features for one later), zero-shot LLM
  classification per item (cost), GLiNER / DictaBERT entity models (need a Python host; typed people come from the
  one LLM read on hot stories instead).

### 4. Trend reading
- **Holt double exponential smoothing** (alpha 0.5, beta 0.3) on hourly independent arrivals: current pace and
  whether it is accelerating.
- **Kleinberg's burst automaton** (s = 3, gamma 0.5, five levels, Poisson emission): burst intensity 0 to 4, read as
  the maximum over the last three hours.
- **Poisson window scan** over 48 hours against the story's own resting rate: the window that is hardest to explain
  by chance. This is the abnormality figure and the "N reports in H hours, X times the resting pace" sentence.
- **Stage rules** on top: new, climbing, exploding, at its peak, steady, cooling, quiet, plus an estimate of hours
  left from the decay of the smoothed level.
- **Over:** Bayesian online change-point detection, CUSUM, ADWIN, STL and S-H-ESD (need longer or more regular
  series than a three-day-old story has), median/MAD z-scores (too many zeros in hourly counts), Hawkes processes
  and SEISMIC (need per-post engagement cascades, which this system cannot see for Instagram, X or TikTok).

### 5. Six readings instead of one score
Each is 0 to 1 and explainable on its own.

| Reading | Built from |
| --- | --- |
| News coverage | decayed independent coverage, scaled to today's strongest stories, weighted by outlet entropy |
| Conversation | social posts (several voices, not one loud post), search, video, Wikipedia, plus "discourse" evidence inside the coverage: reactions, columns, repeated quotes, several angles |
| Social | tone, human interest, video and quote share, social objects, minus process news and hostility; blended with the LLM read |
| Jewish relevance | share of items with a Jewish or Israeli marker, the LLM read when present, spread across communities |
| Momentum | burst level, acceleration, abnormality |
| MM fit | what the account's own past posts say (coverage and performance of the nearest ones), the LLM read, and a pillar and tone prior |

Source diversity uses Shannon entropy over outlets (the effective number of outlets), originality, and the number
of platforms involved.

### 6. The decision
- **Method:** three composites (Timing, Fit, Diversity), `Opportunity = Timing^0.45 * Fit^0.40 * Diversity^0.15 *
  (0.3 + 0.7 * Jewish)`, then a decision table with gates. First matching row wins.
- **Post now** needs the top threshold and a story that is still rising, bursting, with real conversation and fit.
- **Fit-led row:** a fresh story with a strong, LLM-confirmed fit becomes "Post today" even with thin coverage,
  because much of what MM posts is one human story from one outlet. It can never become "Post now".
- **Fit gate:** timing alone never earns a recommendation. MM fit has to clear 0.45, and where the language model
  has read the story its own verdict has to be at least 0.5. A huge story that is not MM's kind is shown as low
  priority with the reason.
- **News-only veto:** heavy coverage with almost no conversation is marked as conventional news.
- **Reasons** are the largest contributors for and against, in plain words. **Formats** (Reel, Carousel, Quote card,
  Static graphic, News graphic) follow from evidence: footage, a repeated quote, several angles, breaking pace.
- **Over:** a single weighted sum (not explainable, no vetoes), a learned ranker now (no outcomes to train on yet;
  every recommendation is logged with its scores so one can be trained later).

### 7. Angles and social objects
- **Method:** average-linkage agglomerative clustering at cosine 0.74 inside a story, class-based TF-IDF
  (`sqrt(tf) * ln(1 + A/f)`) for names, stable angle IDs across runs, "fresh" flag for angles formed in the last
  hours. Embedded YouTube, X, Instagram, TikTok and Telegram IDs are read from feeds and from up to three article
  pages per hot story (HTMLRewriter / regex). Quotes are matched across outlets by 3-gram overlap.
- **Contributes:** the "post ideas" on the home screen, new-subcluster detection, auto-split when one story is
  really two.

### 8. Broad topics
- **Method:** k-nearest-neighbor graph over story centroids (k = 5, minimum similarity 0.61), Louvain community
  detection (`graphology-communities-louvain`, MIT, resolution 1.7), identity carried across runs by member overlap,
  a 45 day daily table per topic for its usual share of the conversation.
- **Over:** leidenalg (GPL, Python only; Louvain at this size gives the same partitions), hierarchical topic
  models (BERTopic hierarchy, HDP) which would have replaced the story level instead of sitting above it.

### 9. The semantic map
- **Method:** UMAP (`umap-js`, PAIR-code, Apache 2.0) on story centroids with a precomputed neighbor graph,
  warm-started from the previous layout, then aligned to it by Procrustes and damped, so the map moves gradually.
  Clustering and scoring stay in 1,024 dimensions; UMAP is for the picture only. Items sit around their story,
  grouped by angle. Topic regions are drawn along a minimum spanning tree of their stories.
- **Rendering:** one canvas, hand-written pan and zoom. Evaluated d3-zoom, d3-delaunay, deck.gl and
  embedding-atlas; a few hundred bubbles do not need them.

### 10. Search and reference signals
Google Trends daily lists for Israel, the US, the states with the largest Jewish communities and several diaspora
countries, filtered for Jewish relevance; Google autocomplete diffs on seed words; Wikipedia most-read lists and a
watchlist against a two-week norm; and a per-story "probe" that asks autocomplete and Wikipedia about each warm
story directly.

### 11. Language model
One batched call, at most every 30 minutes (every 8 when an unread story is already rated actionable). It names
stories and topics, summarizes, gives 0 to 1 reads on relevance, social and MM fit, proposes an opening line and a
format, and rules on borderline merges. It only sees stories the trend engine already ranked at the top, plus a
handful of fresh single-source human stories the prior rates as a good fit. **It never decides what is trending.**

### 12. Learning from Meaningful Minute (`mm` function, `analyst/` job)

**What it learned from.** 636 posts from the account's own grid, 26 March to 9 October 2026, read in a browser:
caption, format, likes and comments for 477 of them, view counts for 391 reels. 553 are the account's own posts
(326 reels, 139 carousels, 88 single images); the rest are collaborations and are left out of performance scoring.
475 posts older than three days with at least one metric are stored as `MMPost` records with their embeddings.

**What the account turned out to be.** The first version of MM fit assumed MM avoids politics and conflict. The
posts say otherwise. Typical performance by kind of post, as a multiple of the format's median:

| Kind of post | Typical | Share that are 2x hits |
| --- | --- | --- |
| Tragedy, mourning, missing people | 1.4x | 29% |
| Kindness, family, heartwarming | 1.4x | 38% |
| Jewish history and "did you know" | 1.3x | 26% |
| Hostages and October 7 stories | 1.3x | 31% |
| Holocaust and survivors | 1.3x | 27% |
| Antisemitism called out | 1.2x | 27% |
| Jewish pride in sports, business, entertainment | 1.2x | 37% |
| Fallen soldiers and the IDF | 1.0x | 30% |
| Torah, faith, the Rebbe | 0.9x | 12% |
| Only in Israel, Kotel and holiday scenes | 0.9x | 17% |
| Political fights (Mamdani, the UN, Netanyahu, Trump) | 0.9x | 22% |

The biggest single posts were advocacy and outrage (two reactions to 9/11 at 2.6M views, Rubio answering
protesters, the NYPD commissioner backing Israel) and individual human stories (the man who stayed with his
friend on 9/11, Edith Eger, a lone soldier). Reels have a median of about 64,000 views and 2,200 likes; carousels a
median of about 4,000 likes. Podcast clips that say "comment link" collect comments that are requests, so their
comments are not counted as engagement.

**How it is used.**
- **Normalization:** `E = likes + 2 comments + 3 (saves + shares)`; `r = ln(1+E)` minus the median of the 30 posts of
  the same format around it in time; `y` = percentile of `r` among posts of that format within four months. Reels
  get the same on views, and a post is judged on the better of the two.
- **History reading for each story (live, every tick):** the 20 nearest past posts by embedding. *Coverage* comes from
  the mean similarity of the nearest eight (0.50 is ordinary Jewish news, 0.63 is home ground, both measured on
  the account). *Performance* is the similarity-weighted percentile of those neighbors. The reading is
  `coverage * (0.75 + 0.5 * (performance - 0.5))`.
- **MM fit** is 40% history, 45% language read, 15% prior once a story has been read; 60% history and 40% prior
  before that. The language model is shown the same nearest posts and how they did.
- **Six new content pillars** came out of the reading: standing up for Israel in public, antisemitism called out,
  only in Israel, Jewish history and did-you-know, heroism under attack, mourning and remembrance.
- **Trained models, checked honestly:** on the most recent fifth of the history, the nearest-neighbor reading has a
  rank correlation of 0.35 with real performance, ridge regression 0.16 and LightGBM 0.22. So nearest neighbors is
  what runs. The daily job (`analyst/run.py`) repeats the comparison as history grows and exports ridge only when
  it is at least as good; LightGBM would need to win by 0.03 to justify a tree walker in the live path.
- **Clustering audit (same job):** all items of the last 60 hours are clustered again with HDBSCAN (scikit-learn)
  on a PCA-50 reduction and compared with the live stories: adjusted Rand index, adjusted mutual information,
  homogeneity, completeness. Stories HDBSCAN would merge are handed to the language read for a ruling. First run on
  live data: 87% agreement (AMI), homogeneity 0.92, completeness 0.93.
- **Auth for the job:** no repository secrets. The workflow presents a GitHub OIDC token; the Worker verifies the
  signature and that it was issued to this repository's `analyst.yml` on `main` (`src/oidc.js`).
- **Feedback loop:** every recommendation is logged (`Recommendation`) with scores, engine version and propensity;
  "We posted this" and "Not for us" are recorded against it. That log is the training set for a ranker.
- **Keeping it current:** new posts can be added from the Engine page (CSV paste or upload). Importing the same post
  twice does nothing.

## Migration from the first generation

- **Kept:** every source and adapter, the collected items, the Worker (embeddings, relay, scheduler), the search
  and Wikipedia probes, the Sources and Search pages.
- **Replaced:** `Topic` (one flat level) by `Story` and `Theme`; the `score` function by `analyze`; heat as a single
  number by six readings and a decision; the Board, Topic and Pipeline pages by Now, Map, Story, Topics and Engine.
- **How the data moved:** nothing was converted in place. Every stored item was replayed through the new engine in
  time order, which rebuilt stories, trends and topics from scratch. `analyze {"rebuild": "yes-delete-all-stories"}`
  followed by repeated `cluster` calls does the same again if the clustering parameters change.
- **Left in place:** the old `Topic` records (unused) and the `board` and `score` endpoints as stubs.
- **Rollback point:** Base44 checkpoint `6ac89a1f6104ae5781af7707` is the last first-generation state.

## Data model (Base44 entities)

`Source`, `Signal` (items, with vector, fingerprint, form, features, story and angle), `Story`, `Theme`, `Snapshot`
(half-hourly state for trajectories), `Recommendation`, `MMPost`, `Blob` (prototype vectors, document frequencies,
hourly volume, the map frame, models), `PipelineRun`, `Config`, `DailyTerms`.

## Tuning

- Decision thresholds: Engine page, Settings (stored in `Config.engine.scoring`).
- Clustering and scoring parameters: `Config.engine.params` and `Config.engine.scoring` override the defaults in
  `base44/shared/zg2/stories.js` and `scores.js`.
- `dev/replay.ts` in the Base44 app replays stored items through the engine offline for testing changes.

## Known limits

- Instagram, X and TikTok are not read. Conversation is inferred from Reddit, Bluesky, Mastodon, Telegram, search,
  YouTube, Wikipedia and from discourse signals inside the coverage.
- MM fit rests on about six months of the account's posts. It should be topped up every month or two.
- Topic baselines need a few weeks of data before "times its usual share" is trustworthy.
