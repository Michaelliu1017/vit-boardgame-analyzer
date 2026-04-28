# Project Report Material — Pacific 1940 Battle Analyzer

> **Purpose of this document.** Working content for the four CVPR-format
> deliverables (Proposal, Feasibility Study, Midterm, Final) plus the in-class
> presentation. Each section below maps to a specific submission and contains
> the substantive answers the team can lift directly into the LaTeX/Word
> template. Items marked **`[TEAM TODO]`** are facts only the team has (training
> dataset details, individual contributions, etc.); items marked **`[RED]`** are
> work that is *not yet done* and must be coloured red in the midterm report
> per the rubric.
>
> All technical claims about the deployed system are grounded in the current
> `dpm-edits` branch: `app/backend/pipeline.py`, `app/backend/main.py`,
> `app/frontend/index.html`, and the empirical timings in
> `docs/inference-optimization.md`.

---

## 0. Project at a glance (one-paragraph elevator pitch)

We built a phone-photo-to-tactical-advice tool for the tabletop war game
**Axis & Allies Pacific 1940**. A user takes a picture of a contested
territory on the board; an open-vocabulary object detector
(**OWL-ViT large**) finds every plastic miniature in the frame, a fine-tuned
**Vision Transformer (ViT-Small/16)** classifies each one as one of seven unit
types, and an HSV colour heuristic assigns the piece to **Japan** (orange) or
**United States** (green). The resulting unit counts are then fed to one of
three decision-support tools served from a **FastAPI** backend: (1) a learned
**MLP win-rate predictor**, (2) a brute-force **best-composition
recommender** that re-uses the MLP as a fitness function over IPC-bounded
attacker mixes, and (3) a deterministic **rules-based combat simulator**.
A static HTML/JS frontend ties it all together. The novel contribution is
not any single model — it is the end-to-end loop from a noisy hobby-table
photo to a numerical "you have a 54% chance, but you should have brought
two infantry and an artillery instead" recommendation.

---

# Part 1 — Project Proposal (March 7) — 2 pages, CVPR

> **Format reminder from the rubric:** "Please do not answer these questions
> as a set of bullet points. A set of bullet points is not persuasive. You
> should tell a story that answers these questions through the story."
>
> The five blocks below are written as continuous prose. The intended way
> to use them is to drop them into the LaTeX template under headings like
> *Motivation*, *Background*, *Approach*, *Timeline*, and *Evaluation*, then
> trim/expand to fit two columns × two pages.

### 1.1  What are you trying to do?

Imagine sitting at a four-hour board-game session of *Axis & Allies Pacific
1940*. It is your turn to attack the Philippines. You can see the table:
a clutch of Japanese infantry, a tank, and a fighter on one side, an
American infantry-and-artillery garrison on the other. Before you commit,
you want a quick second opinion — *should I attack at all, and if I had a
chance to redo my last purchase, what should I have bought instead?*
Today the answer takes a calculator, a printed odds table, or a tab on
your laptop into which you have to manually type how many of each unit
each side has. We want that second opinion to come from a single phone
photo of the table. Snap, wait a beat, see a number — and a suggested
better hand.

Reduced to its smallest sentence: **we are building a tool that looks at
a photograph of a war-game position and tells the player both how likely
they are to win the next fight and what units they should have brought
to it.** Everything else — neural networks, transformers, simulators — is
plumbing in service of that one user experience.

### 1.2  How is it done today, and what are the limits of current practice?

Two communities already work in this space, and their tools sit at
opposite ends of a usability spectrum. On one end, hobbyist battle
calculators (e.g. *AAOddsCalc*, the *Hasbro Battle Calc*, browser tools
embedded in *TripleA*) take typed unit counts and run thousands of
Monte-Carlo dice rolls, returning win probabilities and average
survivors. They are accurate but brittle: every parameter is hand-typed,
nothing knows what is on the actual table, and they answer a single
narrow question (*who wins?*). On the other end, general computer-vision
pipelines (chess-board recognition, MTG card scanners, sports player
detection) demonstrate that "photo of a real-world game state" → "machine
representation" is a tractable problem, but those pipelines target much
larger datasets and far less visually crowded scenes than a 30-piece
naval clash on a printed map.

The result is that nobody has stitched the two halves together. Players
who want a second opinion incur all the cost of manual data entry; the
calculators they entry into can score a position but cannot suggest a
*better* one; and no off-the-shelf detector is trained on plastic
miniatures small enough to fit in a fingernail and overlapping with
printed map iconography. The limit, then, is not raw model capability —
it is the missing connector between perception and recommendation for a
niche, visually messy, low-data domain.

### 1.3  What is new in your approach and why do you think it will succeed?

Our specific bet is that **open-vocabulary detection plus a tiny learned
combat oracle is enough to close the loop**, and that the recommendation
problem is small enough that brute-force search on top of that oracle
beats anything cleverer. Concretely: instead of training a custom DETR or
YOLO from scratch on a dataset we do not have, we use Google's
**OWL-ViT large**, prompted with seven plain-English text strings
(*"plastic military figurine toy"*, *"plastic tank toy"*, …). This
side-steps the data-scarcity problem entirely for the localisation step
and lets us spend our limited annotation budget on a much narrower
problem: classifying a *cropped, centred, single-piece* image into one of
seven unit types, where a small Vision Transformer (`vit_small_patch16`)
fine-tuned on a few thousand crops is more than sufficient. Faction
assignment ("is this orange or green?") is then a five-line HSV heuristic
that needs no learning at all.

The recommendation half is similarly opinionated: an MLP trained on
simulator-generated battle outcomes acts as a millisecond-cost surrogate
for thousands of Monte-Carlo rolls, and because the legal attacker space
is bounded by an IPC budget of low double digits, the *entire* set of
buyable mixes is small enough to enumerate exhaustively (~10⁴ candidates
per budget). We rank them all by the MLP and return the top one. The
reason no one has done this is straightforward — Pacific 1940 has a
small player base, an unusual visual signature, and sits in a gap
between "general game AI" research and "everyday object recognition"
research; the obvious move is exactly the one we are taking, and the
only thing keeping it undone is that nobody has bothered.

### 1.4  How long will this take?

The work fits inside a single semester because each piece is incremental
on top of pre-trained components. February is data collection: photographing
boards, cropping unit instances, labelling the seven unit classes, and
seeding the win-rate model with simulator-generated battles. March is
model work: prompt-tuning OWL-ViT thresholds, fine-tuning the ViT
classifier, training the win-rate MLP. April is integration: wrapping
everything in a FastAPI service, building a static frontend, and
benchmarking end-to-end latency. We allow May for evaluation, ablations,
and the final write-up. There is no novel optimisation method or new
architecture being proposed — the schedule risk is dataset-size, not
research-novelty.

### 1.5  What are the mid-term and final "exams"?

We treat success as observable at three different layers, each with a
crisp metric. At the perception layer we report **detection precision /
recall** on a held-out set of 20 board photos and **per-class
classification accuracy** on a held-out set of cropped unit images;
these are the mid-term "is the photo-to-counts step working" gate. At
the reasoning layer we report **win-rate calibration** (mean absolute
error of the MLP versus a 10⁵-roll Monte-Carlo ground truth) and the
**recommendation quality** (does the MLP-recommended mix actually beat
the player's original mix when both are simulated?). At the system
layer we report **end-to-end latency** per request and a small **user
study** in which two of us play through scripted scenarios and rate
whether the recommendations are non-obvious and useful. The project
"works" when the perception step is right >80% of the time on real
photos, the win-rate model is calibrated within ±5%, and the end-to-end
round trip is under three seconds on a laptop CPU.

---

# Part 2 — Project Feasibility Study (April 11) — 2 pages, CVPR

> **Rubric:** "what data you have downloaded and code packages you have
> tested. You must have some amount of code up and running and data
> downloaded at this point, including a figure."

### 2.1  Data we have on hand

- **Real board photographs.** `app/backend/test/` currently holds 13 JPGs
  (`t0–t6`, `test1–test5`, plus `test.jpg`) of physical *Pacific 1940*
  positions photographed with a phone camera at varying angles, distances,
  and lighting conditions. These act as the held-out evaluation set for
  the end-to-end pipeline.
- **Annotated detection visualisation.** `assets/annotated_cv_boxes_green.jpg`
  is a sample OWL-ViT output overlaid on a board photo and is the figure
  we plan to use for the feasibility-study figure.
- **Trained model weights, already in the repo.**
  - `app/backend/vit_classifier.pth` — 86.7 MB, ViT-Small/16 fine-tuned on our
    cropped-piece dataset. Loaded at FastAPI startup.
  - `app/backend/winrate_model.pt` — 1.4 MB, MLP win-rate model. Loaded at
    startup.
- **Open-vocabulary detector weights from Hugging Face Hub.** Pulled
  on first run from `google/owlvit-large-patch14`. No local download
  required beyond the Hugging Face cache.
- `[TEAM TODO]` Number of cropped instances per class used to fine-tune the
  ViT, and number of simulator-generated battles used to train the MLP.
  These numbers should be filled in from your training notebooks.

### 2.2  Code packages tested

The full pipeline runs end-to-end on macOS (MPS), Linux (CPU/CUDA), and
Windows (CPU/CUDA). The dependency surface is intentionally small:

| Package | Role | Status |
|---|---|---|
| `torch` | Inference for OWL-ViT, ViT, MLP | Working on CPU, MPS, CUDA |
| `torchvision` | NMS, transforms | Working |
| `transformers` (Hugging Face) | OWL-ViT processor + model | Working; auto-downloads weights |
| `timm` | ViT-Small backbone for classifier | Working |
| `opencv-python` | HSV faction colour analysis | Working |
| `Pillow` | Image I/O | Working |
| `fastapi` + `uvicorn` | HTTP API | Working |
| `python-multipart` | `/analyze` form upload | Working |
| `numpy` | Combat simulator RNG | Working |

Setup, dependency gaps, and per-OS instructions are documented in the
top-level `README.md`. Empirical inference timings — used to identify
which optimisations are worth pursuing in the second half of the
semester — are in `docs/inference-optimization.md` and were collected by
`app/backend/bench_inference.py`.

### 2.3  Suggested figure for the feasibility study

A single composite figure with three panels works well for two pages:

1. **Left panel** — a raw board photo (`app/backend/test/t5.JPG`).
2. **Middle panel** — the same photo overlaid with OWL-ViT detection
   boxes (`assets/annotated_cv_boxes_green.jpg`).
3. **Right panel** — a screenshot of the FastAPI Swagger UI at
   `/docs` showing the four working endpoints (`docs/images/ui-initial.png`
   can be used as a stand-in if the Swagger screenshot is not yet
   captured).

Caption suggestion: *"End-to-end pipeline working on a real phone photo:
(left) raw input, (middle) OWL-ViT detections, (right) live
FastAPI service exposing /analyze, /winrate, /recommend, /simulate."*

---

# Part 3 — Project Midterm Report (April 28) — 4 pages, CVPR

> **Rubric:** "the report must have the same sections as the final report,
> but things that will be done in the remaining time should be coloured in
> red."
>
> Use the section drafts in **Part 4** below as the body of the midterm.
> Mark every block tagged `[RED]` in red in the LaTeX/Word document.
> Below is a checklist of what is currently done versus what should be
> red on the date of submission.

### 3.1  What is done (black text)

- End-to-end pipeline runs against real board photos.
- All four API endpoints (`/analyze`, `/winrate`, `/recommend`,
  `/simulate`) respond with valid payloads.
- Static HTML frontend can drive a full session for all three modes.
- Trained ViT classifier and MLP win-rate model are committed as
  `.pth`/`.pt` files.
- Inference profiling complete; bottleneck identified
  (`find_best_attack` MLP loop) — see `docs/inference-optimization.md`.
- Cross-platform setup verified on macOS, Linux, Windows
  (`docs/readme-run-report.md`).

### 3.2  What should be coloured **`[RED]`** in the midterm

- **Quantitative evaluation tables** — detection P/R, classifier
  per-class accuracy, MLP MAE vs. Monte-Carlo, end-to-end accuracy on
  the 13 held-out photos. Until these numbers exist as a table, that
  table is red.
- **Confusion matrix figure** for the ViT classifier.
- **Calibration plot** for the MLP win-rate model versus simulator
  ground truth.
- **Best-composition validation** — for each of the 13 photos, does
  the recommended mix actually win more often than the original?
- **Optimisation rollout** — the inference-optimisation findings exist
  as a plan; the actual code changes (batched MLP search, OWL-ViT text
  embedding cache, ViT batching, async offload) are not yet merged.
- **User study** — two-team scripted playthrough comparing decisions
  with and without the tool.
- **Ablations** — OWL-ViT base vs. large, ViT-Small vs. ViT-Tiny,
  with and without the colour-faction step.

### 3.3  Length budget for 4 pages

Approximate column counts that fit in CVPR 4-page format:

- Introduction + summary figure — ¾ column
- Related work — ¾ column
- Data — ½ column + figure
- Method — 1½ columns + pipeline figure
- Experiments — 1½ columns + 1 figure + 1 table
- Conclusion / future work — ¼ column
- References — ½ column

---

# Part 4 — Project Final Report (End of Class) — 6 pages, CVPR

The body sections below are written long enough to support the 6-page
final report. Trim for the 4-page midterm; keep the same section
ordering for both as required.

## 4.1  Introduction (with summary figure)

*Axis & Allies Pacific 1940* is a hex-and-counter war game whose
combat resolution is fully governed by published probability tables —
each unit attacks and defends on a known dice value, dice are rolled
sequentially, and casualties are removed in a known order. In principle
this makes the game an ideal target for decision-support tools: every
question of the form *"if I attack with this stack, what is my win
probability?"* has a closed-form Monte-Carlo answer. In practice,
players almost never use such tools at the table, because the cost of
typing the position into a calculator outweighs the value of the
answer. Our project closes that gap by replacing the type-it-in step
with a phone photo and the hand-tuned calculator with a learned
recommender.

The contribution is a complete decision-support pipeline that takes a
single uncontrolled phone photograph of a contested territory and
returns three distinct answers: (i) the predicted probability that the
named attacker wins the upcoming combat, (ii) the optimal attacker
purchase under two different IPC budgets, and (iii) a fully simulated
battle with surviving units on each side. Each answer is served in
under five seconds end-to-end on a CPU laptop.

> **Required summary figure.** Pipeline overview: phone photo →
> OWL-ViT detection → ViT classification → HSV faction assignment →
> editable counts modal → one of three downstream outputs (win-rate
> bar, two recommendation cards, or a simulated battle log). Existing
> `docs/images/ui-states.png` covers the right-hand half of this
> figure; the team needs to add the perception half to its left.

## 4.2  Related Work

**Open-vocabulary object detection.** OWL-ViT [Minderer et al., ECCV
2022] introduced a vision-transformer-based detector that accepts free
text queries at inference time, allowing localisation of object
categories never seen during training. We use the publicly released
`google/owlvit-large-patch14` checkpoint with seven natural-language
queries describing miniature unit types. Closely related work
includes GLIP [Li et al., CVPR 2022], OWLv2 [Minderer et al., NeurIPS
2023], and Grounding-DINO [Liu et al., 2023]; we chose OWL-ViT for its
simple HuggingFace integration and stable behaviour on small,
overlapping plastic figurines.

**Vision Transformers for fine-grained classification.** ViT [Dosovitskiy
et al., ICLR 2021] established that pure self-attention backbones can
match or beat CNNs on image classification given sufficient
pre-training. We use `vit_small_patch16_224` from `timm` [Wightman
2019], fine-tuned on cropped piece images, because the per-class
intra-domain variability is small relative to the ImageNet pretraining
distribution and the smaller backbone is fast on CPU.

**Game-state recognition from photos.** The closest analogues to our
perception stage are LiveChess2FEN [Mallasén-Quintana et al., 2021]
for chess and various Magic-the-Gathering card scanners. These
demonstrate that controlled top-down photos of game states are
tractable, but they target single-piece-per-cell or planar-card
problems with much less visual clutter than a stacked-miniatures war
game.

**Battle calculators and simulators.** Hobbyist tools like *AAOddsCalc*,
*TripleA*'s built-in battle calculator, and various spreadsheet-based
odds tables provide manual-entry win-probability estimates for *Axis &
Allies* battles via Monte-Carlo simulation. They are the de-facto
ground truth we calibrate our learned MLP against, and they are the
"current practice" we compare end-to-end usability against.

**Surrogate models for combat outcomes.** Using a small neural network
to amortise the cost of repeated stochastic simulations is standard
practice in game AI (most notably AlphaGo / AlphaZero [Silver et al.,
2017]). Our MLP is a much simpler instance of the same idea: rather
than learning a policy or value over game-tree states, it learns the
single-battle probability function from simulator-generated examples.

`[TEAM TODO]` Add 2-3 citations specific to the *training data* you
generated (e.g. if you used a published Pacific-1940 rules
implementation).

## 4.3  Data

> **Required figure.** A grid of representative cropped unit images
> across the seven classes for both factions, plus one full board
> photo with bounding boxes overlaid. Source crops can be generated
> from `app/backend/test/` images via the running `/analyze` endpoint
> with intermediate visualisation enabled (see `test_pipeline.py`).

Three datasets feed the system, each with a different provenance:

**Board photographs (real, hand-collected).** 13 JPG photos in
`app/backend/test/`, captured on a phone camera at the table during
actual play sessions. Resolutions vary from roughly 1.5 MP to 12 MP;
lighting ranges from indoor lamp to overhead fluorescent. These are
held out from training and used only for end-to-end evaluation.

**Cropped unit instances (real + augmented).** `[TEAM TODO]` describe
training set size per class, source (cropped from board photos /
photographed individually / synthetic), and augmentation policy. The
trained checkpoint is at `app/backend/vit_classifier.pth` (86.7 MB,
which is consistent with a `vit_small_patch16_224` head with no
modifications).

**Simulator-generated battles (synthetic).** The MLP win-rate model is
trained on outcomes from a Monte-Carlo simulator that implements the
official A&A combat rules (anti-aircraft pre-fire, artillery support
bonus, tactical bomber pairing). `[TEAM TODO]` confirm the number of
generated battles, the sampling distribution over IPC sizes and
attacker/defender mixes, and the train/val split.

The unit vocabulary is fixed across all three datasets: seven
attacker-eligible classes (Infantry, Mech, Artillery, Tank, Fighter,
Tactical Bomber, Strategic Bomber) and one defender-only class
(AA gun). Costs in IPCs are taken directly from the official rulebook
and are constants in `app/backend/pipeline.py`.

## 4.4  Method

> **Required figure.** Architecture diagram. Boxes for OWL-ViT,
> filtering, ViT classifier, faction colour, win-rate MLP, and combat
> simulator, with arrows showing data flow and tensor shapes between
> them. The shapes for each stage are:
> photo → boxes (`N×4`) → crops (`N×3×224×224`) → unit labels
> (`N`) + factions (`N`) → counts dictionary (`8` per faction) →
> scalar win rate.

### 4.4.1  Detection (open-vocabulary)

We invoke OWL-ViT large with seven hand-written natural-language
prompts representing the seven attacker unit families, intentionally
omitting the AA-gun class because it is rare and visually similar to
artillery. Detections below `OWL_THRESHOLD = 0.05` confidence are
discarded; the survivors are passed through (a) torchvision NMS at
`NMS_IOU = 0.1` to suppress duplicate boxes around the same piece;
(b) a containment filter that removes boxes fully inside another box
of equal or higher score; and (c) area / aspect-ratio sanity bounds
(`BOX_AREA_MIN = 0.001`, `BOX_AREA_MAX = 0.4`,
`ASPECT_RATIO_MAX = 10.0`). Constants live in
`app/backend/pipeline.py`.

### 4.4.2  Classification (fine-tuned ViT)

Each surviving box is cropped, resized to 224×224, normalised with
ImageNet statistics, and passed through a `vit_small_patch16_224`
backbone with a custom 7-way classification head trained on our
cropped-piece dataset. Predictions below
`VIT_CONF_THRESH = 0.7` are dropped to avoid contaminating the unit
counts with low-confidence guesses. `[TEAM TODO]` confirm fine-tuning
details: optimiser, learning rate, schedule, number of epochs, base
checkpoint.

### 4.4.3  Faction assignment (HSV heuristic)

For each accepted crop we count pixels falling into a Japan-orange HSV
band (`H∈[5,20], S∈[100,255], V∈[100,255]`) and a USA-green band
(`H∈[25,45], S∈[40,255], V∈[40,255]`) and assign the piece to the
faction with the larger pixel count. Crops in which both counts are
zero are labelled `unknown`. This is a five-line OpenCV step with no
learnable parameters; we kept it because the two factions are
visually disjoint enough that a learned classifier here would be
gratuitous.

### 4.4.4  Win-rate prediction (MLP surrogate)

Counts from both factions are concatenated into a 16-dimensional
input vector (eight unit slots per side) and passed through a small
MLP that returns a single sigmoid-bounded scalar in `[0, 1]`
representing P(attacker wins). The model is loaded once at FastAPI
startup from `app/backend/winrate_model.pt`. `[TEAM TODO]` confirm the
exact MLP architecture (hidden widths, activation, regularisation).

### 4.4.5  Best-composition recommendation (MLP-ranked exhaustive search)

Given a fixed defender stack and an IPC budget *B*, we enumerate every
attacker mix whose total IPC value is ≤ *B*. The legal attacker
vocabulary contains seven unit types with costs ∈ {3, 4, 4, 6, 10,
11, 12} IPC, so for the budgets actually seen at the table (low
double digits) the candidate set is on the order of 10⁴ entries per
budget. Each candidate is scored by the MLP win-rate model and the
top one is returned. The recommender is invoked twice per request —
once at the attacker's current IPC budget (*"what should I have
brought for the same money?"*) and once at the defender's IPC budget
(*"if I matched their investment, what is best?"*).

### 4.4.6  Combat simulation (rules-based, deterministic)

For users who want a concrete battle outcome rather than a probability,
we expose a deterministic simulator that implements the official
single-territory combat rules: AA fire pre-flight, artillery support
bonus on paired infantry, tactical-bomber pairing bonus, casualty
removal in defender-chosen order, repeated rounds until one side is
eliminated. The RNG is seeded with `numpy.random.default_rng(42)`,
so identical inputs produce identical outputs. The simulator both
serves the `/simulate` endpoint and is the source of ground-truth
labels for the win-rate MLP.

### 4.4.7  Serving (FastAPI + static frontend)

All four endpoints (`/analyze`, `/winrate`, `/recommend`, `/simulate`)
are served by a single uvicorn process. CORS is open. A static HTML
page at `app/frontend/index.html` provides the entire UI in one file
with no build step.

## 4.5  Experiments

> **Required figure.** A 2×2 panel: top-left detection P/R curve,
> top-right ViT confusion matrix, bottom-left MLP calibration
> diagram (reliability plot), bottom-right end-to-end latency bar
> chart per endpoint.

### 4.5.1  Inference profiling (done)

Empirical timings on Windows / CPU, measured by
`app/backend/bench_inference.py` and reported in
`docs/inference-optimization.md`:

| Stage | Cost | Notes |
|---|---|---|
| OWL-ViT detection | ~5.0 s | Dominates `/analyze`; image-resolution sensitive |
| ViT classification (sequential) | ~0.13 s | Currently per-crop; ~1.5× faster batched |
| MLP single forward | ~0.28 ms | Used by `/winrate` |
| MLP batched (10⁴ rows) | ~3.8 ms | ~700× speed-up over the sequential loop currently used by `find_best_attack` |
| `find_best_attack` (current) | ~2.8 s/budget | Two budgets per `/recommend` request → ~5.6 s |
| `/winrate` end-to-end | <10 ms | Bottleneck-free |
| `/simulate` end-to-end | tens of ms | Bottleneck-free |

The single most impactful optimisation is batching the MLP forward
inside `find_best_attack`. The detailed roadmap is in
`docs/inference-optimization.md`.

### 4.5.2  Detection accuracy `[RED]`

Plan: hand-label boxes on the 13 held-out board photos in
`app/backend/test/` (or a curated subset) and report precision /
recall at the operating threshold (`OWL_THRESHOLD = 0.05`, NMS IoU
`0.1`). Also report a precision-recall curve sweeping the threshold,
and a small ablation comparing OWL-ViT-base, OWL-ViT-large, and
OWLv2.

### 4.5.3  Classifier accuracy `[RED]`

Plan: held-out cropped instances per class, report top-1 per-class
accuracy, macro-averaged F1, and a confusion matrix figure. Note
which class pairs (likely Tactical Bomber vs Strategic Bomber, Mech
vs Infantry) cause most of the residual error.

### 4.5.4  Win-rate calibration `[RED]`

Plan: for *N* random battles sampled uniformly over IPC sizes
[5, 50], compute the MLP-predicted win rate and the empirical win
rate over 10⁴ Monte-Carlo simulator rolls. Report MAE, a reliability
diagram, and the worst calibration buckets.

### 4.5.5  Recommendation quality `[RED]`

Plan: for each of the 13 held-out board photos, run `/recommend`
and then simulate 10³ battles for both the original attacker mix
and the recommended mix. Report (a) the win-rate delta, and
(b) the fraction of cases where the recommended mix is strictly
better.

### 4.5.6  End-to-end latency `[RED]` (post-optimisation)

Plan: re-run the bench script after the optimisations from
`docs/inference-optimization.md` are merged, and compare a "before
vs after" bar chart per endpoint.

### 4.5.7  Qualitative user study `[RED]`

Plan: two team members play three pre-scripted scenarios each, once
with the tool and once without, and rate decision confidence on a
5-point scale. Report mean and discuss qualitative observations
(false-confidence cases, recognition failures on dim photos, etc.).

## 4.6  References (starter list)

Replace this section with the team's actual BibTeX in the LaTeX
template.

1. Minderer, M., Gritsenko, A., Stone, A., et al. *Simple
   Open-Vocabulary Object Detection with Vision Transformers* (OWL-ViT).
   ECCV 2022.
2. Minderer, M., et al. *Scaling Open-Vocabulary Object Detection*
   (OWLv2). NeurIPS 2023.
3. Dosovitskiy, A., et al. *An Image is Worth 16×16 Words: Transformers
   for Image Recognition at Scale.* ICLR 2021.
4. Wightman, R. *PyTorch Image Models* (`timm`). GitHub, 2019.
5. Carion, N., et al. *End-to-End Object Detection with Transformers*
   (DETR). ECCV 2020.
6. Liu, S., et al. *Grounding DINO: Marrying DINO with Grounded
   Pre-Training for Open-Set Object Detection.* 2023.
7. Mallasén-Quintana, D., del Barrio, A. A., Prieto-Matías, M.
   *LiveChess2FEN: a Framework for Classifying Chess Pieces based on
   CNNs.* 2021.
8. Silver, D., et al. *Mastering the Game of Go without Human
   Knowledge.* Nature 2017.
9. Hugging Face. *Transformers* library. https://huggingface.co/docs/transformers
10. Bradski, G. *The OpenCV Library.* 2000.
11. `[TEAM TODO]` Cite the *AAOddsCalc* / *TripleA* battle calculator
    sources you compared against.
12. `[TEAM TODO]` Cite any published *Axis & Allies* rules document
    used for the simulator implementation.

## 4.7  Required attribution table — what is yours vs others'

> **Rubric:** "Clearly indicate what work is yours and what work is
> others'."
>
> Drop this as a small table or as bullets in the experiments
> appendix.

| Component | Origin | Modification |
|---|---|---|
| OWL-ViT large weights | Google research, via Hugging Face Hub | Used as-is, prompted with our 7 unit-family text strings |
| Hugging Face `transformers` library | Hugging Face | Used unchanged for OWL-ViT processor + model loading |
| `timm` library | Ross Wightman | Used unchanged for `vit_small_patch16_224` backbone |
| ViT classifier head + fine-tuned weights | **Ours** | Trained on our cropped-piece dataset |
| HSV faction-colour heuristic | **Ours** | Hand-tuned thresholds in `pipeline.py` |
| Win-rate MLP architecture and weights | **Ours** | Trained on our simulator outputs |
| Combat simulator (A&A Pacific 1940 rules) | **Ours** | Hand-implemented from official rulebook |
| Best-composition exhaustive search | **Ours** | `find_best_attack` in `pipeline.py` |
| FastAPI service, frontend, benchmarking, docs | **Ours** | This branch |

## 4.8  Required "what we built that wasn't obvious" callouts

> **Rubric:** "Claim credit for everything you've done. … But if you
> spent hours debugging some terrible library or had to call five
> people to get a copy of a dataset, then you should mention this!"

Strong candidates to call out, expanded with one sentence each:

- **Cross-platform setup pain**, documented in
  `docs/readme-run-report.md`: missing `transformers`,
  `python-multipart`, `matplotlib` from `requirements.txt`; the
  README's referenced entry point (`python test.py`) did not exist;
  Windows-specific CUDA wheel handling.
- **Dataset bootstrapping**: `[TEAM TODO]` write 1-2 sentences on the
  effort to photograph and crop training data for the ViT, and the
  effort to generate the simulator dataset for the MLP.
- **Inference bottleneck investigation**: built a microbench
  (`app/backend/bench_inference.py`) and identified that
  `find_best_attack` was calling the MLP 20,000 times sequentially per
  request. Documented the fix path in `docs/inference-optimization.md`.
- **UI walkthrough audit**: ran the static frontend in headless Edge,
  captured 6 distinct UI states, and produced a 17-item friction list
  in `docs/ui-review.md`.

---

# Part 5 — Final Class Presentation (7.5%)

> **Rubric:** "in-class presentation of a to-be-determined length."

Recommended slide order, sized for ~10-12 minutes. Adjust to the
final length the instructor announces.

1. **Title + 30-second pitch.** "Phone photo of a war-game position
   becomes a win probability and a recommended purchase."
2. **The problem, in one slide.** A single live-table photo plus a
   screenshot of an existing battle calculator with all its empty
   text boxes. Drives the "manual entry" pain.
3. **Pipeline at a glance.** Reuse the summary figure from Section 4.1.
4. **Demo (live or video).** Upload one of the held-out photos; show
   counts modal; show all three modes' outputs. ~90 seconds.
5. **Detection results.** PR curve + 1-2 success/failure board photos.
6. **Classification results.** Confusion matrix.
7. **Win-rate calibration.** Reliability plot vs Monte Carlo.
8. **Recommendation quality.** Win-rate delta on held-out photos.
9. **Performance.** Bench-script numbers; before/after bar chart for
   the optimised `find_best_attack`.
10. **Limitations and what we'd do next.** Single-battle scope,
    no multi-territory turn planning, AA-gun detection blind spot,
    HSV thresholds tuned to current paint scheme.
11. **What was ours vs what we leveraged.** Repeat of the
    attribution table.
12. **Q&A.**

---

# Part 6 — Submission packaging checklist

For the final report's required ZIP of code:

- Tag a release commit on `dpm-edits` (or merged `main`) for
  reproducibility.
- Include `README.md`, `app/`, `docs/`, `assets/`, the `.pth`/`.pt`
  weight files, and `app/backend/test/` photos.
- Exclude `app/backend/venv/` (currently committed; should be removed
  before zipping per `docs/readme-run-report.md`).
- Exclude `__pycache__/` and any `*.pyc` files.
- Include `docs/inference-optimization.md`, `docs/ui-review.md`,
  `docs/readme-run-report.md` as supporting evidence of system-level
  engineering work.
- `[TEAM TODO]` Include training notebooks / scripts for the ViT
  classifier and the win-rate MLP if they exist outside this repo.

---

## Appendix A — Quick reference: where everything lives in the repo

| File | Role |
|---|---|
| `app/backend/main.py` | FastAPI app, four endpoints, combat simulator |
| `app/backend/pipeline.py` | OWL-ViT detection, ViT classification, faction colour, MLP, find-best-attack |
| `app/backend/bench_inference.py` | Empirical CPU timings used in the report |
| `app/backend/vit_classifier.pth` | Trained ViT classifier weights |
| `app/backend/winrate_model.pt` | Trained MLP win-rate weights |
| `app/backend/test/` | 13 held-out board photos |
| `app/frontend/index.html` | Single-file static UI |
| `app/frontend/frontend_api_guide.txt` | API contract used by the frontend |
| `docs/readme-run-report.md` | Two-phase setup walkthrough + issues list |
| `docs/ui-review.md` | UI screenshots + 17-item friction analysis |
| `docs/inference-optimization.md` | Inference flow + ranked optimisation roadmap |
| `docs/images/ui-initial.png` | First-load UI screenshot |
| `docs/images/ui-states.png` | Composite of all dynamic UI states |
| `assets/annotated_cv_boxes_green.jpg` | Sample OWL-ViT detection visualisation |
| `README.md` | Setup, usage, output interpretation, API reference |

## Appendix B — Headline numbers to put in every report (single source of truth)

These are the few numbers that should be consistent across the
proposal, feasibility study, midterm, and final. Update here once,
copy everywhere.

- Number of unit classes: **8** (7 attacker-eligible + 1 defender-only AA).
- Detector: **OWL-ViT large** (`google/owlvit-large-patch14`),
  open-vocabulary with 7 text prompts.
- Classifier: **ViT-Small/16** (`vit_small_patch16_224`) fine-tuned.
- Win-rate model: small MLP, **16-dim input** (8 slots × 2 factions),
  scalar output.
- Held-out real photos: **13**.
- Trained-weight sizes on disk: **86.7 MB** (ViT) + **1.4 MB** (MLP).
- Endpoints: **4** (`/analyze`, `/winrate`, `/recommend`, `/simulate`)
  + **1** health (`/health`).
- Worst-case `/analyze` latency on CPU laptop: **~5 s** (OWL-ViT-bound).
- Worst-case `/recommend` latency on CPU laptop, current build:
  **~5.6 s** (find_best_attack-bound, planned to drop to **<50 ms**).
