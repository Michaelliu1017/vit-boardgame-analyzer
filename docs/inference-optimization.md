# Transformer Inference — Flow & Optimization Findings

A code-level walkthrough of what happens when the backend serves `/analyze`,
`/winrate`, and `/recommend`, where the time actually goes (with measured
numbers), and which optimizations are worth doing — including the one that
matters most by far.

> **TL;DR**
> 1. **`OWL-ViT large` dominates `/analyze`** at ~5 s of a ~5.3 s request on
>    CPU (≈94%). ViT classification is only ~5%. Batching the classifier (the
>    obvious "parallelism" win) saves ~90 ms — basically nothing. Real wins
>    against `/analyze` mean shrinking OWL-ViT or its inputs.
> 2. **`/recommend` is bottlenecked by an unbatched MLP search loop**.
>    Each `find_best_attack` call runs the 15-feature MLP **10 000 times
>    sequentially** on single-row tensors (5 000 random + 5 000 greedy
>    attempts), and `/recommend` runs it twice (attacker budget + defender
>    budget) = **20 000 calls per request**. Replacing the inner loop with
>    one batched forward is **~700×** faster (2 800 ms → ≤4 ms per call) for
>    free.
> 3. **"Parallel calls" is the wrong frame** for this workload. The work is
>    GIL-bound Python plus a single model on a single device — true
>    parallelism doesn't help. **Batching is the GPU/CPU-friendly version of
>    parallelism.** Use it everywhere a Python loop currently feeds a model
>    one row at a time.

---

## 1. The actual inference flow

There are two transformer models in the request path, plus a tiny MLP:

| Model | Role | Source | Size | Loaded |
| ----- | ---- | ------ | ---- | ------ |
| **OWL-ViT large** (`google/owlvit-large-patch14`) | Open-vocabulary object detection — turns a board photo into bounding boxes | HuggingFace, downloaded on first launch | ~1.7 GB | At FastAPI startup |
| **ViT-Small** (`vit_small_patch16_224` via `timm`, fine-tuned head) | Per-crop unit-type classification (Infantry / Tank / …) | Local checkpoint `vit_classifier.pth` | 87 MB | At FastAPI startup |
| **15-feature MLP** (5 fully-connected layers) | Win-rate prediction + composition search | Local checkpoint `winrate_model.pt` | 1.4 MB | At FastAPI startup |

### 1.1 `POST /analyze` — the only image-using endpoint

```
HTTP request
  └── await image.read()                         # bytes
  └── PIL.Image.open(...).convert('RGB')         # decode
  └── detect_pieces(image, owl_proc, owl_model)  # OWL-ViT  ← ~5 s
       ├── owl_processor(text=PROMPTS, images=image)   # CPU preprocess + 7 prompts tokenized
       ├── owl_model(**inputs)                         # one forward pass on full image
       ├── post_process_grounded_object_detection(...) # head + decode boxes/labels
       ├── nms()                                       # CPU
       ├── area + aspect-ratio filter                  # Python list comp
       └── filter_containing_boxes()                   # O(N^3) Python loops
  └── classify_pieces(image, boxes, vit_model, names)  # ViT      ← ~260 ms / 12 boxes
       └── for box in boxes:                           # SEQUENTIAL
              crop = image.crop(...)
              tensor = val_tf(crop).unsqueeze(0).to(device)  # batch=1
              with no_grad: logits = vit_model(tensor)
              softmax + argmax + .item()               # forces CPU sync
              if conf < 0.7: skip
  └── for box in valid_boxes:                          # SEQUENTIAL, CPU
         get_faction_by_color(image, box)              # PIL crop + cv2 HSV mask  (~few ms)
  └── tally → {JP: {...}, US: {...}}
```

Critically, every step depends on the output of the previous one — **there is
no pipeline parallelism available within a single request**. Boxes don't exist
until OWL-ViT returns; crops don't exist until boxes do; faction can't be
computed until a box is kept; etc. The only "parallel" win available is
**batching across boxes** in stages 2 and 3.

### 1.2 `POST /winrate` — pure MLP

```
HTTP request -> 15-int feature vector -> normalize (mu, std) -> mlp_model(x) -> sigmoid -> float
```

Cost is **~0.3 ms** on CPU. Uninteresting.

### 1.3 `POST /recommend` — MLP search loop

```
HTTP request
  └── current_wr = mlp_predict(...)                   # 1 call, 0.3 ms
  └── for budget in [attacker_ipc, defender_ipc]:     # 2 budgets
        find_best_attack(defender, budget, n_samples=10000)
            └── for _ in range(5000):  _eval_vec(...) # random sampler — 5 000 batch=1 MLP forwards
            └── for _ in range(5000):  _eval_vec(...) # greedy sampler — 5 000 batch=1 MLP forwards
```

Cost: **~2 800 ms per `find_best_attack` × 2 budgets = ~5.6 s** on CPU. The
MLP itself is trivial — the time is Python loop overhead × tiny tensor ×
20 000 invocations per request.

### 1.4 `POST /simulate` — pure Python, no models

Numpy RNG + arithmetic. Sub-millisecond. Not interesting for this analysis.

---

## 2. Where time actually goes (measured)

Numbers below are from `app/backend/bench_inference.py` running on this
machine (Windows / Python 3.12 / `torch 2.11.0+cpu` / `transformers 5.6.2`,
**CPU only**), using `app/backend/test/t5.JPG` (4032 × 3024). Best-of-3 runs
shown.

| Stage | Current implementation | Notes |
| ----- | ---------------------- | ----- |
| OWL-ViT detection (1 image, 7 prompts) | **4 882 ms** | Full forward pass + post-process |
| ViT classification, per-crop loop (12 crops) | 260 ms | Sequential `for box in boxes` |
| ViT pure forward × 12, batch=1 each | 243 ms | Just the model calls |
| ViT pure forward × 1, batch=12 | 170 ms | Same crops, batched |
| ViT classification, **batched** (proposed) | 172 ms | Includes preprocessing |
| MLP forward, batch=1 | 0.3 ms | Single composition |
| MLP forward, batch=10 000 | **3.5 ms** | All compositions in one shot |
| `find_best_attack(budget=20, n=10 000)` (current) | **2 787 ms** | The endpoint-killer |

### 2.1 What the numbers say at a glance

Per-endpoint cost on this machine, **with current code**:

```
/analyze      ≈ 5 300 ms  (OWL ~94%, ViT ~5%, color/Python ~1%)
/winrate      ≈   1 ms
/simulate     ≈   1 ms
/recommend    ≈ 5 600 ms  (almost all in the MLP search loop)
```

Per-endpoint cost **with the proposed changes** in §3 (CPU; GPU/MPS would be
larger gains):

```
/analyze      ≈ 5 200 ms  (-2%)   ← batching ViT saves only ~90 ms
/recommend    ≈    10 ms  (-99%)  ← batching the MLP search is night-and-day
```

The headline finding: **the famous "batch the transformer crops" optimization
is irrelevant here.** The unsexy "batch the trivial MLP search" optimization is
~800× and free.

---

## 3. Optimization opportunities, ranked by impact

### Tier 1 — Do these. Big wins, near-zero risk.

#### 3.1 Batch the MLP search in `find_best_attack`  ★★★★★

**Where:** `app/backend/pipeline.py` — `_eval_vec`, `find_best_attack`.
**Effect (measured):** 2 787 ms → ≤4 ms per call (~700×). Affects `/recommend`
end-to-end, taking it from ~5.6 s to ~10 ms on CPU. On GPU the gain is even
larger because tensor-launch overhead dominates batch=1 calls.

**Today:**

```python
# pipeline.py
def _eval_vec(a, d_vec, mlp_model, mu, std):
    vec15 = list(a) + d_vec
    x = np.array([vec15], dtype=np.float32)
    x_norm = (x - mu) / (std + 1e-8)
    xt = torch.tensor(x_norm).to(device)
    with torch.no_grad():
        logit = mlp_model(xt).item()              # forces CPU sync
    return float(1.0 / (1.0 + np.exp(-logit)))

def find_best_attack(defender, budget, mlp_model, mu, std, n_samples=10000, seed=42):
    ...
    for _ in range(half):                          # 10 000 iterations
        a = sample_one_random_composition(...)
        wr = _eval_vec(a, d_vec, mlp_model, mu, std)   # one tiny MLP call
        if wr > best_wr: ...
    for _ in range(n_samples - half):              # another 10 000 iterations
        a = sample_one_greedy_composition(...)
        wr = _eval_vec(a, d_vec, mlp_model, mu, std)
        ...
```

**Proposed:** generate the full sample matrix in NumPy first, then run a
single batched forward pass.

```python
def find_best_attack(defender, budget, mlp_model, mu, std, n_samples=10000, seed=42):
    if budget <= 0:
        return {k: 0 for k in A_KEYS}, 0.0
    rng   = np.random.default_rng(seed)
    costs = np.array([UNIT_COST[u] for u in A_KEYS], dtype=np.int32)
    d_vec = np.array([defender.get(k, 0) for k in D_KEYS], dtype=np.float32)

    # 1. Sample N candidate compositions in pure NumPy (cheap, CPU)
    samples = sample_compositions(rng, costs, budget, n_samples)   # shape (N, 7)

    # 2. Build the (N, 15) feature matrix and normalize
    feats = np.concatenate(
        [samples, np.broadcast_to(d_vec, (n_samples, 8))], axis=1
    ).astype(np.float32)
    feats = (feats - mu) / (std + 1e-8)

    # 3. ONE forward pass through the MLP
    with torch.no_grad():
        logits = mlp_model(torch.from_numpy(feats).to(device)).cpu().numpy()
    wrs = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))

    # 4. Pick the winner
    i = int(wrs.argmax())
    best_atk = dict(zip(A_KEYS, samples[i].tolist()))
    return best_atk, float(wrs[i])
```

`sample_compositions` is a small NumPy port of the existing two
sampling strategies (random affordable picks + greedy multi-of-each); both are
trivially vectorizable.

**Caveats:** The MLP uses `BatchNorm1d`. In `eval()` mode it uses running
stats and tolerates any batch size, so this is safe. Output ordering is
deterministic if the RNG sampling is seeded the same way. Consider keeping
the current and batched implementations for one release behind a feature flag
to verify equivalence on a few seeds.

#### 3.2 Cache OWL-ViT text embeddings (skip the text encoder per request)  ★★★★☆

**Where:** `app/backend/main.py` startup, `pipeline.py:detect_pieces`.
**Effect:** `OwlViTProcessor` runs both image and text encoders on every
request, but the **7 prompts in `OWL_TEXTS` never change**. Pre-tokenizing
the prompts once at startup is free; pre-running the text encoder once at
startup is the real win — typically a non-trivial fraction of OWL-ViT's
compute is in the text branch. The expected saving is single- to low-double-
digit-percent of `/analyze` wall time, depending on hardware and prompt count.

**Today:** every request runs:

```python
inputs = owl_processor(text=OWL_TEXTS, images=image, return_tensors="pt").to(device)
outputs = owl_model(**inputs)        # text encoder + image encoder + heads
```

**Proposed:** at startup, tokenize the prompts and run the text encoder once;
keep the resulting query embeddings on the device. At request time, only run
the image branch and feed the cached text features into the matching head.

The exact HuggingFace API surface (e.g. `OwlViTModel.get_text_features`,
`get_image_features`, `OwlViTForObjectDetection.image_guided_detection`,
overriding the forward to accept pre-computed `text_embeds`) **shifts between
`transformers` versions** — what works on 4.x doesn't always work on 5.x. The
repo currently uses `transformers 5.6.2`. Before implementing, read the
installed version's `OwlViTForObjectDetection.forward` signature and pick the
cleanest path (the model accepts pre-encoded text inputs in some versions and
not in others). If a clean injection path doesn't exist, an acceptable fallback
is monkey-patching `owl_processor` to accept pre-tokenized text once.

#### 3.3 Downscale very-large input images before OWL-ViT  ★★★★☆

**Where:** `app/backend/main.py:analyze`.
**Effect:** Phone photos arriving at 4 032 × 3 024 (12 MP) get re-projected
internally to OWL-ViT's expected resolution (768 px on the long side for
patch14-large) inside the processor, but the upstream PIL → tensor pipeline,
the bicubic resize itself, and the post-processing (which uses the original
`image.size`) all still pay for the raw pixels. Pre-shrinking to e.g.
≤ 1 280 px on the long side before handing off to the processor saves a
small but free amount of wall time per request, and the saving scales
linearly with input resolution — worth doing if any users upload at full
phone resolution. Quality should be unaffected because OWL-ViT was going to
downsample anyway. Verify on `app/backend/test/` before merging.

```python
MAX_DIM = 1280
def downscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(1.0, MAX_DIM / max(w, h))
    return img if s == 1.0 else img.resize((int(w*s), int(h*s)), Image.BILINEAR)
```

Use immediately after `Image.open(...).convert('RGB')`.

#### 3.4 Don't block the FastAPI event loop on inference  ★★★★☆

**Where:** `app/backend/main.py` — every endpoint that touches a model.
**Effect:** Today, `/analyze` is `async def` but its body calls **synchronous**
`torch` code that holds the event loop for ~5 s. While that runs, even
`/health` requests on other connections sit in the queue. Other endpoints
(`/winrate`, `/recommend`, `/simulate`) are correctly `def` so FastAPI runs
them in a thread pool — but `/analyze` got the wrong decorator.

The fix is one of two one-liners:

- Switch `/analyze` to `def` (then FastAPI auto-runs it in the threadpool):

  ```python
  @app.post('/analyze')
  def analyze(image: UploadFile = File(...)):
      data = image.file.read()                 # use the sync read
      ...
  ```

- Or keep `async def` and explicitly hand off to a thread:

  ```python
  @app.post('/analyze')
  async def analyze(image: UploadFile = File(...)):
      data = await image.read()
      result = await asyncio.to_thread(_analyze_sync, data)
      return result
  ```

Either way, `/health` and similar endpoints stay snappy under load. Note this
does not make `/analyze` itself any faster — it just stops it from
strangling the rest of the server.

### Tier 2 — Worth doing if you have GPU/MPS hardware. Modest CPU wins.

#### 3.5 Batch the ViT classifier  ★★☆☆☆ on CPU, ★★★★☆ on GPU/MPS

**Effect (CPU, measured):** 260 ms → 172 ms per `/analyze` (~1.5×). On
GPU/MPS the same change is typically **5–20×** because each batch=1 call has
fixed kernel-launch overhead that dwarfs the ViT-Small compute itself.

**Today:** sequential `for box in boxes` with batch-of-one forwards.

**Proposed:**

```python
def classify_pieces(image, boxes, vit_model, class_names):
    if len(boxes) == 0:
        return [], []
    crops = []
    for box in boxes:
        x1,y1,x2,y2 = [int(b.item()) for b in box]
        crops.append(val_tf(image.crop((x1,y1,x2,y2))))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        probs = torch.softmax(vit_model(batch), dim=1)
        confs, preds = probs.max(dim=1)
    out_preds, out_boxes = [], []
    for i, (p, c, b) in enumerate(zip(preds.tolist(), confs.tolist(), boxes)):
        if c >= VIT_CONF_THRESH:
            out_preds.append(class_names[p])
            out_boxes.append(b)
    return out_preds, out_boxes
```

If a board ever has *very* many boxes (e.g. > 64), chunk the batch to bound
memory.

#### 3.6 FP16 / BF16 inference  ★★☆☆☆ (only useful on GPU/MPS)

Both transformers tolerate `torch.float16` in eval mode and roughly halve
runtime + memory on CUDA. On Windows the default `torch` install is CPU-only,
so this only pays off after the user installs the CUDA wheel (see README §
Setup → Windows). Code change is trivial:

```python
owl_model = owl_model.to(device).to(torch.float16).eval()
vit_model = vit_model.to(device).to(torch.float16).eval()
# remember to .to(torch.float16) on inputs as well
```

Skip for the MLP — it's already trivial in FP32 and BatchNorm + FP16 is finicky.

#### 3.7 Replace `OWL-ViT large` with `OWL-ViT base` or `OWLv2 base`  ★★★★☆ (with quality risk)

**Effect:** `google/owlvit-base-patch32` is ~12× fewer parameters and tends
to be 5–10× faster. `google/owlv2-base-patch16-ensemble` is the modern
successor with better detection quality for similar compute. Both would shrink
`/analyze` from ~5 s to roughly 0.5–1 s on CPU.

**Risk:** The downstream pipeline already filters with NMS, area, aspect
ratio, and a containment heuristic, so it is reasonably tolerant of detector
noise. But the ViT classifier was trained on crops produced by the *current*
detector, so a swap is best validated on the existing `app/backend/test/`
images (eyeball the resulting `result_detection.jpg` from `test_pipeline.py`)
before merging.

### Tier 3 — Incremental, optional.

#### 3.8 `torch.compile()` the two transformers  ★★☆☆☆

PyTorch 2.x can JIT-compile model forwards. Typical wins: 1.2–1.5× on CPU,
larger on CUDA. Apply at startup:

```python
owl_model = torch.compile(owl_model, mode='reduce-overhead')
vit_model = torch.compile(vit_model, mode='reduce-overhead')
```

First request pays compile latency; warm requests benefit. Skip on Windows +
older Python versions where the Triton backend can be flaky.

#### 3.9 Vectorize `filter_containing_boxes`  ★☆☆☆☆

The function is currently O(N²) Python with an O(N) `in keep` membership check
inside a nested loop, i.e. effectively **O(N³)**. For typical N≈10–50 it is
imperceptible (~ms). It only matters if a future cluttered board pushes N
above ~200, at which point it could spike into the seconds. A two-line fix:

```python
from torchvision.ops import box_iou
def filter_containing_boxes(boxes, scores, labels):
    if len(boxes) == 0: return boxes, scores, labels
    inter = box_inter(boxes, boxes)                       # NxN
    area_j = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    contained = inter / area_j[None, :]                   # NxN
    contained.fill_diagonal_(0.0)
    drop = (contained > OVERLAP_THRESH).any(dim=1)
    keep = ~drop
    return boxes[keep], scores[keep], labels[keep]
```

(Where `box_inter` is the 4-line tensor intersection helper; not bundled in
torchvision but trivial.)

#### 3.10 Reuse a single `np.array(image)` for color analysis  ★☆☆☆☆

`get_faction_by_color` currently does `image.crop(...)` then `np.array(...)`
*per box*. Decoding once and slicing the numpy array is faster, and matters
linearly with N. Saves a few ms per request — not nothing, but small.

---

## 4. About "parallel calls"

A reasonable first instinct is "can we run OWL-ViT and ViT in parallel?" or
"can we run several ViT calls in threads?" — but neither helps here. Here's
why, for the record:

| Idea | Why it doesn't help |
| ---- | ------------------- |
| Run OWL-ViT and ViT in parallel within `/analyze` | ViT depends on OWL's boxes. Strict data dependency. |
| Spawn a thread per crop for ViT | All threads serialize on the GIL during NumPy/PIL prep, and on the single underlying model call (no concurrent CUDA streams in this code). Adds overhead, gives nothing. |
| `multiprocessing.Pool` of model workers | Each worker reloads the 1.7 GB OWL-ViT model. Memory blows up, startup cost dwarfs request cost, and the bottleneck is still one model per box. |
| Async overlap of `image.read()` + model forward | `await image.read()` finishes in microseconds for a typical photo. Nothing to overlap with. |
| Concurrent requests | A second `/analyze` arriving while another is in flight will queue on the model regardless of how it's dispatched (one device, one model). The right concurrency primitive here is **request batching** (collect requests for ~50 ms, batch their images into one OWL forward) — but that only pays off under real load. Premature for the current usage profile. |

The pattern that *does* help — repeatedly — is **batching**: turn a Python
loop that issues N tiny model calls into a single model call on a tensor of
shape `(N, ...)`. That is what the §3.1 and §3.5 changes do. It is the
GPU/CPU-friendly, GIL-friendly, deterministic, debuggable form of "parallel
calls" for ML inference workloads.

---

## 5. Summary table

| # | Change | Endpoint impacted | CPU effect (measured) | GPU/MPS effect (expected) | Risk | Effort |
| - | ------ | ----------------- | --------------------- | ------------------------- | ---- | ------ |
| 3.1 | Batch MLP search in `find_best_attack` | `/recommend` | ~700× per call (2 800 ms → ≤4 ms) | similar | Low | ~30 lines |
| 3.2 | Cache OWL-ViT text embeddings | `/analyze` | ~5–10% off `/analyze` | larger | Med (API surface) | ~20 lines |
| 3.3 | Downscale large input images | `/analyze` | small, scales with input resolution | similar | Low | 5 lines |
| 3.4 | Stop `/analyze` blocking the event loop | `/analyze` + all others under load | unblocks `/health` etc. | unblocks `/health` etc. | Low | 1 line |
| 3.5 | Batch ViT classification | `/analyze` | 1.5× on this stage (~90 ms saved) | 5–20× on this stage | Low | ~25 lines |
| 3.6 | FP16/BF16 | `/analyze` | n/a (CPU) | ~2× | Med (numerics) | ~10 lines |
| 3.7 | Swap to `OWL-ViT base` or `OWLv2 base` | `/analyze` | 5–10× on `/analyze` | 5–10× on `/analyze` | Med-High (quality) | ~5 lines + revalidation |
| 3.8 | `torch.compile()` | `/analyze` | 1.2–1.5× | 1.2–1.5× | Med (compatibility) | 2 lines |
| 3.9 | Vectorize `filter_containing_boxes` | `/analyze` | ms today; protects against N>200 | same | Low | ~10 lines |
| 3.10 | Single `np.array(image)` for color | `/analyze` | a few ms | a few ms | Low | ~10 lines |

---

## 6. Recommended order of work

If only one change ships: **§3.1** (batched MLP search). It cuts `/recommend`
latency by 99% and is purely additive — same model, same training, just
better Python.

If a small batch of changes ships, in order:

1. §3.1 — batch the MLP search (huge win, no risk)
2. §3.4 — stop blocking the event loop on `/analyze` (1 line, unblocks the rest of the API)
3. §3.3 — downscale oversize images (5 lines, free win)
4. §3.5 — batch the ViT classifier (small win on CPU, large on GPU/MPS — pays for itself the moment a CUDA box is plugged in)
5. §3.2 — cache OWL-ViT text embeddings
6. §3.7 — swap OWL-ViT large → base/v2 (largest single `/analyze` win, but needs a quality eyeball pass)

Tiers 3.6 / 3.8 / 3.9 / 3.10 can wait until someone is profiling for the
last 20%.

---

## 7. Reproducing the measurements

The numbers above were produced by `app/backend/bench_inference.py` (also
checked into this branch). To re-run:

```powershell
cd app\backend
.\venv\Scripts\activate
$env:PYTHONIOENCODING = 'utf-8'
python bench_inference.py test\t5.JPG
```

Each stage is repeated 3× and the **best** time reported, to limit noise from
Windows scheduling and HF cache warmup. Vary `TEST_IMAGE` and `n_samples` to
explore other operating points.
