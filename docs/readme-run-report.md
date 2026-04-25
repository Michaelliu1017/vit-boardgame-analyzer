# README Run Report

Empirical walkthrough of the README on a Windows machine, capturing every deviation, missing dependency, and observed runtime issue. No source code was modified during this run.

- Date: 2026-04-25
- Branch: `dpm-edits` (off `main` @ `c57c402`)
- Performed by: developer dry-run, no fixes applied

---

## 1. Environment snapshot

| Field | Value |
| --- | --- |
| OS | Windows 10 (NT 10.0.26200) |
| Shell | PowerShell 5.1.26100.8115 (Desktop edition) |
| Default `python` | 3.12.10 |
| Available via `py -0` | 3.13 (default), 3.12, 3.10 — **no 3.11** |
| GPU | NVIDIA GeForce RTX 4080, driver 595.97, CUDA 13.2 |
| Git | clone present, HEAD `c57c402`, branch `dpm-edits` (clean) |

Key implication: README requires "Python 3.11 or 3.12". Only 3.12 is available on this box; 3.11 is not installed. Also the box has CUDA hardware, which becomes relevant for the torch install discussion below.

---

## 2. Phase 1 — strict literal README walkthrough

Each step was executed exactly as written in [README.md](../README.md). Stop-and-log on first failure of each section.

### 2.1 Setup block

README:

```bash
cd app/backend
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

| Step | Result |
| --- | --- |
| `cd app/backend` | OK |
| `python3.11 -m venv venv` | **FAIL** — `'python3.11' is not recognized as the name of a cmdlet, function, script file, or operable program.` |
| `venv\Scripts\activate` | Skipped (no venv to activate). Tested separately in Phase 2: actually works in PowerShell, see §3.1. |
| `pip install -r requirements.txt` | Skipped (no venv). Run literally in Phase 2, see §3.2. |

Phase 1 ended here for Setup — venv creation blocked the rest.

### 2.2 Run block

README:

```bash
python test.py
```

```
C:\Users\dhair\AppData\Local\Programs\Python\Python312\python.exe: can't open file
'C:\\--DPM-MAIN-DIR--\\--GIT-REPOS--\\vit-boardgame-analyzer\\app\\backend\\test.py':
[Errno 2] No such file or directory
```

`test.py` does not exist anywhere in the repository (`git ls-files | grep test.py` is empty). Closest file is `test_pipeline.py`, but that is a debug visualization script, not the application entry point. Real entry point is `uvicorn main:app`.

The README also claims:

> "Follow the prompts to enter enemy units and IPC budget. The script will output the optimal attack composition and predicted win rate."

There are **no interactive prompts** anywhere in the current code. The system is now a FastAPI backend + static HTML frontend. The README is describing a prior CLI version that has been removed.

### 2.3 Prerequisites check

README states:

> `blr_weights.npz` and `winrate_model.pt` placed in `app/backend/`

| File | Present? | Used by current code? |
| --- | --- | --- |
| `blr_weights.npz` | No (not anywhere on disk, and not in git) | No — only mentioned in README. No code reference. |
| `winrate_model.pt` | **Yes**, present in `app/backend/` (1.4 MB, tracked in git) | Yes — loaded by `main.py` |
| `vit_classifier.pth` | **Yes**, present in `app/backend/` (86.7 MB, tracked in git) — **not mentioned in README** | Yes — loaded by `main.py` |

Net: README requirements are out of sync with code. The `vit_classifier.pth` requirement is undocumented; the `blr_weights.npz` requirement is obsolete.

### 2.4 Phase 1 conclusion

The README cannot be followed end to end on this machine. It fails at:
1. Python interpreter naming (`python3.11` not on PATH).
2. Documented entry point (`test.py` does not exist).
3. Documented prerequisites (`blr_weights.npz` not used; `vit_classifier.pth` not mentioned).

---

## 3. Phase 2 — minimal corrections to push further

Each correction is documented so it can roll back into a README/dep update.

### 3.1 Correction: use `py -3.12` for venv creation

```powershell
py -3.12 -m venv venv
```

Created the venv successfully under `app/backend/venv/`. (Note: this conflicts with the venv that is already committed to git — see §4.2.)

Then tested README's literal Windows hint:

```powershell
venv\Scripts\activate
```

Result: works in PowerShell 5.1 — resolved `activate` to `activate.bat`, set `$env:VIRTUAL_ENV`, and put the venv's `python.exe` first on PATH. So that particular README line is correct on Windows.

### 3.2 Correction: install via the `requirements.txt` literally

`pip install -r requirements.txt` succeeded but produced a behaviorally important result:

| Package | Installed | Notes |
| --- | --- | --- |
| `torch` | `2.11.0+cpu` | **CPU-only wheel.** Despite RTX 4080 + CUDA 13.2 driver being present, default PyPI wheel for `torch` on Windows is CPU. To get GPU support, requirements would need `--index-url https://download.pytorch.org/whl/cu124` (or appropriate CUDA tag) or a separate install line. |
| `pybanner` | git+https | Required `git` on PATH; worked here. Will fail in environments without git. |
| `fastapi` | 0.136.1 | unpinned; got latest. |
| `transformers` | **NOT installed** | Not in `requirements.txt`. Required by `pipeline.py` line 3. |
| `matplotlib` | **NOT installed** | Not in `requirements.txt`. Required by `test_pipeline.py` line 7. |
| `python-multipart` | **NOT installed** | Not in `requirements.txt`. Required by FastAPI for `UploadFile` form data. |
| `scikit-learn` | 1.8.0 | Listed in requirements but not actually imported by any source file. |

After verification:

```powershell
PS> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.11.0+cpu  False
```

So on this hardware the system silently runs on CPU even though a GPU is available. (As a side effect, this masks the device-mismatch bug between [`app/backend/main.py`](../app/backend/main.py) and [`app/backend/pipeline.py`](../app/backend/pipeline.py): with both falling back to `cpu`, the mismatch never triggers.)

### 3.3 Correction: install missing deps to make backend importable

```powershell
python -m pip install transformers       # 5.6.2 installed
python -c "import main"                  # FAILS — RuntimeError: Form data requires "python-multipart" to be installed.
python -m pip install python-multipart   # 0.0.26 installed
python -c "import main"                  # OK
```

Two implicit fixes were required just to import the backend module:
1. `transformers` (used by [`app/backend/pipeline.py`](../app/backend/pipeline.py) for `OwlViTProcessor`/`OwlViTForObjectDetection`)
2. `python-multipart` (transitively required by FastAPI because [`app/backend/main.py`](../app/backend/main.py) line 182 declares `image: UploadFile = File(...)`)

Note on `transformers` v5: it still ships `OwlViTProcessor` and `OwlViTForObjectDetection`, so the import works without code changes. Newer Owlv2 classes are recommended but not required for runtime today.

### 3.4 Discovery: model weights and assets are present after all

A disk-wide search found:

```
app/backend/vit_classifier.pth   86.7 MB   (tracked)
app/backend/winrate_model.pt      1.4 MB   (tracked)
```

`git ls-files` further reveals additional tracked artifacts that the README does not mention:

- `assets/annotated_cv_boxes_green.jpg`, `assets/display.JPG`, `assets/gitowl.png` — README image references resolve.
- `app/backend/test/t0..t6.JPG`, `test/test1..5.jpg` — 13 sample boards usable by `test_pipeline.py` and `/analyze`.
- `app/backend/result.jpg`, `result_compare.jpg`, `result_crops.jpg`, `result_detection.jpg` — leftover output from a previous `test_pipeline.py` run, committed.

So weights and assets are not the blocker. The blocker is the documented setup steps and missing deps.

### 3.5 Run the real entry point

With the venv set up and the missing deps installed:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Startup output (abridged):

```
INFO:     Started server process
INFO:     Waiting for application startup.
Warning: You are sending unauthenticated requests to the HF Hub. ... HF_TOKEN ...
UserWarning: huggingface_hub cache-system uses symlinks ... your machine does not support them ...
   To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator.
Loading weights: 100%|##########| 604/604 [00:00<00:00, 30863.39it/s]
All models loaded  device=cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Startup succeeded after a one-time download of `google/owlvit-large-patch14` from HuggingFace (~90 seconds on this connection). Time from `uvicorn` invocation to "Application startup complete" was about 100 seconds end to end. No errors.

### 3.6 Endpoint probes

All five endpoints returned 200 OK. Selected payloads:

`GET /health`
```json
{"status":"ok"}
```

`POST /winrate` with all-zero units (attacker JP)
```json
{"win_rate":0.5833,"attacker_ipc":0,"defender_ipc":0}
```
Observation: with no units on either side, the model predicts attacker wins 58%. Expected behavior is undefined (no battle), but the magnitude suggests the MLP has a learned bias.

`POST /winrate` with realistic balanced forces (2I+1A vs 3I+1A)
```json
{"win_rate":0.0976,"attacker_ipc":10,"defender_ipc":13}
```
9.76% attacker win rate for what looks like a roughly even match feels very pessimistic. May indicate model bias, label inversion, or a normalization issue. Warrants a model-side review, separate from code bugs.

`POST /simulate` — defender mech bug reproduction. Sent `JP=10 Infantry, US=2 Mech + 1 AA`:
```json
{
  "winner": "JP",
  "attacker_survivors": {"Infantry":10,"Mech":0,"Artillery":0,"Tank":0,"Fighter":0,"TacBmb":0,"StrBmb":0,"AA":0},
  "defender_survivors": {"Infantry":0,"Mech":2,"Artillery":0,"Tank":0,"Fighter":0,"TacBmb":0,"StrBmb":0,"AA":0}
}
```
**Bug confirmed.** Defender lost (`winner: JP`) but the response says 2 mechs survived. Cause: in `combat()` ([`app/backend/main.py`](../app/backend/main.py) lines 50–105), defender mechs are merged into the infantry pool for combat (`di += dm`) but never restored or zeroed afterward. The casualty loop only iterates `['i','a','t','f','tb','sb']` for `D`, and there is no `restore_mech` call for the defender (only the attacker has one). So `D['m']` retains its initial value across combat. Same class of bug applies to `D['aa']`, partly mitigated by line 105 zeroing it when defender loses entirely.

`POST /recommend` with realistic balanced forces:
```json
{
  "current_win_rate": 0.0976,
  "attacker_ipc": 10,
  "defender_ipc": 13,
  "recommendations": [
    { "label":"Attacker budget (10 IPC)","budget":10,
      "units":{"Infantry":2,"Mech":0,"Artillery":1,"Tank":0,"Fighter":0,"TacBmb":0,"StrBmb":0},
      "win_rate":0.0976 },
    { "label":"Defender budget (13 IPC)","budget":13,
      "units":{"Infantry":3,"Mech":0,"Artillery":1,"Tank":0,"Fighter":0,"TacBmb":0,"StrBmb":0},
      "win_rate":0.3039 }
  ]
}
```
Two issues here:
- The `units` dict for each recommendation **omits the `AA` key**, while `/winrate` and `/simulate` always include it. Cause: `mlp_atk_to_frontend()` ([`app/backend/main.py`](../app/backend/main.py) line 134) reverse-maps from `ATK_KEY`, in which `AA` maps to `None`, so AA is dropped. Frontend tolerates this because it does `units[u] || 0` when rendering, but it's a contract inconsistency.
- At the attacker's own budget (10 IPC), the recommended composition is the same as what the user already has, with the same win rate — i.e., random search across 10,000 samples did not find anything better. Not necessarily a bug, but a UX dead-end at small budgets.

`POST /analyze` with `app/backend/test/t5.JPG` (real board photo, 13 units):
- Latency: ~5.8 seconds end to end on CPU.
- Response:
```json
{"JP":{"Infantry":5,"Mech":0,"Artillery":1,"Tank":0,"Fighter":0,"TacBmb":0,"StrBmb":0,"AA":0},
 "US":{"Infantry":2,"Mech":1,"Artillery":1,"Tank":1,"Fighter":1,"TacBmb":0,"StrBmb":0,"AA":0}}
```
- No errors during request. Counts look plausible for a board photo, though human verification against the actual image is needed for accuracy assessment.
- Server log was clean: no Python warnings, no deprecation messages during the request.

### 3.7 Frontend file sanity check

Static check of [`app/frontend/index.html`](../app/frontend/index.html):

- `BASE = 'http://localhost:8000'` — matches backend port. The frontend uses `localhost`; backend was bound to `127.0.0.1`. Both resolve identically in practice.
- All four `fetch` calls hit valid endpoints: `/analyze`, `/winrate`, `/recommend`, `/simulate`.
- `UNITS = ['Infantry','Mech','Artillery','Tank','Fighter','TacBmb','StrBmb','AA']` — matches backend `ALL_UNIT_TYPES`.
- `accept="image/*"` plus an HEIC hint in placeholder text. Backend uses Pillow `Image.open`; HEIC will fail unless `pillow-heif` is also installed — minor UI/backend mismatch.
- No browser was launched in this dry run.

---

## 4. Consolidated issue list

Grouped by category. Severity is from the perspective of "can a fresh dev follow the README and get a running app".

### 4.1 README accuracy (severity: blocker)

1. **Wrong run command.** README says `python test.py`. There is no `test.py`. Real entry: `uvicorn main:app`.
2. **Stale CLI description.** README says "Follow the prompts to enter enemy units and IPC budget." There are no prompts; the system is HTTP/UI-driven.
3. **Wrong required-files list.** Mentions `blr_weights.npz` (unused). Omits `vit_classifier.pth` (required). `winrate_model.pt` is correct.
4. **No mention of frontend.** Nothing in the README explains how to use [`app/frontend/index.html`](../app/frontend/index.html), what URL the backend must be reachable on, or that there is a 3-mode UI flow (the actual user-facing path).
5. **`python3.11` shorthand only works on Unix-like systems.** On Windows the launcher convention is `py -3.11`. Even then, this machine has no 3.11 installed; `py -3.12` works.

### 4.2 Repository hygiene (severity: high)

1. **The entire `app/backend/venv/` is committed to git** (5000+ tracked files, including `bin/python3.11`, all `site-packages` contents). `.gitignore` does list `app/backend/venv/`, but the directory was committed before the ignore was added, so git keeps tracking it. This bloats clones, makes diffs unreadable, and the committed venv is from a macOS box (Unix `bin/` layout) so it's useless on Windows anyway.
2. **Orphan compiled Python files in git**: `app/backend/__pycache__/blr_model.cpython-311.pyc` and `blr_model.cpython-314.pyc` are tracked, but `blr_model.py` does not exist in the repo. Leftover from removed code (likely paired with the obsolete `blr_weights.npz` reference).
3. **Tracked test outputs**: `app/backend/result.jpg`, `result_compare.jpg`, `result_crops.jpg`, `result_detection.jpg` are checked-in artifacts produced by `test_pipeline.py`. Should likely be gitignored and regenerated.

### 4.3 Dependencies (severity: blocker)

1. **`transformers` is missing from `requirements.txt`.** Backend cannot import without it.
2. **`python-multipart` is missing from `requirements.txt`.** Backend cannot start (FastAPI raises during route registration for any `UploadFile` endpoint).
3. **`matplotlib` is missing from `requirements.txt`.** `test_pipeline.py` cannot import.
4. **`scikit-learn` is listed but unused** by any source file.
5. **All versions are unpinned.** `pip install -r requirements.txt` today picks `fastapi 0.136.1`, `pydantic 2.13.x`, `transformers 5.6.x`, `torch 2.11.x`, `numpy 2.x`. Future installs will not be reproducible. Note transformers v5 is a major version bump that has deprecated/removed classes elsewhere in its API; pinning is especially important here.
6. **CPU-only torch by default.** Default PyPI `torch` wheel on Windows is `+cpu`. The README says nothing about getting GPU support; users with NVIDIA GPUs get a silent ~10–100x slowdown. Needs documented CUDA-tagged install line (e.g. `--index-url https://download.pytorch.org/whl/cu124`).
7. **`pybanner` is a `git+https` install** — only used by `test_pipeline.py`, but every install of `requirements.txt` requires `git` on PATH. Could be moved to a `requirements-dev.txt` or replaced with a tiny inline banner.

### 4.4 Runtime / code bugs (severity: medium–high)

1. **`/simulate` defender Mech survivors are wrong** (confirmed empirically above). Defender mechs are never decremented from the casualty loop and never restored after combat, so the response always reports the original mech count regardless of who won. The same class of bug exists for `D['aa']`, partially masked by an explicit zero on full defender loss.
2. **`/recommend` units payload omits `AA`** while `/winrate` and `/simulate` always include it. Contract inconsistency.
3. **MLP behavior worth investigating** (not necessarily a code bug):
   - All-zero units → 58% attacker win rate (model bias with no signal).
   - 2 Inf + 1 Art (JP) vs 3 Inf + 1 Art (US) → only 9.76% attacker win — feels very pessimistic.
   - At small budgets, `find_best_attack`'s 10,000-sample random search frequently returns the user's own composition; effectively a no-op recommendation.
4. **Device mismatch between [`main.py`](../app/backend/main.py) and [`pipeline.py`](../app/backend/pipeline.py)** — `main.py` picks `mps`/`cuda`/`cpu`, while `pipeline.py` picks only `mps`/`cpu`. On a CUDA-only host, `inputs.to(device='cpu')` on line 99 of `pipeline.py` would clash with `owl_model` on `cuda`. Did NOT trigger in this run because torch installed as CPU-only — the bug is dormant here but real.
5. **`weights_only=False` on `torch.load`** will warn on torch 2.4+ and may break on a future major. The checkpoints contain non-tensor objects (`class_names`, `mu`, `std`), so this is a checkpoint-format issue, not just a flag.
6. **`@app.on_event('startup')`** is deprecated in modern FastAPI in favor of lifespan handlers. Works for now, will warn/break later.
7. **Module-level `rng_sim = np.random.default_rng(42)` in `combat()`** is shared across requests and not thread-safe.
8. **Pydantic schema is too loose**: `UnitsRequest.JP/US: dict` with no value typing. No validation of unit names or non-negative integers.

### 4.5 Platform / portability (severity: medium)

1. **Filename case mismatch**: [`app/backend/test_pipeline.py`](../app/backend/test_pipeline.py) line 16 has `IMAGE_PATH = "test/t5.jpg"` but the actual file is `test/t5.JPG`. Resolves on Windows (case-insensitive FS) but breaks on Linux/macOS.
2. **HuggingFace cache symlink warning on Windows** without Developer Mode or admin: model files are duplicated rather than symlinked, costing extra disk for `owlvit-large` (~hundreds of MB extra).
3. **HEIC support hinted in UI but not in backend**: `index.html` placeholder text says "jpg / png / heic" but Pillow does not decode HEIC without `pillow-heif`.

### 4.6 What worked

- README's `venv\Scripts\activate` line on Windows + PowerShell 5.1 (resolved to `activate.bat`).
- After installing `transformers` and `python-multipart`, the backend imported and started cleanly.
- All five HTTP endpoints returned 200 OK with sensible (or in `/simulate`'s case, partially buggy) payloads.
- Real-image `/analyze` returned plausible counts in ~5.8 s on CPU.
- Frontend wiring (`BASE`, endpoints, `UNITS`) is internally consistent with the backend.
- Model weights, assets, and 13 sample test images are all checked in and usable.

---

## 5. Recommended fix scope (proposed for a follow-up branch)

These are *not* part of this report's work; they are the next-step recommendations to share with the dev along with the findings above.

1. **README rewrite** to describe the actual architecture (FastAPI backend + static HTML frontend), correct entry point (`uvicorn main:app`), accurate prerequisites (`vit_classifier.pth` + `winrate_model.pt`), Windows + macOS + Linux setup steps including a CUDA-tagged torch install line, and how to open the frontend.
2. **`requirements.txt` cleanup**: add `transformers`, `python-multipart`, `matplotlib` (or split dev deps), drop `scikit-learn`, pin versions, and either move `pybanner` to dev-only or remove it.
3. **Untrack the venv**: `git rm -r --cached app/backend/venv` plus a follow-up commit; same for the orphan `__pycache__/blr_model*` and the `result*.jpg` outputs.
4. **Fix `combat()` defender-Mech (and AA) accounting** in [`app/backend/main.py`](../app/backend/main.py): mirror the attacker's `restore_mech` behavior on the defender side, or rework the combat representation to track mech and infantry separately.
5. **Make `/recommend` units payload always include `AA`: 0**, for shape consistency with `/winrate` and `/simulate`.
6. **Unify device selection** — have [`pipeline.py`](../app/backend/pipeline.py) reuse the device chosen in [`main.py`](../app/backend/main.py) instead of re-deriving it as MPS-or-CPU.
7. **Investigate the MLP**: bias toward defender, near-zero win rates for balanced forces, and the 58% all-zero answer. May be a training data issue, label inversion, or normalization mismatch with the data the model was trained on.
8. **Tighten Pydantic schemas** (`UnitsRequest.JP: dict[str, int]` plus value validation) and migrate `@app.on_event('startup')` to a lifespan handler.
9. **Fix the `t5.jpg` vs `t5.JPG` case mismatch** in [`test_pipeline.py`](../app/backend/test_pipeline.py).
10. **De-duplicate** the detection/classification logic in [`test_pipeline.py`](../app/backend/test_pipeline.py) by importing from [`pipeline.py`](../app/backend/pipeline.py) rather than re-implementing it.

---

## Appendix A — Exact commands run

```powershell
# Phase 1 - strict
python --version
py -0
nvidia-smi
git rev-parse HEAD
cd app/backend
python3.11 -m venv venv          # FAIL
python test.py                   # FAIL

# Phase 2 - corrections
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import main"          # FAIL: ModuleNotFoundError transformers
python -m pip install transformers
python -c "import main"          # FAIL: RuntimeError needs python-multipart
python -m pip install python-multipart
python -c "import main"          # OK
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Probes
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-RestMethod  http://127.0.0.1:8000/winrate    -Method Post -Body $body -ContentType application/json
Invoke-RestMethod  http://127.0.0.1:8000/simulate   -Method Post -Body $body -ContentType application/json
Invoke-RestMethod  http://127.0.0.1:8000/recommend  -Method Post -Body $body -ContentType application/json
curl.exe -X POST -F "image=@test/t5.JPG" http://127.0.0.1:8000/analyze
```

## Appendix B — Files modified by this run

None of the source files were modified. The following side effects occurred outside source code:

- Created `app/backend/venv/` (a fresh Windows venv; coexists with the macOS venv that is in the git index).
- Populated `~/.cache/huggingface/hub/models--google--owlvit-large-patch14/`.
- `app/backend/venv/pyvenv.cfg` shows as `M` in `git status` because of the local pip upgrade — already-tracked file got rewritten. This is a symptom of issue 4.2.1 above, not a fix to apply.
- Created this report at `docs/readme-run-report.md`.
