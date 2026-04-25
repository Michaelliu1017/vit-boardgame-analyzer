"""Inference micro-benchmark for the FastAPI backend.

Times the current pipeline (OWL-ViT + ViT classifier + MLP search) against
proposed batched variants. Backs the numbers in
``docs/inference-optimization.md``. Run from ``app/backend/`` with the venv
active::

    set PYTHONIOENCODING=utf-8         # Windows / cp1252 console
    python bench_inference.py test/t5.JPG

This is an analysis tool — not a unit test, not part of the API surface.
Safe to delete if the doc is no longer maintained.
"""
import os, sys, time, statistics, contextlib
import numpy as np, torch, torch.nn as nn, timm
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (
    detect_pieces, classify_pieces, get_faction_by_color,
    OWL_TEXTS, val_tf, A_KEYS, D_KEYS, UNIT_COST, find_best_attack,
)

DEVICE = torch.device('cpu')
TEST_IMAGE = sys.argv[1] if len(sys.argv) > 1 else 'test/t5.JPG'
N_TIMING_RUNS = 3   # repeat hot timings this many times

@contextlib.contextmanager
def timed(label, store):
    t = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t
    store.setdefault(label, []).append(elapsed)
    print(f'  {label:42s}  {elapsed*1000:8.1f} ms')

def best(label, store): return min(store[label]) * 1000.0

def main():
    print(f'=== Benchmark on {TEST_IMAGE} (device=cpu) ===\n')
    times = {}

    print('[load] Loading models…')
    t0 = time.perf_counter()
    owl_proc  = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    owl_model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14').to(DEVICE).eval()
    print(f'  OWL-ViT loaded in {time.perf_counter()-t0:.1f}s')

    t0 = time.perf_counter()
    vit_ckpt = torch.load('vit_classifier.pth', map_location=DEVICE, weights_only=False)
    vit_class_names = vit_ckpt['class_names']
    vit_model = timm.create_model('vit_small_patch16_224', pretrained=False,
                                  num_classes=len(vit_class_names)).to(DEVICE).eval()
    vit_model.load_state_dict(vit_ckpt['model_state'])
    print(f'  ViT classifier loaded in {time.perf_counter()-t0:.1f}s')

    t0 = time.perf_counter()
    mlp_ckpt = torch.load('winrate_model.pt', map_location=DEVICE, weights_only=False)
    mlp_model = nn.Sequential(
        nn.Linear(15,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128,64),  nn.ReLU(), nn.Linear(64,1),
    ).to(DEVICE).eval()
    mlp_model.load_state_dict(mlp_ckpt['model_state'])
    mu, std = mlp_ckpt['mu'], mlp_ckpt['std']
    print(f'  MLP loaded in {time.perf_counter()-t0:.1f}s\n')

    image = Image.open(TEST_IMAGE).convert('RGB')
    print(f'[image] Size: {image.size}\n')

    # -----------------------------------------------------------------
    # 1. OWL-ViT detection — current code, repeated
    # -----------------------------------------------------------------
    print('[1] OWL-ViT detection (single image, repeated):')
    boxes = scores = labels = None
    for _ in range(N_TIMING_RUNS):
        with timed('owl_detect (current)', times):
            boxes, scores, labels = detect_pieces(image, owl_proc, owl_model)
    print(f'  -> {len(boxes)} boxes after filter\n')

    # -----------------------------------------------------------------
    # 2. ViT classification — current per-crop loop
    # -----------------------------------------------------------------
    print('[2] ViT classification (per-crop loop, current):')
    for _ in range(N_TIMING_RUNS):
        with timed('vit_classify_loop (current)', times):
            preds, valid = classify_pieces(image, boxes, vit_model, vit_class_names)
    n_crops = len(boxes)
    print(f'  -> {len(valid)} kept after conf filter, processed {n_crops} crops\n')

    # -----------------------------------------------------------------
    # 3. ViT classification — proposed BATCHED variant
    # -----------------------------------------------------------------
    print(f'[3] ViT classification (batched, proposed):  N_crops={n_crops}')
    def classify_batched(image, boxes, model, class_names):
        if len(boxes) == 0: return [], []
        crops = []
        for box in boxes:
            x1,y1,x2,y2 = [int(b.item()) for b in box]
            crops.append(val_tf(image.crop((x1,y1,x2,y2))))
        batch = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
        out_preds, out_boxes = [], []
        for i, (p, c, b) in enumerate(zip(preds.tolist(), confs.tolist(), boxes)):
            if c >= 0.7:
                out_preds.append(class_names[p])
                out_boxes.append(b)
        return out_preds, out_boxes
    for _ in range(N_TIMING_RUNS):
        with timed('vit_classify_batched (proposed)', times):
            preds_b, valid_b = classify_batched(image, boxes, vit_model, vit_class_names)
    print(f'  -> {len(valid_b)} kept (must match {len(valid)})\n')

    # -----------------------------------------------------------------
    # 4. Per-crop sub-cost: just the model forward, batch_size=1 vs N
    # -----------------------------------------------------------------
    print('[4] Pure model forward — batch=1 (called N times) vs batch=N:')
    crops = [val_tf(image.crop(tuple(int(b.item()) for b in box))) for box in boxes]
    batch = torch.stack(crops).to(DEVICE)
    if n_crops > 0:
        for _ in range(N_TIMING_RUNS):
            with timed(f'vit forward x{n_crops} (batch=1 each)', times):
                with torch.no_grad():
                    for c in batch:
                        _ = vit_model(c.unsqueeze(0))
        for _ in range(N_TIMING_RUNS):
            with timed(f'vit forward x1   (batch={n_crops})', times):
                with torch.no_grad():
                    _ = vit_model(batch)
    print()

    # -----------------------------------------------------------------
    # 5. MLP — single sample forward (current find_best_attack pattern)
    # -----------------------------------------------------------------
    print('[5] MLP — single sample (called per attempt in find_best_attack):')
    rng = np.random.default_rng(42)
    sample = np.array([rng.integers(0, 5, size=15)], dtype=np.float32)
    sample_norm = (sample - mu) / (std + 1e-8)
    sample_t = torch.tensor(sample_norm, dtype=torch.float32)
    for _ in range(N_TIMING_RUNS):
        with timed('mlp forward (batch=1)', times):
            with torch.no_grad():
                _ = mlp_model(sample_t).item()
    print()

    # -----------------------------------------------------------------
    # 6. MLP batched — proposed
    # -----------------------------------------------------------------
    print('[6] MLP — batched 10000 samples (proposed find_best_attack inner loop):')
    big = np.array(rng.integers(0, 5, size=(10000, 15)), dtype=np.float32)
    big_norm = (big - mu) / (std + 1e-8)
    big_t = torch.tensor(big_norm, dtype=torch.float32)
    for _ in range(N_TIMING_RUNS):
        with timed('mlp forward (batch=10000)', times):
            with torch.no_grad():
                _ = mlp_model(big_t).cpu().numpy()
    print()

    # -----------------------------------------------------------------
    # 7. find_best_attack as currently written
    # -----------------------------------------------------------------
    print('[7] find_best_attack(budget=20, n=10000) — current implementation:')
    defender = {k: 1 for k in D_KEYS}
    for _ in range(2):  # only 2 — this is slow
        with timed('find_best_attack (current)', times):
            _ = find_best_attack(defender, 20, mlp_model, mu, std, n_samples=10000)
    print()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print('============================================================')
    print('SUMMARY (best-of-N):')
    print('============================================================')
    for label in times:
        print(f'  {label:48s}  {best(label, times):10.1f} ms')

if __name__ == '__main__':
    main()
