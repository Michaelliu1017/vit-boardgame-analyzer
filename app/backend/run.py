# run_pipeline.py
import torch, torch.nn as nn, timm
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from pipeline import run_pipeline, BUDGET_OFFSETS

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading models...")

owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
owl_model     = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-large-patch14").to(device)
owl_model.eval()
print("OWL-ViT loaded")

vit_ckpt    = torch.load("vit_classifier.pth", map_location=device, weights_only=False)
vit_model   = timm.create_model('vit_small_patch16_224', pretrained=False,
                                 num_classes=len(vit_ckpt['class_names'])).to(device)
vit_model.load_state_dict(vit_ckpt['model_state'])
vit_model.eval()
class_names = vit_ckpt['class_names']
print(f"ViT loaded: {class_names}")

mlp_ckpt  = torch.load("winrate_model.pt", map_location=device, weights_only=False)
mlp_model = nn.Sequential(
    nn.Linear(15,256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
    nn.Linear(128,64),  nn.ReLU(), nn.Linear(64,1),
).to(device)
mlp_model.load_state_dict(mlp_ckpt["model_state"])
mlp_model.eval()
mu  = mlp_ckpt["mu"]
std = mlp_ckpt["std"]
print("MLP loaded\n")

models = {
    "owl_processor": owl_processor, "owl_model": owl_model,
    "vit_model":     vit_model,     "class_names": class_names,
    "mlp_model":     mlp_model,     "mu": mu, "std": std,
}

result = run_pipeline(
    image_path     = "test/test1.jpg",
    models         = models,
    budget_offsets = BUDGET_OFFSETS,
)

if result:
    from collections import Counter
    predictions = result["predictions"]
    factions    = result["factions"]
    jp_units = [p for p,f in zip(predictions,factions) if f=="JP"]
    us_units = [p for p,f in zip(predictions,factions) if f=="US"]

    print("\n══════════════════════════════════════════")
    print(f"  Japan  (Attacker) — {len(jp_units)} units")
    for cls,cnt in Counter(jp_units).most_common():
        print(f"    {cls:<16} × {cnt}")
    print(f"\n  US (Defender) — {len(us_units)} units")
    for cls,cnt in Counter(us_units).most_common():
        print(f"    {cls:<16} × {cnt}")
    print(f"\n  Defender IPC:     {result['defender_ipc']}")
    print(f"  Current Win Rate: {result['current_win_rate']*100:.1f}%")
    print("\n── Recommendations ──")
    for r in result['recommendations']:
        units = {k:v for k,v in r['attacker'].items() if v>0}
        print(f"  {r['label']}: {units}  →  {r['win_rate']*100:.1f}%")