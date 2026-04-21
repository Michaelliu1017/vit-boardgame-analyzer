# pipeline.py
import torch, cv2, numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
from collections import Counter
import torch.nn as nn

# ════════════════════════════════════════════════════════
# 全局参数
# ════════════════════════════════════════════════════════
OWL_THRESHOLD    = 0.05
NMS_IOU          = 0.1
BOX_AREA_MIN     = 0.001
BOX_AREA_MAX     = 0.4
ASPECT_RATIO_MAX = 10.0
OVERLAP_THRESH   = 0.8
VIT_CONF_THRESH  = 0.7

JP_HSV_LOW  = (5,  100, 100)
JP_HSV_HIGH = (20, 255, 255)
US_HSV_LOW  = (25, 40,  40)
US_HSV_HIGH = (45, 255, 255)

A_KEYS        = ["ai","am","aa","at","af","atb","asb"]
D_KEYS        = ["di","dm","da","dt","df","dtb","dsb","daa"]
UNIT_COST     = {"ai":3,"am":4,"aa":4,"at":6,"af":10,"atb":11,"asb":12}
DEFENDER_COST = {"di":3,"dm":4,"da":4,"dt":6,"df":10,"dtb":11,"dsb":12,"daa":5}

ATTACKER_FACTION = "JP"
DEFENDER_FACTION = "US"
BUDGET_OFFSETS   = [-5, 0, 5]

UNIT_TYPE_MAP = {
    "Infantry":  {"atk": "ai",  "def": "di"},
    "Mech":      {"atk": "am",  "def": "dm"},
    "Artillery": {"atk": "aa",  "def": "da"},
    "Tank":      {"atk": "at",  "def": "dt"},
    "Fighter":   {"atk": "af",  "def": "df"},
    "TacBmb":    {"atk": "atb", "def": "dtb"},  #
    "StrBmb":    {"atk": "asb", "def": "dsb"},
    "AA":        {"atk": None,  "def": "daa"},
}

OWL_TEXTS = [[
    "plastic military figurine toy",
    "plastic vehicle toy",
    "plastic artillery toy",
    "plastic tank toy",
    "plastic airplane toy",
    "plastic warship toy",
    "plastic anti air gun toy",
]]

device = "mps" if torch.backends.mps.is_available() else "cpu"

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════
def get_faction_by_color(image: Image.Image, box) -> str:
    x1,y1,x2,y2 = [int(b.item()) for b in box]
    crop = np.array(image.crop((x1,y1,x2,y2)))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    jp   = int(cv2.inRange(hsv, np.array(JP_HSV_LOW),  np.array(JP_HSV_HIGH)).sum())
    us   = int(cv2.inRange(hsv, np.array(US_HSV_LOW),  np.array(US_HSV_HIGH)).sum())
    if jp == 0 and us == 0: return "unknown"
    return "JP" if jp >= us else "US"


def filter_containing_boxes(boxes, scores, labels):
    if len(boxes) == 0: return boxes, scores, labels
    keep = list(range(len(boxes)))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or i not in keep or j not in keep: continue
            inter = (max(0, min(boxes[i][2],boxes[j][2]) - max(boxes[i][0],boxes[j][0])) *
                     max(0, min(boxes[i][3],boxes[j][3]) - max(boxes[i][1],boxes[j][1])))
            area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            if area_j > 0 and inter / area_j > OVERLAP_THRESH:
                if i in keep: keep.remove(i)
    k = torch.tensor(keep)
    return boxes[k], scores[k], labels[k]


# ════════════════════════════════════════════════════════
# Step 1: OWL-ViT 检测
# ════════════════════════════════════════════════════════
def detect_pieces(image: Image.Image, owl_processor, owl_model):
    inputs = owl_processor(
        text=OWL_TEXTS, images=image, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)

    results = owl_processor.post_process_grounded_object_detection(
        outputs, threshold=OWL_THRESHOLD,
        target_sizes=torch.tensor([image.size[::-1]])
    )[0]

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # NMS
    keep   = nms(boxes, scores, iou_threshold=NMS_IOU)
    boxes  = boxes[keep]; scores = scores[keep]; labels = labels[keep]

    # 面积 + 长宽比过滤
    img_area = image.size[0] * image.size[1]
    valid = [i for i, box in enumerate(boxes)
             if (BOX_AREA_MIN < (box[2]-box[0])*(box[3]-box[1])/img_area < BOX_AREA_MAX
                 and max(box[2]-box[0], box[3]-box[1]) /
                     (min(box[2]-box[0], box[3]-box[1]) + 1e-6) < ASPECT_RATIO_MAX)]

    if not valid:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    v = torch.tensor(valid)
    boxes  = boxes[v]; scores = scores[v]; labels = labels[v]

    # 包含框过滤
    boxes, scores, labels = filter_containing_boxes(boxes, scores, labels)

    return boxes, scores, labels


# ════════════════════════════════════════════════════════
# Step 2: ViT 分类 + 置信度过滤
# ════════════════════════════════════════════════════════
def classify_pieces(image: Image.Image, boxes, vit_model, class_names):
    predictions = []
    valid_boxes = []

    for box in boxes:
        x1,y1,x2,y2 = [int(b.item()) for b in box]
        crop   = image.crop((x1,y1,x2,y2))
        tensor = val_tf(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(vit_model(tensor)[0], dim=0)
            pred  = probs.argmax().item()
            conf  = probs[pred].item()

        if conf < VIT_CONF_THRESH:
            continue

        predictions.append(class_names[pred])
        valid_boxes.append(box)

    return predictions, valid_boxes


# ════════════════════════════════════════════════════════
# Step 3: 统计兵力
# ════════════════════════════════════════════════════════
def count_units(predictions, factions):
    attacker = {v: 0 for v in A_KEYS}
    defender = {v: 0 for v in D_KEYS}
    for pred, faction in zip(predictions, factions):
        mapping = UNIT_TYPE_MAP.get(pred)
        if not mapping: continue
        if faction == "JP":
            key = mapping["atk"]
            if key: attacker[key] += 1
        elif faction == "US":
            key = mapping["def"]
            if key: defender[key] += 1
    return attacker, defender


# ════════════════════════════════════════════════════════
# Step 4: MLP 预측胜率
# ════════════════════════════════════════════════════════
def mlp_predict(attacker, defender, mlp_model, mu, std):
    vec15  = ([attacker.get(k,0) for k in A_KEYS] +
              [defender.get(k,0) for k in D_KEYS])
    x      = np.array([vec15], dtype=np.float32)
    x_norm = (x - mu) / (std + 1e-8)
    xt     = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = mlp_model(xt).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


# ════════════════════════════════════════════════════════
# Step 5: 搜索最佳进攻配置
# ════════════════════════════════════════════════════════
def calc_ipc(units: dict, cost_table: dict) -> int:
    return sum(units.get(k,0) * v for k,v in cost_table.items())


def _eval_vec(a, d_vec, mlp_model, mu, std):
    vec15  = list(a) + d_vec
    x      = np.array([vec15], dtype=np.float32)
    x_norm = (x - mu) / (std + 1e-8)
    xt     = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = mlp_model(xt).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def find_best_attack(defender, budget, mlp_model, mu, std,
                     n_samples=10000, seed=42):
    if budget <= 0:
        return {k: 0 for k in A_KEYS}, 0.0

    rng   = np.random.default_rng(seed)
    costs = np.array([UNIT_COST[u] for u in A_KEYS], dtype=np.int32)
    d_vec = [defender.get(k, 0) for k in D_KEYS]
    best_wr, best_atk = -1.0, None
    half = n_samples // 2

    for _ in range(half):
        a = np.zeros(7, dtype=np.int32)
        remaining = budget
        while remaining >= costs.min():
            affordable = np.where(costs <= remaining)[0]
            j = int(rng.choice(affordable))
            a[j] += 1; remaining -= int(costs[j])
        if a.sum() == 0: continue
        wr = _eval_vec(a, d_vec, mlp_model, mu, std)
        if wr > best_wr: best_wr = wr; best_atk = dict(zip(A_KEYS, a.tolist()))

    for _ in range(n_samples - half):
        a = np.zeros(7, dtype=np.int32)
        remaining = budget
        for _ in range(64):
            j = int(rng.integers(0, 7))
            c = int(costs[j])
            if c <= remaining:
                k = int(rng.integers(1, remaining // c + 1))
                a[j] += k; remaining -= k * c
            if remaining < costs.min(): break
        if a.sum() == 0: continue
        wr = _eval_vec(a, d_vec, mlp_model, mu, std)
        if wr > best_wr: best_wr = wr; best_atk = dict(zip(A_KEYS, a.tolist()))

    return best_atk, best_wr


# ════════════════════════════════════════════════════════
# 完整 Pipeline
# ════════════════════════════════════════════════════════
def run_pipeline(image_path, models, budget_offsets=BUDGET_OFFSETS):
    owl_processor = models["owl_processor"]
    owl_model     = models["owl_model"]
    vit_model     = models["vit_model"]
    class_names   = models["class_names"]
    mlp_model     = models["mlp_model"]
    mu, std       = models["mu"], models["std"]

    image = Image.open(image_path).convert('RGB')

    print("Step 1: OWL-ViT detection...")
    boxes, scores, labels = detect_pieces(image, owl_processor, owl_model)
    print(f"  Detected: {len(boxes)}")
    if len(boxes) == 0:
        print("No pieces detected"); return None

    print("Step 2: ViT classification...")
    predictions, valid_boxes = classify_pieces(image, boxes, vit_model, class_names)
    print(f"  After confidence filter: {len(valid_boxes)}")
    print(f"  Classes: {Counter(predictions)}")
    if len(valid_boxes) == 0:
        print("No valid pieces after confidence filter"); return None

    factions = [get_faction_by_color(image, box) for box in valid_boxes]
    print(f"  JP: {sum(1 for f in factions if f=='JP')}  "
          f"US: {sum(1 for f in factions if f=='US')}")

    attacker, defender = count_units(predictions, factions)
    defender_ipc = calc_ipc(defender, DEFENDER_COST)
    current_wr   = mlp_predict(attacker, defender, mlp_model, mu, std)

    print(f"\n  Attacker (JP): { {k:v for k,v in attacker.items() if v>0} }")
    print(f"  Defender (US): { {k:v for k,v in defender.items() if v>0} }")
    print(f"  Defender IPC:  {defender_ipc}")
    print(f"  Current win rate: {current_wr*100:.1f}%")

    print("\nStep 3: Searching best compositions...")
    recommendations = []
    for offset in budget_offsets:
        budget = max(3, defender_ipc + offset)
        label  = (f"Same IPC ({budget})" if offset == 0
                  else f"{'Above' if offset>0 else 'Below'} {abs(offset)} IPC ({budget})")
        best_atk, best_wr = find_best_attack(defender, budget, mlp_model, mu, std)
        recommendations.append({
            "label": label, "budget": budget,
            "offset": offset, "attacker": best_atk, "win_rate": best_wr,
        })
        print(f"  {label}: { {k:v for k,v in best_atk.items() if v>0} }  →  {best_wr*100:.1f}%")

    return {
        "attacker": attacker, "defender": defender,
        "defender_ipc": defender_ipc, "current_win_rate": current_wr,
        "recommendations": recommendations,
        "factions": factions, "predictions": predictions,
    }