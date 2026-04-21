# pipeline.py
import torch
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
from collections import Counter
import torch.nn as nn
import cv2

# ── 常量 ──────────────────────────────────────────────
A_KEYS    = ["ai","am","aa","at","af","atb","asb"]
D_KEYS    = ["di","dm","da","dt","df","dtb","dsb","daa"]
UNIT_COST = {"ai":3,"am":4,"aa":4,"at":6,"af":10,"atb":11,"asb":12}
DEFENDER_COST = {
    "di":3,"dm":4,"da":4,"dt":6,
    "df":10,"dtb":11,"dsb":12,"daa":5
}

ATTACKER_FACTION = "JP"
DEFENDER_FACTION = "US"
BUDGET_OFFSETS   = [-5, 0, 5]

UNIT_TYPE_MAP = {
    "Infantry":  {"atk": "ai",  "def": "di"},
    "Mech":      {"atk": "am",  "def": "dm"},
    "Artillery": {"atk": "aa",  "def": "da"},
    "Tank":      {"atk": "at",  "def": "dt"},
    "Fighter":   {"atk": "af",  "def": "df"},
    "TacBomber": {"atk": "atb", "def": "dtb"},
    "StrBomber": {"atk": "asb", "def": "dsb"},
    "AA":        {"atk": None,  "def": "daa"},
}

UNIT_FULL_NAME = {
    "Infantry":"Infantry", "Mech":"Mech Infantry",
    "Artillery":"Artillery", "Tank":"Tank",
    "Fighter":"Fighter", "TacBomber":"Tac Bomber",
    "StrBomber":"Str Bomber", "AA":"Anti-Air",
}

OWL_TEXTS = [[
    "plastic military figurine toy",
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
# 颜色判断阵营
# ════════════════════════════════════════════════════════
def get_faction_by_color(image: Image.Image, box) -> str:
    x1, y1, x2, y2 = [int(b.item()) for b in box]
    crop    = np.array(image.crop((x1, y1, x2, y2)))
    img_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask_jp = cv2.inRange(img_hsv, np.array([5,100,100]), np.array([20,255,255]))
    mask_us = cv2.inRange(img_hsv, np.array([25,40,40]),  np.array([45,255,255]))
    jp = int(mask_jp.sum())
    us = int(mask_us.sum())
    if jp == 0 and us == 0:
        return "unknown"
    return "JP" if jp >= us else "US"


# ════════════════════════════════════════════════════════
# 过滤包含其他框的大框
# ════════════════════════════════════════════════════════
def filter_containing_boxes(boxes, scores, labels, overlap_threshold=0.8):
    if len(boxes) == 0:
        return boxes, scores, labels

    keep = list(range(len(boxes)))

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or i not in keep or j not in keep:
                continue
            xi1 = max(boxes[i][0], boxes[j][0])
            yi1 = max(boxes[i][1], boxes[j][1])
            xi2 = min(boxes[i][2], boxes[j][2])
            yi2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            if area_j > 0 and inter / area_j > overlap_threshold:
                if i in keep:
                    keep.remove(i)

    keep = torch.tensor(keep)
    return boxes[keep], scores[keep], labels[keep]


# ════════════════════════════════════════════════════════
# Step 1: OWL-ViT 检测
# ════════════════════════════════════════════════════════
def detect_pieces(image: Image.Image, owl_processor, owl_model,
                  threshold=0.05):
    inputs = owl_processor(
        text=OWL_TEXTS, images=image, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results      = owl_processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    keep   = nms(boxes, scores, iou_threshold=0.2)
    boxes  = boxes[keep]; scores = scores[keep]; labels = labels[keep]

    img_w, img_h = image.size
    img_area     = img_w * img_h
    valid = []
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box.tolist()
        box_w = x2-x1; box_h = y2-y1
        box_ratio    = box_w*box_h / img_area
        aspect_ratio = max(box_w,box_h) / (min(box_w,box_h) + 1e-6)
        if 0.001 < box_ratio < 0.4 and aspect_ratio < 4.0:
            valid.append(i)

    if not valid:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    valid  = torch.tensor(valid)
    boxes  = boxes[valid]; scores = scores[valid]; labels = labels[valid]
    boxes, scores, labels = filter_containing_boxes(boxes, scores, labels)

    return boxes, scores, labels


# ════════════════════════════════════════════════════════
# Step 2: ViT 分类兵种
# ════════════════════════════════════════════════════════
def classify_pieces(image: Image.Image, boxes, vit_model, class_names):
    predictions = []
    for box in boxes:
        x1,y1,x2,y2 = [int(b.item()) for b in box]
        crop   = image.crop((x1,y1,x2,y2))
        tensor = val_tf(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = vit_model(tensor).argmax(1).item()
        predictions.append(class_names[pred])
    return predictions


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
# Step 4: MLP 预测胜率
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
#
# 策略：混合采样
#   - 一半样本：逐个买（每次1个），充分探索均衡配置
#   - 一半样本：批量买（旧逻辑），探索集中配置
# 两种策略各有优势，合并取最优
# ════════════════════════════════════════════════════════
def calc_ipc(units: dict, cost_table: dict) -> int:
    return sum(units.get(k,0) * v for k,v in cost_table.items())


def _eval_vec(a, d_vec, mlp_model, mu, std):
    """直接用 vec15 评估，和旧 main.py 的 mlp_predict 完全一致"""
    vec15  = list(a) + d_vec
    x      = np.array([vec15], dtype=np.float32)
    x_norm = (x - mu) / (std + 1e-8)
    xt     = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = mlp_model(xt).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def find_best_attack(defender, budget, mlp_model, mu, std,
                     n_samples=20000, seed=42):
    if budget <= 0:
        return {k: 0 for k in A_KEYS}, 0.0

    rng    = np.random.default_rng(seed)
    costs  = np.array([UNIT_COST[u] for u in A_KEYS], dtype=np.int32)
    d_vec  = [defender.get(k, 0) for k in D_KEYS]

    best_wr  = -1.0
    best_atk = None

    half = n_samples // 2

    # ── 策略1：逐个买（每次买1个），充分探索均衡配置 ──────
    # 这是关键：可以生成 2art+3inf 这类不花满预算的优质组合
    for _ in range(half):
        a         = np.zeros(7, dtype=np.int32)
        remaining = budget

        while remaining >= costs.min():
            affordable = np.where(costs <= remaining)[0]
            j          = int(rng.choice(affordable))
            a[j]      += 1
            remaining  -= int(costs[j])

        if a.sum() == 0:
            continue

        wr = _eval_vec(a, d_vec, mlp_model, mu, std)
        if wr > best_wr:
            best_wr  = wr
            best_atk = dict(zip(A_KEYS, a.tolist()))

    # ── 策略2：批量买（旧逻辑），探索集中配置 ────────────
    # 旧 main.py 的原始逻辑，可以生成纯步兵、重坦等极端配置
    for _ in range(n_samples - half):
        a         = np.zeros(7, dtype=np.int32)
        remaining = budget

        for _ in range(64):
            j = int(rng.integers(0, 7))
            c = int(costs[j])
            if c <= remaining:
                k      = int(rng.integers(1, remaining // c + 1))
                a[j]  += k
                remaining -= k * c
            if remaining < costs.min():
                break

        if a.sum() == 0:
            continue

        wr = _eval_vec(a, d_vec, mlp_model, mu, std)
        if wr > best_wr:
            best_wr  = wr
            best_atk = dict(zip(A_KEYS, a.tolist()))

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

    print("Step 1: 检测棋子位置...")
    boxes, scores, labels = detect_pieces(image, owl_processor, owl_model)
    print(f"  检测到 {len(boxes)} 个棋子")
    if len(boxes) == 0:
        print("未检测到棋子")
        return None

    factions = [get_faction_by_color(image, box) for box in boxes]
    print(f"  颜色判断 → JP: {sum(1 for f in factions if f=='JP')}  "
          f"US: {sum(1 for f in factions if f=='US')}")

    print("Step 2: ViT 分类棋子...")
    predictions = classify_pieces(image, boxes, vit_model, class_names)
    print(f"  分类结果: {Counter(predictions)}")

    attacker, defender = count_units(predictions, factions)
    print(f"  进攻方 (JP): {attacker}")
    print(f"  防守方 (US): {defender}")

    defender_ipc = calc_ipc(defender, DEFENDER_COST)
    print(f"  防守方总 IPC: {defender_ipc}")

    current_wr = mlp_predict(attacker, defender, mlp_model, mu, std)
    print(f"\n当前胜率: {current_wr*100:.1f}%")

    print("\n搜索最佳进攻配置...")
    recommendations = []
    for offset in budget_offsets:
        budget = max(3, defender_ipc + offset)
        label  = (f"与敌方相同 ({budget} IPC)" if offset == 0
                  else f"{'高于' if offset>0 else '低于'}敌方 "
                       f"{abs(offset)} IPC ({budget} IPC)")
        best_atk, best_wr = find_best_attack(
            defender, budget, mlp_model, mu, std
        )
        recommendations.append({
            "label": label, "budget": budget,
            "offset": offset, "attacker": best_atk, "win_rate": best_wr,
        })
        print(f"  {label}")
        print(f"    配置: { {k:v for k,v in best_atk.items() if v>0} }")
        print(f"    胜率: {best_wr*100:.1f}%")

    return {
        "attacker_faction": ATTACKER_FACTION,
        "defender_faction": DEFENDER_FACTION,
        "attacker":         attacker,
        "defender":         defender,
        "defender_ipc":     defender_ipc,
        "current_win_rate": current_wr,
        "recommendations":  recommendations,
        "factions":         factions,
        "predictions":      predictions,
    }