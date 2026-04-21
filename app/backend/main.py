# main.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from collections import Counter
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


A_KEYS = ["ai","am","aa","at","af","atb","asb"]
D_KEYS = ["di","dm","da","dt","df","dtb","dsb","daa"]
UNIT_COST = {
    "ai":3,"am":4,"aa":4,"at":6,
    "af":10,"atb":11,"asb":12,
}
VIT_CLASSES = [
    "JP_Inf","JP_Mech","JP_Art","JP_Tank",
    "JP_Ftr","JP_TacBmb","JP_StrBmb","JP_AA",
    "US_Inf","US_Mech","US_Art","US_Tank",
    "US_Ftr","US_TacBmb","US_StrBmb","US_AA",
]

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ════════════════════════════════════════════════════════
# 模型定义
# ════════════════════════════════════════════════════════
def build_mlp():
    return nn.Sequential(
        nn.Linear(15, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ════════════════════════════════════════════════════════
# 启动时加载模型
# ════════════════════════════════════════════════════════
@app.on_event("startup")
def load_all_models():
    global vit_model, vit_classes, mlp_model, mu, std

    # ── ViT ──────────────────────────────────────────
    vit_ckpt    = torch.load("vit_classifier.pth",
                              map_location=DEVICE, weights_only=False)
    vit_classes = vit_ckpt['class_names']
    vit_model   = timm.create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=len(vit_classes)
    ).to(DEVICE)
    vit_model.load_state_dict(vit_ckpt['model_state'])
    vit_model.eval()
    print(f"[ViT] Loaded, classes: {vit_classes}")

    # ── MLP ──────────────────────────────────────────
    mlp_ckpt  = torch.load("winrate_model.pt",
                            map_location=DEVICE, weights_only=False)
    mlp_model = build_mlp().to(DEVICE)
    mlp_model.load_state_dict(mlp_ckpt["model_state"])
    mlp_model.eval()
    mu  = mlp_ckpt["mu"]
    std = mlp_ckpt["std"]
    print(f"[MLP] Loaded, epoch={mlp_ckpt.get('epoch','?')}")
    print("所有模型加载完成")


# ════════════════════════════════════════════════════════
# 颜色分割：裁剪棋子区域
# ════════════════════════════════════════════════════════
def extract_pieces(img_bytes: bytes, min_area=2000, padding=20):
    nparr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_jp = cv2.inRange(hsv,
                          np.array([5,  100, 100]),
                          np.array([20, 255, 255]))
    mask_us = cv2.inRange(hsv,
                          np.array([25, 40, 40]),
                          np.array([45, 255, 255]))

    mask   = cv2.bitwise_or(mask_jp, mask_us)
    kernel = np.ones((5,5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crops = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_rgb.shape[1], x + w + padding)
        y2 = min(img_rgb.shape[0], y + h + padding)
        crops.append(Image.fromarray(img_rgb[y1:y2, x1:x2]))

    return crops


# ════════════════════════════════════════════════════════
# MLP 推理
# ════════════════════════════════════════════════════════
def mlp_predict(vec15: list) -> float:
    x      = np.array([vec15], dtype=np.float32)
    x_norm = (x - mu) / (std + 1e-8)
    xt     = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logit = mlp_model(xt).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


# ════════════════════════════════════════════════════════
# 搜索最佳进攻配置
# 随机采样 n_samples 个合法配置，用 MLP 评估，返回最优
# ════════════════════════════════════════════════════════
def find_best_attack(defender: dict, budget: int,
                     n_samples: int = 10000,
                     seed: int = 42) -> dict:
    rng   = np.random.default_rng(seed)
    costs = np.array([UNIT_COST[u] for u in A_KEYS], dtype=np.int32)
    d_vec = [defender.get(k, 0) for k in D_KEYS]

    best_wr  = -1.0
    best_atk = None

    for _ in range(n_samples):
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

        vec15 = list(a) + d_vec
        wr    = mlp_predict(vec15)

        if wr > best_wr:
            best_wr  = wr
            best_atk = dict(zip(A_KEYS, a.tolist()))

    return {"attacker": best_atk, "win_rate": best_wr}


# ════════════════════════════════════════════════════════
# 接口1：图片识别
# POST /analyze
# ════════════════════════════════════════════════════════
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    img_bytes = await image.read()

    crops = extract_pieces(img_bytes)
    if len(crops) == 0:
        return {
            "units":   {c: 0 for c in VIT_CLASSES},
            "warning": "未检测到棋子，请检查图片或调整光线"
        }

    predictions = []
    for crop in crops:
        tensor = val_tf(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = vit_model(tensor).argmax(1).item()
        predictions.append(vit_classes[pred])

    counts = Counter(predictions)
    units  = {c: counts.get(c, 0) for c in VIT_CLASSES}
    return {"units": units}


# ════════════════════════════════════════════════════════
# 接口2：胜率预测 + 最佳兵力推荐
# POST /recommend
# ════════════════════════════════════════════════════════
class RecommendRequest(BaseModel):
    attacker: dict
    defender: dict
    budget:   int


@app.post("/recommend")
def recommend(req: RecommendRequest):
    # 当前配置的胜率
    vec15 = (
        [req.attacker.get(k, 0) for k in A_KEYS] +
        [req.defender.get(k, 0) for k in D_KEYS]
    )
    current_wr = mlp_predict(vec15)

    # 搜索最佳进攻配置
    result = find_best_attack(
        defender  = req.defender,
        budget    = req.budget,
        n_samples = 10000,
    )

    return {
        "current_win_rate": round(current_wr, 4),
        "best_attacker":    result["attacker"],
        "best_win_rate":    round(result["win_rate"], 4),
    }


# ════════════════════════════════════════════════════════
# 调试接口
# GET /debug
# ════════════════════════════════════════════════════════
@app.get("/debug")
def debug():
    test_cases = [
        ("1i vs 1i",          [1,0,0,0,0,0,0, 1,0,0,0,0,0,0,0]),
        ("3i+1a+1t vs 3i+1a", [3,0,1,1,0,0,0, 3,0,1,0,0,0,0,0]),
        ("5i+2a vs 4i+1a+1t", [5,0,2,0,0,0,0, 4,0,1,1,0,0,0,0]),
        ("2t+1f vs 3i+2a",    [0,0,0,2,1,0,0, 3,0,2,0,0,0,0,0]),
    ]
    results = []
    for name, vec in test_cases:
        mlp = mlp_predict(vec)
        results.append({
            "case": name,
            "mlp":  round(mlp, 4),
        })
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok"}
