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
from typing import Optional
import io

from pipeline import (
    detect_pieces, classify_pieces, count_units,
    get_faction_by_color, mlp_predict, find_best_attack,
    calc_ipc, UNIT_TYPE_MAP, DEFENDER_COST, A_KEYS, D_KEYS
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 单位名映射 ────────────────────────────────────────
# pipeline ViT 输出 8 类 → MLP 字段
ATK_KEY = {"Infantry":"ai","Mech":"am","Artillery":"aa","Tank":"at",
           "Fighter":"af","TacBomber":"atb","StrBomber":"asb","AA":None}
DEF_KEY = {"Infantry":"di","Mech":"dm","Artillery":"da","Tank":"dt",
           "Fighter":"df","TacBomber":"dtb","StrBomber":"dsb","AA":"daa"}

ALL_UNIT_TYPES = ["Infantry","Mech","Artillery","Tank",
                  "Fighter","TacBomber","StrBomber","AA"]


# ════════════════════════════════════════════════════════
# 仿真器（单局）
# ════════════════════════════════════════════════════════
rng_sim = np.random.default_rng(42)

def combat(ai,am,aa,at,af,atb,asb, di,dm,da,dt,df,dtb,dsb,daa):
    ai += am
    di += dm

    A = dict(i=ai,m=am,a=aa,t=at,f=af,tb=atb,sb=asb)
    D = dict(i=di,m=dm,a=da,t=dt,f=df,tb=dtb,sb=dsb,aa=daa)

    aa_fired = False

    while True:
        # Anti-air
        A_air = A["f"] + A["tb"] + A["sb"]
        if (not aa_fired) and D["aa"] > 0 and A_air > 0:
            shots   = min(3 * D["aa"], A_air)
            aa_hits = rng_sim.binomial(shots, 1/6)
            k = aa_hits
            kill = min(A["f"],  k); A["f"]  -= kill; k -= kill
            kill = min(A["tb"], k); A["tb"] -= kill; k -= kill
            kill = min(A["sb"], k); A["sb"] -= kill
            aa_fired = True

        # Attacker hits
        sup   = min(A["i"], A["a"])
        unsup = A["i"] - sup
        boosted_tb = min(A["tb"], A["f"] + A["t"])
        normal_tb  = A["tb"] - boosted_tb

        a_hits = (rng_sim.binomial(sup,        1/3) +
                  rng_sim.binomial(unsup,       1/6) +
                  rng_sim.binomial(A["a"],      1/3) +
                  rng_sim.binomial(A["t"],      1/2) +
                  rng_sim.binomial(A["f"],      1/2) +
                  rng_sim.binomial(boosted_tb,  2/3) +
                  rng_sim.binomial(normal_tb,   1/2) +
                  rng_sim.binomial(A["sb"],     2/3))

        # Defender hits
        d_hits = (rng_sim.binomial(D["i"],  1/3) +
                  rng_sim.binomial(D["a"],  1/3) +
                  rng_sim.binomial(D["t"],  1/2) +
                  rng_sim.binomial(D["f"],  2/3) +
                  rng_sim.binomial(D["tb"], 1/2) +
                  rng_sim.binomial(D["sb"], 1/6))

        # Attacker casualties
        rem = int(d_hits)
        for key in ["i","a","t","f","tb","sb"]:
            kill = min(A[key], rem); A[key] -= kill; rem -= kill

        # Defender casualties
        rem = int(a_hits)
        for key in ["i","a","t","f","tb","sb"]:
            kill = min(D[key], rem); D[key] -= kill; rem -= kill

        A_alive = sum(A[k] for k in ["i","a","t","f","tb","sb"]) > 0
        D_alive = sum(D[k] for k in ["i","a","t","f","tb","sb"]) > 0

        def restore_mech(A, am_orig):
            if A["i"] >= am_orig:
                A["i"] -= am_orig; A["m"] = am_orig
            elif A["i"] > 0:
                A["m"] = A["i"]; A["i"] = 0

        if not A_alive and not D_alive:
            restore_mech(A, am); return "D", A, D
        if not A_alive:
            restore_mech(A, am); return "D", A, D
        if not D_alive:
            D["aa"] = 0
            restore_mech(A, am); return "A", A, D


def units_dict_to_display(u: dict) -> dict:
    """把内部字段转成前端显示格式"""
    return {
        "Infantry":  u.get("i", 0),
        "Mech":      u.get("m", 0),
        "Artillery": u.get("a", 0),
        "Tank":      u.get("t", 0),
        "Fighter":   u.get("f", 0),
        "TacBomber": u.get("tb", 0),
        "StrBomber": u.get("sb", 0),
        "AA":        u.get("aa", 0),
    }


# ════════════════════════════════════════════════════════
# 启动时加载模型
# ════════════════════════════════════════════════════════
@app.on_event("startup")
def load_all_models():
    global owl_processor, owl_model, vit_model, vit_class_names
    global mlp_model, mu, std

    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    owl_model     = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-large-patch14"
    ).to(DEVICE)
    owl_model.eval()

    vit_ckpt       = torch.load("vit_classifier.pth",
                                 map_location=DEVICE, weights_only=False)
    vit_class_names = vit_ckpt["class_names"]
    vit_model       = timm.create_model(
        "vit_small_patch16_224", pretrained=False,
        num_classes=len(vit_class_names)
    ).to(DEVICE)
    vit_model.load_state_dict(vit_ckpt["model_state"])
    vit_model.eval()

    mlp_ckpt  = torch.load("winrate_model.pt",
                            map_location=DEVICE, weights_only=False)
    mlp_model = nn.Sequential(
        nn.Linear(15,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128,64),  nn.ReLU(),
        nn.Linear(64,1),
    ).to(DEVICE)
    mlp_model.load_state_dict(mlp_ckpt["model_state"])
    mlp_model.eval()
    mu  = mlp_ckpt["mu"]
    std = mlp_ckpt["std"]

    print("所有模型加载完成")


# ════════════════════════════════════════════════════════
# 工具函数：前端兵种格式 → MLP 向量
# ════════════════════════════════════════════════════════
def frontend_to_attacker(units: dict) -> dict:
    return {ATK_KEY[k]: units.get(k, 0)
            for k in ALL_UNIT_TYPES if ATK_KEY[k] is not None}

def frontend_to_defender(units: dict) -> dict:
    return {DEF_KEY[k]: units.get(k, 0) for k in ALL_UNIT_TYPES}

def attacker_to_frontend(atk: dict) -> dict:
    rev = {v:k for k,v in ATK_KEY.items() if v}
    return {rev.get(k, k): v for k,v in atk.items()}


# ════════════════════════════════════════════════════════
# 接口1：图片识别
# POST /analyze
# ════════════════════════════════════════════════════════
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    boxes, scores, labels = detect_pieces(img, owl_processor, owl_model)
    if len(boxes) == 0:
        empty = {u: 0 for u in ALL_UNIT_TYPES}
        return {"JP": empty, "US": empty, "warning": "未检测到棋子"}

    factions    = [get_faction_by_color(img, box) for box in boxes]
    predictions = classify_pieces(img, boxes, vit_model, vit_class_names)

    jp = {u: 0 for u in ALL_UNIT_TYPES}
    us = {u: 0 for u in ALL_UNIT_TYPES}

    for pred, faction in zip(predictions, factions):
        if pred in ALL_UNIT_TYPES:
            if faction == "JP":
                jp[pred] += 1
            elif faction == "US":
                us[pred] += 1

    return {"JP": jp, "US": us}


# ════════════════════════════════════════════════════════
# 共用请求体
# ════════════════════════════════════════════════════════
class UnitsRequest(BaseModel):
    attacker:   str        # "JP" or "US"
    JP:         dict
    US:         dict
    ipc_offset: Optional[int] = 0


# ════════════════════════════════════════════════════════
# 接口2：预测胜率
# POST /winrate
# ════════════════════════════════════════════════════════
@app.post("/winrate")
def winrate(req: UnitsRequest):
    if req.attacker == "JP":
        atk = frontend_to_attacker(req.JP)
        dfn = frontend_to_defender(req.US)
    else:
        atk = frontend_to_attacker(req.US)
        dfn = frontend_to_defender(req.JP)

    wr = mlp_predict(atk, dfn, mlp_model, mu, std)
    return {"win_rate": round(wr, 4)}


# ════════════════════════════════════════════════════════
# 接口3：推理最佳组合
# POST /recommend
# ════════════════════════════════════════════════════════
@app.post("/recommend")
def recommend(req: UnitsRequest):
    if req.attacker == "JP":
        atk = frontend_to_attacker(req.JP)
        dfn = frontend_to_defender(req.US)
    else:
        atk = frontend_to_attacker(req.US)
        dfn = frontend_to_defender(req.JP)

    current_wr   = mlp_predict(atk, dfn, mlp_model, mu, std)
    defender_ipc = calc_ipc(dfn, DEFENDER_COST)
    base_budget  = defender_ipc + req.ipc_offset

    recommendations = []
    for delta in [-5, 0, 5]:
        budget = max(3, base_budget + delta)
        label  = (f"Same IPC ({budget})" if delta == 0
                  else f"{'Above' if delta > 0 else 'Below'} "
                       f"{abs(delta)} IPC ({budget})")
        best_atk, best_wr = find_best_attack(
            dfn, budget, mlp_model, mu, std
        )
        recommendations.append({
            "label":    label,
            "budget":   budget,
            "units":    attacker_to_frontend(best_atk),
            "win_rate": round(best_wr, 4),
        })

    return {
        "current_win_rate": round(current_wr, 4),
        "defender_ipc":     defender_ipc,
        "recommendations":  recommendations,
    }


# ════════════════════════════════════════════════════════
# 接口4：模拟对战
# POST /simulate
# ════════════════════════════════════════════════════════
@app.post("/simulate")
def simulate(req: UnitsRequest):
    if req.attacker == "JP":
        atk_units = req.JP
        def_units = req.US
    else:
        atk_units = req.US
        def_units = req.JP

    # 转成仿真器参数
    ai  = atk_units.get("Infantry",  0)
    am  = atk_units.get("Mech",      0)
    aa  = atk_units.get("Artillery", 0)
    at  = atk_units.get("Tank",      0)
    af  = atk_units.get("Fighter",   0)
    atb = atk_units.get("TacBomber", 0)
    asb = atk_units.get("StrBomber", 0)

    di  = def_units.get("Infantry",  0)
    dm  = def_units.get("Mech",      0)
    da  = def_units.get("Artillery", 0)
    dt  = def_units.get("Tank",      0)
    df  = def_units.get("Fighter",   0)
    dtb = def_units.get("TacBomber", 0)
    dsb = def_units.get("StrBomber", 0)
    daa = def_units.get("AA",        0)

    winner_code, A_survivors, D_survivors = combat(
        ai,am,aa,at,af,atb,asb,
        di,dm,da,dt,df,dtb,dsb,daa
    )

    winner = req.attacker if winner_code == "A" else (
        "US" if req.attacker == "JP" else "JP"
    )

    return {
        "winner":              winner,
        "attacker_survivors":  units_dict_to_display(A_survivors),
        "defender_survivors":  units_dict_to_display(D_survivors),
    }


@app.get("/health")
def health():
    return {"status": "ok"}