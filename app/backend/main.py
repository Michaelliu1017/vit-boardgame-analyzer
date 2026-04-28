# main.py
import io
from pathlib import Path
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import (
    detect_pieces, classify_pieces, count_units,
    get_faction_by_color, mlp_predict, find_best_attack,
    calc_ipc, A_KEYS, D_KEYS, DEFENDER_COST
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent.parent / "frontend"

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(_FRONTEND / "index.html")

DEVICE = torch.device(
    'mps' if torch.backends.mps.is_available()
    else ('cuda' if torch.cuda.is_available() else 'cpu')
)

ALL_UNIT_TYPES = ['Infantry','Mech','Artillery','Tank',
                  'Fighter','TacBmb','StrBmb','AA']

ATK_KEY = {
    'Infantry':'ai', 'Mech':'am', 'Artillery':'aa', 'Tank':'at',
    'Fighter':'af', 'TacBmb':'atb', 'StrBmb':'asb', 'AA':None
}

DEF_KEY = {
    'Infantry':'di', 'Mech':'dm', 'Artillery':'da', 'Tank':'dt',
    'Fighter':'df', 'TacBmb':'dtb', 'StrBmb':'dsb', 'AA':'daa'
}


# ════════════════════════════════════════════════════════
# 仿真器
# ════════════════════════════════════════════════════════
rng_sim = np.random.default_rng(42)

def combat(ai,am,aa,at,af,atb,asb, di,dm,da,dt,df,dtb,dsb,daa):
    ai += am; di += dm
    A = dict(i=ai,m=am,a=aa,t=at,f=af,tb=atb,sb=asb)
    D = dict(i=di,m=dm,a=da,t=dt,f=df,tb=dtb,sb=dsb,aa=daa)
    aa_fired = False

    while True:
        A_air = A['f'] + A['tb'] + A['sb']
        if (not aa_fired) and D['aa'] > 0 and A_air > 0:
            shots = min(3*D['aa'], A_air)
            k = rng_sim.binomial(shots, 1/6)
            for key in ['f','tb','sb']:
                kill = min(A[key],k); A[key]-=kill; k-=kill
            aa_fired = True

        sup = min(A['i'],A['a']); unsup = A['i']-sup
        boosted_tb = min(A['tb'], A['f']+A['t'])
        normal_tb  = A['tb'] - boosted_tb

        a_hits = (rng_sim.binomial(sup,       1/3) +
                  rng_sim.binomial(unsup,      1/6) +
                  rng_sim.binomial(A['a'],     1/3) +
                  rng_sim.binomial(A['t'],     1/2) +
                  rng_sim.binomial(A['f'],     1/2) +
                  rng_sim.binomial(boosted_tb, 2/3) +
                  rng_sim.binomial(normal_tb,  1/2) +
                  rng_sim.binomial(A['sb'],    2/3))

        d_hits = (rng_sim.binomial(D['i'],  1/3) +
                  rng_sim.binomial(D['a'],  1/3) +
                  rng_sim.binomial(D['t'],  1/2) +
                  rng_sim.binomial(D['f'],  2/3) +
                  rng_sim.binomial(D['tb'], 1/2) +
                  rng_sim.binomial(D['sb'], 1/6))

        rem = int(d_hits)
        for key in ['i','a','t','f','tb','sb']:
            kill=min(A[key],rem); A[key]-=kill; rem-=kill

        rem = int(a_hits)
        for key in ['i','a','t','f','tb','sb']:
            kill=min(D[key],rem); D[key]-=kill; rem-=kill

        A_alive = sum(A[k] for k in ['i','a','t','f','tb','sb']) > 0
        D_alive = sum(D[k] for k in ['i','a','t','f','tb','sb']) > 0

        def restore_mech(A, am_orig):
            if A['i'] >= am_orig: A['i']-=am_orig; A['m']=am_orig
            elif A['i'] > 0:      A['m']=A['i'];   A['i']=0

        if not A_alive and not D_alive:
            restore_mech(A,am); return 'D', A, D
        if not A_alive:
            restore_mech(A,am); return 'D', A, D
        if not D_alive:
            D['aa']=0; restore_mech(A,am); return 'A', A, D


def sim_to_display(u):
    return {
        'Infantry':u.get('i',0), 'Mech':u.get('m',0),
        'Artillery':u.get('a',0), 'Tank':u.get('t',0),
        'Fighter':u.get('f',0), 'TacBmb':u.get('tb',0),
        'StrBmb':u.get('sb',0), 'AA':u.get('aa',0),
    }


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════
def calc_ipc_frontend(units):
    ipc = {
        'Infantry':3, 'Mech':4, 'Artillery':4, 'Tank':6,
        'AA':5, 'Fighter':10, 'TacBmb':11, 'StrBmb':12,
    }
    return sum(units.get(u,0) * ipc[u] for u in ipc)

def frontend_to_attacker(units):
    return {ATK_KEY[u]: units.get(u,0)
            for u in ALL_UNIT_TYPES if ATK_KEY[u] is not None}

def frontend_to_defender(units):
    return {DEF_KEY[u]: units.get(u,0) for u in ALL_UNIT_TYPES}

def mlp_atk_to_frontend(atk):
    rev = {v:k for k,v in ATK_KEY.items() if v}
    return {rev.get(k,k): v for k,v in atk.items()}


# ════════════════════════════════════════════════════════
# 启动加载模型
# ════════════════════════════════════════════════════════
@app.on_event('startup')
def load_all_models():
    global owl_processor, owl_model, vit_model, vit_class_names
    global mlp_model, mu, std

    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    owl_processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
    owl_model     = OwlViTForObjectDetection.from_pretrained(
        'google/owlvit-large-patch14').to(DEVICE)
    owl_model.eval()

    vit_ckpt        = torch.load('vit_classifier.pth',
                                  map_location=DEVICE, weights_only=False)
    vit_class_names = vit_ckpt['class_names']
    vit_model       = timm.create_model(
        'vit_small_patch16_224', pretrained=False,
        num_classes=len(vit_class_names)).to(DEVICE)
    vit_model.load_state_dict(vit_ckpt['model_state'])
    vit_model.eval()

    mlp_ckpt  = torch.load('winrate_model.pt',
                            map_location=DEVICE, weights_only=False)
    mlp_model = nn.Sequential(
        nn.Linear(15,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128,64),  nn.ReLU(), nn.Linear(64,1),
    ).to(DEVICE)
    mlp_model.load_state_dict(mlp_ckpt['model_state'])
    mlp_model.eval()
    mu  = mlp_ckpt['mu']
    std = mlp_ckpt['std']
    print(f'All models loaded  device={DEVICE}')


# ════════════════════════════════════════════════════════
# POST /analyze
# ════════════════════════════════════════════════════════
@app.post('/analyze')
async def analyze(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')

    boxes, scores, labels = detect_pieces(img, owl_processor, owl_model)
    if len(boxes) == 0:
        empty = {u:0 for u in ALL_UNIT_TYPES}
        return {'JP':empty, 'US':empty, 'warning':'No pieces detected'}

    predictions, valid_boxes = classify_pieces(img, boxes, vit_model, vit_class_names)
    if len(valid_boxes) == 0:
        empty = {u:0 for u in ALL_UNIT_TYPES}
        return {'JP':empty, 'US':empty, 'warning':'No pieces passed confidence filter'}

    factions = [get_faction_by_color(img, box) for box in valid_boxes]

    jp = {u:0 for u in ALL_UNIT_TYPES}
    us = {u:0 for u in ALL_UNIT_TYPES}
    for pred, faction in zip(predictions, factions):
        if pred in ALL_UNIT_TYPES:
            if faction == 'JP':   jp[pred] += 1
            elif faction == 'US': us[pred] += 1

    return {'JP':jp, 'US':us}


# ════════════════════════════════════════════════════════
# 共용 请求体
# ════════════════════════════════════════════════════════
class UnitsRequest(BaseModel):
    attacker: str
    JP:       dict
    US:       dict


# ════════════════════════════════════════════════════════
# POST /winrate
# ════════════════════════════════════════════════════════
@app.post('/winrate')
def winrate(req: UnitsRequest):
    atk_units = req.JP if req.attacker=='JP' else req.US
    def_units = req.US if req.attacker=='JP' else req.JP
    atk = frontend_to_attacker(atk_units)
    dfn = frontend_to_defender(def_units)
    wr  = mlp_predict(atk, dfn, mlp_model, mu, std)
    return {
        'win_rate':     round(wr, 4),
        'attacker_ipc': calc_ipc_frontend(atk_units),
        'defender_ipc': calc_ipc_frontend(def_units),
    }


# ════════════════════════════════════════════════════════
# POST /recommend
# ════════════════════════════════════════════════════════
@app.post('/recommend')
def recommend(req: UnitsRequest):
    atk_units = req.JP if req.attacker=='JP' else req.US
    def_units = req.US if req.attacker=='JP' else req.JP
    atk = frontend_to_attacker(atk_units)
    dfn = frontend_to_defender(def_units)

    current_wr   = mlp_predict(atk, dfn, mlp_model, mu, std)
    attacker_ipc = calc_ipc_frontend(atk_units)
    defender_ipc = calc_ipc_frontend(def_units)

    recommendations = []
    for budget, label in [
        (attacker_ipc, f'Attacker budget ({attacker_ipc} IPC)'),
        (defender_ipc, f'Defender budget ({defender_ipc} IPC)'),
    ]:
        best_atk, best_wr = find_best_attack(
            dfn, max(3, budget), mlp_model, mu, std
        )
        recommendations.append({
            'label':    label,
            'budget':   budget,
            'units':    mlp_atk_to_frontend(best_atk),
            'win_rate': round(best_wr, 4),
        })

    return {
        'current_win_rate': round(current_wr, 4),
        'attacker_ipc':     attacker_ipc,
        'defender_ipc':     defender_ipc,
        'recommendations':  recommendations,
    }


# ════════════════════════════════════════════════════════
# POST /simulate
# ════════════════════════════════════════════════════════
@app.post('/simulate')
def simulate(req: UnitsRequest):
    atk_units = req.JP if req.attacker=='JP' else req.US
    def_units = req.US if req.attacker=='JP' else req.JP

    winner_code, A_surv, D_surv = combat(
        atk_units.get('Infantry',0),  atk_units.get('Mech',0),
        atk_units.get('Artillery',0), atk_units.get('Tank',0),
        atk_units.get('Fighter',0),   atk_units.get('TacBmb',0),
        atk_units.get('StrBmb',0),
        def_units.get('Infantry',0),  def_units.get('Mech',0),
        def_units.get('Artillery',0), def_units.get('Tank',0),
        def_units.get('Fighter',0),   def_units.get('TacBmb',0),
        def_units.get('StrBmb',0),    def_units.get('AA',0),
    )

    winner = req.attacker if winner_code=='A' else (
        'US' if req.attacker=='JP' else 'JP'
    )

    return {
        'winner':             winner,
        'attacker_survivors': sim_to_display(A_surv),
        'defender_survivors': sim_to_display(D_surv),
    }


@app.get('/health')
def health():
    return {'status': 'ok'}