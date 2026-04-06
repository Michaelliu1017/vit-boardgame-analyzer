from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from blr_model import load_model, predict

app = FastAPI(title="Wargame Win-Rate API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_model("winrate_model.pt")
    print("Model loaded ✓")


class BattleInput(BaseModel):
    # attacker
    ai:  int = Field(0, ge=0, description="Infantry")
    am:  int = Field(0, ge=0, description="Mechanized Infantry")
    aa:  int = Field(0, ge=0, description="Artillery")
    at:  int = Field(0, ge=0, description="Tank")
    af:  int = Field(0, ge=0, description="Fighter")
    atb: int = Field(0, ge=0, description="Tactical Bomber")
    asb: int = Field(0, ge=0, description="Strategic Bomber")
    # defenser
    di:  int = Field(0, ge=0)
    dm:  int = Field(0, ge=0)
    da:  int = Field(0, ge=0)
    dt:  int = Field(0, ge=0)
    df:  int = Field(0, ge=0)
    dtb: int = Field(0, ge=0)
    dsb: int = Field(0, ge=0)
    daa: int = Field(0, ge=0, description="Anti-Air Artillery ")


@app.post("/predict")
def predict_endpoint(battle: BattleInput):
    vec15 = [
        battle.ai, battle.am, battle.aa, battle.at, battle.af, battle.atb, battle.asb,
        battle.di, battle.dm, battle.da, battle.dt, battle.df, battle.dtb, battle.dsb,
        battle.daa
    ]
    result = predict(vec15)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}