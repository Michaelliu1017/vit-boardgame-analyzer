# blr_model.py
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def phi_from_x15(x15: np.ndarray) -> np.ndarray:

    x = x15.astype(np.float64)
    A = x[:7]; D = x[7:]
    ai, am, aa, at, af, atb, asb = A
    di, dm, da, dt, df, dtb, dsb, daa = D

    Ai          = ai + am
    Di          = di + dm
    supA        = min(Ai, aa)
    boosted_tb  = min(atb, af + at)
    air_total   = af + atb + asb

    feats = [1.0]
    feats += A.tolist()
    feats += D.tolist()
    feats += [A.sum(), D.sum(), A.sum() - D.sum()]
    feats += [Ai, Di, supA, boosted_tb, daa, air_total, daa * air_total]
    feats += [aa * Di, at * dt, af * df, atb * dtb]
    return np.array(feats, dtype=np.float64)


def predict_prob_mean_mc(Phi_star, m, S, beta, mc=64, seed=0):
    rng      = np.random.default_rng(seed)
    mu_pred  = Phi_star @ m
    var      = (1.0 / beta) + np.sum((Phi_star @ S) * Phi_star, axis=1)
    std_pred = np.sqrt(np.maximum(var, 1e-12))
    samples  = rng.normal(mu_pred[:, None], std_pred[:, None],
                          size=(len(mu_pred), mc))
    return sigmoid(samples).mean(axis=1)


# ── Load weights ─────────────────────────────────────────
def load_blr(path: str = "blr_weights.npz") -> dict:
    data = np.load(path)
    return {
        "m":    data["m"],
        "S":    data["S"],
        "beta": float(data["beta"][0]),
    }


# ── interface ─────────────────────────────────────────
def predict(vec15: list, weights: dict, mc: int = 64) -> float:
    """
    input:  [ai,am,aa,at,af,atb,asb, di,dm,da,dt,df,dtb,dsb,daa]
    output: winrate [0, 1]
    """
    x15 = np.array(vec15, dtype=np.float32)
    phi = phi_from_x15(x15).reshape(1, -1)
    prob = predict_prob_mean_mc(phi, weights["m"], weights["S"],
                                weights["beta"], mc=mc)
    return float(prob[0])


def predict_from_dicts(attacker: dict, defender: dict, weights: dict) -> float:
    """
    attacker: dict(ai=1, am=0, aa=1, ...)
    defender: dict(di=3, dm=0, da=1, ..., daa=0)
    """
    A_KEYS = ["ai", "am", "aa", "at", "af", "atb", "asb"]
    D_KEYS = ["di", "dm", "da", "dt", "df", "dtb", "dsb", "daa"]

    vec15 = (
        [attacker.get(k, 0) for k in A_KEYS] +
        [defender.get(k, 0) for k in D_KEYS]
    )
    return predict(vec15, weights)