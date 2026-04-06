# test.py
import numpy as np
from pyBanner import banner, info,effect
from blr_model import load_blr, predict_from_dicts

weights = load_blr("blr_weights.npz")

UNIT_COST = {
    "ai": 3, "am": 4, "aa": 4, "at": 6,
    "af": 10, "atb": 11, "asb": 12,
}
A_KEYS = list(UNIT_COST.keys())

D_KEYS = ["di", "dm", "da", "dt", "df", "dtb", "dsb", "daa"]
D_NAMES = {
    "di":  "Infantry",
    "dm":  "Mechanized Infantry",
    "da":  "Artillery",
    "dt":  "Tank",
    "df":  "Fighter",
    "dtb": "Tactical Bomber",
    "dsb": "Strategic Bomber",
    "daa": "Anti-air Artillery",
}
A_NAMES = {
    "ai":  "Infantry",
    "am":  "Mechanized Infantry",
    "aa":  "Artillery",
    "at":  "Tank",
    "af":  "Fighter",
    "atb": "Tactical Bomber",
    "asb": "Strategic Bomber",
}


# ════════════════════════════════════════════════════════
# input
# ════════════════════════════════════════════════════════
def input_int(prompt: str, min_val: int = 0, max_val: int = 999) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if min_val <= v <= max_val:
                return v
            print(f"  Type in integer between {min_val}~{max_val} ")
        except ValueError:
            print("  Please type in integer")


def input_defender() -> dict:
    print("\n─── Enemy Forces（Press enter to skip）───")
    defender = {}
    for key, name in D_NAMES.items():
        while True:
            raw = input(f"  {name} ({key}): ").strip()
            if raw == "":
                defender[key] = 0
                break
            try:
                v = int(raw)
                if v >= 0:
                    defender[key] = v
                    break
                print("Invalid number")
            except ValueError:
                print("Invalid number")
    return defender


def input_budget() -> int:
    print("")
    return input_int("─── Input IPC Budget:", min_val=1)


# ════════════════════════════════════════════════════════
# search
# ════════════════════════════════════════════════════════
def find_best_attack(defender: dict, budget: int,
                     n_samples: int = 8000, seed: int = 42) -> dict:
    rng   = np.random.default_rng(seed)
    costs = np.array([UNIT_COST[u] for u in A_KEYS], dtype=np.int32)

    best_wr  = -1.0
    best_atk = None

    for _ in range(n_samples):
        a         = np.zeros(7, dtype=np.int32)
        remaining = budget

        for _ in range(32):
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

        attacker = dict(zip(A_KEYS, a.tolist()))
        wr       = predict_from_dicts(attacker, defender, weights)

        if wr > best_wr:
            best_wr  = wr
            best_atk = attacker

    return {"attacker": best_atk, "win_rate": best_wr,
            "cost": budget - sum(
                best_atk[u] * UNIT_COST[u] for u in A_KEYS
            )}


# ════════════════════════════════════════════════════════
# output
# ════════════════════════════════════════════════════════
def print_defender(defender: dict):
    units = {D_NAMES[k]: v for k, v in defender.items() if v > 0}
    if not units:
        print("  （No Unit）")
        return
    for name, cnt in units.items():
        print(f"    {name}: {cnt}")


def print_result(result: dict, budget: int):
    print("\n─── The Optimal Plan " + "─" * 35)
    atk   = result["attacker"]
    units = {A_NAMES[k]: v for k, v in atk.items() if v > 0}
    for name, cnt in units.items():
        print(f"    {name}: {cnt}")
    print(f"\n  Budget:     {budget} IPC")
    print(f"  Cost:     {budget - result['cost']} IPC")
    print(f"  Left:     {result['cost']} IPC")
    print(f"  Win Rate: {result['win_rate'] * 100:.1f}%")
    print("─" * 50)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    # ── Startup banner  ─────────────────────────────
    banner(6)
    effect(1,description="VIT Boardgame Analyzer")
    info(0,
         project="VIT Boardgame Analyzer [TEST]",
         version="1.0",
         environment="Development",
         description="",
         status="Initializing... "
         )

    print("╔══════════════════════════════════════╗")
    print("║      Optimal Combination Planner     ║")
    print("╚══════════════════════════════════════╝")

    while True:
        defender = input_defender()
        budget   = input_budget()

        if sum(defender.values()) == 0:
            print("\n  Cannot be all 0")
            continue

        print("\n  Thinking...")
        result = find_best_attack(defender, budget)

        print("\n  Enemy Forces:")
        print_defender(defender)
        print_result(result, budget)

        again = input("\nContinue Planing？(y/n): ").strip().lower()
        if again != "y":
            print("\nQuit。")
            break


if __name__ == "__main__":
    main()