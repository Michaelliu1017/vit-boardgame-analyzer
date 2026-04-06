# vit-boardgame-analyzer
A board game decision support platform that combines simulation, machine learning, and computer vision.

Work in Progress...


<p align="center">
  <img src="assets/gitowl.png" width="100">
</p>

# VIT Boardgame Analyzer — Test Script

## Prerequisites

- Python 3.11 or 3.12
- `blr_weights.npz` and `winrate_model.pt` placed in `app/backend/`

---

## Setup
```bash
cd app/backend
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run
```bash
python test.py
```

Follow the prompts to enter enemy units and IPC budget.
The script will output the optimal attack composition and predicted win rate.

---

## Exit
```bash
deactivate
```
