# 🅿️ AI-Driven Parking Intelligence for Bengaluru

### Detecting Illegal-Parking Hotspots & Quantifying Their Impact on Traffic Flow

> **Problem statement** — *How can AI-driven parking intelligence detect illegal-parking
> hotspots and quantify their impact on traffic flow to enable targeted, predictive enforcement?*

Bengaluru's traffic police log **hundreds of thousands** of parking-violation records every year,
but enforcement is largely **reactive** — patrols respond *after* congestion has formed. This project
turns that historical log into a **proactive, hour-by-hour enforcement plan** using a three-stage
machine-learning pipeline.

---

## 🎯 What this project does

| Stage | Question it answers | Method |
|-------|--------------------|--------|
| **1 · Detect** | *Where are the true hotspots?* | **DBSCAN** spatial clustering (haversine metric) groups violations into geographic hotspots and discards scattered noise. |
| **2 · Quantify** | *Which hotspots actually hurt traffic?* | A transparent **Congestion-Impact Score (CIS)** blends volume, dwell-time, junction proximity, peak-hour concentration and vehicle footprint into a single **0–100** priority index. |
| **3 · Target** | *When should we send patrols?* | A leakage-free **spatio-temporal demand forecaster** predicts violations per zone per hour, producing a *where + when* dispatch schedule. |

The final output is a ranked **dispatch table** (which hotspot, which day, which 3-hour window),
a **weekly enforcement heatmap**, and an **interactive map** — everything a planner needs to move
from guesswork to data-driven deployment.

---

## 🧠 Why this analysis is rigorous

- ⏱️ **Correct local time.** Source timestamps are treated as **UTC** and converted to **IST
  (Asia/Kolkata, UTC+5:30)** before any temporal analysis. Without this, "peak hours" are shifted
  5.5 hours and every time-based conclusion is wrong.
- 🔮 **Honest forecasting, no leakage.** The model trains on the **past** and is tested on the
  **future** via a chronological split. No target-derived feature leaks into the inputs, so the
  reported **MAE / RMSE / R²** are trustworthy out-of-sample estimates — and they are benchmarked
  against a **naïve baseline** to prove the model adds real skill.
- 🔍 **Interpretable by design.** The impact weights are explicit and the forecast drivers are
  measured by **permutation importance** — essential for a public-enforcement decision tool.

---

## 📊 The Congestion-Impact Score (CIS)

A raw violation *count* is not congestion: a bus blocking a junction at 6 PM hurts far more than
ten scooters on an empty side-street at noon. The CIS captures that:

$$\text{CIS} = 100 \times \big(0.40\,V + 0.20\,D + 0.20\,J + 0.12\,P + 0.08\,F\big)$$

| Symbol | Driver | Why it reduces traffic flow |
|:--:|---|---|
| `V` | **Volume** (log-scaled) | chronic, repeated obstruction pressure |
| `D` | **Duration** | how long each vehicle actually blocks the road |
| `J` | **Junction share** | blocking a junction backs up *all* directions |
| `P` | **Peak-hour share** | impact is worst when roads are already full |
| `F` | **Vehicle footprint** | larger vehicles block more carriageway width |

Each driver is min-max normalised to `[0, 1]`, so the score is bounded **0–100** and comparable
across the city. Hotspots are then tiered **Low / Medium / High / Critical**.

---

## 🗂️ Dataset

- **Source:** [Traffic Bangalore dataset](https://www.kaggle.com/datasets/jyotismritabasisthya/traffic-bangalore) (Kaggle)
- **Scale:** ~250k+ parking-violation records
- **Key fields used:** `created_datetime`, `closed_datetime`, `latitude`, `longitude`,
  `vehicle_type`, `violation_type`, `junction_name`, `police_station`, `center_code`

---

## 🚀 How to run

### Option A — Kaggle (recommended)

1. Open the notebook on Kaggle (or upload `traffic-bangalore-gridlock.ipynb`).
2. Add the dataset **`jyotismritabasisthya/traffic-bangalore`** via *Add Data*.
3. **Run All**. The notebook auto-detects the dataset path, so no edits are needed.

### Option B — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset CSV into this folder (or any sub-folder)
#    The loader searches ./**/*.csv automatically.

# 3. Launch Jupyter and run all cells
jupyter notebook traffic-bangalore-gridlock.ipynb
```

The data-loading cell tries the known Kaggle path first, then falls back to a recursive glob
search, so it runs unchanged in both environments.

---

## 📦 Generated artifacts

Running the notebook writes the following to the working directory:

| File | Contents |
|---|---|
| `hotspot_impact_scores.csv` | Every detected hotspot with its CIS, tier and drivers |
| `enforcement_priorities.csv` | Ranked dispatch table: **where + when** to deploy |
| `demand_forecaster.joblib` | The trained forecasting model + feature list |
| `parking_hotspots_map.html` | Standalone interactive Folium dispatch map |

---

## 🛠️ Tech stack

- **Python** · pandas · NumPy
- **scikit-learn** — DBSCAN, HistGradientBoostingRegressor, permutation importance
- **Matplotlib** · **Seaborn** — exploratory & diagnostic visuals
- **Folium** — interactive geospatial dispatch map
- **joblib** — model persistence

---

## ⚠️ Limitations & next steps

- Volume is a **proxy** for congestion; integrating live **speed / GPS-probe** data would calibrate
  the CIS against measured delay.
- The window spans ~6 months; a full year would expose monsoon / seasonal effects.
- **Next:** live camera / ANPR ingestion, an alerting micro-service, and feedback loops that learn
  from enforcement outcomes.

---

## 📁 Repository structure

```
.
├── traffic-bangalore-gridlock.ipynb   # Main solution notebook (3-stage pipeline)
├── README.md                          # This file
├── DOCUMENTATION.md                   # Technical deep-dive
└── requirements.txt                   # Python dependencies
```
