# Technical Documentation — AI-Driven Parking Intelligence

This document is a deep-dive into the methodology, design decisions, and engineering rigour behind
`traffic-bangalore-gridlock.ipynb`. It is intended for reviewers and engineers who want to
understand *why* each step is built the way it is.

---

## 1. Pipeline overview

```
Raw violation logs
        │
        ▼
[A] Cleaning  ──►  drop junk/ID cols · parse timestamps · UTC→IST · geo-filter · feature engineering
        │
        ▼
[B] EDA  ──►  temporal (UTC vs IST) · day×hour surface · spatial scatter · vehicle/violation mix
        │
        ▼
[1] DETECT     DBSCAN (haversine)            ──►  geographic hotspots (+ noise removed)
        │
        ▼
[2] QUANTIFY   Congestion-Impact Score       ──►  ranked 0–100 priority per hotspot
        │
        ▼
[3] FORECAST   HistGradientBoostingRegressor ──►  hourly demand per zone (chronological eval)
        │
        ▼
[4] TARGET     prioritisation + scheduling   ──►  where+when dispatch table · weekly heatmap
        │
        ▼
[5] DELIVER    Folium map · CSV/joblib exports
```

---

## 2. Data cleaning & feature engineering

### 2.1 Column hygiene
- **Junk columns** (`Unnamed: *`) arise from trailing commas in the source CSV and are dropped.
- **Identifiers / pipeline-internal fields** (`id`, `vehicle_number`, `device_id`, `created_by_id`,
  validation/SCITA timestamps, etc.) carry no predictive signal for *where/when congestion forms*
  and are removed to avoid noise and accidental leakage.

### 2.2 The timezone correction (critical)
The raw timestamps are stored in **UTC**. Plotting the raw hour-of-day produces an implausible
**04:00–06:00 peak** — nobody is mass-issuing parking tickets before dawn. This is a textbook UTC
artefact. We fix it:

```python
def to_local(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)   # parse as UTC
    return s.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)  # shift +5:30, drop tz
```

After conversion the load moves into the realistic commercial-day / evening window. **Every
temporal feature** (`Hour`, `IsRushHour`, day-of-week, the forecasting panel) is derived from this
corrected local clock. We retain `HourRawUTC` *only* to visualise the before/after in EDA.

### 2.3 Geographic filtering
Points are restricted to the Bengaluru bounding box (`lat ∈ [12.7, 13.3]`, `lon ∈ [77.3, 77.9]`).
This discards `(0, 0)` GPS errors and out-of-region noise that would otherwise distort clustering.

### 2.4 Engineered signals
| Feature | Definition | Purpose |
|---|---|---|
| `duration_min` | `closed_datetime − created_datetime` (clamped 0–24 h, median-imputed) | direct, timezone-invariant measure of how long a road stays blocked |
| `is_junction` | 1 if `junction_name` is a real junction | junction obstructions block multiple directions |
| `violation_class` | normalised to `WRONG_PARKING / NO_PARKING / OTHER` | consistent categorical signal |
| `vehicle_footprint` | mapped from `vehicle_type` (scooter 0.6 → truck 4.0) | larger vehicles obstruct more carriageway |
| `IsRushHour` | hour ∈ {08–11, 17–21} IST | peak-hour overlap amplifies impact |

All downstream columns are defensively created with sensible defaults if absent, so the notebook is
robust to schema differences.

---

## 3. Stage 1 — Hotspot detection (DBSCAN)

We use **DBSCAN with the haversine metric** on `radians(latitude, longitude)`:

```python
eps_rad = HOTSPOT_RADIUS_M / EARTH_RADIUS_M    # 60 m / 6,371,000 m
DBSCAN(eps=eps_rad, min_samples=30, metric="haversine", algorithm="ball_tree")
```

**Why DBSCAN (not K-Means)?**
- It needs **no preset number of clusters** — we don't know how many hotspots exist a priori.
- It clusters by **true ground distance** (haversine), so `eps` is a real metric radius (~60 m,
  roughly one street segment).
- It explicitly labels **scattered one-off violations as noise (`-1`)**, separating chronic
  choke-points from background activity. K-Means would force every point into a cluster.

**Parameters**
- `HOTSPOT_RADIUS_M = 60` — neighbourhood radius (street-segment scale).
- `HOTSPOT_MIN_SAMPLES = 30` — minimum violations to qualify as a persistent hotspot.

---

## 4. Stage 2 — Congestion-Impact Score (CIS)

### 4.1 Rationale
Volume alone is a poor proxy for congestion. The CIS is a **transparent linear blend** of five
normalised drivers, chosen because each maps to a physical mechanism of flow disruption:

$$\text{CIS} = 100 \times \big(0.40\,V + 0.20\,D + 0.20\,J + 0.12\,P + 0.08\,F\big)$$

| Term | Source aggregate | Normalisation |
|---|---|---|
| `V` Volume | count of violations in hotspot | `minmax(log1p(volume))` — log damps the heavy tail |
| `D` Duration | median `duration_min` | `minmax` |
| `J` Junction | mean `is_junction` | already in `[0,1]`, clipped |
| `P` Peak | mean `IsRushHour` | already in `[0,1]`, clipped |
| `F` Footprint | mean `vehicle_footprint` | `minmax` |

### 4.2 Weight design
Weights sum to **1.0**, so CIS ∈ `[0, 100]`. Volume gets the largest weight (chronic pressure is
the strongest signal), followed by duration and junction-blocking (both directly throttle flow).
**Graceful degradation:** if `closed_datetime` is unavailable (`HAS_DURATION == False`), the
duration weight `0.20` folds into volume so the score remains bounded and meaningful.

### 4.3 Tiering
`pd.cut` with bins `[-0.1, 25, 50, 75, 100.1]` → **Low / Medium / High / Critical**, giving planners
an immediately interpretable triage.

---

## 5. Stage 3 — Spatio-temporal demand forecasting

This is the part most prone to mistakes; the design specifically **avoids target leakage**.

### 5.1 Panel construction
For the **top-K hotspots** (`K = min(40, n_hotspots)`) we build a complete
`(cluster × date × hour)` grid over each hotspot's active date span and **fill genuine zeros** for
hours with no violations. Without zero-filling the model would only ever see positive counts and
could not learn *absence* of demand.

### 5.2 Features (leakage-free)
```python
FEATURES = ["hour_sin", "hour_cos", "Hour", "DayOfWeek", "IsWeekend", "IsRushHour",
            "Month", "latitude", "longitude", "junction_share", "footprint",
            "zone_hour_baseline"]
```
- **Cyclical hour encoding** (`hour_sin/cos`) so 23:00 and 00:00 are adjacent.
- **Calendar** features capture weekly/seasonal rhythm.
- **Static zone descriptors** (`latitude`, `longitude`, `junction_share`, `footprint`) describe the
  *place*, not the target.
- **`zone_hour_baseline`** = mean count per `(zone, hour)` computed **from training dates only** —
  a historical prior that is *not* derived from any test-period target.

> No feature is computed from the label on the rows being predicted. The target (`count`) never
> appears, directly or transformed, among the inputs for a given row.

### 5.3 Chronological evaluation
```python
cutoff = panel["Date"].quantile(0.75)
train = panel[panel["Date"] <= cutoff]   # earliest ~75% of dates
test  = panel[panel["Date"] >  cutoff]   # most recent ~25%, unseen future
```
This is a **real forecast**: train on the past, test on a later period the model has never seen.
A random shuffle split would leak future information and inflate scores.

### 5.4 Model
`HistGradientBoostingRegressor(max_depth=7, learning_rate=0.08, max_iter=400,
l2_regularization=1.0)` — a fast, regularised gradient-boosting regressor that handles the mixed
numeric features well and trains quickly on the large panel. Predictions are clipped at `0`
(counts cannot be negative).

### 5.5 Honest metrics
We report **MAE, RMSE, R²** on the held-out future, and compare MAE against a **naïve baseline**
(the `zone × hour` average from training). *Skill vs naïve* = `1 − MAE_model / MAE_naive`. Beating
the baseline proves the model captures real spatio-temporal structure rather than memorising
coordinates.

> **Contrast with the discarded approach.** An earlier version effectively used a
> coordinate-to-severity lookup, which can trivially report "100% accuracy" yet generalises to
> nothing. RMSE is computed as `mean_squared_error(...) ** 0.5` to stay compatible across
> scikit-learn versions (avoids the deprecated `squared=False` argument).

### 5.6 Interpretability
**Permutation importance** on a test-set sample shows which signals the model truly relies on
(typically time-of-day and the zone baseline dominate). A predicted-vs-actual daily time series
provides a visual sanity check over the future weeks.

---

## 6. Stage 4 — Prioritisation & scheduling

For each priority hotspot we predict a **typical weekly demand surface** (7 days × 24 hours) using
the trained model, then derive:
- **`busiest_day`** — day-of-week with the highest mean predicted demand.
- **`peak_window`** — the contiguous **3-hour** window maximising summed predicted demand.
- **`pred_peak_per_hr`** — the peak hourly intensity.

The result is the **dispatch table** (`enforcement_priorities.csv`): *which hotspot, which day,
which window* — ranked by CIS. Averaging predicted demand across all priority hotspots yields the
**city-wide weekly enforcement heatmap**.

---

## 7. Stage 5 — Delivery

- **Interactive Folium map** (`parking_hotspots_map.html`): a sampled violation-density heat layer
  plus impact-tiered hotspot markers (radius ∝ CIS) with click-through popups (score, volume,
  junction share, peak share, police station). Sampling keeps the HTML browser-friendly.
- **Exports:** impact scores CSV, enforcement-priorities CSV, and the serialised model
  (`joblib`) with its feature list, ready to power a dashboard or API.

---

## 8. Reproducibility

- `RANDOM_STATE = 42` is applied to NumPy, DBSCAN (implicitly deterministic), the regressor,
  permutation importance, and all sampling steps.
- The loader resolves the dataset path deterministically (known Kaggle path first, then sorted
  glob), so runs are repeatable across environments.

---

## 9. Configuration reference

| Constant | Value | Meaning |
|---|---|---|
| `ASSUME_UTC_SOURCE` | `True` | treat raw timestamps as UTC |
| `LOCAL_TZ` | `"Asia/Kolkata"` | target timezone (IST) |
| `HOTSPOT_RADIUS_M` | `60` | DBSCAN neighbourhood radius (metres) |
| `HOTSPOT_MIN_SAMPLES` | `30` | DBSCAN minimum cluster size |
| `EARTH_RADIUS_M` | `6,371,000` | for converting metres → radians |
| `MORNING_PEAK` | `08:00–11:59` | morning rush window (IST) |
| `EVENING_PEAK` | `17:00–21:59` | evening rush window (IST) |
| `K` (forecast zones) | `min(40, n_hotspots)` | hotspots carried into forecasting |
| forecast split | `Date.quantile(0.75)` | chronological train/test cutoff |
