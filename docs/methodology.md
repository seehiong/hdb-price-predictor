# Methodology

Design notes for anyone auditing how the numbers in the [README](../README.md) were produced. The numerical claims here were measured on this repository's data; where an approach was tried and rejected, the measurement that rejected it is included. Explanatory passages are labelled as reasoning rather than results.

The exact feature construction, cutoff handling, training commands and export steps live in [`pipeline_public/`](../pipeline_public) and [`pipeline_private/`](../pipeline_private). The published figures in `models/metrics.json` are written by those same training runs, so anything quoted below can be regenerated with `run_pipeline.py`.

## Contents

- [Validation](#validation)
- [Forecasting past the data cutoff](#forecasting-past-the-data-cutoff)
  - [How the projection is applied](#how-the-projection-is-applied)
  - [Two prediction modes](#two-prediction-modes)
- [MRT proximity](#mrt-proximity)
- [Approaches tested and rejected](#approaches-tested-and-rejected)
- [Published metrics](#published-metrics)

## Validation

Both models use a **chronological holdout**: training ends at a fixed historical cutoff and every later transaction is reserved for evaluation. This mirrors what the app actually does — price a sale that has not happened yet.

The models train on all complete historical calendar years (up through 2025-12), ensuring the trees genuinely learn price dynamics up to the cutoff for historical comparable valuation. Subsequent unseen future sales (2026-01 onward) are held out for out-of-sample forward evaluation:

| Model | Data spans | Trained on | Evaluated on | Holdout size |
| :--- | :--- | :--- | :--- | :--- |
| Public (HDB) | 2017-01 → 2026-09 | ≤ 2025-12 | 2026-01 onward | 18,278 sales |
| Private (non-landed) | 2021-06 → 2026-06 | ≤ 2025-12 | 2026-01 onward | 5,458 sales |

The holdout boundary sits cleanly at the end of the last complete calendar year (December 31, 2025). Any user query within the training range (`sale_year <= 2025`) runs as a pure comparable valuation; any query past the cutoff (`sale_year >= 2026`) is projected using the official indices.

### Why not a random split

A shuffled split lets the model see later sales of the same block or project, and lets it learn the price level of years it is then tested on. Both leak. Measured on the shipped models — identical features and hyperparameters, only the split changed, holdout sizes matched:

| Model | Random split | Chronological (published) |
| :--- | :--- | :--- |
| Public (HDB) | 4.08% MAPE · R² 97.61% | **4.63% MAPE · R² 95.53%** |
| Private (non-landed) | 5.73% MAPE · R² 94.44% | **6.27% MAPE · R² 95.97%** |

A random split makes the HDB model look nearly twice as accurate as it is. The published figures are deliberately the less flattering ones.

These are **model-level validation metrics**, not a guarantee that any individual property will be valued within the stated range. MAPE is listed first in the README because it answers the question a reader actually has — how far off is a typical estimate — more directly than R².

## Forecasting past the data cutoff

Gradient-boosted trees cannot extrapolate: any `sale_year` beyond the training range falls into the same terminal leaf, so the raw model returns a flat forecast. Before this was addressed, the private model returned an identical price for 2027 through 2032, and the HDB model drifted slightly *down*.

Prices are therefore projected using the **official published indices** (HDB RPI; URA PPI by market segment) rather than a trend fitted to our own transactions. The segment split matters: CCR, RCR and OCR grew roughly +20%, +33% and +46% between 2021-Q1 and 2026-Q2, so a single blended rate overstates a prime-district six-year forecast by about 24% — around S$876k on a S$3M unit.

### How the projection is applied

The model is always evaluated at the last year it was trained on, with a neutral mid-year month. Its output is then multiplied by `(1 + rate) ^ years_elapsed`, measured from the **end** of that year.

Two details that are easy to get wrong, and were:

- **Measure elapsed time from the end of the cutoff year, not the start.** Measuring from the start counts an extra year: December of the cutoff year is zero years out, but the following January would score 1.000 instead of 0.083. That overstated every user-selectable date by 6.4%.
- **Do not feed the target month to the model.** `sale_month` is a raw numeric feature, so the trees split it at thresholds that mostly encode within-year price drift rather than real seasonality. Passing a projected month through produced a 7.9% cliff between December and January. Elapsed time is carried by the growth factor instead.
- **Two-tier market segment resolution (CCR / RCR / OCR).** URA defines market segments by Planning Areas rather than postal districts, so several districts cross regional boundaries (e.g. District 20 spans both RCR in Bishan and OCR in Ang Mo Kio). The engine uses a two-tier lookup: project-level ground truth from official URA caveats is checked first (`project_segment` in `condo_project_encoding.json`, e.g. resolving *Nuovo* correctly to OCR and *Sky Habitat* to RCR), falling back to the predominant district default only when a project is unlisted.

### Two prediction modes

The year picker accepts dates from 2017 (the start of the HDB dataset) through the current year + 6.

1. **Within training cutoff (`sale_year <= cutoff`)**: `getForecastAdjustment()` returns `growthFactor = 1.0` and `yearsProjected = 0` — the model runs pure XGBoost directly on the selected year and month, with no index multiplication. This serves both past years (asking "what should a comparable flat have sold for in 2022?") and valuations within the training window. The measured holdout MAPE (4.63% HDB / 6.27% Condo) applies directly.
2. **Beyond training cutoff (`sale_year > cutoff`)**: XGBoost is evaluated at the cutoff year base with a neutral mid-year month, scaled forward by the compounding market index.

The UI reflects which path is active via the year-label and hint text (see the [two-modes table in the README](../README.md#two-prediction-modes)).

## MRT proximity

Each block is compared against all 563 station exits and the nearest is kept, so blocks served by several stations need no special handling and no spatial database — it is a 12,847 × 563 scan that runs in seconds.

The trainer joins on (block, town), which is all the resale data provides. Around 596 buildings share such a key with another building — "Block 12, Bukit Merah" is four separate blocks between 0.13 km and 1.24 km from a station — and those keys carry the mean of their members, blurring the feature for about 7% of transactions.

Measured before deciding whether to fix it:

| | MAE | MAPE | R² |
| :--- | :--- | :--- | :--- |
| With MRT features (7% blurred, **shipped**) | S$49,293 | 7.22% | 90.33% |
| Without MRT features | S$52,038 | 7.54% | 88.64% |

The feature is worth S$2,745 of MAE even blurred, and the remaining 93% of rows are already exact, so perfect disambiguation would recover only a fraction of that — not worth an external geocoder dependency in the build. The web app looks up by postal code, which is unique and unaffected.

## Approaches tested and rejected

### Deflating the training target by the index

The textbook approach: train on `price ÷ index` so the model learns time-invariant quality, then multiply the index back at prediction time. Implemented and measured with the index truncated at 2024-Q4 so no variant may read an index beyond the training boundary:

| Approach | MAE | MAPE | R² |
| :--- | :--- | :--- | :--- |
| Raw price, no index (**shipped**) | S$159,498 | 7.52% | 93.63% |
| Index-deflated target | S$177,515 | 10.06% | 93.10% |
| Clamp + single blended CAGR | S$218,103 | 11.05% | 88.76% |

Deflation works mechanically — it flattens 2021→2025 from +32% raw to within 3.4% per segment — but loses on accuracy and over-predicts in every segment (+5.3% to +7.9% mean signed error). The results suggest that the market's sharp deceleration, rather than the deflation step itself, is the main reason the trailing trend overshoots: RCR grew 1.2% over the holdout where its own trailing trend implied 12%. That reading is consistent with the measurements, though these experiments do not isolate it as the sole cause.

The models therefore predict raw prices, and the index is applied only to the forward curve.

### Adding landed property to the private model

The private dataset carries 12,305 landed transactions -- terrace, semi-detached, detached and their strata variants -- which the model filters out. Folding them in was measured rather than assumed:

| Model | Rows | MAE | MAPE | R² |
| :--- | ---: | ---: | ---: | ---: |
| Non-landed only (**shipped**) | 71,538 | S$159,498 | 7.52% | 93.63% |
| Landed only | 12,078 | S$785,503 | 12.47% | 83.18% |
| All private combined | 83,616 | S$260,624 | 8.60% | 93.11% |

Scoring the combined model per segment shows why one model for both is the wrong shape:

| Segment | Combined | Own model |
| :--- | ---: | ---: |
| Non-landed | 7.91% MAPE | **7.52%** |
| Landed | 12.45% MAPE | 12.47% |

Combining costs the non-landed model 0.39pp and gains landed nothing, because the two are different markets: landed sells at roughly three times the price (median S$4.4M against S$1.5M) with wider dispersion (p90/p10 unit price 2.73x against 2.11x).

Landed is also harder to model with the fields available. `floor_range` is `-` for 99% of landed rows, so that feature is dead. 42% of landed transactions are named `LANDED HOUSING DEVELOPMENT` rather than a real project, so project encoding -- the strongest single feature on the non-landed side -- cannot work for them. And land area, which drives landed pricing, is not in the dataset at all.

Postal coverage is the other constraint. The URA dwelling-units layer does carry 72,681 landed entries with postal codes, but only 33% of landed transactions can be matched to one, again because of the placeholder project names.

If landed is added later it should be a **third property class with its own model**, not a third category folded into this one, and its ~12.5% MAPE should be surfaced as plainly as the current 6.27% is. The naming already anticipates this: the two shipped classes are **Public (HDB)** and **Private (non-landed)**, leaving **Private (landed)** free for a third.

### Block-level price encoding for HDB

Block+street explains 72.9% of HDB unit-price variance against 23.4% for town alone, so target-encoding it looked promising. It made accuracy *worse*, and a drift-free town-relative variant did not help either:

| Variant | MAE | MAPE |
| :--- | :--- | :--- |
| Baseline (no block encoding) | S$69,920 | 10.40% |
| + block PSM encoding | S$71,553 | 10.57% |
| + drift-free town-relative encoding | S$71,421 | 10.61% |

*Measured before MRT proximity was added, so the baseline here is the pre-MRT model rather than the shipped one (S$49,293). The comparison between variants is like-for-like; only the absolute level differs from the published figures.*

Block price is stable (r = 0.934 across the split) but **absolute**, so it anchors the model to stale price levels in a market that rose 31% over the holdout window. MRT distance works where this fails because it is a geographic fact that stays valid as prices move.

The same encoding on the private side does work — 2,220 distinct projects are genuinely heterogeneous where 26 towns are not — and cut private-model MAE by 27% (S$218,103 → S$159,498 on the chronological holdout).

## Published metrics

Accuracy figures are written to `models/metrics.json` by the trainers and read by the web page at load, so the About panel can never drift from the model actually shipped. A copy is also inlined into `app.js` as `DEFAULT_METRICS`, because browsers block `fetch()` on `file://` — it is regenerated on every training run, so it cannot go stale either.

**After a retrain there is nothing to edit in `index.html`.**

### Tree counts

Early stopping keeps training past the best iteration to prove the score is not improving, so the saved booster holds more trees than the model that was validated. Python's `predict()` truncates at `best_iteration` silently, but the browser walks whatever is exported — so the export is truncated at training time and `metrics.json` reports the exported count.

| Model | Trees built | Trees shipped | Why |
| :--- | :--- | :--- | :--- |
| Public (HDB) | 600 | 600 | Never triggered early stopping; all estimators kept |
| Private (non-landed) | 211 | 181 | Stopped improving at iteration 180; the trailing 30 are discarded |

The number in this table, in `models/metrics.json`, and the tree count in the browser bundle are the same by construction — the export is what `metrics.json` counts.
