# Dataset folder

Place your training CSVs here as:

```
data/fake.csv
data/real.csv
```

## Expected columns

Each CSV needs at least one text column. The loader auto-detects common
column names (case-insensitive):

- A **text** column: `text`, `content`, `body`, or `article` (required)
- A **title** column: `title` (optional — combined with text if present)

Example (`real.csv`):

| title | text |
|---|---|
| Markets rally on rate decision | Stocks rose Tuesday after... |

If your dataset uses different column names, rename them to match, or edit
`utils/baseline.py` → `_load_csv_texts()` to add your column names to the
detection list.

## What happens at startup

- `data/fake.csv` rows are labelled **Misleading** (1)
- `data/real.csv` rows are labelled **Genuine** (0)
- The pipeline trains a TF-IDF (1-2 grams, 20k features) + XGBoost classifier
- The trained pipeline is cached to `model_cache/xgb_pipeline.joblib`
- On every subsequent startup, if `fake.csv`/`real.csv` haven't changed
  (same file size + modified time), the cached model loads instantly instead
  of retraining

## No CSVs yet?

The app still runs — it falls back to a tiny 80-example inline placeholder
dataset so you can develop/demo the rest of the pipeline. Swap in your real
20k-row dataset whenever it's ready; the next startup will detect the new
files and retrain automatically.

## Force a retrain

If you edit the CSVs in place without changing their size (rare), or just
want to force a fresh training run, delete the cache:

```bash
rm -rf model_cache/
```
