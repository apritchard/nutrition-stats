# nutrition-stats

Interactive charts for personal health data — weight, nutrition, TDEE, and exercise — exported from MyFitnessPal.

## Setup

```bash
pip install -r requirements.txt
```

## Adding new data

Drop new export CSVs into `data/exports/`. The pipeline auto-discovers files by name pattern, so no code changes are needed:

| File pattern | Source |
|---|---|
| `Nutrition-Summary-*.csv` | MyFitnessPal nutrition export |
| `Exercise-Summary-*.csv` | MyFitnessPal exercise export |
| `Measurement-Summary-*.csv` | MyFitnessPal measurements export (primary weight source) |
| `Weight-Datapoints*.csv` | Manual weight log (historical; no longer updated) |

Overlapping date ranges across multiple files of the same type are deduplicated automatically. Where `Weight-Datapoints` and `Measurement-Summary` have the same date, `Weight-Datapoints` wins.

## Running the pipeline

```bash
python run_all.py
```

This runs four steps in order:

1. `aggregate.py` — merges all CSVs into `data/processed/daily_unified.csv`
2. `tdee.py` — computes rolling TDEE estimates → `data/processed/tdee_results.csv`
3. `visualize.py` — generates main charts → `charts/`
4. `visualize_extra.py` — generates supplemental charts → `charts/`

Open `charts/chart_full_dashboard.html` when done.

## Running individual steps

```bash
python aggregate.py       # rebuild daily_unified.csv only
python tdee.py            # rebuild tdee_results.csv (requires daily_unified.csv)
python visualize.py       # rebuild main charts (requires tdee_results.csv)
python visualize_extra.py # rebuild supplemental charts (requires both processed CSVs)
```

## Output charts

| File | Contents |
|---|---|
| `chart_full_dashboard.html` | All charts combined — start here |
| `chart_dashboard.html` | 3-panel: weight · TDEE · calories & exercise |
| `chart_weight.html` | Weight progress with rolling averages |
| `chart_tdee.html` | TDEE estimates and calorie breakdown |
| `chart_tdee_trend.html` | TDEE trend over time (metabolic adaptation) |
| `chart_dow_patterns.html` | Calorie patterns by day of week |
| `chart_projection.html` | Weight goal projection with confidence bands |
| `chart_calorie_targets.html` | Recommended intake targets toward goal weight |

## Project structure

```
data/
  exports/      ← drop new export CSVs here
  processed/    ← auto-generated intermediates (gitignored)
charts/         ← auto-generated HTML output (gitignored)
archive/        ← one-off scripts and old outputs
aggregate.py
tdee.py
visualize.py
visualize_extra.py
run_all.py
requirements.txt
```
