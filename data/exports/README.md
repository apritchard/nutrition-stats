# Data Exports

Drop your MyFitnessPal export CSVs into this folder, then run `python run_all.py` from the repo root to regenerate all charts.

## MyFitnessPal exports

From the MyFitnessPal app or website, export and place the following files here:

- **`Nutrition-Summary-<start>-to-<end>.csv`** — meal-level nutrition data (Breakfast, Lunch, Dinner, Snacks)
- **`Exercise-Summary-<start>-to-<end>.csv`** — cardio and strength exercise logs
- **`Measurement-Summary-<start>-to-<end>.csv`** — body weight measurements (primary weight source)

The pipeline auto-discovers all files matching each pattern, so you can drop in a new export covering a newer date range without removing old ones. Overlapping rows are deduplicated automatically.

## Supplemental weight data (optional)

If you have weight estimates from outside MyFitnessPal (e.g. manual daily weigh-ins), add them to:

- **`Weight-Datapoints.csv`** — two columns: `Date`, `Weight`

Example:
```
Date,Weight
9/3/2025,250
9/4/2025,249.6
```

`Weight-Datapoints` and `Measurement-Summary` are merged automatically. Where both sources have an entry for the same date, `Weight-Datapoints` takes priority.
