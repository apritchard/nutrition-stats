"""
aggregate.py — Build unified daily dataframe from all CSV sources.

Outputs: daily_unified.csv

Columns in output:
  weight_lbs          — interpolated weight (linear, from Weight-Datapoints + Measurement-Summary)
  weight_raw          — measured weight on that day (NaN if not measured)
  Calories            — total daily calories (estimated if meals were missing)
  Fat (g), Protein (g), Carbohydrates (g), ... — summed macros/micros
  meals_logged        — how many of the 4 standard meals were recorded
  has_estimated_meals — True if any meals were imputed from averages
  estimated_calories_added — extra calories added via imputation
  exercise_calories   — total calories burned via exercise (0 if no log = rest day)
  exercise_minutes    — total exercise minutes
  exercise_sessions   — number of exercise entries logged
  net_calories        — Calories - exercise_calories
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR      = Path(__file__).parent
EXPORTS_DIR   = DATA_DIR / 'data' / 'exports'
PROCESSED_DIR = DATA_DIR / 'data' / 'processed'
STANDARD_MEALS = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']

# Nutrition columns to aggregate (all numeric columns except Date, Meal, Note)
SKIP_COLS = {'Date', 'Meal', 'Note'}


def load_weight():
    """Merge both weight sources, interpolate across the full date range."""
    wp_files = sorted(EXPORTS_DIR.glob('Weight-Datapoints*.csv'))
    wp = pd.concat([pd.read_csv(f) for f in wp_files]).drop_duplicates()
    wp = wp.dropna(subset=['Date', 'Weight'])
    wp['Date'] = pd.to_datetime(wp['Date'])
    wp = wp.set_index('Date')['Weight']

    ms_files = sorted(EXPORTS_DIR.glob('Measurement-Summary-*.csv'))
    ms = pd.concat([pd.read_csv(f) for f in ms_files]).drop_duplicates()
    ms = ms.dropna(subset=['Date', 'Weight'])
    ms['Date'] = pd.to_datetime(ms['Date'])
    ms = ms.set_index('Date')['Weight']

    # Weight-Datapoints is primary on any date where both sources have a value
    # (it was manually curated and wins conflicts). Measurement-Summary fills all
    # other dates — including the historical gap it uniquely covers, and all future
    # data going forward now that Weight-Datapoints is no longer being updated.
    combined = wp.combine_first(ms).sort_index()

    full_range = pd.date_range(combined.index.min(), combined.index.max(), freq='D')
    interpolated = combined.reindex(full_range).interpolate(method='linear')
    interpolated.index.name = 'Date'

    return interpolated, combined  # (interpolated series, raw measured points)


def load_nutrition():
    """
    Aggregate nutrition to one row per day.

    Incomplete days (fewer than 4 meals logged) have the missing meals estimated
    using the per-meal-type average across all fully-logged days. The amount added
    and a flag are recorded so downstream code can treat these days appropriately.

    Days with zero meals logged are left as NaN (not imputed).
    """
    files = sorted(EXPORTS_DIR.glob('Nutrition-Summary-*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Date'])

    numeric_cols = [c for c in df.columns if c not in SKIP_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Per-meal-type averages for imputation — computed only from fully-logged days
    full_days = df.groupby('Date')['Meal'].apply(set)
    full_day_dates = full_days[full_days.apply(lambda s: set(STANDARD_MEALS).issubset(s))].index
    meal_avgs = df[df['Date'].isin(full_day_dates)].groupby('Meal')[numeric_cols].mean()

    rows = []
    for date, group in df.groupby('Date'):
        logged_meals = set(group['Meal'])
        missing_meals = [m for m in STANDARD_MEALS if m not in logged_meals]

        day_totals = group[numeric_cols].sum()
        estimated = False
        estimated_calories_added = 0.0

        for meal in missing_meals:
            if meal in meal_avgs.index:
                day_totals = day_totals.add(meal_avgs.loc[meal])
                estimated_calories_added += meal_avgs.loc[meal].get('Calories', 0)
                estimated = True

        row = day_totals.to_dict()
        row['Date'] = date
        row['meals_logged'] = len(logged_meals)
        row['has_estimated_meals'] = estimated
        row['estimated_calories_added'] = round(estimated_calories_added, 1)
        rows.append(row)

    result = pd.DataFrame(rows).set_index('Date').sort_index()
    return result


def load_exercise():
    """
    Aggregate exercise to one row per day.

    Note on exercise calories: the app likely reports net additional calories burned
    above baseline (not gross calories including BMR during exercise time). Either way,
    the TDEE calculation in tdee.py derives TDEE from food intake + weight change and
    does not require exercise calories — they are used here for visualization only.
    """
    files = sorted(EXPORTS_DIR.glob('Exercise-Summary-*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Date'])

    df['Exercise Calories'] = pd.to_numeric(df['Exercise Calories'], errors='coerce').fillna(0)
    df['Exercise Minutes'] = pd.to_numeric(df['Exercise Minutes'], errors='coerce').fillna(0)

    daily = df.groupby('Date').agg(
        exercise_calories=('Exercise Calories', 'sum'),
        exercise_minutes=('Exercise Minutes', 'sum'),
        exercise_sessions=('Exercise', 'count'),
    )
    return daily


def build_unified():
    """Merge all sources into a single daily CSV."""
    weight_interp, weight_raw = load_weight()
    nutrition = load_nutrition()
    exercise = load_exercise()

    # Analysis window: where we have both weight and nutrition
    start = max(nutrition.index.min(), weight_interp.index.min())
    end = min(nutrition.index.max(), weight_interp.index.max())
    date_range = pd.date_range(start, end, freq='D')

    unified = pd.DataFrame(index=date_range)
    unified.index.name = 'Date'

    unified['weight_lbs'] = weight_interp.reindex(date_range)
    unified['weight_raw'] = weight_raw.reindex(date_range)

    unified = unified.join(nutrition, how='left')
    unified = unified.join(exercise, how='left')

    unified['exercise_calories'] = unified['exercise_calories'].fillna(0)
    unified['exercise_minutes'] = unified['exercise_minutes'].fillna(0)
    unified['exercise_sessions'] = unified['exercise_sessions'].fillna(0).astype(int)
    unified['net_calories'] = unified['Calories'] - unified['exercise_calories']

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(PROCESSED_DIR / 'daily_unified.csv')

    total_days = len(unified)
    days_with_nutrition = unified['Calories'].notna().sum()
    days_estimated = unified['has_estimated_meals'].fillna(False).sum()
    days_no_nutrition = total_days - days_with_nutrition

    print(f"daily_unified.csv written: {total_days} days ({start.date()} to {end.date()})")
    print(f"  Nutrition data:     {days_with_nutrition} days logged")
    print(f"  Meals imputed:      {days_estimated} days had 1+ missing meal estimated")
    print(f"  No nutrition data:  {days_no_nutrition} days (will be NaN — excluded from TDEE windows)")
    print(f"  Exercise data:      {(unified['exercise_calories'] > 0).sum()} days with exercise logged")

    return unified


if __name__ == '__main__':
    build_unified()
