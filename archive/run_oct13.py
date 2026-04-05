"""
run_oct13.py — Generate a comparison dashboard using data from Oct 13, 2025 onwards.

Excludes the early spotty data period (Sep 3 – Oct 12, 2025).
Writes: chart_full_dashboard_oct13.html

Note: the individual chart HTML files (chart_weight.html, chart_tdee.html, etc.)
will be overwritten with oct13 versions as a side effect. Run 'python visualize.py'
afterwards to restore them to the full-period versions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR   = Path(__file__).parent
START_DATE = '2025-10-13'


def build_oct13_tdee(unified):
    """Recompute all tdee_results columns on the oct13-filtered unified data."""
    from tdee import rolling_tdee, CALS_PER_LB

    df = unified.copy()
    r  = pd.DataFrame(index=df.index)

    r['weight_lbs']          = df['weight_lbs']
    r['weight_raw']          = df['weight_raw']
    r['calories_in']         = df['Calories']
    r['protein_g']           = df.get('Protein (g)')
    r['carbs_g']             = df.get('Carbohydrates (g)')
    r['fat_g']               = df.get('Fat (g)')
    r['exercise_calories']   = df['exercise_calories']
    r['exercise_minutes']    = df['exercise_minutes']
    r['net_calories']        = df['net_calories']
    r['has_estimated_meals'] = df['has_estimated_meals'].fillna(False)

    # Rolling weight averages
    r['weight_7d_avg']  = df['weight_lbs'].rolling(7,  center=True, min_periods=4).mean()
    r['weight_14d_avg'] = df['weight_lbs'].rolling(14, center=True, min_periods=7).mean()

    # Rolling calorie averages
    r['calories_7d_avg']  = df['Calories'].rolling(7,  min_periods=5).mean()
    r['calories_14d_avg'] = df['Calories'].rolling(14, min_periods=10).mean()

    # Rolling exercise averages
    r['exercise_7d_avg']  = df['exercise_calories'].rolling(7,  min_periods=4).mean()
    r['exercise_14d_avg'] = df['exercise_calories'].rolling(14, min_periods=7).mean()

    # TDEE rolling windows
    r['tdee_7d']  = rolling_tdee(df, window_days=7)
    r['tdee_14d'] = rolling_tdee(df, window_days=14)

    r['tdee_7d_smoothed']  = r['tdee_7d'].rolling(7,  min_periods=4).mean()
    r['tdee_14d_smoothed'] = r['tdee_14d'].rolling(7, min_periods=4).mean()
    r['tdee_30d_avg']      = r['tdee_14d'].rolling(30, min_periods=15).mean()

    # Exercise-adjusted TDEE
    r['tdee_adj_14d']          = r['tdee_14d'] - r['exercise_14d_avg']
    r['tdee_adj_14d_smoothed'] = r['tdee_adj_14d'].rolling(7,  min_periods=4).mean()
    r['tdee_adj_30d_avg']      = r['tdee_adj_14d'].rolling(30, min_periods=15).mean()

    # Low-activity flag
    ex_median        = r['exercise_14d_avg'].median()
    r['low_activity'] = r['exercise_14d_avg'] < (ex_median * 0.30)

    # Deficit / weekly change
    r['daily_deficit_14d']   = r['tdee_14d'] - r['calories_in']
    r['deficit_7d_avg']      = r['daily_deficit_14d'].rolling(7, min_periods=4).mean()
    r['weekly_weight_change'] = r['weight_7d_avg'].diff(7)

    return r


def main():
    from visualize import load_data, chart_full_dashboard

    unified_full, _ = load_data()

    # Filter to Oct 13+
    unified = unified_full[unified_full.index >= START_DATE].copy()
    print(f"Filtered to {START_DATE}+: {len(unified)} days "
          f"({unified.index.min().date()} to {unified.index.max().date()})")

    tdee = build_oct13_tdee(unified)

    # Print a quick comparison to the full-period stats
    tdee14 = tdee['tdee_14d'].dropna()
    print(f"\nOct-13 window TDEE (14-day rolling):")
    print(f"  Mean:   {tdee14.mean():.0f} cal/day")
    print(f"  Median: {tdee14.median():.0f} cal/day")
    print(f"  Range:  {tdee14.min():.0f} - {tdee14.max():.0f} cal/day")

    out = DATA_DIR / 'chart_full_dashboard_oct13.html'
    chart_full_dashboard(unified, tdee, output_path=out)
    print(f"\nOpen chart_full_dashboard_oct13.html to compare.")
    print("Run 'python visualize.py' to restore the individual chart files.")


if __name__ == '__main__':
    main()
