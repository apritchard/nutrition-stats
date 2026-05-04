"""
tdee.py — Calculate rolling TDEE estimates from unified daily data.

Reads:  daily_unified.csv
Writes: tdee_results.csv

TDEE Method (energy balance):
  Over any window of N days:
    total_TDEE = calories_consumed - (weight_change_lbs × 3500)
    daily_TDEE = total_TDEE / N

  If you ate 28,000 cal over 14 days and lost 1 lb (3,500 cal deficit),
  your TDEE was (28,000 + 3,500) / 14 ≈ 2,250 cal/day.

  Windows containing any day with missing nutrition data are skipped (NaN),
  rather than imputed, to avoid silently hiding data gaps in TDEE estimates.

  Exercise calories are not used in this formula — TDEE is derived purely from
  food intake and body weight change, which already captures exercise's effect
  on total energy expenditure. Exercise data is preserved here for visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import CALS_PER_LB, DOB, HEIGHT_CM, SEX, MSJ_ACTIVITY_MULTIPLIER

DATA_DIR      = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / 'data' / 'processed'


def rolling_tdee(df, window_days, weight_col='weight_smooth'):
    """
    Compute rolling TDEE using a trailing window.

    Returns a Series aligned to df.index. The value on day i represents
    the average daily TDEE over the window ending on day i.
    Windows with any NaN in Calories or the weight column are returned as NaN.

    weight_col should be a pre-smoothed weight series (default: 'weight_smooth')
    to prevent noisy endpoint weigh-ins from swinging the TDEE calculation.
    """
    calories = df['Calories'].values
    weight = df[weight_col].values
    n = len(df)
    result = np.full(n, np.nan)

    for i in range(window_days - 1, n):
        cal_window = calories[i - window_days + 1 : i + 1]
        wt_window = weight[i - window_days + 1 : i + 1]

        if np.any(np.isnan(cal_window)) or np.any(np.isnan(wt_window)):
            continue

        total_cals = cal_window.sum()
        weight_change = wt_window[-1] - wt_window[0]  # negative = weight lost
        total_tdee = total_cals - (weight_change * CALS_PER_LB)
        result[i] = total_tdee / window_days

    return pd.Series(result, index=df.index)


def build_tdee_results():
    df = pd.read_csv(PROCESSED_DIR / 'daily_unified.csv', index_col='Date', parse_dates=True)

    results = pd.DataFrame(index=df.index)
    results['weight_lbs']           = df['weight_lbs']
    results['weight_raw']           = df['weight_raw']
    results['calories_in']          = df['Calories']
    results['protein_g']            = df.get('Protein (g)')
    results['carbs_g']              = df.get('Carbohydrates (g)')
    results['fat_g']                = df.get('Fat (g)')
    results['exercise_calories']    = df['exercise_calories']
    results['exercise_minutes']     = df['exercise_minutes']
    results['net_calories']         = df['net_calories']
    results['has_estimated_meals']  = df['has_estimated_meals'].fillna(False)

    # Rolling weight averages (centered for visual accuracy)
    results['weight_7d_avg']  = df['weight_lbs'].rolling(7,  center=True, min_periods=4).mean()
    results['weight_14d_avg'] = df['weight_lbs'].rolling(14, center=True, min_periods=7).mean()

    # Trailing smoothed weight used as TDEE input — prevents noisy endpoint weigh-ins
    # (post-workout water loss, recovery bounce) from swinging the TDEE calculation.
    # A single 0.5 lb endpoint error moves 7d TDEE by ~250 cal/day; smoothing removes this.
    df['weight_smooth'] = df['weight_lbs'].rolling(7, min_periods=4).mean()

    # Rolling calorie intake averages
    results['calories_7d_avg']  = df['Calories'].rolling(7,  min_periods=5).mean()
    results['calories_14d_avg'] = df['Calories'].rolling(14, min_periods=10).mean()

    # Rolling exercise averages (trailing to match tdee windows)
    results['exercise_7d_avg']  = df['exercise_calories'].rolling(7,  min_periods=4).mean()
    results['exercise_14d_avg'] = df['exercise_calories'].rolling(14, min_periods=7).mean()

    # TDEE estimates (trailing window — represents TDEE as of the end of the window)
    results['tdee_7d']  = rolling_tdee(df, window_days=7)
    results['tdee_14d'] = rolling_tdee(df, window_days=14)

    # Smooth the noisy 7-day estimate with a further 7-day rolling average
    results['tdee_7d_smoothed']  = results['tdee_7d'].rolling(7,  min_periods=4).mean()
    results['tdee_14d_smoothed'] = results['tdee_14d'].rolling(7, min_periods=4).mean()

    # 30-day rolling average of gross TDEE — slowest/smoothest trend view
    results['tdee_30d_avg'] = results['tdee_14d'].rolling(30, min_periods=15).mean()

    # Daily calorie deficit (positive = deficit, negative = surplus)
    results['daily_deficit_14d'] = results['tdee_14d'] - results['calories_in']
    results['deficit_7d_avg']    = results['daily_deficit_14d'].rolling(7, min_periods=4).mean()

    # Weekly weight change (lbs/week) from rolling avg — useful for progress tracking
    results['weekly_weight_change'] = (
        results['weight_7d_avg'].diff(7)  # change over 7 days
    )

    # Mifflin-St Jeor TDEE reference line
    # BMR (male)   = 10×kg + 6.25×cm − 5×age + 5
    # BMR (female) = 10×kg + 6.25×cm − 5×age − 161
    # TDEE = BMR × 1.2 (sedentary baseline) + 14-day rolling exercise avg
    # Weight input: weight_smooth (same 7-day trailing avg used by rolling_tdee)
    # Age input: calculated per row from DOB so birthday transitions are captured
    if DOB and HEIGHT_CM and SEX:
        dob_ts  = pd.Timestamp(DOB)
        ages    = ((df.index - dob_ts).days / 365.25).astype(int)
        wt_kg   = df['weight_smooth'] / 2.20462
        offset  = 5 if SEX == 'male' else -161
        mst_bmr = 10 * wt_kg + 6.25 * HEIGHT_CM - 5 * ages + offset
        results['mst_tdee_static'] = mst_bmr * MSJ_ACTIVITY_MULTIPLIER

    results.to_csv(PROCESSED_DIR / 'tdee_results.csv')
    print("tdee_results.csv written")

    # Summary
    tdee14 = results['tdee_14d'].dropna()
    tdee7  = results['tdee_7d_smoothed'].dropna()
    if len(tdee14):
        print(f"\n14-day rolling TDEE ({len(tdee14)} data points):")
        print(f"  Mean:   {tdee14.mean():.0f} cal/day")
        print(f"  Median: {tdee14.median():.0f} cal/day")
        print(f"  Range:  {tdee14.min():.0f} – {tdee14.max():.0f} cal/day")

    deficit = results['daily_deficit_14d'].dropna()
    if len(deficit):
        print(f"\nAverage daily deficit (14-day TDEE basis): {deficit.mean():.0f} cal/day")
        implied_loss = deficit.mean() * len(df) / CALS_PER_LB
        print(f"Implied total weight loss over period:     {implied_loss:.1f} lbs")

    return results


if __name__ == '__main__':
    build_tdee_results()
