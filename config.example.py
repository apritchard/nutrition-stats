"""
config.example.py — Template for personal configuration.

Copy this file to config.py and fill in your own values.
config.py is gitignored so your personal data stays local.

    cp config.example.py config.py
"""

# ---------------------------------------------------------------------------
# Goal settings
# ---------------------------------------------------------------------------

# Your target body weight in lbs
GOAL_WEIGHT = 175

# Intermediate weight milestones to annotate on the projection chart
GOAL_WEIGHT_MILESTONES = [200, 190, 180, 175]

# Estimated TDEE (cal/day) at your goal weight with moderate exercise.
# Calculate using Mifflin-St Jeor + activity multiplier, or use an online TDEE calculator.
# Example for a 175 lb male with moderate exercise: ~2743 cal/day
GOAL_TDEE = 2743

# Calorie deficit to maintain once you reach your goal weight (cal/day).
# 250 = ~0.5 lbs/week — a gentle maintenance buffer.
GOAL_DEFICIT_AT_TARGET = 250

# Minimum daily calorie intake — the simulation will never recommend below this.
# Adjust based on your size, medical guidance, and activity level.
SAFE_FLOOR = 1500

# ---------------------------------------------------------------------------
# Projection settings
# ---------------------------------------------------------------------------

# How many recent days of smoothed weight to fit the trend line from
PROJECTION_LOOKBACK_DAYS = 60

# How many days forward to project weight
PROJECTION_DAYS_FORWARD = 270

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

# Calories per pound of body fat — used in all energy balance calculations.
# The commonly cited value is 3500; some research suggests 3400–3600.
CALS_PER_LB = 3500

# Meal names as they appear in your nutrition export.
# A "complete" day requires all four of these to be logged.
# Partial days have missing meals estimated from per-meal averages.
STANDARD_MEALS = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']

# Days where 14-day rolling exercise average falls below this fraction of the
# overall median are flagged as low-activity periods (shaded on TDEE trend chart).
LOW_ACTIVITY_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Macro targets (grams/day) — set to None to omit the target line from charts
# ---------------------------------------------------------------------------

PROTEIN_TARGET_G = None   # e.g. 180 — grams of protein per day
CARBS_TARGET_G   = None   # e.g. 200 — grams of carbohydrates per day
FAT_TARGET_G     = None   # e.g. 70  — grams of fat per day

# ---------------------------------------------------------------------------
# Biometrics — used for Mifflin-St Jeor BMR reference line
# ---------------------------------------------------------------------------

# Date of birth as YYYY-MM-DD string — age is calculated dynamically per row
DOB        = None    # e.g. '1990-06-15'
HEIGHT_CM  = None    # e.g. 177.8 (= 70 inches / 5'10")
SEX        = None    # 'male' or 'female'

# Activity multiplier for the static MSJ reference line (BMR × multiplier, no exercise data).
# Common values: 1.2 sedentary, 1.375 light, 1.55 moderate, 1.725 active, 1.9 very active
MSJ_ACTIVITY_MULTIPLIER = 1.55
