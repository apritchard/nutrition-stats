"""
style_config.py — Visual style settings for all charts.

Safe to commit. Edit freely to change colors, chart theme, or hover behavior.
"""

# Plotly chart theme and hover mode
TEMPLATE = 'plotly_white'
HOVER    = 'x unified'

# Color palette — used across visualize.py and visualize_extra.py
COLORS = {
    # Weight
    'weight_raw':         '#5B9BD5',
    'weight_7d':          '#2E75B6',
    'weight_14d':         '#1F4E79',

    # TDEE
    'tdee_7d':            '#ED7D31',
    'tdee_14d':           '#C55A11',
    'tdee_30d':           '#7B3F00',

    # Calories
    'calories_in':        '#FF6B6B',
    'calories_avg':       '#C0392B',

    # Exercise
    'exercise':           '#27AE60',
    'exercise_avg':       '#1E8449',

    # Deficit / estimated meals
    'deficit':            '#8E44AD',
    'estimated':          '#F39C12',

    # Trend lines
    'ols_trend':          '#C55A11',
    'ts_trend':           '#C00000',
    'ts_adj_trend':       '#1F4E79',

    # Net calories
    'net_calories':       '#2C3E50',

    # Projection
    'projection':         '#C00000',
    'projection_band':    'rgba(192,0,0,0.08)',

    # Goal / recommendation
    'goal_line':          '#27AE60',
    'goal_label':         '#1E8449',
    'intake_rec':         '#27AE60',

    # Low-activity shading
    'low_activity_fill':  'rgba(100,100,100,0.10)',
    'low_activity_text':  '#666666',

    # Reference lines
    'today_line':         '#aaaaaa',
    'safety_floor':       '#C00000',

    # Mifflin-St Jeor formula reference
    'mst_tdee':           '#00ACC1',
    'mst_tdee_static':    '#0057A8',

    # Macronutrients
    'macro_protein':      '#9B59B6',
    'macro_protein_fill': 'rgba(155, 89, 182, 0.4)',
    'macro_carbs':        '#E67E22',
    'macro_carbs_fill':   'rgba(230, 126, 34, 0.4)',
    'macro_fat':          '#16A085',
    'macro_fat_fill':     'rgba(22, 160, 133, 0.4)',
}
