"""
visualize.py — Generate interactive HTML charts from aggregated health data.

Reads:  daily_unified.csv, tdee_results.csv
Writes: chart_weight.html     — weight progress with rolling averages
        chart_tdee.html       — TDEE estimates + calorie intake breakdown
        chart_dashboard.html  — unified 3-panel view (weight, TDEE, calories/exercise)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from config import GOAL_WEIGHT
from style_config import COLORS, HOVER, TEMPLATE

DATA_DIR      = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / 'data' / 'processed'
CHARTS_DIR    = DATA_DIR / 'charts'


def load_data():
    unified = pd.read_csv(PROCESSED_DIR / 'daily_unified.csv', index_col='Date', parse_dates=True)
    tdee    = pd.read_csv(PROCESSED_DIR / 'tdee_results.csv',  index_col='Date', parse_dates=True)
    return unified, tdee


def chart_weight(unified, tdee):
    """Weight over time: raw measurements, 7-day and 14-day rolling averages."""
    fig = go.Figure()

    raw_mask = tdee['weight_raw'].notna()
    fig.add_trace(go.Scatter(
        x=tdee.index[raw_mask],
        y=tdee.loc[raw_mask, 'weight_raw'],
        mode='markers',
        name='Measured Weight',
        marker=dict(color=COLORS['weight_raw'], size=7, symbol='circle'),
        hovertemplate='%{x|%b %d}: <b>%{y:.1f} lbs</b><extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['weight_7d_avg'],
        mode='lines', name='7-Day Rolling Avg',
        line=dict(color=COLORS['weight_7d'], width=2),
        hovertemplate='7d avg: %{y:.1f} lbs<extra></extra>',
    ))

    # Shade region where meals were estimated (data quality indicator)
    est_days = tdee[tdee['has_estimated_meals'] == True].index
    if len(est_days):
        fig.add_trace(go.Scatter(
            x=est_days, y=tdee.loc[est_days, 'weight_lbs'],
            mode='markers', name='Day w/ Estimated Meals',
            marker=dict(color=COLORS['estimated'], size=4, symbol='diamond'),
            opacity=0.5,
            hovertemplate='Estimated meals: %{x|%b %d}<extra></extra>',
        ))

    first = tdee['weight_raw'].dropna()
    if len(first):
        start_w, end_w = first.iloc[0], first.iloc[-1]
        fig.add_annotation(
            x=0.01, y=0.05, xref='paper', yref='paper',
            text=f'Total loss: <b>{start_w - end_w:.1f} lbs</b>  ({start_w:.1f} → {end_w:.1f})',
            showarrow=False, bgcolor='white', bordercolor='#ccc', borderwidth=1,
            font=dict(size=12),
        )

    fig.update_layout(
        title='Weight Progress',
        xaxis_title='Date',
        yaxis_title='Weight (lbs)',
        hovermode=HOVER,
        template=TEMPLATE,
        margin=dict(b=100),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
    )

    out = CHARTS_DIR / 'chart_weight.html'
    fig.write_html(str(out))
    print(f"Wrote {out.name}")
    return fig


def chart_tdee(unified, tdee):
    """TDEE estimates + calorie intake and exercise breakdown."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Estimated TDEE (energy balance method)', 'Daily Calories In vs Exercise'),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )

    # --- TDEE estimates ---
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['tdee_7d_smoothed'],
        mode='lines', name='TDEE via 7-day window (smoothed)',
        line=dict(color=COLORS['tdee_7d'], width=2),
        hovertemplate='TDEE 7d: %{y:.0f} cal<extra></extra>',
    ), row=1, col=1)

    # 30-day rolling average
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['tdee_30d_avg'],
        mode='lines', name='TDEE 30-day avg',
        line=dict(color='#7B3F00', width=2.5),
        hovertemplate='TDEE 30d: %{y:.0f} cal<extra></extra>',
    ), row=1, col=1)

    # Reference: 7-day calorie intake avg
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['calories_7d_avg'],
        mode='lines', name='Avg Calories Eaten (7-day)',
        line=dict(color=COLORS['calories_avg'], width=1.5, dash='dot'),
        hovertemplate='Avg intake: %{y:.0f} cal<extra></extra>',
    ), row=1, col=1)

    # --- Calorie intake bars ---
    fig.add_trace(go.Bar(
        x=tdee.index, y=tdee['calories_in'],
        name='Calories Eaten', marker_color=COLORS['calories_in'], opacity=0.6,
        hovertemplate='%{x|%b %d} eaten: %{y:.0f} cal<extra></extra>',
    ), row=2, col=1)

    # --- Exercise calories bars (overlaid) ---
    fig.add_trace(go.Bar(
        x=tdee.index, y=tdee['exercise_calories'],
        name='Exercise Calories', marker_color=COLORS['exercise'], opacity=0.75,
        hovertemplate='%{x|%b %d} exercise: %{y:.0f} cal<extra></extra>',
    ), row=2, col=1)

    # Exercise 7-day avg line
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['exercise_7d_avg'],
        mode='lines', name='Exercise 7-Day Avg',
        line=dict(color=COLORS['exercise_avg'], width=2),
        hovertemplate='Ex 7d avg: %{y:.0f} cal<extra></extra>',
    ), row=2, col=1)

    fig.update_layout(
        title='TDEE & Calorie Analysis',
        hovermode=HOVER,
        template=TEMPLATE,
        barmode='overlay',
        height=700,
        margin=dict(b=120),
        legend=dict(orientation='h', yanchor='top', y=-0.10, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='cal/day', row=1, col=1)
    fig.update_yaxes(title_text='Calories', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    out = CHARTS_DIR / 'chart_tdee.html'
    fig.write_html(str(out))
    print(f"Wrote {out.name}")
    return fig


def chart_dashboard(unified, tdee):
    """Unified 3-panel dashboard: weight · TDEE · calories + exercise."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            'Weight Progress',
            'Estimated TDEE',
            'Calories In & Exercise',
        ),
        vertical_spacing=0.07,
        row_heights=[0.33, 0.34, 0.33],
    )

    # === Row 1: Weight ===
    raw_mask = tdee['weight_raw'].notna()
    fig.add_trace(go.Scatter(
        x=tdee.index[raw_mask], y=tdee.loc[raw_mask, 'weight_raw'],
        mode='markers', name='Measured Weight',
        marker=dict(color=COLORS['weight_raw'], size=5),
        hovertemplate='%{x|%b %d}: <b>%{y:.1f} lbs</b><extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['weight_7d_avg'],
        mode='lines', name='Weight 7-Day Avg',
        line=dict(color=COLORS['weight_7d'], width=2),
        hovertemplate='7d: %{y:.1f} lbs<extra></extra>',
    ), row=1, col=1)

    # === Row 2: TDEE ===
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['tdee_7d_smoothed'],
        mode='lines', name='TDEE (7-day)',
        line=dict(color=COLORS['tdee_7d'], width=2),
        hovertemplate='TDEE 7d: %{y:.0f}<extra></extra>',
    ), row=2, col=1)

    # 30-day rolling average
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['tdee_30d_avg'],
        mode='lines', name='TDEE 30-day avg',
        line=dict(color='#7B3F00', width=2.5),
        hovertemplate='TDEE 30d: %{y:.0f}<extra></extra>',
    ), row=2, col=1)

    # 7-day calorie intake avg for reference on TDEE panel
    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['calories_7d_avg'],
        mode='lines', name='Avg Intake (7-day)',
        line=dict(color=COLORS['calories_avg'], width=1.5, dash='dot'),
        hovertemplate='Avg intake: %{y:.0f}<extra></extra>',
    ), row=2, col=1)

    # === Row 3: Calories + Exercise ===
    fig.add_trace(go.Bar(
        x=tdee.index, y=tdee['calories_in'],
        name='Calories Eaten', marker_color=COLORS['calories_in'], opacity=0.55,
        hovertemplate='%{x|%b %d} eaten: %{y:.0f}<extra></extra>',
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=tdee.index, y=tdee['exercise_calories'],
        name='Exercise Calories', marker_color=COLORS['exercise'], opacity=0.8,
        hovertemplate='%{x|%b %d} exercise: %{y:.0f}<extra></extra>',
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=tdee.index, y=tdee['exercise_7d_avg'],
        mode='lines', name='Exercise 7-Day Avg',
        line=dict(color=COLORS['exercise_avg'], width=2),
        hovertemplate='Ex avg: %{y:.0f}<extra></extra>',
    ), row=3, col=1)

    fig.update_layout(
        title='<b>Health Dashboard</b> — Weight · TDEE · Calories',
        height=950,
        hovermode=HOVER,
        template=TEMPLATE,
        barmode='overlay',
        margin=dict(b=120),
        legend=dict(orientation='h', yanchor='top', y=-0.06, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='lbs',      row=1, col=1)
    fig.update_yaxes(title_text='cal/day',  row=2, col=1)
    fig.update_yaxes(title_text='calories', row=3, col=1)
    fig.update_xaxes(title_text='Date',     row=3, col=1)

    out = CHARTS_DIR / 'chart_dashboard.html'
    fig.write_html(str(out))
    print(f"Wrote {out.name}")
    return fig


def chart_full_dashboard(unified, tdee, output_path=None):
    """
    Assemble all charts into a single HTML page using plotly's to_html().
    output_path overrides the default DATA_DIR/chart_full_dashboard.html.
    """
    from visualize_extra import (
        chart_tdee_trend, chart_dow_patterns, chart_calorie_targets,
        chart_macros, chart_weekly_summary,
    )

    figs = {
        'cal_targets':      chart_calorie_targets(unified, tdee),
        'dashboard':        chart_dashboard(unified, tdee),
        'tdee_trend':       chart_tdee_trend(unified, tdee),
        'dow':              chart_dow_patterns(unified, tdee),
        'macros':           chart_macros(unified, tdee),
        'weekly_summary':   chart_weekly_summary(unified, tdee),
    }

    _first_chart = [True]  # load plotly.js CDN once via the first chart div

    def to_div(fig, height=None, div_id=None):
        if height:
            fig.update_layout(height=height)
        if _first_chart[0]:
            _first_chart[0] = False
            include = 'cdn'   # emits the correct versioned plotly.js CDN script
        else:
            include = False   # plotly.js already loaded; skip redundant tag
        kwargs = {'full_html': False, 'include_plotlyjs': include, 'config': {'responsive': True}}
        if div_id:
            kwargs['div_id'] = div_id
        return fig.to_html(**kwargs)

    date_min = tdee.index.min().strftime('%Y-%m-%d')
    date_max = tdee.index.max().strftime('%Y-%m-%d')

    # Chart IDs that share a date x-axis and respond to the global date filter
    LINKED_CHART_IDS = ['chart-dashboard', 'chart-tdee-trend', 'chart-macros', 'chart-weekly']

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Health Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f2f5;
      padding: 24px;
      color: #2c3e50;
    }}
    h1 {{
      font-size: 1.6em;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .subtitle {{
      font-size: 0.9em;
      color: #7f8c8d;
      margin-bottom: 16px;
    }}
    .date-filter {{
      display: flex;
      align-items: center;
      gap: 10px;
      background: white;
      border-radius: 10px;
      padding: 12px 18px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .date-filter label {{
      font-size: 0.85em;
      font-weight: 600;
      color: #555;
      white-space: nowrap;
    }}
    .date-filter input[type="date"] {{
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 0.85em;
      color: #2c3e50;
    }}
    .date-filter button {{
      border: none;
      border-radius: 6px;
      padding: 6px 14px;
      font-size: 0.85em;
      cursor: pointer;
      font-weight: 600;
    }}
    .btn-apply {{ background: #2E75B6; color: white; }}
    .btn-apply:hover {{ background: #1F4E79; }}
    .btn-reset {{ background: #eee; color: #555; }}
    .btn-reset:hover {{ background: #ddd; }}
    .date-filter .quick-btns {{ display: flex; gap: 6px; margin-left: 6px; flex-wrap: wrap; }}
    .date-filter .quick-btns button {{
      background: #f0f2f5; color: #2c3e50; border: 1px solid #ddd;
    }}
    .date-filter .quick-btns button:hover {{ background: #e0e4ea; }}
    .card {{
      background: white;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
      margin-bottom: 20px;
    }}
    .card h2 {{
      font-size: 1em;
      font-weight: 600;
      color: #555;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #eee;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .grid-2 .card {{ margin-bottom: 0; }}
    @media (max-width: 900px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Health Dashboard</h1>
  <p class="subtitle">Generated {pd.Timestamp.now().strftime('%B %d, %Y')} &mdash; data through {tdee.index.max().strftime('%B %d, %Y')}</p>

  <div class="date-filter">
    <label>Date range:</label>
    <input type="date" id="date-start" min="{date_min}" max="{date_max}" value="{date_min}">
    <span style="color:#999;font-size:0.85em">&rarr;</span>
    <input type="date" id="date-end"   min="{date_min}" max="{date_max}" value="{date_max}">
    <button class="btn-apply" onclick="applyDateFilter()">Apply</button>
    <button class="btn-reset" onclick="resetDateFilter()">Reset</button>
    <div class="quick-btns">
      <button onclick="setLast(30)">30d</button>
      <button onclick="setLast(60)">60d</button>
      <button onclick="setLast(90)">90d</button>
    </div>
  </div>

  <div class="card">
    <h2>Daily Overview &mdash; Weight &middot; TDEE &middot; Calories &amp; Exercise</h2>
    {to_div(figs['dashboard'], height=900, div_id='chart-dashboard')}
  </div>

  <div class="card">
    <h2>TDEE Trend &mdash; Metabolic Adaptation</h2>
    {to_div(figs['tdee_trend'], height=500, div_id='chart-tdee-trend')}
  </div>

  <div class="card">
    <h2>Calorie Patterns by Day of Week</h2>
    {to_div(figs['dow'], height=500)}
  </div>

  <div class="card">
    <h2>Recommended Calorie Targets &mdash; Goal: {GOAL_WEIGHT} lbs</h2>
    {to_div(figs['cal_targets'], height=520)}
  </div>

  <div class="card">
    <h2>Macronutrient Breakdown</h2>
    {to_div(figs['macros'], height=850, div_id='chart-macros')}
  </div>

  <div class="card">
    <h2>Week-by-Week Summary</h2>
    {to_div(figs['weekly_summary'], height=950, div_id='chart-weekly')}
  </div>

  <script>
    var LINKED_CHARTS = {str(LINKED_CHART_IDS)};
    var DATA_MIN = '{date_min}';
    var DATA_MAX = '{date_max}';

    function relayoutCharts(start, end) {{
      LINKED_CHARTS.forEach(function(id) {{
        var el = document.getElementById(id);
        if (el) Plotly.relayout(el, {{'xaxis.range': [start, end], 'xaxis.autorange': false}});
      }});
    }}

    function applyDateFilter() {{
      var start = document.getElementById('date-start').value;
      var end   = document.getElementById('date-end').value;
      if (!start || !end) return;
      relayoutCharts(start, end);
    }}

    function resetDateFilter() {{
      document.getElementById('date-start').value = DATA_MIN;
      document.getElementById('date-end').value   = DATA_MAX;
      LINKED_CHARTS.forEach(function(id) {{
        var el = document.getElementById(id);
        if (el) Plotly.relayout(el, {{'xaxis.autorange': true}});
      }});
    }}

    function setLast(days) {{
      var end   = new Date(DATA_MAX);
      var start = new Date(end);
      start.setDate(start.getDate() - days + 1);
      var fmt = function(d) {{ return d.toISOString().slice(0,10); }};
      document.getElementById('date-start').value = fmt(start);
      document.getElementById('date-end').value   = fmt(end);
      relayoutCharts(fmt(start), fmt(end));
    }}
  </script>
</body>
</html>"""

    out = Path(output_path) if output_path else CHARTS_DIR / 'chart_full_dashboard.html'
    out.write_text(html, encoding='utf-8')
    print(f"Wrote {out.name}")


def main():
    unified, tdee = load_data()
    chart_weight(unified, tdee)
    chart_tdee(unified, tdee)
    chart_dashboard(unified, tdee)
    chart_full_dashboard(unified, tdee)
    print("\nDone. Open chart_full_dashboard.html for the complete view.")


if __name__ == '__main__':
    main()
