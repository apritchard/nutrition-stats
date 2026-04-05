"""
visualize_extra.py — Three additional insight charts.

Reads:  daily_unified.csv, tdee_results.csv
Writes: chart_tdee_trend.html   — TDEE over time with linear trend (metabolic adaptation)
        chart_dow_patterns.html — Calorie patterns by day of week
        chart_projection.html   — Weight goal projection with confidence bands
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import timedelta

DATA_DIR      = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / 'data' / 'processed'
CHARTS_DIR    = DATA_DIR / 'charts'
TEMPLATE = 'plotly_white'
HOVER    = 'x unified'

# Target weights to annotate on the projection chart
GOAL_WEIGHTS = [200, 190, 180, 175]

# How many days of recent smoothed weight to fit the trend from
PROJECTION_LOOKBACK_DAYS = 60
# How many days forward to project
PROJECTION_DAYS_FORWARD  = 270


def load_data():
    unified = pd.read_csv(PROCESSED_DIR / 'daily_unified.csv', index_col='Date', parse_dates=True)
    tdee    = pd.read_csv(PROCESSED_DIR / 'tdee_results.csv',  index_col='Date', parse_dates=True)
    return unified, tdee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ols_fit(x, y):
    """Return (b0, b1, residual_std, Sxx, x_mean) for y = b0 + b1*x."""
    n      = len(x)
    x_mean = x.mean()
    Sxx    = ((x - x_mean) ** 2).sum()
    b1     = ((x - x_mean) * (y - y.mean())).sum() / Sxx
    b0     = y.mean() - b1 * x_mean
    y_hat  = b0 + b1 * x
    s      = np.sqrt(((y - y_hat) ** 2).sum() / max(n - 2, 1))
    return b0, b1, s, Sxx, x_mean


def theil_sen_fit(x, y):
    """
    Robust linear regression via Theil-Sen estimator.

    Slope = median of all pairwise slopes. Intercept = median(y) - b1*median(x).
    Automatically down-weights outlier periods (e.g. holiday exercise breaks)
    without needing to identify them manually. O(n^2) — fine for n < 500.
    """
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    b1 = np.median(slopes)
    b0 = np.median(y) - b1 * np.median(x)
    return b0, b1


def find_low_activity_spans(low_activity_series):
    """Return list of (start_date, end_date) for contiguous low-activity runs."""
    spans = []
    in_span = False
    start = None
    for date, val in low_activity_series.items():
        if val and not in_span:
            in_span = True
            start = date
        elif not val and in_span:
            spans.append((start, date))
            in_span = False
    if in_span:
        spans.append((start, low_activity_series.index[-1]))
    return spans


def prediction_interval(x_future, b0, b1, s, Sxx, x_mean, n, z=1.96):
    """95% prediction interval at each point in x_future."""
    y_pred = b0 + b1 * x_future
    se     = s * np.sqrt(1 + 1/n + (x_future - x_mean)**2 / Sxx)
    return y_pred, y_pred - z * se, y_pred + z * se


# ---------------------------------------------------------------------------
# Chart 1: TDEE trend over time
# ---------------------------------------------------------------------------

def chart_tdee_trend(unified, tdee):
    """
    TDEE trend chart with three improvements over the original:

    1. Exercise-adjusted TDEE series (raw TDEE minus rolling exercise avg) isolates
       baseline metabolic rate from fluctuating activity levels.
    2. Theil-Sen robust trend line alongside OLS — Theil-Sen automatically ignores
       outlier periods (e.g. holiday exercise breaks) by using the median slope.
    3. Low-activity periods are shaded so it's visually clear where exercise dipped
       and why the raw TDEE trend was pulled down.
    """
    raw_series = tdee['tdee_30d_avg'].dropna()
    adj_series = tdee['tdee_adj_30d_avg'].dropna()

    if len(raw_series) < 10:
        print("Not enough TDEE data for trend chart.")
        return

    # Fit both OLS and Theil-Sen to the RAW series
    x_raw = np.arange(len(raw_series))
    y_raw = raw_series.values
    b0_ols, b1_ols, s_ols, Sxx, x_mean = ols_fit(x_raw, y_raw)
    b0_ts,  b1_ts                       = theil_sen_fit(x_raw, y_raw)

    # Fit Theil-Sen to the ADJUSTED series (exercise removed)
    # Align to same index as raw for comparability
    adj_aligned = adj_series.reindex(raw_series.index)
    valid_adj   = adj_aligned.dropna()
    x_adj = np.array([raw_series.index.get_loc(d) for d in valid_adj.index], dtype=float)
    y_adj = valid_adj.values
    b0_adj, b1_adj = theil_sen_fit(x_adj, y_adj)

    def fmt_slope(b1):
        monthly = b1 * 30
        direction = "declining" if b1 < 0 else "increasing"
        return f"{direction} {abs(monthly):.0f} cal/month"

    fig = go.Figure()

    # --- Shade low-activity periods ---
    low_act_spans = find_low_activity_spans(tdee['low_activity'].fillna(False))
    for i, (start, end) in enumerate(low_act_spans):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='rgba(100,100,100,0.10)',
            line_width=0,
            annotation_text='Low activity' if i == 0 else '',
            annotation_position='top left',
            annotation_font=dict(size=10, color='#666'),
        )

    # --- Raw TDEE ---
    fig.add_trace(go.Scatter(
        x=raw_series.index, y=raw_series.values,
        mode='lines', name='TDEE gross (14d smoothed)',
        line=dict(color='#ED7D31', width=2),
        hovertemplate='%{x|%b %d} gross: %{y:.0f} cal<extra></extra>',
    ))

    # --- Exercise-adjusted TDEE ---
    fig.add_trace(go.Scatter(
        x=adj_series.index, y=adj_series.values,
        mode='lines', name='TDEE ex-adjusted (14d smoothed)',
        line=dict(color='#5B9BD5', width=2),
        hovertemplate='%{x|%b %d} adj: %{y:.0f} cal<extra></extra>',
    ))

    # --- OLS trend on raw (the original, outlier-sensitive line) ---
    ols_y = b0_ols + b1_ols * x_raw
    fig.add_trace(go.Scatter(
        x=raw_series.index, y=ols_y,
        mode='lines', name=f'OLS trend (gross): {fmt_slope(b1_ols)}',
        line=dict(color='#C55A11', width=1.5, dash='dash'),
        hovertemplate='OLS: %{y:.0f} cal<extra></extra>',
    ))

    # --- Theil-Sen trend on raw (robust, ignores holiday outliers) ---
    ts_y = b0_ts + b1_ts * x_raw
    fig.add_trace(go.Scatter(
        x=raw_series.index, y=ts_y,
        mode='lines', name=f'Theil-Sen trend (gross): {fmt_slope(b1_ts)}',
        line=dict(color='#C00000', width=2.5, dash='longdash'),
        hovertemplate='Theil-Sen: %{y:.0f} cal<extra></extra>',
    ))

    # --- Theil-Sen trend on exercise-adjusted ---
    adj_trend_y = b0_adj + b1_adj * x_raw
    fig.add_trace(go.Scatter(
        x=raw_series.index, y=adj_trend_y,
        mode='lines', name=f'Theil-Sen trend (ex-adj): {fmt_slope(b1_adj)}',
        line=dict(color='#1F4E79', width=2.5, dash='longdash'),
        hovertemplate='Adj trend: %{y:.0f} cal<extra></extra>',
    ))

    # --- Summary annotation ---
    ols_diff  = (b0_ols + b1_ols * x_raw[-1]) - (b0_ols + b1_ols * x_raw[0])
    ts_diff   = (b0_ts  + b1_ts  * x_raw[-1]) - (b0_ts  + b1_ts  * x_raw[0])
    adj_diff  = (b0_adj + b1_adj * x_raw[-1]) - (b0_adj + b1_adj * x_raw[0])
    fig.add_annotation(
        x=0.99, y=0.97, xref='paper', yref='paper', align='left',
        xanchor='right',
        text=(
            f"<b>Trend comparison (start to end)</b><br>"
            f"OLS gross:        {ols_diff:+.0f} cal total ({b1_ols*30:+.0f}/month)<br>"
            f"Theil-Sen gross:  {ts_diff:+.0f} cal total ({b1_ts*30:+.0f}/month)<br>"
            f"Theil-Sen ex-adj: {adj_diff:+.0f} cal total ({b1_adj*30:+.0f}/month)"
        ),
        showarrow=False,
        bgcolor='white', bordercolor='#ccc', borderwidth=1,
        font=dict(size=11, family='monospace'),
        valign='top',
    )

    fig.update_layout(
        title='<b>TDEE Trend — Gross vs Exercise-Adjusted, OLS vs Robust (Theil-Sen)</b>',
        xaxis_title='Date',
        yaxis_title='Estimated TDEE (cal/day)',
        hovermode=HOVER,
        template=TEMPLATE,
        height=550,
        margin=dict(b=130),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
    )

    out = CHARTS_DIR / 'chart_tdee_trend.html'
    fig.write_html(str(out))
    print(
        f"Wrote {out.name}\n"
        f"  OLS gross:        {fmt_slope(b1_ols)} ({b1_ols*7:+.1f} cal/week)\n"
        f"  Theil-Sen gross:  {fmt_slope(b1_ts)}  ({b1_ts*7:+.1f} cal/week)\n"
        f"  Theil-Sen ex-adj: {fmt_slope(b1_adj)} ({b1_adj*7:+.1f} cal/week)"
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Day-of-week patterns
# ---------------------------------------------------------------------------

def chart_dow_patterns(unified, tdee):
    """
    Average calories eaten, exercise calories, and net calories by day of week.
    Also shows exercise minutes by day. Individual-day scatter overlaid on bars.
    """
    DOW_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    DOW_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    df = unified[['Calories', 'exercise_calories', 'exercise_minutes', 'net_calories']].copy()
    df = df[df['Calories'].notna()]
    df['dow'] = pd.Categorical(df.index.day_name(), categories=DOW_ORDER, ordered=True)

    stats = df.groupby('dow', observed=True).agg(
        cal_mean=('Calories', 'mean'),
        cal_std=('Calories', 'std'),
        ex_mean=('exercise_calories', 'mean'),
        ex_std=('exercise_calories', 'std'),
        net_mean=('net_calories', 'mean'),
        exmin_mean=('exercise_minutes', 'mean'),
        count=('Calories', 'count'),
    ).reindex(DOW_ORDER)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Average Daily Calories by Day of Week',
            'Average Exercise Minutes by Day of Week',
        ),
        vertical_spacing=0.18,
        row_heights=[0.6, 0.4],
    )

    # --- Row 1: calorie bars ---
    fig.add_trace(go.Bar(
        x=DOW_SHORT, y=stats['cal_mean'],
        name='Calories Eaten',
        marker_color='#FF6B6B', opacity=0.85,
        error_y=dict(type='data', array=stats['cal_std'], visible=True, color='#c0392b'),
        hovertemplate='%{x} eaten: <b>%{y:.0f}</b> cal (avg)<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=DOW_SHORT, y=stats['ex_mean'],
        name='Exercise Calories',
        marker_color='#27AE60', opacity=0.85,
        error_y=dict(type='data', array=stats['ex_std'], visible=True, color='#1E8449'),
        hovertemplate='%{x} exercise: <b>%{y:.0f}</b> cal (avg)<extra></extra>',
    ), row=1, col=1)

    # Net calories as a line
    fig.add_trace(go.Scatter(
        x=DOW_SHORT, y=stats['net_mean'],
        mode='lines+markers', name='Net Calories (eaten - exercise)',
        line=dict(color='#2C3E50', width=2.5),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='%{x} net: <b>%{y:.0f}</b> cal<extra></extra>',
    ), row=1, col=1)

    # Individual day scatter (jittered) for calories eaten
    for i, dow in enumerate(DOW_ORDER):
        day_data = df[df['dow'] == dow]['Calories']
        jitter   = np.random.default_rng(42).uniform(-0.25, 0.25, len(day_data))
        fig.add_trace(go.Scatter(
            x=[DOW_SHORT[i]] * len(day_data),
            y=day_data.values,
            mode='markers',
            marker=dict(color='#C0392B', size=4, opacity=0.3),
            showlegend=(i == 0),
            name='Individual days',
            hovertemplate=f'{DOW_SHORT[i]}: %{{y:.0f}} cal<extra></extra>',
        ), row=1, col=1)

    # --- Row 2: exercise minutes ---
    fig.add_trace(go.Bar(
        x=DOW_SHORT, y=stats['exmin_mean'],
        name='Exercise Minutes',
        marker_color='#5B9BD5', opacity=0.85,
        hovertemplate='%{x}: <b>%{y:.0f}</b> min (avg)<extra></extra>',
    ), row=2, col=1)

    # Sample count annotation
    count_text = '  |  '.join(f"{DOW_SHORT[i]}: n={int(stats['count'].iloc[i])}"
                               for i in range(7))
    fig.add_annotation(
        x=0.5, y=-0.12, xref='paper', yref='paper',
        text=count_text, showarrow=False, font=dict(size=10, color='#666'),
        align='center',
    )

    fig.update_layout(
        title='<b>Calorie & Exercise Patterns by Day of Week</b>',
        hovermode='x unified',
        template=TEMPLATE,
        barmode='group',
        height=650,
        margin=dict(b=130),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='Calories', row=1, col=1)
    fig.update_yaxes(title_text='Minutes', row=2, col=1)

    out = CHARTS_DIR / 'chart_dow_patterns.html'
    fig.write_html(str(out))
    fig_to_return = fig  # capture before the print block

    # Print the most interesting finding
    cal_by_dow = stats['cal_mean'].reindex(DOW_ORDER)
    highest = cal_by_dow.idxmax()
    lowest  = cal_by_dow.idxmin()
    print(f"Wrote {out.name}  |  Highest avg intake: {highest} ({cal_by_dow[highest]:.0f} cal), "
          f"Lowest: {lowest} ({cal_by_dow[lowest]:.0f} cal)")
    return fig_to_return


# ---------------------------------------------------------------------------
# Chart 3: Goal projection
# ---------------------------------------------------------------------------

def chart_projection(unified, tdee):
    """
    Fit a linear trend to recent smoothed weight, project forward with 95%
    prediction intervals, and mark goal weight crossings.
    """
    smoothed = tdee['weight_7d_avg'].dropna()
    raw_mask = tdee['weight_raw'].notna()

    # Fit trend from last PROJECTION_LOOKBACK_DAYS of smoothed data
    recent = smoothed.tail(PROJECTION_LOOKBACK_DAYS)
    if len(recent) < 14:
        print("Not enough data for projection chart.")
        return

    x = np.arange(len(recent), dtype=float)
    y = recent.values
    b0, b1, s, Sxx, x_mean = ols_fit(x, y)
    n = len(recent)

    # Build future date range
    last_date     = recent.index[-1]
    future_dates  = pd.date_range(last_date + timedelta(days=1),
                                   periods=PROJECTION_DAYS_FORWARD, freq='D')
    future_x      = np.arange(n, n + PROJECTION_DAYS_FORWARD, dtype=float)

    proj_y, pi_lo, pi_hi = prediction_interval(future_x, b0, b1, s, Sxx, x_mean, n)

    # Historical trend line (over the lookback window only)
    hist_trend_y = b0 + b1 * x

    fig = go.Figure()

    # All historical smoothed weight
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed.values,
        mode='lines', name='7-Day Avg Weight',
        line=dict(color='#2E75B6', width=2),
        hovertemplate='%{x|%b %d}: %{y:.1f} lbs<extra></extra>',
    ))

    # Raw measurements
    fig.add_trace(go.Scatter(
        x=tdee.index[raw_mask], y=tdee.loc[raw_mask, 'weight_raw'],
        mode='markers', name='Measured Weight',
        marker=dict(color='#5B9BD5', size=6, symbol='circle'),
        hovertemplate='%{x|%b %d}: %{y:.1f} lbs (measured)<extra></extra>',
    ))

    # Trend line over lookback window
    fig.add_trace(go.Scatter(
        x=recent.index, y=hist_trend_y,
        mode='lines', name=f'Trend fit (last {PROJECTION_LOOKBACK_DAYS}d)',
        line=dict(color='#C00000', width=1.5, dash='dash'),
        hovertemplate='Trend: %{y:.1f} lbs<extra></extra>',
    ))

    # Projected line
    fig.add_trace(go.Scatter(
        x=future_dates, y=proj_y,
        mode='lines', name='Projected (linear)',
        line=dict(color='#C00000', width=2),
        hovertemplate='%{x|%b %d} projected: %{y:.1f} lbs<extra></extra>',
    ))

    # 95% prediction interval band
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([pi_hi, pi_lo[::-1]]),
        fill='toself',
        fillcolor='rgba(192,0,0,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% prediction interval',
        hoverinfo='skip',
    ))

    # Goal weight lines + date labels
    for goal in GOAL_WEIGHTS:
        current_min = smoothed.iloc[-1]
        if goal >= current_min:
            continue  # already past this goal

        # Find when trend line crosses this weight
        if b1 >= 0:
            continue  # not losing — won't cross
        x_cross = (goal - b0) / b1
        if x_cross < 0 or x_cross > n + PROJECTION_DAYS_FORWARD:
            continue

        days_from_last = x_cross - (n - 1)
        cross_date     = last_date + timedelta(days=int(days_from_last))

        fig.add_hline(
            y=goal,
            line=dict(color='#27AE60', width=1.5, dash='dot'),
            annotation_text=f"  {goal} lbs ~ {cross_date.strftime('%b %d, %Y')}",
            annotation_position='top left',
            annotation_font=dict(size=11, color='#1E8449'),
        )

    # Rate annotation
    weekly_rate = b1 * 7
    fig.add_annotation(
        x=0.01, y=0.05, xref='paper', yref='paper',
        text=f"Current rate: <b>{weekly_rate:+.2f} lbs/week</b> "
             f"(based on last {PROJECTION_LOOKBACK_DAYS} days)",
        showarrow=False, align='left',
        bgcolor='white', bordercolor='#ccc', borderwidth=1,
        font=dict(size=12),
    )

    fig.update_layout(
        title='<b>Weight Projection</b> — Linear Trend with 95% Prediction Interval',
        xaxis_title='Date',
        yaxis_title='Weight (lbs)',
        hovermode=HOVER,
        template=TEMPLATE,
        margin=dict(b=120),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        height=600,
    )

    out = CHARTS_DIR / 'chart_projection.html'
    fig.write_html(str(out))
    fig_to_return = fig

    print(f"Wrote {out.name}  |  Rate: {weekly_rate:+.2f} lbs/week")
    for goal in GOAL_WEIGHTS:
        current_min = smoothed.iloc[-1]
        if goal >= current_min:
            continue
        if b1 >= 0:
            break
        x_cross = (goal - b0) / b1
        days_from_last = x_cross - (n - 1)
        if days_from_last < 0 or days_from_last > PROJECTION_DAYS_FORWARD:
            continue
        cross_date = last_date + timedelta(days=int(days_from_last))
        print(f"  {goal} lbs projected: {cross_date.strftime('%B %d, %Y')} "
              f"({int(days_from_last)} days from now)")
    return fig_to_return


# ---------------------------------------------------------------------------
# Chart 4: Recommended calorie targets as TDEE declines
# ---------------------------------------------------------------------------

def chart_calorie_targets(unified, tdee, goal_weight=175, goal_deficit_at_target=250, goal_tdee=2743):
    """
    Simulates weight loss with a weight-keyed tapering deficit and derives the timeline.

    Design:
      - Deficit tapers from current level → goal_deficit_at_target as weight → goal_weight.
        This means intake drops gently and non-linearly, never aggressively.
      - TDEE interpolates linearly between current observed TDEE and goal_tdee, keyed to
        weight progress rather than elapsed time. This anchors the endpoint to a physiological
        estimate (e.g. Mifflin-St Jeor at goal weight + moderate exercise) rather than
        extrapolating the historical trend, which overshoots at lower body weights.
      - Timeline is derived from the simulation — not pre-specified.
      - Recommended intake = projected_TDEE(weight) - tapered_deficit(weight)

    Two panels:
      1. Calories over time: TDEE, recommended intake, monthly targets
      2. Simulated weight progress toward goal
    """
    CALS_PER_LB  = 3500
    SAFE_FLOOR   = 1500
    HIST_CONTEXT = 60
    MAX_SIM_DAYS = 730   # safety cap on simulation (2 years)

    last_date      = tdee.index.max()
    start_weight   = tdee['weight_7d_avg'].dropna().iloc[-1]
    lbs_to_lose    = max(start_weight - goal_weight, 0)

    # --- TDEE anchors ---
    tdee_hist    = tdee['tdee_14d_smoothed'].dropna()
    start_tdee   = float(tdee_hist.iloc[-1])
    start_intake = float(tdee['calories_14d_avg'].dropna().iloc[-1])
    start_deficit = start_tdee - start_intake

    # --- Day-by-day simulation ---
    # tdee(weight)    interpolates linearly from start_tdee → goal_tdee as weight → goal_weight.
    #                 Anchoring to a physiological target TDEE prevents the trend-line from
    #                 over-projecting metabolic adaptation at lower body weights.
    # deficit(weight) tapers linearly from start_deficit → goal_deficit_at_target
    # intake(t)       = tdee(weight) - deficit(weight)
    records = []
    w = start_weight

    for day in range(MAX_SIM_DAYS):
        if w <= goal_weight:
            break
        frac    = (w - goal_weight) / lbs_to_lose          # 1.0 at start → 0.0 at goal
        t       = goal_tdee + (start_tdee - goal_tdee) * frac
        deficit = goal_deficit_at_target + (start_deficit - goal_deficit_at_target) * frac
        intake  = max(t - deficit, SAFE_FLOOR)

        records.append({
            'date':    last_date + timedelta(days=day + 1),
            'weight':  w,
            'tdee':    t,
            'deficit': deficit,
            'intake':  intake,
            'rate_lbs_week': deficit * 7 / CALS_PER_LB,
        })

        w -= deficit / CALS_PER_LB

    sim = pd.DataFrame(records).set_index('date')
    goal_date   = sim.index[-1]
    total_weeks = len(sim) / 7

    # Monthly reference rows for intake labels
    monthly = sim['intake'].resample('MS').mean()

    # --- Historical context ---
    hist_tdee   = tdee_hist.tail(HIST_CONTEXT)
    hist_intake = tdee['calories_7d_avg'].dropna().tail(HIST_CONTEXT)

    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Calories: TDEE vs Recommended Intake', 'Simulated Weight Progress'),
        vertical_spacing=0.10,
        row_heights=[0.58, 0.42],
    )

    # --- Row 1: calories ---

    # Shade deficit zone (between TDEE and intake)
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(sim.index), pd.Series(sim.index[::-1])]),
        y=np.concatenate([sim['tdee'].values, sim['intake'].values[::-1]]),
        fill='toself', fillcolor='rgba(39,174,96,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Deficit zone', hoverinfo='skip', showlegend=True,
    ), row=1, col=1)

    # Historical TDEE
    fig.add_trace(go.Scatter(
        x=hist_tdee.index, y=hist_tdee.values,
        mode='lines', name='TDEE (historical)',
        line=dict(color='#ED7D31', width=2),
        hovertemplate='%{x|%b %d} TDEE: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Historical intake
    fig.add_trace(go.Scatter(
        x=hist_intake.index, y=hist_intake.values,
        mode='lines', name='Actual intake (14d avg)',
        line=dict(color='#FF6B6B', width=2),
        hovertemplate='%{x|%b %d} intake: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Projected TDEE
    fig.add_trace(go.Scatter(
        x=sim.index, y=sim['tdee'],
        mode='lines', name='Projected TDEE (ex-adj decline)',
        line=dict(color='#ED7D31', width=2, dash='dash'),
        hovertemplate='%{x|%b %d} proj TDEE: %{y:.0f}<extra></extra>',
    ), row=1, col=1)

    # Recommended intake
    fig.add_trace(go.Scatter(
        x=sim.index, y=sim['intake'],
        mode='lines', name='Recommended intake (tapering deficit)',
        line=dict(color='#27AE60', width=2.5),
        hovertemplate='%{x|%b %d} target: %{y:.0f} cal (%{customdata:.1f} lbs/wk deficit)<extra></extra>',
        customdata=sim['rate_lbs_week'],
    ), row=1, col=1)

    # Monthly intake labels
    for month_start, intake_val in monthly.items():
        if pd.isna(intake_val):
            continue
        fig.add_annotation(
            x=month_start, y=intake_val, row=1, col=1,
            text=f"<b>{intake_val:.0f}</b>",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor='#1E8449', ax=0, ay=-30,
            font=dict(size=10, color='#1E8449'),
            bgcolor='white', bordercolor='#27AE60', borderwidth=1,
        )

    # Safety floor
    fig.add_shape(type='line', xref='x', yref='y',
        x0=sim.index[0], x1=sim.index[-1], y0=SAFE_FLOOR, y1=SAFE_FLOOR,
        line=dict(color='#C00000', width=1, dash='dot'), row=1, col=1,
    )

    # Today line (row 1 + row 2)
    for row in [1, 2]:
        fig.add_shape(type='line', xref='x', yref='paper',
            x0=last_date, x1=last_date,
            y0=(0.42 if row == 1 else 0), y1=(1.0 if row == 1 else 0.38),
            line=dict(color='#aaa', width=1, dash='dot'),
        )
    fig.add_annotation(
        x=last_date, y=1.01, xref='x', yref='paper',
        text='Today', showarrow=False,
        font=dict(size=10, color='#888'), xanchor='left',
    )

    # --- Row 2: weight ---

    # Historical weight (7d avg)
    hist_weight = tdee['weight_7d_avg'].dropna().tail(HIST_CONTEXT)
    fig.add_trace(go.Scatter(
        x=hist_weight.index, y=hist_weight.values,
        mode='lines', name='Weight (7d avg, historical)',
        line=dict(color='#2E75B6', width=2),
        hovertemplate='%{x|%b %d}: %{y:.1f} lbs<extra></extra>',
    ), row=2, col=1)

    # Simulated weight
    fig.add_trace(go.Scatter(
        x=sim.index, y=sim['weight'],
        mode='lines', name='Simulated weight',
        line=dict(color='#2E75B6', width=2, dash='dash'),
        hovertemplate='%{x|%b %d} sim: %{y:.1f} lbs<extra></extra>',
    ), row=2, col=1)

    # Goal weight line
    fig.add_shape(type='line', xref='x', yref='y',
        x0=hist_weight.index[0], x1=goal_date, y0=goal_weight, y1=goal_weight,
        line=dict(color='#27AE60', width=1.5, dash='dot'), row=2, col=1,
    )
    fig.add_annotation(
        x=goal_date, y=goal_weight, row=2, col=1,
        text=f'  {goal_weight} lbs  {goal_date.strftime("%b %Y")}',
        showarrow=False, font=dict(size=11, color='#27AE60'),
        xanchor='left',
    )

    # Summary box
    end_tdee   = sim['tdee'].iloc[-1]
    end_intake = sim['intake'].iloc[-1]
    fig.add_annotation(
        x=0.01, y=0.50, xref='paper', yref='paper',
        align='left', valign='middle',
        text=(
            f"<b>Plan Summary</b><br>"
            f"Start intake:  {start_intake:.0f} cal \u2192 Goal intake: {end_intake:.0f} cal<br>"
            f"Start deficit: {start_deficit:.0f} cal/day \u2192 {goal_deficit_at_target} cal/day at goal<br>"
            f"TDEE at goal:  ~{end_tdee:.0f} cal/day<br>"
            f"Timeline:      {total_weeks:.0f} weeks ({total_weeks/4.33:.1f} months)"
        ),
        showarrow=False,
        bgcolor='white', bordercolor='#ccc', borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        title=f'<b>Calorie Targets to Reach {goal_weight} lbs</b> — Tapering Deficit, TDEE-Adjusted',
        hovermode=HOVER,
        template=TEMPLATE,
        height=650,
        margin=dict(b=130),
        legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='cal/day', row=1, col=1)
    fig.update_yaxes(title_text='lbs', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    out = CHARTS_DIR / 'chart_calorie_targets.html'
    fig.write_html(str(out))
    fig_to_return = fig

    print(f"Wrote {out.name}")
    print(f"  Start intake:   {start_intake:.0f} cal/day  (deficit: {start_deficit:.0f})")
    print(f"  End intake:     {end_intake:.0f} cal/day  (deficit: {goal_deficit_at_target})")
    print(f"  TDEE at goal:   {end_tdee:.0f} cal/day")
    print(f"  Timeline:       {total_weeks:.0f} weeks  ({total_weeks/4.33:.1f} months)")
    print(f"  Goal date:      {goal_date.strftime('%B %d, %Y')}")
    return fig_to_return


# ---------------------------------------------------------------------------

def main():
    unified, tdee = load_data()
    chart_tdee_trend(unified, tdee)
    chart_dow_patterns(unified, tdee)
    chart_projection(unified, tdee)
    chart_calorie_targets(unified, tdee)
    print("\nDone.")


if __name__ == '__main__':
    main()
