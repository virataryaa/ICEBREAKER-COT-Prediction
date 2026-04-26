import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

st.set_page_config(page_title="COT Prediction", layout="wide", page_icon="📊")

# ── HardMiner Abyss theme — exact gradient from icebreaker.html ───────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 60% at 20% 80%, rgba(26,74,90,.50) 0%, transparent 65%),
        radial-gradient(ellipse 60% 50% at 80% 20%, rgba(42,85,104,.45) 0%, transparent 60%),
        radial-gradient(ellipse 80% 70% at 50% 30%, rgba(79,176,200,.10) 0%, transparent 55%),
        #0D1620;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(13,22,32,.85) !important;
    backdrop-filter: saturate(180%) blur(16px);
    -webkit-backdrop-filter: saturate(180%) blur(16px);
    border-bottom: 1px solid rgba(188,212,222,.14);
}
[data-testid="stSidebar"] {
    background: rgba(21,34,46,.92);
}
div[data-testid="stDataFrame"] td { padding: 3px 10px !important; font-size: 13px !important; }
div[data-testid="stDataFrame"] th { padding: 4px 10px !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).resolve().parent.parent / "Database" / "cot_prediction.parquet"

DISPLAY_ORDER = ['KC', 'RC', 'CC', 'LCC', 'SB', 'CT']
COLORS = {
    'KC': '#4e9af1', 'RC': '#f1a94e', 'CC': '#6bcb77',
    'LCC': '#f16b6b', 'SB': '#bf8fd0', 'CT': '#f1d34e'
}

# ── Load ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load():
    df  = pd.read_parquet(DATA_PATH)
    hist = df[~df['Is_Simulation']].copy().sort_values(['Commodity', 'Date'])
    sim  = df[df['Is_Simulation']].copy().set_index('Commodity')
    return hist, sim

hist, sim = load()
name_map = sim['Name'].to_dict()

# ── Helpers ───────────────────────────────────────────────────────────────────

def fit(df, x_col, y_col):
    """Robust linear regression with z-score outlier removal."""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 10:
        return np.nan, np.nan, np.nan
    z = np.abs(stats.zscore(sub[y_col]))
    sub = sub[z < 10]
    sl, ic, r, _, _ = stats.linregress(sub[x_col], sub[y_col])
    return sl, ic, r ** 2

def get_spec_cols(include_index):
    net_col = 'Spec_Net_Full'    if include_index else 'Spec_Net_NoIdx'
    chg_col = 'Spec_Net_Full_Chg' if include_index else 'Spec_Net_NoIdx_Chg'
    last_col = 'Last_Spec_Full'  if include_index else 'Last_Spec_NoIdx'
    return net_col, chg_col, last_col

def compute_pred_spec_net(sym_hist, net_col, chg_col, slope, intercept):
    """1-step-ahead predicted spec net: last actual + predicted change."""
    df = sym_hist.dropna(subset=[net_col, chg_col]).copy()
    df['_prev'] = df[net_col].shift(1)
    df['Pred_Chg'] = slope * df['Rollex_Pct'] + intercept
    df['Pred_Spec_Net'] = df['_prev'] + df['Pred_Chg']
    return df

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("## COT Prediction Dashboard")

sim_as_of  = sim['Sim_As_Of'].max()
last_cot   = sim['Last_COT_Date'].max()

c1, c2, c3 = st.columns([2, 2, 4])
with c1:
    st.markdown(f"**Last COT:** {last_cot.strftime('%d-%b-%y')}")
with c2:
    st.markdown(f"**Sim as of:** {sim_as_of.strftime('%d-%b-%y')}")

st.caption(
    "NYC spec = Non Com + Non Rep + Index  |  "
    "European spec = Managed Money + Others + Non Rep"
)
st.divider()

# ── Global controls ───────────────────────────────────────────────────────────

ctrl1, ctrl2 = st.columns([3, 1])

with ctrl1:
    min_yr = int(hist['Date'].dt.year.min())
    max_yr = int(hist['Date'].dt.year.max())
    yr_range = st.slider(
        "Regression date range (governs R² and all predictions below)",
        min_yr, max_yr, (min_yr, max_yr), step=1
    )

with ctrl2:
    index_choice = st.radio(
        "Spec definition",
        ["Include Index", "Exclude Index"],
        horizontal=True
    )

include_index = (index_choice == "Include Index")
net_col, chg_col, last_col = get_spec_cols(include_index)

# Filter history to selected date window
hist_w = hist[hist['Date'].dt.year.between(*yr_range)].copy()

st.divider()

# ── Summary Table ─────────────────────────────────────────────────────────────

st.subheader("Spec Change Prediction & Actual")

rows = []
for sym in DISPLAY_ORDER:
    if sym not in sim.index:
        continue
    s = sim.loc[sym]

    sym_data = hist_w[hist_w['Commodity'] == sym].dropna(subset=['Rollex_Pct', chg_col])
    slope, intercept, r2 = fit(sym_data, 'Rollex_Pct', chg_col)

    last_spec  = s[last_col]
    partial    = s['Rollex_Pct']
    pred_chg   = (slope * partial + intercept) if not np.isnan(slope) else np.nan
    sim_net    = last_spec + pred_chg if not np.isnan(pred_chg) else np.nan

    rows.append({
        'Spec Inclusion': s['Spec_Full_Label'] if include_index else s['Spec_NoIdx_Label'],
        'Commodity':      s['Name'],
        'COT Date':       s['Last_COT_Date'].strftime('%d-%b-%y'),
        'Spec Net':       last_spec,
        'Prediction':     sim_net,
        'Pred Change':    pred_chg,
        'Actual':         'Awaiting',
        'Px Change':      f"{partial:+.1%}",
        'R²':             r2,
    })

table_df = pd.DataFrame(rows)

def color_pred_change(val):
    try:
        v = float(val)
        if np.isnan(v):
            return ''
        return 'color: #6bcb77; text-align: center' if v > 0 else 'color: #f16b6b; text-align: center'
    except Exception:
        return ''


styled = (
    table_df.style
    .map(color_pred_change, subset=['Pred Change'])
    .set_properties(**{'text-align': 'center'})
    .set_properties(subset=['Spec Inclusion', 'Commodity'], **{'text-align': 'left'})
    .set_properties(subset=['Commodity'], **{'font-weight': 'bold'})
    .set_properties(subset=['Actual'], **{'color': '#888888', 'font-style': 'italic', 'text-align': 'center'})
    .format({
        'Spec Net':   '{:.1f}',
        'Prediction': '{:.1f}',
        'Pred Change':'{:+.1f}',
        'R²':         '{:.1f}',
    }, na_rep='—')
)

st.dataframe(styled, width='stretch', hide_index=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_scatter, tab_ts = st.tabs(["Scatter & R²", "Spec Net: Actual vs Predicted"])

# ── Scatter tab ───────────────────────────────────────────────────────────────

with tab_scatter:
    sel_col, chart_col = st.columns([1, 4])

    with sel_col:
        sel = st.selectbox("Commodity", DISPLAY_ORDER, format_func=lambda x: name_map.get(x, x))

    sym_data = hist_w[hist_w['Commodity'] == sel].dropna(subset=['Rollex_Pct', chg_col])
    slope, intercept, r2 = fit(sym_data, 'Rollex_Pct', chg_col)

    x_vals = sym_data['Rollex_Pct'].values
    y_vals = sym_data[chg_col].values
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_line = slope * x_line + intercept

    sim_row   = sim.loc[sel]
    partial   = sim_row['Rollex_Pct']
    pred_chg_sim = slope * partial + intercept if not np.isnan(slope) else np.nan
    color     = COLORS.get(sel, '#ffffff')

    fig_s = go.Figure()

    fig_s.add_trace(go.Scatter(
        x=sym_data['Rollex_Pct'],
        y=sym_data[chg_col],
        mode='markers',
        marker=dict(
            color=sym_data['Date'].dt.year,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Year', thickness=12),
            size=7, opacity=0.75,
            line=dict(width=0.5, color='white')
        ),
        text=sym_data['Date'].dt.strftime('%d-%b-%y'),
        hovertemplate='%{text}<br>Rollex: %{x:.2%}<br>Spec Chg: %{y:.1f}k<extra></extra>',
        name='Weekly obs'
    ))

    fig_s.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Regression  R²={r2:.1f}'
    ))

    fig_s.add_trace(go.Scatter(
        x=[partial], y=[pred_chg_sim],
        mode='markers',
        marker=dict(color='yellow', size=14, symbol='star', line=dict(width=1, color='black')),
        name=f"This week: {pred_chg_sim:+.1f}k" if not np.isnan(pred_chg_sim) else "This week",
        hovertemplate=f"Rollex: {partial:+.2%}<br>Pred chg: {pred_chg_sim:+.1f}k<extra></extra>"
    ))

    fig_s.add_hline(y=0, line_color='gray', line_width=0.8)
    fig_s.add_vline(x=0, line_color='gray', line_width=0.8)

    fig_s.update_layout(
        title=f"{name_map[sel]} — Rollex Weekly % vs Spec Net Change  "
              f"({yr_range[0]}–{yr_range[1]}  |  R²={r2:.1f}  |  slope={slope:+.1f}  |  n={len(sym_data)})",
        xaxis=dict(title='Rollex Weekly % Change', tickformat='.1%'),
        yaxis=dict(title='Spec Net Change (000 contracts)'),
        template='plotly_dark',
        height=520,
        legend=dict(x=0.01, y=0.99),
    )

    with chart_col:
        st.plotly_chart(fig_s, use_container_width=True)

    # Inline stats (no ugly metric cards)
    last_spec_s  = sim_row[last_col]
    sim_net_s    = last_spec_s + pred_chg_sim if not np.isnan(pred_chg_sim) else np.nan
    st.markdown(
        f"**Predicted spec chg this week:** `{pred_chg_sim:+.1f}k`  &nbsp;|&nbsp;  "
        f"**Last spec net:** `{last_spec_s:.1f}k`  &nbsp;|&nbsp;  "
        f"**Sim spec net:** `{sim_net_s:.1f}k`  &nbsp;|&nbsp;  "
        f"**Rollex partial-week move:** `{partial:+.1%}`"
    )

# ── Time series tab ───────────────────────────────────────────────────────────

with tab_ts:
    sel2_col, chart2_col = st.columns([1, 4])

    with sel2_col:
        sel2    = st.selectbox("Commodity", DISPLAY_ORDER, format_func=lambda x: name_map.get(x, x), key='ts_sym')
        lookback = st.slider("Show last N weeks", 26, 600, 156, step=26)

    sym_hist2 = hist[hist['Commodity'] == sel2].copy()

    # Refit on selected window for pred line, but show full history in chart
    sym_w2 = hist_w[hist_w['Commodity'] == sel2].dropna(subset=['Rollex_Pct', chg_col])
    slope2, intercept2, r2_2 = fit(sym_w2, 'Rollex_Pct', chg_col)

    sym_ts = compute_pred_spec_net(sym_hist2, net_col, chg_col, slope2, intercept2).tail(lookback)

    sim_row2 = sim.loc[sel2]
    partial2  = sim_row2['Rollex_Pct']
    last2     = sim_row2[last_col]
    pred_chg2 = slope2 * partial2 + intercept2 if not np.isnan(slope2) else np.nan
    sim_net2  = last2 + pred_chg2 if not np.isnan(pred_chg2) else np.nan
    sim_date2 = sim_row2['Date'] if 'Date' in sim_row2.index else sim_row2.name

    color2 = COLORS.get(sel2, '#ffffff')

    fig_t = go.Figure()

    fig_t.add_trace(go.Scatter(
        x=sym_ts['Date'], y=sym_ts[net_col],
        name='Actual Spec Net',
        line=dict(color=color2, width=2),
        hovertemplate='%{x|%d-%b-%y}<br>Actual: %{y:.1f}k<extra></extra>'
    ))

    fig_t.add_trace(go.Scatter(
        x=sym_ts['Date'], y=sym_ts['Pred_Spec_Net'],
        name='Predicted (1-step)',
        line=dict(color='orange', width=1.5, dash='dot'),
        hovertemplate='%{x|%d-%b-%y}<br>Predicted: %{y:.1f}k<extra></extra>'
    ))

    fig_t.add_trace(go.Scatter(
        x=[sim_row2['Date']], y=[sim_net2],
        mode='markers',
        marker=dict(color='yellow', size=14, symbol='star', line=dict(width=1, color='black')),
        name=f"Sim: {sim_net2:.1f}k",
        hovertemplate=f"Sim: {sim_net2:.1f}k<extra></extra>"
    ))

    fig_t.add_hline(y=0, line_color='gray', line_width=0.8, line_dash='dot')

    fig_t.update_layout(
        title=f"{name_map[sel2]} — Actual vs Predicted Spec Net  "
              f"(regression {yr_range[0]}–{yr_range[1]}  |  R²={r2_2:.1f})",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Spec Net (000 contracts)'),
        template='plotly_dark',
        height=520,
        legend=dict(x=0.01, y=0.99),
    )

    with chart2_col:
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown(
        f"**Last spec net:** `{last2:.1f}k`  &nbsp;|&nbsp;  "
        f"**Predicted this week:** `{sim_net2:.1f}k`  &nbsp;|&nbsp;  "
        f"**Pred change:** `{pred_chg2:+.1f}k`  &nbsp;|&nbsp;  "
        f"**Rollex partial-week:** `{partial2:+.1%}`"
    )
