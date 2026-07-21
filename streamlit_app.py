"""
Bui Dam Climate Risk Dashboard
ML-based streamflow prediction under CMIP6 climate scenarios
Black Volta Basin, Ghana

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config 
st.set_page_config(
    page_title="Bui Dam Climate Risk Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS 
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px 14px;
    text-align: center;
    transition: border-color .2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
  .metric-label { font-size: 0.75rem; color: #8b949e; letter-spacing: .04em; text-transform: uppercase; }
  .metric-delta { font-size: 0.82rem; margin-top: 4px; }
  .delta-pos { color: #3fb950; }
  .delta-neg { color: #f85149; }
  .delta-neu { color: #8b949e; }

  /* risk badge */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: .03em;
  }
  .badge-low    { background:#1a3a1e; color:#3fb950; border:1px solid #3fb950; }
  .badge-medium { background:#3a2a00; color:#d29922; border:1px solid #d29922; }
  .badge-high   { background:#3a1a1a; color:#f85149; border:1px solid #f85149; }
  .badge-critical { background:#5a0a0a; color:#ff7b72; border:1px solid #ff7b72; }

  /* section titles */
  .section-title {
    font-size: 1.05rem; font-weight: 600; color: #58a6ff;
    border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-bottom: 14px;
  }

  /* disclaimer */
  .disclaimer {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 14px; font-size: 0.78rem; color: #8b949e;
  }

  h1, h2, h3 { color: #e6edf3 !important; }
  .stSelectbox label, .stRadio label { color: #8b949e !important; font-size:0.82rem !important; }
  hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# 
#  DATA — actual values from notebook outputs
# 

# Historical baseline 
HISTORICAL = {
    "mean_flow": 136.5, "drought_days": 259.8, "flood_days": 8.2,
    "q5": 0.95, "q95": 724.68,
    "wet_season": 293.6, "dry_season": 23.0,
    "temp": 28.0, "precip": 2.68,
}

ANNUAL_HIST = {
    1990:72.2,1991:156.7,1992:55.8,1993:63.9,1994:99.4,1995:107.7,1996:124.8,
    1997:54.4,1998:65.2,1999:153.7,2000:100.4,2001:70.7,2002:63.0,2003:172.6,
    2004:87.8,2005:55.2,2006:124.2,2007:248.5,2008:243.2,2009:41.8,2010:259.7,
    2011:152.2,2012:129.2,2013:142.2,2014:191.7,2015:298.2,2016:144.1,
    2017:161.1,2018:302.4,2019:192.0,2020:96.0,
}
FLOOD_HIST = {
    1990:0,1991:11,1992:0,1993:0,1994:0,1995:0,1996:0,1997:0,1998:0,1999:0,
    2000:0,2001:0,2002:0,2003:0,2004:0,2005:0,2006:0,2007:32,2008:32,2009:0,
    2010:25,2011:7,2012:0,2013:0,2014:26,2015:53,2016:0,2017:0,2018:52,
    2019:15,2020:0,
}
DROUGHT_HIST = {
    1990:224,1991:240,1992:286,1993:279,1994:275,1995:262,1996:265,1997:290,
    1998:278,1999:257,2000:277,2001:282,2002:286,2003:248,2004:290,2005:308,
    2006:286,2007:230,2008:234,2009:319,2010:228,2011:278,2012:248,2013:248,
    2014:245,2015:252,2016:248,2017:203,2018:187,2019:243,2020:259,
}

# Scenario projections 
SCENARIOS = {
    "SSP1–2.6 · 2021–2050": {
        "mean_flow":115.8,"drought_days":261.8,"flood_days":0,"q5":19.90,"q95":382.11,
        "q5_chg":2005.5,"q95_chg":-47.3,"wet_season":182.2,"dry_season":67.8,
        "temp":28.09,"precip":2.71,"delta_temp":+0.09,"delta_precip":+0.03,
        "flow_chg":-15.2,"power_mw":60.9,"energy_gwh":534.0,"mol_pct":14.1,
        "storage_mm3":7196,"reliability":0.141,"resilience":0.007,"vulnerability":0.850,
        "recovery_days":136.4,"ssi_extreme":0.0,"ssi_severe":3.6,"ssi_moderate":7.8,"ssi_normal":63.7,
        "horizon":"Near-term","ssp":"SSP1–2.6","color":"#3fb950",
    },
    "SSP2–4.5 · 2021–2050": {
        "mean_flow":175.9,"drought_days":220.4,"flood_days":0,"q5":16.60,"q95":770.49,
        "q5_chg":1656.1,"q95_chg":6.3,"wet_season":261.3,"dry_season":114.3,
        "temp":28.33,"precip":2.69,"delta_temp":+0.33,"delta_precip":+0.01,
        "flow_chg":+28.8,"power_mw":96.7,"energy_gwh":847.9,"mol_pct":19.1,
        "storage_mm3":7299,"reliability":0.267,"resilience":0.011,"vulnerability":0.852,
        "recovery_days":89.2,"ssi_extreme":0.0,"ssi_severe":0.6,"ssi_moderate":2.2,"ssi_normal":73.5,
        "horizon":"Near-term","ssp":"SSP2–4.5","color":"#d29922",
    },
    "SSP5–8.5 · 2021–2050": {
        "mean_flow":138.8,"drought_days":226.0,"flood_days":0,"q5":21.01,"q95":410.69,
        "q5_chg":2123.5,"q95_chg":-43.3,"wet_season":208.7,"dry_season":88.3,
        "temp":28.51,"precip":2.65,"delta_temp":+0.51,"delta_precip":-0.03,
        "flow_chg":+1.7,"power_mw":75.8,"energy_gwh":664.4,"mol_pct":19.5,
        "storage_mm3":7196,"reliability":0.196,"resilience":0.014,"vulnerability":0.830,
        "recovery_days":68.8,"ssi_extreme":0.0,"ssi_severe":1.1,"ssi_moderate":6.7,"ssi_normal":68.7,
        "horizon":"Near-term","ssp":"SSP5–8.5","color":"#f85149",
    },
    "SSP1–2.6 · 2051–2075": {
        "mean_flow":347.4,"drought_days":187.1,"flood_days":0,"q5":21.12,"q95":862.52,
        "q5_chg":2135.2,"q95_chg":19.0,"wet_season":None,"dry_season":None,
        "temp":28.90,"precip":3.16,"delta_temp":+0.90,"delta_precip":+0.48,
        "flow_chg":+154.5,"power_mw":169.2,"energy_gwh":1483.3,"mol_pct":10.3,
        "storage_mm3":8062,"reliability":0.410,"resilience":0.003,"vulnerability":0.862,
        "recovery_days":336.8,"ssi_extreme":0.0,"ssi_severe":3.4,"ssi_moderate":8.7,"ssi_normal":76.2,
        "horizon":"Long-term","ssp":"SSP1–2.6","color":"#3fb950",
    },
    "SSP2–4.5 · 2051–2075": {
        "mean_flow":112.0,"drought_days":261.3,"flood_days":0,"q5":9.39,"q95":377.83,
        "q5_chg":894.0,"q95_chg":-47.9,"wet_season":None,"dry_season":None,
        "temp":30.14,"precip":3.28,"delta_temp":+2.14,"delta_precip":+0.60,
        "flow_chg":-17.9,"power_mw":59.0,"energy_gwh":516.9,"mol_pct":40.6,
        "storage_mm3":7169,"reliability":0.152,"resilience":0.006,"vulnerability":0.874,
        "recovery_days":161.4,"ssi_extreme":0.0,"ssi_severe":2.3,"ssi_moderate":7.7,"ssi_normal":63.1,
        "horizon":"Long-term","ssp":"SSP2–4.5","color":"#d29922",
    },
    "SSP5–8.5 · 2051–2075": {
        "mean_flow":108.7,"drought_days":262.0,"flood_days":0,"q5":19.42,"q95":394.62,
        "q5_chg":1954.6,"q95_chg":-45.5,"wet_season":None,"dry_season":None,
        "temp":31.14,"precip":3.09,"delta_temp":+3.14,"delta_precip":+0.41,
        "flow_chg":-20.4,"power_mw":56.9,"energy_gwh":498.4,"mol_pct":40.4,
        "storage_mm3":7181,"reliability":0.135,"resilience":0.003,"vulnerability":0.871,
        "recovery_days":303.9,"ssi_extreme":0.0,"ssi_severe":2.0,"ssi_moderate":6.7,"ssi_normal":59.4,
        "horizon":"Long-term","ssp":"SSP5–8.5","color":"#f85149",
    },
}

MODEL_PERF = {
    "Random Forest (Chache)":  {"nse":0.991,"kge":0.987,"rmse":23.91,"mae":8.88,  "period":"Test 2019–2020"},
    "XGBoost (Chache)":        {"nse":0.990,"kge":0.980,"rmse":24.99,"mae":10.17, "period":"Test 2019–2020"},
    "LSTM (Chache)":           {"nse":0.971,"kge":0.929,"rmse":42.80,"mae":21.91, "period":"Test 2019–2020"},
    "Random Forest (Lawra)":   {"nse":0.997,"kge":0.990,"rmse":10.29,"mae":4.78,  "period":"Test 2019–2020"},
    "XGBoost (Lawra)":         {"nse":0.996,"kge":0.981,"rmse":12.69,"mae":5.86,  "period":"Test 2019–2020"},
    "LSTM (Lawra)":            {"nse":0.991,"kge":0.948,"rmse":18.36,"mae":10.11, "period":"Test 2019–2020"},
}

COLORS = {"SSP1–2.6":"#3fb950","SSP2–4.5":"#d29922","SSP5–8.5":"#f85149","Historical":"#58a6ff"}
PLOT_BG = "#0d1117"
PAPER_BG = "#0d1117"
GRID_COLOR = "#21262d"
FONT_COLOR = "#e6edf3"

def plotly_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(size=13, color=FONT_COLOR)),
        height=height, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor="#30363d", zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor="#30363d", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=50, r=20, t=45, b=40),
    )

def risk_badge(label):
    cls = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high","Critical":"badge-critical"}.get(label,"badge-medium")
    return f'<span class="badge {cls}">{label}</span>'

def flow_risk(d):
    if d["drought_days"] > 280: return "Critical"
    if d["drought_days"] > 260: return "High"
    if d["drought_days"] > 230: return "Medium"
    return "Low"

def power_risk(d):
    if d["mol_pct"] > 35: return "Critical"
    if d["mol_pct"] > 20: return "High"
    if d["mol_pct"] > 12: return "Medium"
    return "Low"

def delta_html(val, unit="", good="positive"):
    """Return coloured delta string."""
    if val is None: return '<span class="delta-neu">—</span>'
    pos_good = good == "positive"
    cls = "delta-pos" if (val > 0 and pos_good) or (val < 0 and not pos_good) else \
          "delta-neg" if (val < 0 and pos_good) or (val > 0 and not pos_good) else "delta-neu"
    arrow = "▲" if val > 0 else "▼" if val < 0 else "●"
    return f'<span class="{cls}">{arrow} {abs(val):.1f}{unit}</span>'

 
#  SIDEBAR
 
with st.sidebar:
    st.markdown("## 💧 Bui Dam\n### Climate Risk Dashboard")
    st.markdown("---")

    st.markdown("**Select SSP Scenario**")
    scenario_key = st.selectbox(
        "Scenario", list(SCENARIOS.keys()),
        label_visibility="collapsed"
    )
    sc = SCENARIOS[scenario_key]

    st.markdown("**Planning Horizon**")
    horizon_filter = st.radio(
        "Horizon", ["All","Near-term (2021–2050)","Long-term (2051–2075)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Compare Scenarios**")
    compare_mode = st.checkbox("Enable multi-scenario overlay", value=False)
    if compare_mode:
        compare_keys = st.multiselect(
            "Scenarios to compare", list(SCENARIOS.keys()),
            default=list(SCENARIOS.keys())[:3],
            label_visibility="collapsed"
        )

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    <b>Study area:</b> Black Volta Basin, Ghana<br>
    <b>Station:</b> Chache (primary)<br>
    <b>Models:</b> RF · XGBoost · LSTM<br>
    <b>Projections:</b> CMIP6 SSP scenarios<br>
    <b>Baseline:</b> 1990–2020<br><br>
    <!-- # Dzahene R.K. Elorm · KNUST 2025 -->
    </div>
    """, unsafe_allow_html=True)

 
#  HEADER
 
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"## Bui Hydroelectric Dam — Climate Risk Assessment")
    st.markdown(f"*ML-based streamflow prediction under CMIP6 scenarios · Black Volta Basin, Ghana*")
with col_h2:
    ssp_label = sc["ssp"]
    color = sc["color"]
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {color};border-radius:10px;
    padding:12px 16px;text-align:center;margin-top:4px;">
      <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;">Active Scenario</div>
      <div style="font-size:1.3rem;font-weight:700;color:{color};margin:4px 0;">{ssp_label}</div>
      <div style="font-size:0.82rem;color:#8b949e;">{sc['horizon']} · {scenario_key.split('·')[1].strip()}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

 
#  TABS
 
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Summary", "🌊 Streamflow", "⚡ Hydropower", "🌡️ Climate Inputs", "🤖 Model Performance"
])

#  TAB 1 — RISK SUMMARY
with tab1:
    st.markdown('<div class="section-title">Key Risk Indicators</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Mean Annual Flow", f"{sc['mean_flow']:.1f}", "m³/s",
         sc['flow_chg'], "%", "positive"),
        (c2, "Drought Days / yr", f"{sc['drought_days']:.1f}", "days",
         sc['drought_days'] - HISTORICAL['drought_days'], "d", "negative"),
        (c3, "Min Flow (Q5)", f"{sc['q5']:.2f}", "m³/s",
         sc['q5_chg'], "%", "positive"),
        (c4, "High Flow (Q95)", f"{sc['q95']:.2f}", "m³/s",
         sc['q95_chg'], "%", "positive"),
        (c5, "Days below MOL", f"{sc['mol_pct']:.1f}", "%",
         sc['mol_pct'] - 70.7, "pp", "negative"),
    ]
    for col, label, val, unit, delta_val, delta_unit, good in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{sc['color']}">{val}</div>
              <div style="font-size:0.8rem;color:#8b949e;">{unit}</div>
              <div class="metric-delta">{delta_html(delta_val, delta_unit, good)}
                vs baseline</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns([1, 2])

    with col_r1:
        st.markdown('<div class="section-title">Risk Classification</div>', unsafe_allow_html=True)
        fr = flow_risk(sc); pr = power_risk(sc)
        drought_delta = sc['drought_days'] - HISTORICAL['drought_days']
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
        <tr style="border-bottom:1px solid #30363d;">
          <td style="padding:8px 4px;color:#8b949e;">Streamflow risk</td>
          <td style="padding:8px 4px;text-align:right;">{risk_badge(fr)}</td>
        </tr>
        <tr style="border-bottom:1px solid #30363d;">
          <td style="padding:8px 4px;color:#8b949e;">Hydropower risk</td>
          <td style="padding:8px 4px;text-align:right;">{risk_badge(pr)}</td>
        </tr>
        <tr style="border-bottom:1px solid #30363d;">
          <td style="padding:8px 4px;color:#8b949e;">Drought frequency</td>
          <td style="padding:8px 4px;text-align:right;">{risk_badge("High" if drought_delta>5 else "Medium" if drought_delta>-10 else "Low")}</td>
        </tr>
        <tr style="border-bottom:1px solid #30363d;">
          <td style="padding:8px 4px;color:#8b949e;">Flood risk</td>
          <td style="padding:8px 4px;text-align:right;">{risk_badge("Low")}</td>
        </tr>
        <tr>
          <td style="padding:8px 4px;color:#8b949e;">Mean power gen.</td>
          <td style="padding:8px 4px;text-align:right;color:{sc['color']};font-weight:600;">{sc['power_mw']:.1f} MW</td>
        </tr>
        </table>
        <br>
        <div style="font-size:0.78rem;color:#8b949e;">
        Installed capacity: 400 MW<br>
        Capacity factor: {sc['power_mw']/400*100:.1f}%<br>
        Mean recovery from drought: {sc['recovery_days']:.0f} days
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="section-title">SSI-3 Drought Category Distribution</div>', unsafe_allow_html=True)
        ssi_cats   = ["Extreme\n(SSI≤−2)", "Severe\n(−2 to −1.5)", "Moderate\n(−1.5 to −1)", "Normal / Wet\n(SSI>−1)"]
        hist_vals  = [1.1, 3.8, 13.2, 48.9]
        scen_vals  = [sc["ssi_extreme"], sc["ssi_severe"], sc["ssi_moderate"], sc["ssi_normal"]]
        bar_colors = ["#f85149","#d29922","#e3b341","#3fb950"]

        fig_ssi = go.Figure()
        fig_ssi.add_trace(go.Bar(
            name="Historical (1990–2020)", x=ssi_cats, y=hist_vals,
            marker_color=["rgba(248,81,73,0.3)","rgba(210,153,34,0.3)",
                          "rgba(227,179,65,0.3)","rgba(63,185,80,0.3)"],
            marker_line_color=bar_colors, marker_line_width=1.5,
        ))
        fig_ssi.add_trace(go.Bar(
            name=scenario_key, x=ssi_cats, y=scen_vals,
            marker_color=bar_colors, marker_opacity=0.85,
        ))
        fig_ssi.update_layout(**plotly_layout("SSI-3 Category Frequency (% of months)", 280))
        fig_ssi.update_layout(barmode="group", yaxis_title="% of months",
                               legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_ssi, use_container_width=True)

    # RRV metrics
    st.markdown('<div class="section-title">Reservoir Reliability–Resilience–Vulnerability (RRV)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    rrv_items = [
        (c1, "Reliability", sc["reliability"], HISTORICAL.get("reliability",0.170),
         "Fraction of time above MOL", "positive"),
        (c2, "Resilience", sc["resilience"], 0.004,
         "Prob. of recovery from failure", "positive"),
        (c3, "Vulnerability", sc["vulnerability"], 0.916,
         "Mean normalised deficit during failure", "negative"),
        (c4, "Recovery time", sc["recovery_days"], 228.6,
         "Mean days to recover from drought event", "negative"),
    ]
    for col, label, val, hist_val, desc, good in rrv_items:
        delta_val = val - hist_val
        with col:
            fmt = f"{val:.3f}" if label != "Recovery time" else f"{val:.0f} d"
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{sc['color']};font-size:1.6rem;">{fmt}</div>
              <div class="metric-delta">{delta_html(delta_val, "", good)} vs baseline</div>
              <div style="font-size:0.7rem;color:#6e7681;margin-top:6px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


#  TAB 2 — STREAMFLOW
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Historical Annual Mean Streamflow (1990–2020)</div>', unsafe_allow_html=True)
        years = list(ANNUAL_HIST.keys())
        flows = list(ANNUAL_HIST.values())
        colors_hist = [COLORS["SSP5–8.5"] if f < 80 else
                       COLORS["SSP2–4.5"] if f < 140 else
                       COLORS["SSP1–2.6"] for f in flows]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=years, y=flows, marker_color=colors_hist, marker_opacity=0.85, name="Annual mean",
        ))
        fig_hist.add_hline(y=HISTORICAL["mean_flow"], line_dash="dash",
                            line_color=COLORS["Historical"], annotation_text="Mean 136.5",
                            annotation_font_color=COLORS["Historical"])
        fig_hist.add_hline(y=100, line_dash="dot", line_color="#d29922",
                            annotation_text="Min turbine flow", annotation_font_color="#d29922")
        fig_hist.update_layout(**plotly_layout("", 320), yaxis_title="Mean flow (m³/s)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        if not compare_mode:
            st.markdown('<div class="section-title">Seasonal Flow Profile — Projected vs Historical</div>', unsafe_allow_html=True)
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            # Approximate monthly climatology from seasonal means
            hist_monthly  = [18,12,9,14,35,95,280,520,380,190,60,22]
            # Scale projected months to match seasonal means
            if sc["wet_season"] is not None:
                wet_scale  = sc["wet_season"]  / HISTORICAL["wet_season"]
                dry_scale  = sc["dry_season"]  / HISTORICAL["dry_season"]
            else:
                wet_scale  = sc["mean_flow"] / HISTORICAL["mean_flow"]
                dry_scale  = wet_scale
            wet_m  = [5,6,7,8,9,10]; dry_m = [0,1,2,3,4,11]
            scen_monthly = []
            for i, v in enumerate(hist_monthly):
                scen_monthly.append(v * (wet_scale if i+1 in [6,7,8,9,10] else dry_scale))

            fig_seas = go.Figure()
            fig_seas.add_trace(go.Scatter(
                x=months, y=hist_monthly, name="Historical", mode="lines+markers",
                line=dict(color=COLORS["Historical"], width=2),
                marker=dict(size=6),
            ))
            fig_seas.add_trace(go.Scatter(
                x=months, y=scen_monthly, name=sc["ssp"], mode="lines+markers",
                line=dict(color=sc["color"], width=2),
                marker=dict(size=6),
                fill="tonexty", fillcolor=f"rgba({','.join(str(int(sc['color'].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08)"
            ))
            fig_seas.add_vrect(x0="Jun", x1="Oct", fillcolor="rgba(88,166,255,0.06)",
                                line_width=0, annotation_text="Wet season", annotation_position="top left",
                                annotation_font_color="#8b949e", annotation_font_size=10)
            fig_seas.update_layout(**plotly_layout("", 320), yaxis_title="Mean discharge (m³/s)")
            st.plotly_chart(fig_seas, use_container_width=True)
        else:
            st.markdown('<div class="section-title">Multi-Scenario Seasonal Comparison</div>', unsafe_allow_html=True)
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            hist_monthly = [18,12,9,14,35,95,280,520,380,190,60,22]
            fig_ms = go.Figure()
            fig_ms.add_trace(go.Scatter(x=months, y=hist_monthly, name="Historical",
                mode="lines+markers", line=dict(color=COLORS["Historical"],width=2,dash="dash")))
            for k in (compare_keys if compare_mode else [scenario_key]):
                s = SCENARIOS[k]
                ws = s["wet_season"]; ds = s["dry_season"]
                if ws is None: ws_sc = s["mean_flow"]/HISTORICAL["mean_flow"]; ds_sc = ws_sc
                else: ws_sc = ws/HISTORICAL["wet_season"]; ds_sc = ds/HISTORICAL["dry_season"]
                sm = [v*(ws_sc if i+1 in [6,7,8,9,10] else ds_sc) for i,v in enumerate(hist_monthly)]
                fig_ms.add_trace(go.Scatter(x=months, y=sm, name=k, mode="lines+markers",
                    line=dict(color=s["color"],width=2)))
            fig_ms.update_layout(**plotly_layout("",320),yaxis_title="Mean discharge (m³/s)")
            st.plotly_chart(fig_ms, use_container_width=True)

    # Q5 / Q95 panel
    st.markdown('<div class="section-title">Extreme Flow Indices — All Scenarios vs Baseline</div>', unsafe_allow_html=True)
    col_q1, col_q2 = st.columns(2)
    scen_labels = [k.replace(" · ", "\n") for k in SCENARIOS.keys()]

    with col_q1:
        q5_vals  = [HISTORICAL["q5"]]  + [SCENARIOS[k]["q5"]  for k in SCENARIOS]
        q5_labs  = ["Historical"] + scen_labels
        q5_cols  = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_q5 = go.Figure(go.Bar(x=q5_labs, y=q5_vals, marker_color=q5_cols, marker_opacity=0.85))
        fig_q5.update_layout(**plotly_layout("Q5 — Minimum Flow (m³/s)", 300),
                              yaxis_title="Q5 (m³/s)", xaxis_tickfont_size=9)
        st.plotly_chart(fig_q5, use_container_width=True)

    with col_q2:
        q95_vals = [HISTORICAL["q95"]] + [SCENARIOS[k]["q95"] for k in SCENARIOS]
        q95_labs = ["Historical"] + scen_labels
        q95_cols = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_q95 = go.Figure(go.Bar(x=q95_labs, y=q95_vals, marker_color=q95_cols, marker_opacity=0.85))
        fig_q95.add_hline(y=HISTORICAL["q95"], line_dash="dash",
                           line_color=COLORS["Historical"], annotation_text="Baseline Q95",
                           annotation_font_color=COLORS["Historical"])
        fig_q95.update_layout(**plotly_layout("Q95 — High Flow (m³/s)", 300),
                               yaxis_title="Q95 (m³/s)", xaxis_tickfont_size=9)
        st.plotly_chart(fig_q95, use_container_width=True)

    # Drought/flood days
    st.markdown('<div class="section-title">Annual Drought &amp; Flood Day Counts</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        dd_vals = [HISTORICAL["drought_days"]] + [SCENARIOS[k]["drought_days"] for k in SCENARIOS]
        dd_labs = ["Historical"] + scen_labels
        dd_cols = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_dd = go.Figure(go.Bar(x=dd_labs, y=dd_vals, marker_color=dd_cols, marker_opacity=0.85))
        fig_dd.add_hline(y=183, line_dash="dot", line_color="#8b949e",
                          annotation_text="50% of year", annotation_font_color="#8b949e")
        fig_dd.update_layout(**plotly_layout("Drought Days / yr (below 100 m³/s)", 280),
                              yaxis_title="Days/yr", xaxis_tickfont_size=9)
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_d2:
        fd_vals = [HISTORICAL["flood_days"]] + [SCENARIOS[k]["flood_days"] for k in SCENARIOS]
        fd_labs = ["Historical"] + scen_labels
        fd_cols = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_fd = go.Figure(go.Bar(x=fd_labs, y=fd_vals, marker_color=fd_cols, marker_opacity=0.85))
        fig_fd.update_layout(**plotly_layout("Flood Days / yr (above 1,000 m³/s)", 280),
                              yaxis_title="Days/yr", xaxis_tickfont_size=9)
        st.plotly_chart(fig_fd, use_container_width=True)

#  TAB 3 — HYDROPOWER
with tab3:
    st.markdown('<div class="section-title">Hydropower Generation Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    hp_metrics = [
        (c1, "Mean Power", f"{sc['power_mw']:.1f} MW", f"of 400 MW installed", sc['power_mw']/400*100),
        (c2, "Annual Energy", f"{sc['energy_gwh']:.0f} GWh/yr", "projected yield", None),
        (c3, "Days below MOL", f"{sc['mol_pct']:.1f}%", "minimum operating level", None),
        (c4, "Mean Storage", f"{sc['storage_mm3']:,} Mm³", "mean reservoir volume", None),
    ]
    for col, label, val, sub, pct in hp_metrics:
        with col:
            extra = f'<div style="background:#21262d;border-radius:4px;height:6px;margin:8px 0;overflow:hidden;"><div style="background:{sc["color"]};width:{min(pct,100):.0f}%;height:100%;"></div></div>' if pct else ""
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{sc['color']};font-size:1.5rem;">{val}</div>
              <div style="font-size:0.75rem;color:#8b949e;">{sub}</div>
              {extra}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown('<div class="section-title">Mean Power Generation — All Scenarios</div>', unsafe_allow_html=True)
        pw_vals = [75.1] + [SCENARIOS[k]["power_mw"] for k in SCENARIOS]
        pw_labs = ["Historical\n(simulated)"] + scen_labels
        pw_cols = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_pw = go.Figure(go.Bar(x=pw_labs, y=pw_vals, marker_color=pw_cols, marker_opacity=0.85))
        fig_pw.add_hline(y=400, line_dash="dot", line_color="#8b949e",
                          annotation_text="Installed capacity (400 MW)", annotation_font_color="#8b949e")
        fig_pw.update_layout(**plotly_layout("", 320), yaxis_title="Mean power (MW)", xaxis_tickfont_size=9)
        st.plotly_chart(fig_pw, use_container_width=True)

    with col_p2:
        st.markdown('<div class="section-title">Days Below Minimum Operating Level (%)</div>', unsafe_allow_html=True)
        mol_vals = [70.7] + [SCENARIOS[k]["mol_pct"] for k in SCENARIOS]
        mol_labs = ["Historical\n(simulated)"] + scen_labels
        mol_cols = [COLORS["Historical"]] + [SCENARIOS[k]["color"] for k in SCENARIOS]
        fig_mol = go.Figure(go.Bar(x=mol_labs, y=mol_vals, marker_color=mol_cols, marker_opacity=0.85))
        fig_mol.add_hline(y=50, line_dash="dot", line_color="#8b949e",
                           annotation_text="50% threshold", annotation_font_color="#8b949e")
        fig_mol.update_layout(**plotly_layout("", 320), yaxis_title="% of days", xaxis_tickfont_size=9)
        st.plotly_chart(fig_mol, use_container_width=True)

    # RRV radar
    st.markdown('<div class="section-title">RRV Comparison — All Scenarios</div>', unsafe_allow_html=True)
    rrv_data = {
        "Historical":   [0.170, 0.004, 1-0.916],
        "SSP1–2.6 NT":  [0.141, 0.007, 1-0.850],
        "SSP2–4.5 NT":  [0.267, 0.011, 1-0.852],
        "SSP5–8.5 NT":  [0.196, 0.014, 1-0.830],
        "SSP1–2.6 LT":  [0.410, 0.003, 1-0.862],
        "SSP2–4.5 LT":  [0.152, 0.006, 1-0.874],
        "SSP5–8.5 LT":  [0.135, 0.003, 1-0.871],
    }
    rrv_labels = ["Reliability","Resilience","1 − Vulnerability"]
    col_rv1, col_rv2 = st.columns(2)

    with col_rv1:
        fig_radar = go.Figure()
        rrv_colors_map = {
            "Historical":COLORS["Historical"],
            "SSP1–2.6 NT":COLORS["SSP1–2.6"],"SSP2–4.5 NT":COLORS["SSP2–4.5"],"SSP5–8.5 NT":COLORS["SSP5–8.5"],
            "SSP1–2.6 LT":COLORS["SSP1–2.6"],"SSP2–4.5 LT":COLORS["SSP2–4.5"],"SSP5–8.5 LT":COLORS["SSP5–8.5"],
        }
        for name, vals in rrv_data.items():
            r = vals + [vals[0]]
            theta = rrv_labels + [rrv_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r, theta=theta, mode="lines", name=name,
                line=dict(color=rrv_colors_map[name], width=1.8,
                          dash="dash" if "LT" in name else "solid"),
                opacity=0.85,
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#161b22",
                       radialaxis=dict(visible=True, range=[0,0.5],
                                       gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                                       tickfont=dict(color="#8b949e",size=9)),
                       angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                                        tickfont=dict(color=FONT_COLOR,size=10))),
            paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG, height=340,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1,
                        font=dict(size=10)),
            margin=dict(l=40,r=40,t=30,b=30), font=dict(color=FONT_COLOR),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_rv2:
        # Recovery time chart
        rec_scens = ["Historical","SSP1-2.6\nNT","SSP2-4.5\nNT","SSP5-8.5\nNT",
                     "SSP1-2.6\nLT","SSP2-4.5\nLT","SSP5-8.5\nLT"]
        rec_vals  = [228.6,136.4,89.2,68.8,336.8,161.4,303.9]
        rec_nevt  = [41,69,90,128,16,48,26]
        rec_cols  = [COLORS["Historical"],
                     COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"],
                     COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"]]
        fig_rec = make_subplots(specs=[[{"secondary_y":True}]])
        fig_rec.add_trace(go.Bar(x=rec_scens, y=rec_vals, name="Mean recovery (days)",
                                  marker_color=rec_cols, marker_opacity=0.8), secondary_y=False)
        fig_rec.add_trace(go.Scatter(x=rec_scens, y=rec_nevt, name="No. events",
                                      mode="lines+markers", line=dict(color="#58a6ff",width=2),
                                      marker=dict(size=7)), secondary_y=True)
        fig_rec.update_layout(**plotly_layout("Drought Recovery Time & Event Count", 340))
        fig_rec.update_yaxes(title_text="Mean recovery (days)", secondary_y=False, gridcolor=GRID_COLOR)
        fig_rec.update_yaxes(title_text="Number of events", secondary_y=True, gridcolor=GRID_COLOR)
        st.plotly_chart(fig_rec, use_container_width=True)

#  TAB 4 — CLIMATE INPUTS
with tab4:
    st.markdown('<div class="section-title">Bias-Corrected CMIP6 Climate Variables</div>', unsafe_allow_html=True)

    ssp_list   = ["Historical", "SSP1–2.6 NT","SSP2–4.5 NT","SSP5–8.5 NT","SSP1–2.6 LT","SSP2–4.5 LT","SSP5–8.5 LT"]
    temp_vals  = [28.0, 28.09, 28.33, 28.51, 28.90, 30.14, 31.14]
    precip_vals= [2.68, 2.71,  2.69,  2.65,  3.16,  3.28,  3.09]
    t_cols     = [COLORS["Historical"],
                  COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"],
                  COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"]]

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Bar(x=ssp_list, y=temp_vals, marker_color=t_cols,
                                   marker_opacity=0.85, name="Mean temp"))
        fig_temp.add_hline(y=28.0, line_dash="dash", line_color=COLORS["Historical"],
                            annotation_text="Obs baseline (28.0°C)",
                            annotation_font_color=COLORS["Historical"])
        fig_temp.update_layout(**plotly_layout("Mean Temperature (°C) by Scenario", 300),
                                yaxis_title="Temperature (°C)", yaxis_range=[27.5, 31.8],
                                xaxis_tickfont_size=9)
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_t2:
        fig_prec = go.Figure()
        fig_prec.add_trace(go.Bar(x=ssp_list, y=precip_vals, marker_color=t_cols,
                                   marker_opacity=0.85, name="Mean precip"))
        fig_prec.add_hline(y=2.68, line_dash="dash", line_color=COLORS["Historical"],
                            annotation_text="Obs baseline (2.68 mm/day)",
                            annotation_font_color=COLORS["Historical"])
        fig_prec.update_layout(**plotly_layout("Mean Precipitation (mm/day) by Scenario", 300),
                                yaxis_title="Precipitation (mm/day)", yaxis_range=[2.5, 3.5],
                                xaxis_tickfont_size=9)
        st.plotly_chart(fig_prec, use_container_width=True)

    # Warming trajectory scatter
    st.markdown('<div class="section-title">Warming vs Flow Change — Scenario Space</div>', unsafe_allow_html=True)
    delta_temp_all  = [0.09, 0.33, 0.51, 0.90, 2.14, 3.14]
    delta_flow_all  = [-15.2, 28.8, 1.7, 154.5, -17.9, -20.4]
    bubble_ssp      = ["SSP1–2.6 NT","SSP2–4.5 NT","SSP5–8.5 NT","SSP1–2.6 LT","SSP2–4.5 LT","SSP5–8.5 LT"]
    bub_cols        = [COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"],
                       COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"]]

    fig_space = go.Figure()
    fig_space.add_hline(y=0, line_dash="dot", line_color="#8b949e")
    fig_space.add_vline(x=0, line_dash="dot", line_color="#8b949e")
    fig_space.add_vrect(x0=0,x1=4, fillcolor="rgba(248,81,73,0.04)", line_width=0)
    fig_space.add_annotation(x=3.5, y=160, text="High warming", showarrow=False,
                               font=dict(color="#8b949e",size=10))

    for label, dt, df_, col in zip(bubble_ssp, delta_temp_all, delta_flow_all, bub_cols):
        fig_space.add_trace(go.Scatter(
            x=[dt], y=[df_], mode="markers+text", name=label,
            marker=dict(size=16, color=col, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            text=[label.replace(" ","\n")], textposition="top center",
            textfont=dict(size=9, color=col),
        ))
    fig_space.update_layout(
        **plotly_layout("ΔTemp (°C) vs Δ Mean Annual Flow (%) — Scenario Space", 380),
        xaxis_title="ΔTemp vs Baseline (°C)",
        yaxis_title="Δ Mean Annual Flow (%)",
        showlegend=False,
    )
    st.plotly_chart(fig_space, use_container_width=True)

    # OAT sensitivity
    st.markdown('<div class="section-title">OAT Sensitivity — Discharge Response to Climate Perturbations</div>', unsafe_allow_html=True)
    pert      = [-30,-20,-10,-5,5,10,20,30]
    rf_prec   = [0.018,0.011,-0.015,-0.009,-0.021,-0.006,-0.003,-0.005]
    xgb_prec  = [0.019,-0.029,-0.041,-0.031,0.009,-0.009,-0.062,-0.068]
    rf_temp   = [0.365,0.232,0.149,0.142,-0.179,-0.274,-0.464,-0.550]
    xgb_temp  = [-0.340,0.002,0.317,0.110,-0.261,-0.400,-1.157,-2.607]

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        fig_oat1 = go.Figure()
        fig_oat1.add_trace(go.Scatter(x=pert, y=rf_prec, name="RF", mode="lines+markers",
                                       line=dict(color=COLORS["SSP2–4.5"],width=2), marker=dict(size=7)))
        fig_oat1.add_trace(go.Scatter(x=pert, y=xgb_prec, name="XGBoost", mode="lines+markers",
                                       line=dict(color=COLORS["SSP5–8.5"],width=2), marker=dict(size=7)))
        fig_oat1.add_hline(y=0, line_dash="dot", line_color="#8b949e")
        fig_oat1.update_layout(**plotly_layout("Precipitation Perturbation → Discharge Change (%)", 280),
                                xaxis_title="Precipitation change (%)", yaxis_title="Δ Discharge (%)")
        st.plotly_chart(fig_oat1, use_container_width=True)

    with col_o2:
        fig_oat2 = go.Figure()
        fig_oat2.add_trace(go.Scatter(x=pert, y=rf_temp, name="RF", mode="lines+markers",
                                       line=dict(color=COLORS["SSP2–4.5"],width=2), marker=dict(size=7)))
        fig_oat2.add_trace(go.Scatter(x=pert, y=xgb_temp, name="XGBoost", mode="lines+markers",
                                       line=dict(color=COLORS["SSP5–8.5"],width=2), marker=dict(size=7)))
        fig_oat2.add_hline(y=0, line_dash="dot", line_color="#8b949e")
        fig_oat2.update_layout(**plotly_layout("Temperature Perturbation → Discharge Change (%)", 280),
                                xaxis_title="Temperature change (%)", yaxis_title="Δ Discharge (%)")
        st.plotly_chart(fig_oat2, use_container_width=True)

#  TAB 5 — MODEL PERFORMANCE
with tab5:
    st.markdown('<div class="section-title">Model Evaluation Metrics — Test Period (2019–2020)</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([2, 1])

    with col_m1:
        models_list  = list(MODEL_PERF.keys())
        nse_vals     = [MODEL_PERF[m]["nse"]  for m in models_list]
        kge_vals     = [MODEL_PERF[m]["kge"]  for m in models_list]
        rmse_vals    = [MODEL_PERF[m]["rmse"] for m in models_list]
        mae_vals     = [MODEL_PERF[m]["mae"]  for m in models_list]
        model_colors = [COLORS["SSP1–2.6"],COLORS["SSP2–4.5"],COLORS["SSP5–8.5"]]*2

        fig_perf = make_subplots(rows=1, cols=2, subplot_titles=["NSE & KGE","RMSE & MAE (m³/s)"])
        fig_perf.add_trace(go.Bar(x=models_list, y=nse_vals, name="NSE",
                                   marker_color=COLORS["SSP1–2.6"], marker_opacity=0.85), row=1, col=1)
        fig_perf.add_trace(go.Bar(x=models_list, y=kge_vals, name="KGE",
                                   marker_color=COLORS["SSP2–4.5"], marker_opacity=0.85), row=1, col=1)
        fig_perf.add_hline(y=0.75, line_dash="dot", line_color="#8b949e", row=1, col=1)
        fig_perf.add_trace(go.Bar(x=models_list, y=rmse_vals, name="RMSE",
                                   marker_color=COLORS["SSP5–8.5"], marker_opacity=0.85), row=1, col=2)
        fig_perf.add_trace(go.Bar(x=models_list, y=mae_vals, name="MAE",
                                   marker_color=COLORS["Historical"], marker_opacity=0.85), row=1, col=2)
        fig_perf.update_layout(
            height=340, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR, size=10), barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=40,r=20,t=50,b=80),
        )
        for ax in ["xaxis","xaxis2"]:
            fig_perf.update_layout(**{ax: dict(gridcolor=GRID_COLOR, tickangle=-30, tickfont_size=9)})
        for ax in ["yaxis","yaxis2"]:
            fig_perf.update_layout(**{ax: dict(gridcolor=GRID_COLOR)})
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-title">Ablation — NSE vs Feature Count</div>', unsafe_allow_html=True)
        n_feat = list(range(1,16))
        rf_abl  = [0.9852,0.9903,0.9901,0.9889,0.9889,0.9889,0.9895,0.9908,
                   0.9900,0.9903,0.9902,0.9904,0.9902,0.9905,0.9902]
        xgb_abl = [0.9906,0.9910,0.9897,0.9891,0.9879,0.9891,0.9902,0.9907,
                   0.9909,0.9907,0.9910,0.9913,0.9901,0.9904,0.9906]
        fig_abl = go.Figure()
        fig_abl.add_trace(go.Scatter(x=n_feat, y=rf_abl, name="RF", mode="lines+markers",
                                      line=dict(color=COLORS["SSP1–2.6"],width=2), marker=dict(size=5)))
        fig_abl.add_trace(go.Scatter(x=n_feat, y=xgb_abl, name="XGBoost", mode="lines+markers",
                                      line=dict(color=COLORS["SSP2–4.5"],width=2), marker=dict(size=5)))
        fig_abl.add_hline(y=0.75, line_dash="dot", line_color="#8b949e")
        fig_abl.update_layout(**plotly_layout("", 320),
                               xaxis_title="No. features", yaxis_title="NSE",
                               yaxis_range=[0.98,0.995], xaxis_dtick=2)
        st.plotly_chart(fig_abl, use_container_width=True)

    # Feature importance table
    st.markdown('<div class="section-title">Feature Importance Ranking (RF Ablation Order)</div>', unsafe_allow_html=True)
    feat_data = pd.DataFrame({
        "Rank": range(1,16),
        "Feature": ["chache_lag1","chache_lag3","chache_lag7","doy_cos","soil_moisture",
                    "precip_7d","temp_c","lawra_discharge","precip_mm","doy_sin",
                    "precip_30d","lawra_lag7","lawra_lag3","lawra_lag1","dam"],
        "RF NSE at n": rf_abl,
        "XGB NSE at n": xgb_abl,
        "Type": ["Lag flow","Lag flow","Lag flow","Seasonality","Antecedent moisture",
                 "Rolling precip","Temperature","Upstream flow","Precipitation","Seasonality",
                 "Rolling precip","Lag flow","Lag flow","Lag flow","Dam flag"],
    })
    feat_data["RF NSE at n"]  = feat_data["RF NSE at n"].map(lambda x: f"{x:.4f}")
    feat_data["XGB NSE at n"] = feat_data["XGB NSE at n"].map(lambda x: f"{x:.4f}")
    st.dataframe(
        feat_data, use_container_width=True, hide_index=True,
        column_config={
            "Rank":         st.column_config.NumberColumn(width="small"),
            "Feature":      st.column_config.TextColumn(width="medium"),
            "RF NSE at n":  st.column_config.TextColumn("RF NSE"),
            "XGB NSE at n": st.column_config.TextColumn("XGB NSE"),
            "Type":         st.column_config.TextColumn(width="medium"),
        }
    )

    st.markdown("""
    <div class="disclaimer" style="margin-top:12px;">
    <b>Methodology note:</b> All three models trained on 1990–2015 (9,467 daily records) with
    chronological validation (2016–2018) and test (2019–2020) splits.
    Random Forest (200 trees) was selected as the projection driver based on best overall test performance.
    LSTM architecture: stacked 128→64 units, 30-day sequence window, 150 epochs, early stopping patience=15.
    Climate projections use a delta bias-correction anchored to the 2000–2020 observational reference.
    SSP1–2.6 derived by scaling SSP2–4.5 anomalies (T×0.64, P×0.80).
    </div>
    """, unsafe_allow_html=True)

#  Footer 
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#8b949e;font-size:0.78rem;padding:8px 0;">
ML-based Streamflow Prediction under Climate Change Scenarios · Bui Hydroelectric Dam · Black Volta Basin, Ghana<br>
</div>
""", unsafe_allow_html=True)