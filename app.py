# =============================================================
# app.py — PISA Streamlit Dashboard
# Run: streamlit run app.py
#
# Tabs:
#   1. Overview         — today's headline, freshness runway, wastage figures
#   2. Demand forecast  — SKU-level forecast + newsvendor orders
#   3. Spoilage alerts  — model-scored lots, lot tags, cold-room sensors
#   4. Order planner    — newsvendor recommendations per SKU
#   5. Model health     — MAPE, F1, survival, drift
#   6. Ask PISA         — Claude analyst over the DuckDB warehouse
#
# Theme: .streamlit/config.toml (light + dark); tokens mirrored in TOKENS below.
# =============================================================

import html, json, os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import SKUS, WAREHOUSES, DB_PATH, REPORTS_DIR
import ml_models

# ── Auto-setup: generate data + train models if missing ──────
@st.cache_resource(show_spinner=False)
def auto_setup():
    """First launch: run the nightly pipeline (ingest → train → score → DuckDB)."""
    need_ingest = not os.path.exists("data/sensor_readings.csv")  # pre-v2 data lacks sensors/vendors
    if need_ingest or not os.path.exists("models/spoilage_model.pkl") or not os.path.exists(DB_PATH):
        import pipeline
        pipeline.run(skip_ingest=not need_ingest)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PISA",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)

# ── Design tokens ("cold-room desk") ──────────────────────────
# High-contrast light mode palette by default, ensuring all text, tags and charts
# remain crisp and perfectly readable.
THEME = "light"
TOKENS = {
    "light": dict(bg="#F2F5F7", surface="#FFFFFF", surface2="#E4EAEE", ink="#142029", steel="#475569",
                  rule="#CBD5E1", blue="#1F6F8B", red="#CB202D", amber="#D9731A", mustard="#B8901C",
                  green="#2F7D5B"),
    "dark":  dict(bg="#0E1922", surface="#13212B", surface2="#1B2C38", ink="#F1F5F9", steel="#94A3B8",
                  rule="#334155", blue="#5DB0CF", red="#E5414D", amber="#EE8A3A", mustard="#D9B444",
                  green="#4DA67E"),
}
T = TOKENS[THEME]


def hex_alpha_early(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RISK_COLOR = {"CRITICAL": T["red"], "HIGH": T["amber"], "MEDIUM": T["mustard"], "LOW": T["green"]}
RISK_LABEL = {"CRITICAL": "Critical", "HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
CAT_COLORS = [T["blue"], T["steel"], T["amber"], T["green"], T["mustard"], T["red"]]
SEQ        = [hex_alpha_early(T["blue"], 0.35), T["blue"]]   # neutral magnitude; low end stays visible
FONT       = "Archivo, sans-serif"


hex_alpha = hex_alpha_early


def inr(v):
    if v >= 1e7:
        return f"₹{v / 1e7:.1f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:.1f}L"
    return f"₹{v:,.0f}"


# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  :root, .stApp {
    --pisa-ink: #142029;
    --pisa-steel: #475569;
    --pisa-surface: #FFFFFF;
    --pisa-surface2: #E4EAEE;
    --pisa-rule: #CBD5E1;
    --pisa-blue: #1F6F8B;
    --pisa-red: #CB202D;
  }

  .stApp { font-variant-numeric: tabular-nums; }
  .block-container { padding-top: 3.75rem; max-width: 1360px; }
  h1, h2, h3 { font-stretch: 75%; letter-spacing: -0.01em; color: var(--pisa-ink) !important; }
  :focus-visible { outline: 2px solid var(--pisa-blue); outline-offset: 2px; }

  /* category chips: crisp neutral surface with dark text */
  [data-testid="stMultiSelectTagsContainer"] [data-tag] {
      background: var(--pisa-surface) !important; color: var(--pisa-ink) !important; border: 1px solid var(--pisa-rule) !important; }
  [data-testid="stMultiSelectTagsContainer"] [data-tag] * { color: var(--pisa-ink) !important; fill: var(--pisa-ink) !important; }

  /* masthead */
  .mast { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;
           gap: .75rem 2rem; padding-bottom: .9rem; border-bottom: 2px solid var(--pisa-ink); }
  .mast-mark { font-stretch: 62%; font-weight: 900; font-size: 3rem; line-height: .9;
                letter-spacing: -0.02em; color: var(--pisa-red); }
  .mast-name { font-weight: 650; font-size: 1.05rem; color: var(--pisa-ink) !important; margin-top: .35rem; }
  .mast-sub  { color: var(--pisa-steel) !important; font-size: .9rem; }
  .mast-meta { text-align: right; color: var(--pisa-steel) !important; font-size: .9rem; line-height: 1.45; }
  .mast-meta b { color: var(--pisa-ink) !important; font-weight: 650; }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 1.75rem; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; font-stretch: 87.5%; font-size: 1rem;
                                  padding-left: 0; padding-right: 0; color: var(--pisa-steel); }
  .stTabs [aria-selected="true"] { color: var(--pisa-red) !important; }

  /* overview headline */
  .headline { font-stretch: 68%; font-weight: 800; font-size: clamp(2.1rem, 4.6vw, 3.6rem);
               line-height: 1; letter-spacing: -0.015em; color: var(--pisa-ink) !important; max-width: 20ch;
               margin: 1.4rem 0 .6rem; }
  .lede { color: var(--pisa-steel) !important; font-size: 1.05rem; line-height: 1.5; max-width: 62ch; margin: 0 0 .25rem; }

  /* tab intros + section heads */
  .tab-title { font-stretch: 72%; font-weight: 800; font-size: 2rem; line-height: 1.05;
                color: var(--pisa-ink) !important; margin: 1.1rem 0 .3rem; }
  .tab-note  { color: var(--pisa-steel) !important; font-size: .975rem; line-height: 1.5; max-width: 70ch; margin-bottom: .75rem; }
  .sec { font-stretch: 80%; font-weight: 700; font-size: 1.2rem; color: var(--pisa-ink) !important;
          margin: 1.6rem 0 .35rem; padding-top: .9rem; border-top: 1px solid var(--pisa-rule); }

  /* figure strip */
  .figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
           border-top: 1px solid var(--pisa-rule); border-bottom: 1px solid var(--pisa-rule); margin: 1rem 0 .5rem; }
  .fig { padding: .9rem 1rem .95rem 0; }
  .fig + .fig { border-left: 1px solid var(--pisa-rule); padding-left: 1rem; }
  .fig.risk { box-shadow: inset 0 3px 0 var(--c); }
  .fig-v { font-stretch: 75%; font-weight: 750; font-size: 2.15rem; line-height: 1.05; color: var(--pisa-ink) !important; }
  .fig-l { color: var(--pisa-steel) !important; font-size: .875rem; margin-top: .2rem; }
  .fig-d { font-size: .8125rem; font-weight: 600; margin-top: .35rem; }
  .good { color: #2F7D5B; }
  .bad  { color: #CB202D; }

  /* lot tags */
  .tags { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .75rem; margin: .4rem 0 .5rem; }
  .tag { background: var(--pisa-surface) !important; border: 1px solid var(--pisa-rule) !important; border-top: 4px solid var(--c) !important;
          border-radius: 4px; padding: .85rem 1rem .8rem; }
  .tag-sku  { font-stretch: 80%; font-weight: 700; font-size: 1.2rem; color: var(--pisa-ink) !important; }
  .tag-id   { color: var(--pisa-steel) !important; font-size: .8125rem; }
  .tag-odds { font-stretch: 68%; font-weight: 800; font-size: 2.4rem; line-height: 1; color: var(--c); margin-top: .55rem; }
  .tag-odds-l { color: var(--pisa-steel) !important; font-size: .8125rem; }
  .tag dl { display: grid; grid-template-columns: auto 1fr; gap: .15rem .75rem; margin: .65rem 0 .55rem; font-size: .875rem; }
  .tag dt { color: var(--pisa-steel) !important; }
  .tag dd { margin: 0; color: var(--pisa-ink) !important; text-align: right; }
  .tag-act { font-weight: 600; font-size: .875rem; color: var(--pisa-ink) !important; border-top: 1px solid var(--pisa-rule) !important; padding-top: .5rem; }

  /* metrics */
  [data-testid="stMetric"] { border-left: 1px solid var(--pisa-rule); padding-left: .9rem; }
  [data-testid="stMetricValue"] { font-stretch: 75%; font-weight: 750; color: var(--pisa-ink) !important; }
  [data-testid="stMetricLabel"] { color: var(--pisa-steel) !important; }

  /* sidebar */
  .side-mark { font-stretch: 62%; font-weight: 900; font-size: 2.2rem; line-height: .9; color: var(--pisa-red); }
  .side-sub  { color: var(--pisa-steel) !important; font-size: .875rem; margin: .3rem 0 1rem; }
  .side-about { color: var(--pisa-steel) !important; font-size: .85rem; line-height: 1.5; }

  @media (max-width: 640px) {
    .mast-meta { text-align: left; }
    .fig + .fig { border-left: none; padding-left: 0; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=T["ink"], size=13),
    title_font=dict(family=FONT, size=15, color=T["ink"]),
    margin=dict(l=8, r=8, t=40, b=8),
)


def show(fig):
    """Apply the shared chart chrome and render full width."""
    fig.update_xaxes(
        gridcolor=T["rule"], linecolor=T["rule"], zerolinecolor=T["rule"],
        tickfont=dict(color=T["ink"]), title_font=dict(color=T["ink"]),
        automargin=True
    )
    fig.update_yaxes(
        gridcolor=T["rule"], linecolor=T["rule"], zerolinecolor=T["rule"],
        tickfont=dict(color=T["ink"]), title_font=dict(color=T["ink"]),
        automargin=True
    )
    fig.update_layout(
        legend=dict(font=dict(color=T["ink"])),
        legend_bgcolor="rgba(255,255,255,0.6)",
        hoverlabel=dict(font_family=FONT, font_color=T["ink"], bgcolor=T["surface"])
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def tab_intro(title, note):
    st.markdown(f'<div class="tab-title" role="heading" aria-level="2">{title}</div>'
                f'<div class="tab-note">{note}</div>', unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="sec" role="heading" aria-level="3">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_demand():
    df = pd.read_csv("data/demand_data.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_lots():
    df = pd.read_csv("data/lot_data.csv")
    return df

@st.cache_resource
def load_models():
    return ml_models.load_artifacts()

@st.cache_data
def load_active_lots():
    """Today's lots, risk-scored by the spoilage models (RF + Cox)."""
    sp, _ = load_models()
    return ml_models.score_active_lots(pd.read_csv("data/active_lots.csv"), sp)

@st.cache_data
def load_sensors():
    return pd.read_csv("data/sensor_readings.csv", parse_dates=["reading_ts"])

def load_drift_report():
    path = os.path.join(REPORTS_DIR, "drift_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_llm_key():
    try:
        if "user_llm_key" in st.session_state and st.session_state["user_llm_key"]:
            return st.session_state["user_llm_key"].strip()
    except Exception:
        pass
    for k in ["GROQ_API_KEY", "ANTHROPIC_API_KEY"]:
        v = os.getenv(k)
        if v:
            return v
        try:
            if hasattr(st, "secrets") and k in st.secrets:
                return st.secrets[k]
        except Exception:
            pass
    return None

get_anthropic_key = get_llm_key


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
def render_header(sensors):
    now  = pd.Timestamp.now()
    mins = max(0, int((now - sensors["reading_ts"].max()).total_seconds() // 60))
    fresh = (f"{mins} min ago" if mins < 120 else f"{mins // 60} hours ago" if mins < 2880
             else f"{mins // 1440} days ago")
    cities = ", ".join(w["city"] for w in WAREHOUSES[:-1]) + f" and {WAREHOUSES[-1]['city']}"
    st.markdown(
        f'<div class="mast">'
        f'  <div><div class="mast-mark">PISA</div>'
        f'       <div class="mast-name">Hyperpure cold-chain desk</div>'
        f'       <div class="mast-sub">Predictive inventory and spoilage alerts for the {cities} hubs</div></div>'
        f'  <div class="mast-meta"><b>{now:%A}, {now.day} {now:%B}</b><br>Cold-room sensors read {fresh}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def figure_strip(items):
    """items: (value, label, delta_html or '', css_var_color or None)"""
    cells = ""
    for v, l, d, c in items:
        attrs = f'class="fig risk" style="--c:{c}"' if c else 'class="fig"'
        delta = f'<div class="fig-d">{d}</div>' if d else ""
        cells += f'<div {attrs}><div class="fig-v">{v}</div><div class="fig-l">{l}</div>{delta}</div>'
    st.markdown(f'<div class="figs">{cells}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────
def tab_overview(lot_df, demand_df, active_lots):
    # ── Headline: the decision for today ──
    if active_lots.empty:
        st.markdown('<div class="headline" role="heading" aria-level="1">No lots in stock for this selection.</div>'
                    '<p class="lede">Pick another hub or add categories in the sidebar to see today\'s stock.</p>',
                    unsafe_allow_html=True)
        return
    urgent = active_lots[active_lots["spoil_prob_48h"] >= 50]
    n, at_stake = len(urgent), urgent["lot_value_inr"].sum()
    if n:
        head = f"{n} lot{'s are' if n != 1 else ' is'} likely to spoil within 48 hours."
        lede = (f"{inr(at_stake)} of stock is at stake. Dispatch, discount or move these lots first; "
                f"each one is listed under Spoilage alerts.")
    else:
        head = "Nothing in stock is likely to spoil within 48 hours."
        lede = "Every active lot has less than a 50% chance of spoiling before the day after tomorrow."
    st.markdown(f'<div class="headline" role="heading" aria-level="1">{head}</div>'
                f'<p class="lede">{lede}</p>', unsafe_allow_html=True)

    # ── Freshness runway: every lot in stock by days of shelf life left ──
    lane_order = (active_lots.groupby("category")["shelf_life_days"].mean()
                  .sort_values(ascending=False).index.tolist())  # fastest-spoiling lane on top
    lots = active_lots.copy()
    rng  = np.random.default_rng(7)  # fixed jitter so dots don't move on rerun
    lots["x"]    = lots["days_remaining"] + rng.uniform(-0.28, 0.28, len(lots))
    lots["size"] = np.clip(np.sqrt(lots["lot_value_inr"]) / 11, 8, 30)

    fig = go.Figure()
    fig.add_vrect(x0=-0.6, x1=2, fillcolor=T["red"], opacity=0.07, line_width=0,
                  annotation_text="Next 48 hours", annotation_position="top left",
                  annotation_font=dict(color=T["red"], size=12, family=FONT))
    for level in reversed(RISK_ORDER):  # draw critical last so it sits on top
        d = lots[lots["risk_level"] == level]
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["category"], mode="markers", name=RISK_LABEL[level],
            marker=dict(size=d["size"], color=RISK_COLOR[level], opacity=0.92,
                        line=dict(width=1.5, color=T["bg"])),
            customdata=d[["sku_name", "lot_id", "warehouse_name", "lot_value_inr",
                          "spoil_prob_48h", "days_remaining"]],
            hovertemplate=("<b>%{customdata[0]}</b>  %{customdata[1]}<br>%{customdata[2]}<br>"
                           "%{customdata[5]} days left, %{customdata[4]:.0f}% chance of spoiling in 48h<br>"
                           "₹%{customdata[3]:,.0f}<extra></extra>"),
        ))
    fig.update_layout(**LAYOUT_BASE, height=330, hovermode="closest",
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.14, traceorder="reversed",
                                  title_text="Risk  "),
                      xaxis=dict(title="Days of shelf life left", dtick=1, range=[-0.6, lots["days_remaining"].max() + 0.8]),
                      yaxis=dict(title=None, categoryorder="array", categoryarray=lane_order))  # first entry = bottom lane
    show(fig)
    st.caption("Each dot is one lot in stock today, sized by value. Lanes are ordered from the fastest-spoiling category down.")

    # ── Figure strip ──
    total_value   = lot_df["lot_value"].sum()
    wastage_pct   = lot_df["spoiled_value"].sum() / total_value * 100

    months = pd.to_datetime(lot_df["procurement_date"]).dt.to_period("M")
    by_m   = lot_df.groupby(months).agg(sp=("spoiled_value", "sum"), tot=("lot_value", "sum"))
    by_m   = by_m["sp"] / by_m["tot"] * 100
    delta_html = ""
    if len(by_m) >= 2:
        diff = by_m.iloc[-1] - by_m.iloc[-2]
        word = "lower" if diff <= 0 else "higher"
        delta_html = (f'<span class="{"good" if diff <= 0 else "bad"}">{abs(diff):.1f} pts {word} '
                      f'in {by_m.index[-1].strftime("%b")} than {by_m.index[-2].strftime("%b")}</span>')

    total_demand   = demand_df["actual_demand_kg"].sum()
    total_supplied = lot_df["quantity_kg"].sum() - lot_df["spoiled_kg"].sum()
    fill_rate      = min(100.0, total_supplied / total_demand * 100)

    annual_cogs   = demand_df["revenue"].sum()
    avg_inv_value = lot_df.groupby("sku_id")["quantity_kg"].mean().sum() * lot_df["price_per_kg"].mean()
    inv_turnover  = min(annual_cogs / max(avg_inv_value, 1), 68.0)  # capped to a realistic perishables range

    at_risk = active_lots[active_lots["risk_level"].isin(["CRITICAL", "HIGH"])]["lot_value_inr"].sum()
    figure_strip([
        (f"{wastage_pct:.1f}%", "Stock value lost to spoilage, FY2024", delta_html, None),
        (f"{fill_rate:.1f}%",   "Demand filled from stock", "", None),
        (f"{inv_turnover:.0f}×", "Inventory turnover", "", None),
        (inr(at_risk),          "Stock at high or critical risk today", "", T["red"]),
    ])

    # ── Where the money goes ──
    col_a, col_b = st.columns(2)
    with col_a:
        sec("Wastage by category")
        cat_waste = (lot_df.groupby("category")
                     .agg(total_value=("lot_value", "sum"), spoiled_value=("spoiled_value", "sum"))
                     .assign(wastage_pct=lambda x: x.spoiled_value / x.total_value * 100)
                     .reset_index().sort_values("wastage_pct"))
        fig = go.Figure(go.Bar(
            x=cat_waste["wastage_pct"], y=cat_waste["category"], orientation="h",
            marker_color=T["blue"], text=cat_waste["wastage_pct"].map(lambda v: f"{v:.1f}%"),
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}% of stock value spoiled<extra></extra>"))
        fig.update_layout(**LAYOUT_BASE, height=300, xaxis_title="% of stock value spoiled", yaxis_title=None,
                          xaxis_range=[0, cat_waste["wastage_pct"].max() * 1.18])
        show(fig)

    with col_b:
        sec("Monthly wastage")
        lot_m = lot_df.assign(month=pd.to_datetime(lot_df["procurement_date"]).dt.month)
        monthly = (lot_m.groupby("month").agg(spoiled=("spoiled_value", "sum"), total=("lot_value", "sum"))
                   .assign(wastage_pct=lambda x: x.spoiled / x.total * 100).reset_index())
        monthly["month_name"] = pd.to_datetime(monthly["month"].astype(str), format="%m").dt.strftime("%b")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=monthly["month_name"], y=monthly["spoiled"] / 1e5, name="Spoiled value (₹ lakh)",
                              marker_color=hex_alpha(T["steel"], 0.45)))
        fig2.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["wastage_pct"], name="Wastage %",
                                  line=dict(color=T["blue"], width=2.5), mode="lines+markers"),
                       secondary_y=True)
        fig2.update_layout(**LAYOUT_BASE, height=300, hovermode="x unified",
                           legend=dict(orientation="h", x=0, y=1.15))
        fig2.update_yaxes(title_text="₹ lakh", secondary_y=False)
        fig2.update_yaxes(title_text="Wastage", secondary_y=True, showgrid=False, tickformat=".0f", ticksuffix="%")
        show(fig2)

    sec("SKUs losing the most money")
    top = (lot_df.groupby(["sku_name", "category"])["spoiled_value"].sum()
           .reset_index().sort_values("spoiled_value").tail(10))
    fig3 = go.Figure(go.Bar(
        x=top["spoiled_value"] / 1e5, y=top["sku_name"], orientation="h", marker_color=T["blue"],
        customdata=top["category"], text=(top["spoiled_value"] / 1e5).map(lambda v: f"₹{v:.1f}L"),
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y} (%{customdata}): ₹%{x:.1f} lakh spoiled<extra></extra>"))
    fig3.update_layout(**LAYOUT_BASE, height=360, xaxis_title="Spoiled value, FY2024 (₹ lakh)", yaxis_title=None,
                       xaxis_range=[0, top["spoiled_value"].max() / 1e5 * 1.12])
    show(fig3)


# ─────────────────────────────────────────────────────────────
# TAB 2: DEMAND FORECAST
# ─────────────────────────────────────────────────────────────
def tab_demand(demand_df, dm_artifact, sku_models=None):
    tab_intro("Demand forecast", "Choose a SKU and hub to see expected demand and the order quantity that keeps both spoilage and stockouts low.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sku_options = demand_df[["sku_id","sku_name"]].drop_duplicates()
        sku_names   = sku_options["sku_name"].tolist()
        sku_ids     = sku_options["sku_id"].tolist()
        sel_idx     = st.selectbox("Select SKU", range(len(sku_names)),
                                    format_func=lambda i: sku_names[i])
        sel_sku_id  = sku_ids[sel_idx]
        sel_sku_nm  = sku_names[sel_idx]

    with col2:
        warehouses  = demand_df["warehouse_id"].unique().tolist()
        wh_names    = demand_df[["warehouse_id","warehouse_name"]].drop_duplicates()
        wh_map      = dict(zip(wh_names["warehouse_id"], wh_names["warehouse_name"]))
        sel_wh      = st.selectbox("Select Warehouse", warehouses,
                                    format_func=lambda w: wh_map.get(w, w))

    with col3:
        horizon     = st.slider("Forecast Days", 7, 30, 14)

    # Filter data
    df_sku = demand_df[
        (demand_df["sku_id"] == sel_sku_id) &
        (demand_df["warehouse_id"] == sel_wh)
    ].sort_values("date")

    if len(df_sku) == 0:
        st.warning("No data for this combination.")
        return

    # Generate forecast
    from ml_models import generate_forecast
    forecast_df = generate_forecast(demand_df, sel_sku_id, sel_wh, horizon,
                                    sku_models=sku_models)

    # Build chart: historical + forecast
    fig = go.Figure()

    # Historical — last 60 days
    hist = df_sku.tail(60)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["actual_demand_kg"],
        name="Actual demand",
        line=dict(color=T["steel"], width=2),
        mode="lines+markers", marker=dict(size=4),
    ))

    if forecast_df is not None:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
            y=list(forecast_df["upper_bound"]) + list(forecast_df["lower_bound"])[::-1],
            fill="toself", fillcolor=hex_alpha(T["blue"], 0.14),
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band", showlegend=True,
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast_kg"],
            name="Forecast",
            line=dict(color=T["blue"], width=2.5, dash="dash"),
            mode="lines+markers", marker=dict(size=5),
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=380,
        title=f"Demand Forecast — {sel_sku_nm} @ {wh_map.get(sel_wh, sel_wh)}",
        xaxis_title="Date", yaxis_title="Demand (kg)",
        legend=dict(x=0.01, y=0.99),
    )
    show(fig)

    # ── Recommended Orders ──
    if forecast_df is not None:
        sec("Recommended orders")
        src = ("live OpenWeatherMap forecasts" if (forecast_df["temp_source"] == "api").any()
               else "seasonal averages (set OPENWEATHER_API_KEY for live forecasts)")
        st.caption(f"Orders sit at the 20th percentile of forecast demand, because spoilage costs 4× a stockout. "
                   f"Temperature comes from {src}.")

        disp = forecast_df[["date","forecast_kg","lower_bound","upper_bound","optimal_order"]].copy()
        disp.columns = ["Date","Forecast (kg)","Lower Bound","Upper Bound","✅ Optimal Order (kg)"]

        def highlight_order(row):
            return [""] * 4 + [f"background-color: {hex_alpha(T['blue'], 0.16)}; font-weight:600"]

        st.dataframe(
            disp.style.apply(highlight_order, axis=1),
            width="stretch", hide_index=True
        )

    # ── Demand patterns ──
    sec("Average demand by weekday")
    dow_avg = (
        df_sku.assign(dow=df_sku["date"].dt.day_name())
        .groupby("dow")["actual_demand_kg"].mean()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .reset_index()
    )
    fig_dow = px.bar(dow_avg, x="dow", y="actual_demand_kg",
                      color="actual_demand_kg",
                      color_continuous_scale=SEQ,
                      labels={"dow":"Day","actual_demand_kg":"Avg Demand (kg)"},
                      text=dow_avg["actual_demand_kg"].round(1))
    fig_dow.update_traces(textposition="outside")
    fig_dow.update_layout(**LAYOUT_BASE, height=280, showlegend=False,
                           coloraxis_showscale=False)
    show(fig_dow)

    # ── Model MAPE per SKU ──
    sec("Forecast error by SKU (MAPE, lower is better)")
    mape_data = [
        {"SKU ID": k, "MAPE (%)": v["mape"]}
        for k, v in dm_artifact["metrics"].items()
    ]
    mape_df   = pd.DataFrame(mape_data).sort_values("MAPE (%)")
    fig_mape  = px.bar(mape_df, x="SKU ID", y="MAPE (%)",
                        color="MAPE (%)",
                        color_continuous_scale=SEQ,
                        title="Lower MAPE = Better Forecast")
    fig_mape.update_layout(**LAYOUT_BASE, height=280, showlegend=False,
                            coloraxis_showscale=False)
    fig_mape.add_hline(y=mape_df["MAPE (%)"].mean(), line_dash="dash",
                        line_color="grey", annotation_text="Avg MAPE")
    show(fig_mape)


# ─────────────────────────────────────────────────────────────
# TAB 3: SPOILAGE ALERTS
# ─────────────────────────────────────────────────────────────
def lot_tag(row):
    c = RISK_COLOR[row["risk_level"]]
    esc = lambda k: html.escape(str(row[k]))
    return (
        f'<div class="tag" style="--c:{c}">'
        f'<div class="tag-sku">{esc("sku_name")}</div>'
        f'<div class="tag-id">{esc("lot_id")}, {esc("warehouse_name")}</div>'
        f'<div class="tag-odds">{row["spoil_prob_48h"]:.0f}%</div>'
        f'<div class="tag-odds-l">chance of spoiling in the next 48 hours</div>'
        f'<dl><dt>Shelf life left</dt><dd>{int(row["days_remaining"])} of {int(row["shelf_life_days"])} days</dd>'
        f'<dt>Quantity</dt><dd>{row["quantity_kg"]:,.1f} kg</dd>'
        f'<dt>Value</dt><dd>₹{row["lot_value_inr"]:,.0f}</dd>'
        f'<dt>Cold-room temperature</dt><dd>+{row["temp_deviation_c"]:.1f}°C over ideal</dd></dl>'
        f'<div class="tag-act">{esc("recommended_action")}</div>'
        f'</div>'
    )


def tab_alerts(active_lots, sensors):
    tab_intro("Spoilage alerts",
              "Every lot in stock, scored overnight. The risk score comes from the Random Forest, the 48-hour "
              "odds from the Cox survival model, and temperature from the last 24 hours of cold-room sensors.")
    if active_lots.empty:
        st.info("No lots in stock for this hub and category selection. Change the filters in the sidebar.")
        return

    counts = active_lots["risk_level"].value_counts()
    val_at_risk = active_lots[active_lots["risk_level"].isin(["CRITICAL", "HIGH"])]["lot_value_inr"].sum()
    figure_strip(
        [(str(counts.get(lvl, 0)), f"{RISK_LABEL[lvl]} risk lots", "", RISK_COLOR[lvl]) for lvl in RISK_ORDER]
        + [(inr(val_at_risk), "Value in critical and high lots", "", None)]
    )

    # ── Lots to act on first ──
    critical = active_lots[active_lots["risk_level"] == "CRITICAL"].sort_values("spoil_prob_48h", ascending=False)
    if len(critical):
        sec("Act on these first")
        shown = critical.head(6)
        st.markdown(f'<div class="tags">{"".join(lot_tag(r) for _, r in shown.iterrows())}</div>',
                    unsafe_allow_html=True)
        if len(critical) > len(shown):
            st.caption(f"{len(critical) - len(shown)} more critical lots are in the table below.")

    # ── Full table ──
    sec("All lots in stock")
    filter_risk = st.multiselect("Show risk levels", RISK_ORDER, default=["CRITICAL", "HIGH", "MEDIUM"],
                                 format_func=RISK_LABEL.get)
    filtered = active_lots[active_lots["risk_level"].isin(filter_risk)].copy()
    filtered["Risk"] = filtered["risk_level"].map(RISK_LABEL)

    display_df = filtered[["lot_id", "sku_name", "category", "warehouse_name", "days_remaining",
                           "pct_shelf_remaining", "quantity_kg", "lot_value_inr", "risk_score",
                           "spoil_prob_48h", "Risk", "recommended_action"]].copy()
    display_df.columns = ["Lot", "SKU", "Category", "Hub", "Days left", "Shelf life left",
                          "Qty (kg)", "Value (₹)", "Risk score", "Spoil in 48h", "Risk", "Action"]
    st.dataframe(
        display_df, width="stretch", hide_index=True,
        column_config={
            "Days left":       st.column_config.NumberColumn(format="%d"),
            "Qty (kg)":        st.column_config.NumberColumn(format="%.1f"),
            "Value (₹)":       st.column_config.NumberColumn(format="₹%d"),
            "Risk score":      st.column_config.NumberColumn(format="%d", help="Random Forest spoilage score, 0 to 100"),
            "Shelf life left": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "Spoil in 48h":    st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                               help="Cox survival model"),
        },
    )

    # ── Where the risk sits ──
    col_a, col_b = st.columns(2)
    with col_a:
        sec("Lots by risk level")
        rc = active_lots["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
        fig_risk = go.Figure(go.Bar(
            x=[RISK_LABEL[l] for l in RISK_ORDER], y=rc.values,
            marker_color=[RISK_COLOR[l] for l in RISK_ORDER], text=rc.values, textposition="outside",
            cliponaxis=False, hovertemplate="%{x}: %{y} lots<extra></extra>"))
        fig_risk.update_layout(**LAYOUT_BASE, height=280, yaxis_title="Lots", xaxis_title=None)
        show(fig_risk)

    with col_b:
        sec("Value at critical or high risk, by category")
        cat_risk = (active_lots[active_lots["risk_level"].isin(["CRITICAL", "HIGH"])]
                    .groupby("category")["lot_value_inr"].sum().reset_index().sort_values("lot_value_inr"))
        fig_cat = go.Figure(go.Bar(
            x=cat_risk["lot_value_inr"], y=cat_risk["category"], orientation="h", marker_color=T["red"],
            text=cat_risk["lot_value_inr"].map(inr), textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        fig_cat.update_layout(**LAYOUT_BASE, height=280, xaxis_title="Value (₹)", yaxis_title=None,
                          xaxis_range=[0, max(cat_risk["lot_value_inr"].max() if len(cat_risk) else 0, 1) * 1.2])
        show(fig_cat)

    # ── IoT cold-chain sensors ──
    sec("Cold-room temperature, last 7 days")
    wh_names = {w["id"]: w["name"] for w in WAREHOUSES}
    sel = st.selectbox("Hub", sorted(sensors["warehouse_id"].unique()), format_func=lambda w: wh_names.get(w, w),
                       key="sensor_wh")
    fig_s = go.Figure()
    for zone, color, setpoint in [("CHILLER", T["blue"], 3), ("COOL_ROOM", T["steel"], 12)]:
        d = sensors[(sensors["warehouse_id"] == sel) & (sensors["zone"] == zone)]
        label = "Chiller" if zone == "CHILLER" else "Cool room"
        fig_s.add_trace(go.Scatter(x=d["reading_ts"], y=d["temp_c"], name=label, mode="lines",
                                   line=dict(color=color, width=1.6),
                                   hovertemplate=f"{label}: %{{y:.1f}}°C<extra></extra>"))
        fig_s.add_hline(y=setpoint, line=dict(color=color, width=1, dash="dot"),
                        annotation_text=f"{label} setpoint {setpoint}°C", annotation_position="top left",
                        annotation_font=dict(color=color, size=11, family=FONT))
    fig_s.update_layout(**LAYOUT_BASE, height=300, hovermode="x unified", yaxis_title="°C",
                        legend=dict(orientation="h", x=1, xanchor="right", y=1.15))
    show(fig_s)
    st.caption("Spikes above the setpoint are door openings during morning dispatch or compressor failures. "
               "They raise the temperature deviation of every lot stored in that room.")


# ─────────────────────────────────────────────────────────────
# TAB 4: INVENTORY OPTIMIZER
# ─────────────────────────────────────────────────────────────
def tab_inventory(lot_df, demand_df):
    tab_intro("Order planner", "Today's order for each SKU. A spoiled kilo costs four times more than "
              "a missed sale, so PISA deliberately orders below average demand.")

    # ── Newsvendor explainer ──
    with st.expander("How the order quantity is calculated"):
        st.markdown("""
**The Core Problem:** Should we order MORE (risk spoilage) or LESS (risk stockout)?

**The Math:**
- **Cu** = Cost of Under-ordering = 0.25 (25% margin lost + customer trust)
- **Co** = Cost of Over-ordering = 1.00 (100% loss if spoiled)
- **Critical Ratio** = Cu / (Cu + Co) = 0.25 / 1.25 = **0.20**

**What this means:** We order at the **20th percentile** of forecasted demand.
Because spoilage is 4x more expensive than a stockout, we lean toward ordering less.

**Formula:** Optimal Qty = F⁻¹(0.20, μ, σ)  
Where μ = predicted demand, σ = demand uncertainty

**In one line:** *"We don't just forecast mean demand — we pair the ML forecast
distribution with cost logic to find the order quantity that minimises total cost,
not just prediction error."*
        """)

    # ── Per-category recommendations ──
    sec("Today's orders")

    from ml_models import compute_optimal_order
    from scipy import stats

    recs = []
    for sku_row in SKUS:
        df_sku = demand_df[demand_df["sku_id"] == sku_row["sku_id"]]
        if len(df_sku) == 0:
            continue
        last30 = df_sku["actual_demand_kg"].values[-30:]
        mu, sigma = last30.mean(), last30.std()
        optimal   = compute_optimal_order(mu, sigma)
        naive     = round(mu, 1)
        saving_pct= round((naive - optimal) / naive * 100, 1) if naive > 0 else 0

        recs.append({
            "SKU":               sku_row["name"],
            "Category":          sku_row["category"],
            "Avg Demand (kg)":   round(mu, 1),
            "Naive Order (kg)":  naive,
            "✅ Optimal Order (kg)": optimal,
            "Est. Over-order Saved (%)": saving_pct,
            "Shelf Life (days)": sku_row["shelf_life_days"],
            "Price/kg (₹)":     sku_row["price_per_kg"],
        })

    recs_df = pd.DataFrame(recs)

    def highlight_savings(val):
        if isinstance(val, float) and val > 20:
            return f"background-color: {hex_alpha(T['green'], 0.16)}; font-weight:600"
        return ""

    st.dataframe(
        recs_df.style.map(highlight_savings, subset=["Est. Over-order Saved (%)"]),
        width="stretch", hide_index=True
    )

    # ── Over-ordering impact ──
    sec("Over-ordering against spoilage, FY2024")
    cat_over = (
        lot_df.groupby("category")
        .agg(avg_over_order=("over_order_factor","mean"),
             total_spoiled_value=("spoiled_value","sum"))
        .reset_index()
    )
    fig_over = px.scatter(
        cat_over, x="avg_over_order", y="total_spoiled_value",
        size="total_spoiled_value", color="category",
        color_discrete_sequence=CAT_COLORS,
        labels={"avg_over_order":"Avg Over-Order Factor",
                "total_spoiled_value":"Total Spoiled Value (₹)"},
        text="category",
    )
    fig_over.update_traces(textposition="top center")
    fig_over.update_layout(**LAYOUT_BASE, height=350,
                            title="Higher over-ordering → more spoilage value lost")
    fig_over.add_vline(x=1.0, line_dash="dash", line_color="grey",
                        annotation_text="Perfect ordering (1.0x)")
    show(fig_over)


# ─────────────────────────────────────────────────────────────
# TAB 5: MODEL HEALTH
# ─────────────────────────────────────────────────────────────
def tab_model_health(sp_artifact, dm_artifact, lot_df):
    tab_intro("Model health", "How each engine performed on data it never saw in training, and whether incoming data has drifted since.")

    col_a, col_b = st.columns(2)

    # ── Spoilage Model ──
    with col_a:
        sec("Spoilage risk (Random Forest and Cox)")

        m = sp_artifact["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
        mc2.metric("F1 Score",  f"{m['f1_score']:.3f}")
        mc3.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
        mc4.metric("Cox C-index", f"{sp_artifact['survival_metrics']['c_index']:.3f}",
                   help="Time-to-spoilage ranking accuracy (0.5 = random, 1.0 = perfect)")

        # Confusion matrix
        cm  = np.array(m["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: No Spoil","Predicted: Spoil"],
            y=["Actual: No Spoil","Actual: Spoil"],
            colorscale=[[0, T["surface2"]], [1, T["blue"]]],
            text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig_cm.update_layout(**LAYOUT_BASE, height=250,
                              title="Confusion Matrix")
        show(fig_cm)

        # Feature importance
        fi = sp_artifact["feature_importance"]
        if isinstance(fi, pd.DataFrame):
            fi_df = fi
        else:
            fi_df = pd.DataFrame(fi)

        fi_df = fi_df.sort_values("importance", ascending=True).tail(8)
        fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                         color="importance",
                         color_continuous_scale=SEQ,
                         title="Feature Importance")
        fig_fi.update_layout(**LAYOUT_BASE, height=280,
                              coloraxis_showscale=False)
        show(fig_fi)

        top_feat = fi_df.iloc[-1]["feature"]
        second_feat = fi_df.iloc[-2]["feature"] if len(fi_df) >= 2 else "N/A"
        st.info(
            "`" + top_feat + "` is the most important feature driving spoilage predictions. "
            + "`" + second_feat + "` is the second most important driver. Together, these top features "
            + "capture the core spoilage dynamics - lot aging and cold-chain deviations "
            + "are the primary levers a warehouse manager can act on."
        )

    # ── Demand Model ──
    with col_b:
        sec("Demand forecast (XGBoost)")

        mape_vals = [v["mape"] for v in dm_artifact["metrics"].values()]
        avg_mape  = np.mean(mape_vals)
        base_mape = np.mean([v["baseline_mape"] for v in dm_artifact["metrics"].values()])

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Avg MAPE",  str(round(avg_mape, 1)) + "%",  help="Mean Absolute % Error on the last 30 days")
        dc2.metric("Naive Baseline", str(round(base_mape, 1)) + "%", help="Same day last week")
        dc3.metric("SKUs",      len(mape_vals))

        # MAPE distribution
        fig_mape_hist = px.histogram(
            x=mape_vals, nbins=15,
            color_discrete_sequence=[T["blue"]],
            labels={"x":"MAPE (%)","y":"Number of SKUs"},
            title="MAPE Distribution across SKUs"
        )
        avg_mape_text = "Avg: " + str(round(avg_mape, 1)) + "%"
        fig_mape_hist.add_vline(x=avg_mape, line_dash="dash",
                                  annotation_text=avg_mape_text)
        fig_mape_hist.update_layout(**LAYOUT_BASE, height=250)
        show(fig_mape_hist)

        # Actual vs Predicted for one SKU
        sample_sku   = list(dm_artifact["metrics"].keys())[0]
        sample_data  = dm_artifact["metrics"][sample_sku]
        actuals      = sample_data["test_actual"]
        preds        = sample_data["test_predicted"]

        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(y=actuals, name="Actual", line=dict(color=T["steel"], width=2)))
        fig_ap.add_trace(go.Scatter(y=preds,   name="Predicted",
                                     line=dict(color=T["blue"], width=2, dash="dash")))
        fig_ap.update_layout(**LAYOUT_BASE, height=260,
                               title="Actual vs Predicted - " + sample_sku + " (Test Set)",
                               xaxis_title="Day", yaxis_title="Demand (kg)",
                               legend=dict(x=0.01, y=0.99))
        show(fig_ap)

        st.info(
            f"MAPE below 15% is considered good for perishable demand forecasting. XGBoost cuts error "
            f"by {(1 - avg_mape / base_mape) * 100:.0f}% versus the same-day-last-week baseline "
            f"({avg_mape:.1f}% vs {base_mape:.1f}%) on a 30-day time-based holdout."
        )

    # ── Hazard ratios + data drift ──
    col_h, col_d = st.columns(2)
    with col_h:
        sec("What makes lots spoil sooner")
        hr = pd.DataFrame(sp_artifact["survival_metrics"]["hazard_ratios"].items(),
                          columns=["Covariate", "Hazard Ratio"])
        st.dataframe(hr, width="stretch", hide_index=True)
        st.caption("HR > 1 → spoils sooner, per unit of the covariate (e.g. per °C of deviation).")
    with col_d:
        sec("Data drift, last 30 days against the rest of the year")
        drift = load_drift_report()
        if drift is None:
            st.info("No drift report yet — run `python pipeline.py`.")
        else:
            st.dataframe(pd.DataFrame(drift["features"]), width="stretch", hide_index=True)
            if drift["retrain_recommended"]:
                st.warning(f"Drift in {', '.join(drift['drifted_features'])} — the nightly pipeline retrains automatically.")
            else:
                st.success("No significant drift.")

    # ── KPI Summary for Presentation ──
    sec("Projected impact")

    total_value   = lot_df["lot_value"].sum()
    spoiled_value = lot_df["spoiled_value"].sum()
    baseline_wpt  = (spoiled_value / total_value) * 100
    target_wpt    = baseline_wpt * 0.65  # 35% relative reduction

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Wastage %",       str(round(baseline_wpt, 1)) + "%", "Before PISA")
    col2.metric("Target Wastage %",         str(round(target_wpt, 1)) + "%",   "After PISA", delta_color="inverse")
    recovery = round((spoiled_value - spoiled_value*0.65)/1e5)
    col3.metric("Projected Recovery",       "Rs " + str(recovery) + "L/yr", "35% reduction")
    col4.metric("Demand Forecast Accuracy", str(round(100-avg_mape)) + "%", "Avg MAPE: " + str(round(avg_mape, 1)) + "%")


# ─────────────────────────────────────────────────────────────
# TAB 6: ASK PISA (Claude analyst)
# ─────────────────────────────────────────────────────────────
def tab_ask():
    tab_intro("Ask PISA", "Ask a question in plain English. The AI analyst writes SQL against the DuckDB "
              "warehouse database, runs it read-only, and answers with live figures and tables.")

    key = get_llm_key()
    if not key:
        st.info("🔑 **Groq API Key required on deployed app**")
        st.caption("API keys are kept secure and not stored on GitHub. Enter your Groq API key below to start chatting, or configure `GROQ_API_KEY` permanently in Streamlit Cloud's App Settings → Secrets.")
        user_input = st.text_input("Groq API Key (starts with gsk_):", type="password", placeholder="gsk_...")
        if user_input:
            st.session_state["user_llm_key"] = user_input.strip()
            st.rerun()
        return

    engine_name = "⚡ Groq (Fast Inference)" if key.startswith("gsk_") else "Claude 3.5 Analyst"
    st.caption(f"Powered by **{engine_name}** with direct read-only SQL tool access to `data/pisa.duckdb`.")

    history = st.session_state.setdefault("ask_history", [])
    examples = ["Which vendor has the highest spoilage rate and what did it cost?",
                "Which lots will most likely spoil in the next 48 hours?",
                "How did Dairy wastage % change month by month?"]
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples) if c.button(q, width="stretch")), None)

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask about demand, spoilage, vendors, sensors…") or clicked
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        from analyst import ask
        with st.spinner("Querying warehouse DuckDB..."):
            try:
                answer, sqls = ask(question, history=history, api_key=key)
            except Exception as e:
                st.error(f"Analyst error: {e}")
                return
        st.markdown(answer)
        if sqls:
            with st.expander(f"SQL run ({len(sqls)})"):
                for q in sqls:
                    st.code(q, language="sql")
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
ALL_HUBS = "All hubs"


def render_sidebar():
    """Renders the hub/category filters. Returns (selected_warehouse, selected_categories)."""
    with st.sidebar:
        st.markdown('<div class="side-mark">PISA</div>'
                    '<div class="side-sub">Predictive inventory and spoilage alerts</div>',
                    unsafe_allow_html=True)

        sel_warehouse = st.selectbox("Hub", [ALL_HUBS] + [w["name"] for w in WAREHOUSES])
        categories = sorted({s["category"] for s in SKUS})
        sel_categories = st.multiselect("Categories", categories, default=categories)

        st.divider()
        st.markdown(
            '<div class="side-about">Three engines run every night: an XGBoost demand forecast, '
            'a spoilage model (Random Forest plus Cox survival), and a newsvendor order calculator.'
            '<br><br>We aim to lower wastage % without letting the fill rate drop.</div>',
            unsafe_allow_html=True)
        st.divider()
        st.caption("Built by Satyam Jha, B.Tech IT. Zomato Hyperpure case study.")

    return sel_warehouse, sel_categories


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def apply_filters(df, sel_warehouse, sel_categories, warehouse_col="warehouse_name", category_col="category"):
    """Apply sidebar filters to any DataFrame."""
    filtered = df.copy()
    if sel_warehouse != ALL_HUBS:
        filtered = filtered[filtered[warehouse_col] == sel_warehouse]
    if sel_categories:
        filtered = filtered[filtered[category_col].isin(sel_categories)]
    return filtered


def main():
    # First-run setup
    with st.spinner("First run: generating data and training models. This takes a few minutes."):
        auto_setup()

    # Load data
    demand_df   = load_demand()
    lot_df      = load_lots()
    active_lots = load_active_lots()
    sp_artifact, dm_artifact = load_models()

    # Extract trained XGBoost models for forecast tab
    sku_models = dm_artifact.get("models", None)

    sensors = load_sensors()
    sel_warehouse, sel_categories = render_sidebar()
    render_header(sensors)

    # Apply sidebar filters to data
    lot_df_f      = apply_filters(lot_df, sel_warehouse, sel_categories)
    demand_df_f   = apply_filters(demand_df, sel_warehouse, sel_categories)
    active_lots_f = apply_filters(active_lots, sel_warehouse, sel_categories)

    # Fallback: if filters produce empty data, show warning
    if len(lot_df_f) == 0 or len(demand_df_f) == 0:
        st.warning("No data matches these filters. Pick at least one category in the sidebar.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Demand forecast",
        "Spoilage alerts",
        "Order planner",
        "Model health",
        "Ask PISA",
    ])

    with tab1: tab_overview(lot_df_f, demand_df_f, active_lots_f)
    with tab2: tab_demand(demand_df_f, dm_artifact, sku_models=sku_models)
    with tab3: tab_alerts(active_lots_f, sensors)
    with tab4: tab_inventory(lot_df_f, demand_df_f)
    with tab5: tab_model_health(sp_artifact, dm_artifact, lot_df_f)
    with tab6: tab_ask()


if __name__ == "__main__":
    main()
