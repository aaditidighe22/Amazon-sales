import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
C_ORANGE   = "#FF9900"
C_DARK_ORG = "#E65C00"
C_BLUE     = "#146EB4"
C_GREEN    = "#1DB954"
C_RED      = "#E53935"
C_BG       = "#0F1117"
C_CARD     = "#1E2130"
C_BORDER   = "#2E3250"

PLOTLY_PALETTE = [C_ORANGE, C_BLUE, C_GREEN, "#AB47BC", "#26C6DA", "#EF5350", "#FFA726", "#66BB6A"]

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {C_BG};
    color: #E8EAF0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}}
[data-testid="stMain"] {{ background-color: {C_BG}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D0F1A 0%, #141726 60%, #1a1e30 100%);
    border-right: 1px solid {C_BORDER};
}}
[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
[data-testid="stSidebar"] .stRadio > label {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7B8DB0 !important;
    margin-bottom: 0.3rem;
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #CBD5E1 !important;
}}

/* ── Header banner ── */
.hero-banner {{
    background: linear-gradient(135deg, #FF6D00 0%, #FF9900 45%, #FFB300 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}}
.hero-banner::after {{
    content: "🛒";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}}
.hero-banner h1 {{
    margin: 0;
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}}
.hero-banner p {{
    margin: 0.4rem 0 0;
    font-size: 0.97rem;
    color: rgba(255,255,255,0.88);
}}

/* ── Page title pill ── */
.page-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-left: 4px solid {C_ORANGE};
    border-radius: 8px;
    padding: 0.55rem 1.1rem;
    margin: 1.2rem 0 1rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: #E8EAF0;
}}
.step-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: {C_ORANGE};
    color: #000;
    font-size: 0.72rem;
    font-weight: 800;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    flex-shrink: 0;
}}

/* ── KPI Cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.6rem;
}}
.kpi-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}}
.kpi-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, {C_ORANGE});
    border-radius: 14px 14px 0 0;
}}
.kpi-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}}
.kpi-value {{
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--accent, {C_ORANGE});
    line-height: 1.1;
    letter-spacing: -0.03em;
}}
.kpi-label {{
    font-size: 0.78rem;
    color: #7B8DB0;
    margin-top: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Section headings ── */
.sec-head {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #E8EAF0;
    margin: 1.5rem 0 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {C_BORDER};
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* ── Insight cards ── */
.insight-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-left: 4px solid var(--accent, {C_ORANGE});
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
}}
.insight-card h4 {{
    margin: 0 0 0.45rem;
    font-size: 0.97rem;
    color: #E8EAF0;
}}
.insight-card ul {{
    margin: 0; padding-left: 1.2rem;
}}
.insight-card li {{
    font-size: 0.87rem;
    color: #A0AABF;
    margin-bottom: 0.3rem;
    line-height: 1.55;
}}

/* ── Rec cards ── */
.rec-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.65rem;
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
}}
.rec-icon {{
    font-size: 1.4rem;
    flex-shrink: 0;
    margin-top: 0.05rem;
}}
.rec-title {{
    font-size: 0.92rem;
    font-weight: 700;
    color: #E8EAF0;
    margin-bottom: 0.2rem;
}}
.rec-desc {{
    font-size: 0.83rem;
    color: #8899BB;
    line-height: 1.5;
}}

/* ── Prediction result ── */
.pred-yes {{
    background: linear-gradient(135deg, #0D2F1A, #1B4D2E);
    border: 2px solid {C_GREEN};
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}}
.pred-no {{
    background: linear-gradient(135deg, #2F0D0D, #4D1B1B);
    border: 2px solid {C_RED};
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}}
.pred-label {{
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
}}
.pred-sub {{
    font-size: 0.92rem;
    margin-top: 0.4rem;
    color: rgba(255,255,255,0.75);
}}

/* ── Prob bar ── */
.prob-bar-wrap {{
    background: #1A1E2E;
    border-radius: 99px;
    height: 10px;
    margin: 0.5rem 0;
    overflow: hidden;
}}
.prob-bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}}

/* ── Tables ── */
[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {C_BORDER};
}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: #7B8DB0 !important;
    font-weight: 600;
    font-size: 0.85rem;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_ORANGE} !important;
    border-bottom-color: {C_ORANGE} !important;
}}

/* ── Form submit button ── */
[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(135deg, {C_ORANGE}, {C_DARK_ORG}) !important;
    color: #000 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    letter-spacing: 0.03em;
}}

/* ── Sliders, inputs ── */
[data-testid="stSlider"] {{ color: {C_ORANGE}; }}
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div {{
    background: #1A1E2E !important;
    border-color: {C_BORDER} !important;
    color: #E8EAF0 !important;
}}

/* ── Metric widget ── */
[data-testid="stMetric"] {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}
[data-testid="stMetricValue"] {{ color: {C_ORANGE} !important; font-weight: 800 !important; }}
[data-testid="stMetricLabel"] {{ color: #7B8DB0 !important; font-size: 0.8rem !important; }}

/* ── divider ── */
hr {{ border-color: {C_BORDER}; }}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {{ background-color: {C_ORANGE} !important; }}

/* ── Sidebar logo area ── */
.sidebar-logo {{
    background: linear-gradient(135deg, {C_ORANGE}, {C_DARK_ORG});
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    text-align: center;
}}
.sidebar-logo h2 {{
    margin: 0;
    font-size: 1.1rem;
    font-weight: 800;
    color: #000 !important;
}}
.sidebar-logo p {{
    margin: 0.15rem 0 0;
    font-size: 0.72rem;
    color: rgba(0,0,0,0.7) !important;
}}

/* ── Badge pill ── */
.badge {{
    display: inline-block;
    background: rgba(255,153,0,0.15);
    color: {C_ORANGE};
    border: 1px solid rgba(255,153,0,0.3);
    border-radius: 99px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 0.4rem;
}}

/* Plotly charts background */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161929",
    font=dict(family="Inter, Segoe UI, system-ui", color="#CBD5E1", size=12),
    title_font=dict(size=14, color="#E8EAF0", family="Inter, Segoe UI, system-ui"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C_BORDER,
        borderwidth=1,
        font=dict(color="#CBD5E1"),
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1E2130", linecolor=C_BORDER, tickfont=dict(color="#7B8DB0")),
    yaxis=dict(gridcolor="#1E2130", linecolor=C_BORDER, tickfont=dict(color="#7B8DB0")),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title_text=title)
    return fig

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Amazon_Sales_Customer_Purchase_1000.csv")

@st.cache_data
def train_models(df_raw):
    df = df_raw.copy()
    df["Purchase_Binary"] = (df["Purchase_Status"] == "Purchased").astype(int)
    le = LabelEncoder()
    cat_enc = ["Gender", "Product_Category", "Payment_Method", "Device", "Return_Status", "State", "City"]
    df_ml = df.copy()
    for c in cat_enc:
        df_ml[c + "_enc"] = le.fit_transform(df_ml[c].astype(str))
    feat = [
        "Age", "Price", "Quantity", "Discount_Percent", "Customer_Rating",
        "Previous_Purchases", "Website_Visits", "Cart_Additions",
        "Time_on_Site_Min", "Delivery_Days",
        "Gender_enc", "Product_Category_enc", "Payment_Method_enc",
        "Device_enc", "Return_Status_enc",
    ]
    X, y = df_ml[feat], df_ml["Purchase_Binary"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_tr_s, y_tr)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_tr)
    return rf, lr, sc, feat, X_te_s, y_te

df_raw = load_data()
rf_model, lr_model, scaler, FEATURE_COLS, X_test_sc, y_test = train_models(df_raw)

rf_pred = rf_model.predict(X_test_sc);  rf_prob = rf_model.predict_proba(X_test_sc)[:, 1]
lr_pred = lr_model.predict(X_test_sc);  lr_prob = lr_model.predict_proba(X_test_sc)[:, 1]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def kpi(icon, value, label, accent=C_ORANGE, col_width=None):
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
      <span class="kpi-icon">{icon}</span>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

def sec(icon, text):
    st.markdown(f'<div class="sec-head">{icon} {text}</div>', unsafe_allow_html=True)

def page_pill(step, text):
    st.markdown(
        f'<div class="page-pill"><span class="step-badge">{step}</span>{text}</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <h2>🛒 Amazon Analytics</h2>
      <p>Sales & Purchase Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Data Quality",
            "💰 Sales & Revenue",
            "📦 Products & Categories",
            "👥 Customer Behavior",
            "📊 Visualizations",
            "📈 Sales Trends",
            "🤖 ML Training",
            "🎯 Prediction & Evaluation",
            "💡 Business Insights",
        ],
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:0.8rem 1rem;">
      <div style="font-size:0.72rem;color:#7B8DB0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Dataset Info</div>
      <div style="font-size:0.85rem;color:#CBD5E1;margin-bottom:0.25rem;">📄 1,000 records</div>
      <div style="font-size:0.85rem;color:#CBD5E1;margin-bottom:0.25rem;">🔢 22 features</div>
      <div style="font-size:0.85rem;color:#CBD5E1;">🏷️ Binary classification</div>
    </div>
    """, unsafe_allow_html=True)

    purchased_pct = (df_raw["Purchase_Status"] == "Purchased").mean() * 100
    st.markdown(f"""
    <div style="margin-top:0.8rem;background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:0.8rem 1rem;">
      <div style="font-size:0.72rem;color:#7B8DB0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">Conversion Rate</div>
      <div style="font-size:1.3rem;font-weight:800;color:{C_ORANGE};">{purchased_pct:.1f}%</div>
      <div class="prob-bar-wrap" style="margin-top:0.4rem;">
        <div class="prob-bar-fill" style="width:{purchased_pct:.0f}%;background:{C_ORANGE};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO BANNER  (shown on every page)
# ─────────────────────────────────────────────
PAGE_SUBTITLES = {
    "🏠 Dashboard":            "End-to-end snapshot of Amazon sales performance and customer intelligence.",
    "🔍 Data Quality":         "Missing values, outliers, duplicates — full data health check.",
    "💰 Sales & Revenue":      "Revenue KPIs, payment channels, and regional breakdown.",
    "📦 Products & Categories":"Category revenue, top products, discount impact analysis.",
    "👥 Customer Behavior":    "Age, gender, loyalty segments, city-level purchase patterns.",
    "📊 Visualizations":       "Interactive Plotly charts for every key dimension.",
    "📈 Sales Trends":         "Loyalty, discount, time-on-site and cart-addition trend analysis.",
    "🤖 ML Training":          "Feature engineering, model training, feature importance.",
    "🎯 Prediction & Evaluation": "Confusion matrix, ROC, classification report & live predictor.",
    "💡 Business Insights":    "Data-driven recommendations and strategic action plan.",
}
st.markdown(f"""
<div class="hero-banner">
  <h1>🛒 Amazon Sales &amp; Purchase Prediction</h1>
  <p>{PAGE_SUBTITLES.get(page, "")}</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    page_pill(1, "Collect & Load Dataset")

    df = df_raw.copy()
    total_rev = df["Sales_Amount"].sum()
    avg_order = df[df["Sales_Amount"] > 0]["Sales_Amount"].mean()
    total_ord = (df["Purchase_Status"] == "Purchased").sum()
    avg_disc  = df["Discount_Percent"].mean()
    ret_rate  = (df["Return_Status"] == "Returned").mean() * 100
    avg_rat   = df["Customer_Rating"].mean()

    # Row 1 — 3 KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("💵", f"₹{total_rev:,.0f}", "Total Revenue", C_ORANGE), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("🛒", f"{total_ord:,}", "Total Orders", C_BLUE), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("📦", f"₹{avg_order:,.0f}", "Avg Order Value", "#AB47BC"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Row 2 — 3 KPIs
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(kpi("🏷️", f"{avg_disc:.1f}%", "Avg Discount", "#26C6DA"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("↩️", f"{ret_rate:.1f}%", "Return Rate", C_RED), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("⭐", f"{avg_rat:.2f}", "Avg Rating", C_GREEN), unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Mini-charts row
    col_l, col_r = st.columns(2)

    with col_l:
        sec("📊", "Purchase Status Split")
        ps = df["Purchase_Status"].value_counts().reset_index()
        ps.columns = ["Status", "Count"]
        fig = px.pie(ps, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"Purchased": C_ORANGE, "Not Purchased": C_BLUE},
                     hole=0.55)
        fig.update_traces(textfont_size=13, textfont_color="white",
                          marker=dict(line=dict(color="#0F1117", width=3)))
        fig.update_layout(**PLOTLY_LAYOUT, height=280,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sec("📊", "Revenue by Category")
        cat_rev = df.groupby("Product_Category")["Sales_Amount"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(cat_rev, x="Sales_Amount", y="Product_Category", orientation="h",
                     color="Sales_Amount", color_continuous_scale=[[0, "#1E2130"], [1, C_ORANGE]],
                     labels={"Sales_Amount": "Revenue (₹)", "Product_Category": ""})
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Dataset preview
    sec("📄", "Raw Dataset Preview")
    rows = st.slider("Rows to preview", 5, 50, 10, key="ds_rows")
    st.dataframe(
        df.head(rows).style.set_properties(**{"background-color": C_CARD, "color": "#E8EAF0"}),
        use_container_width=True,
    )

    # Column dictionary
    with st.expander("📖 Column Dictionary", expanded=False):
        col_desc = {
            "Customer_ID": "Unique customer identifier",
            "Age": "Customer age (years)",
            "Gender": "Customer gender (Male / Female / Other)",
            "City": "Customer city",
            "State": "Customer state",
            "Product_Category": "Category of purchased product",
            "Product_Name": "Name of product",
            "Price": "Product price (₹)",
            "Quantity": "Units ordered",
            "Discount_Percent": "Discount applied (%)",
            "Payment_Method": "Mode of payment",
            "Device": "Device used for purchase",
            "Customer_Rating": "Customer rating (1–5)",
            "Previous_Purchases": "No. of prior purchases",
            "Website_Visits": "No. of website visits",
            "Cart_Additions": "Items added to cart",
            "Time_on_Site_Min": "Time spent on site (minutes)",
            "Delivery_Days": "Days taken for delivery",
            "Return_Status": "Whether item was returned",
            "Purchase_Status": "Target — Purchased / Not Purchased",
            "Purchase_Probability": "Estimated purchase probability (0–1)",
            "Sales_Amount": "Revenue generated (₹)",
        }
        st.dataframe(
            pd.DataFrame(list(col_desc.items()), columns=["Column", "Description"]),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════════════════
#  PAGE: DATA QUALITY
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Data Quality":
    page_pill(2, "Check for Missing & Incorrect Values")

    df = df_raw.copy()
    missing  = df.isnull().sum()
    miss_pct = (missing / len(df) * 100).round(2)
    miss_df  = pd.DataFrame({"Missing Count": missing, "Missing %": miss_pct})
    dupes    = df.duplicated().sum()

    # Quick health row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("✅", f"{missing.sum()}", "Total Missing Cells", C_GREEN if missing.sum()==0 else C_RED), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("🔁", f"{dupes}", "Duplicate Rows", C_GREEN if dupes==0 else C_RED), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("🔢", f"{df.select_dtypes('number').shape[1]}", "Numeric Columns", C_BLUE), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("🔤", f"{df.select_dtypes('object').shape[1]}", "Categorical Columns", "#AB47BC"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if missing.sum() == 0:
        st.success("✅  No missing values detected — the dataset is clean and complete.")
    else:
        st.warning(f"⚠️  {missing.sum()} missing cells found.")

    page_pill(3, "Data Cleaning & Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        sec("❓", "Missing Values")
        st.dataframe(miss_df, use_container_width=True)

    with col2:
        sec("🔤", "Data Types")
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)
        dtype_df["Nullable"] = df.isnull().any().values
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    sec("📊", "Categorical Value Distributions")
    cat_cols = ["Gender", "Product_Category", "Payment_Method", "Device", "Return_Status", "Purchase_Status"]
    tabs = st.tabs(cat_cols)
    for tab, col in zip(tabs, cat_cols):
        with tab:
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "Count"]
            vc["Percentage (%)"] = (vc["Count"] / len(df) * 100).round(2)
            tc1, tc2 = st.columns([1, 1.5])
            with tc1:
                st.dataframe(vc, use_container_width=True, hide_index=True)
            with tc2:
                fig = px.bar(vc, x=col, y="Count",
                             color="Count", color_continuous_scale=[[0, "#1E2130"], [1, C_ORANGE]],
                             text="Count")
                fig.update_coloraxes(showscale=False)
                fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
                fig.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    sec("📐", "Outlier Summary (IQR Method)")
    num_cols = ["Age", "Price", "Quantity", "Discount_Percent", "Customer_Rating",
                "Previous_Purchases", "Website_Visits", "Cart_Additions",
                "Time_on_Site_Min", "Delivery_Days", "Sales_Amount"]
    q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_cnt = ((df[num_cols] < lower) | (df[num_cols] > upper)).sum()
    out_df = pd.DataFrame({
        "Outliers": out_cnt,
        "Lower Fence": lower.round(2),
        "Upper Fence": upper.round(2),
        "Min": df[num_cols].min().round(2),
        "Max": df[num_cols].max().round(2),
    })
    st.dataframe(out_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: SALES & REVENUE
# ═══════════════════════════════════════════════════════════
elif page == "💰 Sales & Revenue":
    page_pill(4, "Calculate Total Sales & Revenue")

    df = df_raw.copy()
    total_rev = df["Sales_Amount"].sum()
    avg_ord   = df[df["Sales_Amount"] > 0]["Sales_Amount"].mean()
    tot_ord   = (df["Purchase_Status"] == "Purchased").sum()
    avg_disc  = df["Discount_Percent"].mean()
    tot_qty   = df["Quantity"].sum()
    ret_rate  = (df["Return_Status"] == "Returned").mean() * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("💵", f"₹{total_rev:,.0f}", "Total Revenue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("🛒", f"₹{avg_ord:,.0f}", "Avg Order Value", C_BLUE), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("📦", f"{tot_ord:,}", "Total Orders", "#AB47BC"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(kpi("🏷️", f"{avg_disc:.1f}%", "Avg Discount", "#26C6DA"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("📬", f"{tot_qty:,}", "Total Units Sold", C_GREEN), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("↩️", f"{ret_rate:.1f}%", "Return Rate", C_RED), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        sec("💳", "Revenue by Payment Method")
        pay_rev = df.groupby("Payment_Method")["Sales_Amount"].sum().sort_values(ascending=False).reset_index()
        pay_rev.columns = ["Payment Method", "Revenue"]
        fig = px.bar(pay_rev, x="Payment Method", y="Revenue",
                     color="Revenue", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text=pay_rev["Revenue"].apply(lambda x: f"₹{x/1000:.1f}K"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sec("📱", "Revenue by Device")
        dev_rev = df.groupby("Device")["Sales_Amount"].sum().reset_index()
        dev_rev.columns = ["Device", "Revenue"]
        fig = px.pie(dev_rev, names="Device", values="Revenue",
                     color_discrete_sequence=[C_ORANGE, C_BLUE, C_GREEN], hole=0.5)
        fig.update_traces(textfont_size=13, textfont_color="white",
                          marker=dict(line=dict(color="#0F1117", width=3)))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    sec("🗺️", "Top 15 States by Revenue")
    state_rev = df.groupby("State")["Sales_Amount"].sum().sort_values(ascending=False).head(15).reset_index()
    state_rev.columns = ["State", "Revenue"]
    fig = px.bar(state_rev, x="State", y="Revenue",
                 color="Revenue", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                 text=state_rev["Revenue"].apply(lambda x: f"₹{x/1000:.0f}K"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    sec("📋", "Revenue Tables")
    t1, t2, t3 = st.tabs(["By Payment Method", "By Device", "By State (Top 15)"])
    with t1:
        st.dataframe(pay_rev.assign(**{"Revenue (₹)": pay_rev["Revenue"].round(2)}).drop(columns="Revenue"),
                     use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(dev_rev.assign(**{"Revenue (₹)": dev_rev["Revenue"].round(2)}).drop(columns="Revenue"),
                     use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(state_rev.assign(**{"Revenue (₹)": state_rev["Revenue"].round(2)}).drop(columns="Revenue"),
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: PRODUCTS & CATEGORIES
# ═══════════════════════════════════════════════════════════
elif page == "📦 Products & Categories":
    page_pill(5, "Analyze Sales by Product & Category")

    df = df_raw.copy()

    sec("🏷️", "Revenue by Product Category")
    cat_agg = df.groupby("Product_Category").agg(
        Total_Revenue=("Sales_Amount", "sum"),
        Avg_Price=("Price", "mean"),
        Orders=("Purchase_Status", lambda x: (x == "Purchased").sum()),
        Avg_Rating=("Customer_Rating", "mean"),
    ).round(2).sort_values("Total_Revenue", ascending=False).reset_index()

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        fig = px.bar(cat_agg.sort_values("Total_Revenue"), x="Total_Revenue", y="Product_Category",
                     orientation="h", color="Total_Revenue",
                     color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text=cat_agg.sort_values("Total_Revenue")["Total_Revenue"].apply(lambda x: f"₹{x/1000:.0f}K"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                          xaxis_title="Revenue (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(cat_agg.rename(columns={
            "Product_Category": "Category",
            "Total_Revenue": "Revenue (₹)",
            "Avg_Price": "Avg Price (₹)",
            "Avg_Rating": "Avg Rating"
        }), use_container_width=True, hide_index=True)

    sec("🥇", "Top 20 Products by Revenue")
    prod_rev = df.groupby("Product_Name").agg(
        Total_Revenue=("Sales_Amount", "sum"),
        Units_Sold=("Quantity", "sum"),
        Avg_Price=("Price", "mean"),
        Avg_Rating=("Customer_Rating", "mean"),
    ).round(2).sort_values("Total_Revenue", ascending=False).head(20).reset_index()
    st.dataframe(prod_rev, use_container_width=True, hide_index=True)

    sec("🏷️", "Discount Impact on Sales")
    disc_bins = pd.cut(df["Discount_Percent"], bins=[0, 10, 20, 30, 40, 50, 60], include_lowest=True)
    disc_agg  = df.groupby(disc_bins, observed=True)["Sales_Amount"].agg(["mean","count"]).round(2)
    disc_agg.columns = ["Avg Sales Amount (₹)", "Orders"]
    disc_agg.index   = disc_agg.index.astype(str)
    disc_agg = disc_agg.reset_index()
    disc_agg.columns = ["Discount Range", "Avg Sales (₹)", "Orders"]

    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=disc_agg["Discount Range"], y=disc_agg["Avg Sales (₹)"],
            marker_color=C_ORANGE, name="Avg Sales (₹)", opacity=0.85,
            text=[f"₹{v:,.0f}" for v in disc_agg["Avg Sales (₹)"]],
            textposition="outside", textfont_color="#E8EAF0"
        ))
        fig.add_trace(go.Scatter(
            x=disc_agg["Discount Range"], y=disc_agg["Orders"],
            mode="lines+markers", name="Orders",
            yaxis="y2", line=dict(color=C_BLUE, width=2.5),
            marker=dict(size=7)
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=300, barmode="overlay",
            yaxis=dict(title="Avg Sales (₹)", gridcolor="#1E2130"),
            yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False, tickfont=dict(color=C_BLUE)),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(disc_agg, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: CUSTOMER BEHAVIOR
# ═══════════════════════════════════════════════════════════
elif page == "👥 Customer Behavior":
    page_pill(6, "Analyze Customer Purchasing Behavior")

    df = df_raw.copy()
    df["Age_Group"] = pd.cut(df["Age"], bins=[18,25,35,45,55,70],
                              labels=["18–25","26–35","36–45","46–55","56+"])
    df["Purchase_Segment"] = pd.cut(df["Previous_Purchases"], bins=[-1,2,5,10,30],
                                     labels=["New (0–2)","Regular (3–5)","Loyal (6–10)","Champion (11+)"])

    col_l, col_r = st.columns(2)

    with col_l:
        sec("🎂", "Purchase Rate by Age Group")
        age_p = df.groupby("Age_Group", observed=True).apply(
            lambda x: round((x["Purchase_Status"]=="Purchased").mean()*100, 2)
        ).reset_index(); age_p.columns = ["Age Group", "Purchase Rate (%)"]
        fig = px.bar(age_p, x="Age Group", y="Purchase Rate (%)",
                     color="Purchase Rate (%)", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text=age_p["Purchase Rate (%)"].apply(lambda v: f"{v}%"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sec("⚤", "Purchase Rate by Gender")
        gen_p = df.groupby("Gender").apply(
            lambda x: round((x["Purchase_Status"]=="Purchased").mean()*100, 2)
        ).reset_index(); gen_p.columns = ["Gender", "Purchase Rate (%)"]
        fig = px.bar(gen_p, x="Gender", y="Purchase Rate (%)",
                     color="Gender",
                     color_discrete_sequence=[C_ORANGE, C_BLUE, C_GREEN],
                     text=gen_p["Purchase Rate (%)"].apply(lambda v: f"{v}%"))
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    sec("🏆", "Customer Loyalty Segmentation")
    seg = df.groupby("Purchase_Segment", observed=True).agg(
        Count=("Customer_ID","count"),
        Avg_Spend=("Sales_Amount","mean"),
        Purchase_Rate=("Purchase_Status", lambda x: round((x=="Purchased").mean()*100,2)),
        Avg_Rating=("Customer_Rating","mean"),
    ).round(2).reset_index()
    seg.columns = ["Segment","Count","Avg Spend (₹)","Purchase Rate (%)","Avg Rating"]

    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        fig = px.bar(seg, x="Segment", y="Purchase Rate (%)",
                     color="Avg Spend (₹)",
                     color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text=seg["Purchase Rate (%)"].apply(lambda v: f"{v}%"))
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                          coloraxis_colorbar=dict(title="Avg Spend (₹)", tickfont=dict(color="#CBD5E1")))
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(seg, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sec("🏙️", "Top 10 Cities by Revenue")
        city_rev = df.groupby("City")["Sales_Amount"].sum().sort_values(ascending=False).head(10).reset_index()
        city_rev.columns = ["City","Revenue"]
        fig = px.bar(city_rev.sort_values("Revenue"), x="Revenue", y="City", orientation="h",
                     color="Revenue", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False, xaxis_title="Revenue (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sec("🔗", "Feature Correlations with Purchase")
        num_c = ["Age","Price","Discount_Percent","Customer_Rating",
                 "Previous_Purchases","Website_Visits","Cart_Additions",
                 "Time_on_Site_Min","Sales_Amount"]
        df["Purchase_Binary"] = (df["Purchase_Status"]=="Purchased").astype(int)
        corr = df[num_c+["Purchase_Binary"]].corr()[["Purchase_Binary"]].drop("Purchase_Binary").sort_values("Purchase_Binary")
        fig = px.bar(corr.reset_index(), x="Purchase_Binary", y="index", orientation="h",
                     color="Purchase_Binary",
                     color_continuous_scale=[[0,C_RED],[0.5,"#1E2130"],[1,C_GREEN]],
                     range_color=[-0.3, 0.3])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                          xaxis_title="Correlation", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    page_pill(7, "Charts & Visualizations")

    df = df_raw.copy()

    # Filter widget
    with st.expander("🎛️ Filter Data", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_cat = st.multiselect("Product Category", sorted(df["Product_Category"].unique()),
                                      default=sorted(df["Product_Category"].unique()))
        with fc2:
            sel_gender = st.multiselect("Gender", sorted(df["Gender"].unique()),
                                         default=sorted(df["Gender"].unique()))
        with fc3:
            sel_device = st.multiselect("Device", sorted(df["Device"].unique()),
                                         default=sorted(df["Device"].unique()))
        df = df[df["Product_Category"].isin(sel_cat) &
                df["Gender"].isin(sel_gender) &
                df["Device"].isin(sel_device)]
        st.caption(f"Showing **{len(df):,}** records after filters")

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        sec("🥧", "Purchase Status Distribution")
        ps = df["Purchase_Status"].value_counts().reset_index(); ps.columns = ["Status","Count"]
        fig = px.pie(ps, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"Purchased": C_ORANGE, "Not Purchased": C_BLUE},
                     hole=0.55)
        fig.update_traces(textfont_size=13, textfont_color="white",
                          marker=dict(line=dict(color="#0F1117", width=3)))
        fig.update_layout(**PLOTLY_LAYOUT, height=310,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("💳", "Payment Method Usage")
        pm = df["Payment_Method"].value_counts().reset_index(); pm.columns = ["Method","Count"]
        fig = px.bar(pm, x="Method", y="Count",
                     color="Count", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text="Count")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=310, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        sec("🎂", "Age Distribution by Purchase Status")
        fig = px.histogram(df, x="Age", color="Purchase_Status", barmode="overlay",
                           color_discrete_map={"Purchased": C_ORANGE, "Not Purchased": C_BLUE},
                           nbins=25, opacity=0.8)
        fig.update_layout(**PLOTLY_LAYOUT, height=310,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("⭐", "Customer Rating Distribution")
        fig = px.histogram(df, x="Customer_Rating", nbins=25,
                           color_discrete_sequence=[C_ORANGE], opacity=0.85)
        fig.update_layout(**PLOTLY_LAYOUT, height=310, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    c5, c6 = st.columns(2)

    with c5:
        sec("💵", "Price vs Sales Amount")
        fig = px.scatter(df, x="Price", y="Sales_Amount",
                         color="Purchase_Status",
                         color_discrete_map={"Purchased": C_ORANGE, "Not Purchased": C_BLUE},
                         opacity=0.6, size_max=8)
        fig.update_layout(**PLOTLY_LAYOUT, height=310,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        sec("🚚", "Delivery Days Distribution")
        dd = df["Delivery_Days"].value_counts().sort_index().reset_index()
        dd.columns = ["Days","Count"]
        fig = px.bar(dd, x="Days", y="Count",
                     color="Count", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                     text="Count")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=310, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4 — Correlation Heatmap (seaborn, full width)
    sec("🔥", "Correlation Heatmap")
    num_cols = ["Age","Price","Quantity","Discount_Percent","Customer_Rating",
                "Previous_Purchases","Website_Visits","Cart_Additions",
                "Time_on_Site_Min","Delivery_Days","Sales_Amount"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig_h, ax_h = plt.subplots(figsize=(12, 6))
    fig_h.patch.set_facecolor("#0F1117")
    ax_h.set_facecolor("#161929")
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.4, ax=ax_h, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 8, "color": "white"})
    ax_h.tick_params(colors="#CBD5E1", labelsize=8)
    ax_h.set_title("Feature Correlation Heatmap", color="#E8EAF0", fontsize=13, fontweight="bold", pad=12)
    plt.setp(ax_h.get_xticklabels(), rotation=35, ha="right", color="#CBD5E1")
    plt.setp(ax_h.get_yticklabels(), rotation=0, color="#CBD5E1")
    ax_h.collections[0].colorbar.ax.tick_params(labelcolor="#CBD5E1")
    plt.tight_layout()
    st.pyplot(fig_h); plt.close()

    # Row 5
    c7, c8 = st.columns(2)
    with c7:
        sec("📱", "Purchase Rate by Device")
        dev_r = df.groupby("Device")["Purchase_Status"].apply(
            lambda x: round((x=="Purchased").mean()*100, 2)
        ).reset_index(); dev_r.columns = ["Device","Purchase Rate (%)"]
        fig = px.bar(dev_r, x="Device", y="Purchase Rate (%)",
                     color="Device", color_discrete_sequence=[C_ORANGE, C_BLUE, C_GREEN],
                     text=dev_r["Purchase Rate (%)"].apply(lambda v: f"{v}%"))
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        sec("📦", "Revenue by Category")
        cat_r = df.groupby("Product_Category")["Sales_Amount"].sum().sort_values(ascending=True).reset_index()
        cat_r.columns = ["Category","Revenue"]
        fig = px.bar(cat_r, x="Revenue", y="Category", orientation="h",
                     color="Revenue", color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: SALES TRENDS
# ═══════════════════════════════════════════════════════════
elif page == "📈 Sales Trends":
    page_pill(8, "Identify Important Sales Trends")

    df = df_raw.copy()

    # Trend 1
    sec("🏅", "Trend 1 — Customer Loyalty vs Purchase Rate")
    loyalty_bins = pd.cut(df["Previous_Purchases"], bins=[-1,2,5,10,30],
                          labels=["0–2 (New)","3–5 (Regular)","6–10 (Loyal)","11+ (Champion)"])
    loyal_r = df.groupby(loyalty_bins, observed=True)["Purchase_Status"].apply(
        lambda x: round((x=="Purchased").mean()*100,2)
    ).reset_index(); loyal_r.columns = ["Tier","Purchase Rate (%)"]

    col_l, col_r = st.columns([1.6,1])
    with col_l:
        colors_tier = [C_ORANGE, "#FF6D00", "#E65100", C_RED]
        fig = go.Figure(go.Bar(
            x=loyal_r["Tier"], y=loyal_r["Purchase Rate (%)"],
            marker_color=colors_tier,
            text=[f"{v}%" for v in loyal_r["Purchase Rate (%)"]],
            textposition="outside", textfont_color="#E8EAF0",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False, yaxis_title="Purchase Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(loyal_r, use_container_width=True, hide_index=True)

    # Trend 2
    sec("🏷️", "Trend 2 — Discount % vs Purchase Rate")
    disc_bins = pd.cut(df["Discount_Percent"], bins=[-1,10,20,30,40,50,60],
                       labels=["0–10%","11–20%","21–30%","31–40%","41–50%","51–60%"])
    disc_r = df.groupby(disc_bins, observed=True)["Purchase_Status"].apply(
        lambda x: round((x=="Purchased").mean()*100,2)
    ).reset_index(); disc_r.columns = ["Discount Range","Purchase Rate (%)"]

    col_l, col_r = st.columns([1.6,1])
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=disc_r["Discount Range"], y=disc_r["Purchase Rate (%)"],
            mode="lines+markers",
            line=dict(color=C_ORANGE, width=2.8),
            marker=dict(size=9, color=C_ORANGE),
            fill="tozeroy", fillcolor="rgba(255,153,0,0.12)",
            text=[f"{v}%" for v in disc_r["Purchase Rate (%)"]],
            textposition="top center", textfont_color="#E8EAF0",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False, yaxis_title="Purchase Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(disc_r, use_container_width=True, hide_index=True)

    # Trend 3
    sec("⏱️", "Trend 3 — Time on Site vs Purchase Rate")
    time_bins = pd.cut(df["Time_on_Site_Min"], bins=[0,5,10,15,20,30,60],
                       labels=["0–5 min","5–10 min","10–15 min","15–20 min","20–30 min","30+ min"])
    time_r = df.groupby(time_bins, observed=True)["Purchase_Status"].apply(
        lambda x: round((x=="Purchased").mean()*100,2)
    ).reset_index(); time_r.columns = ["Time on Site","Purchase Rate (%)"]

    col_l, col_r = st.columns([1.6,1])
    with col_l:
        fig = px.bar(time_r, x="Time on Site", y="Purchase Rate (%)",
                     color="Purchase Rate (%)", color_continuous_scale=[[0,"#1E2130"],[1,C_GREEN]],
                     text=time_r["Purchase Rate (%)"].apply(lambda v: f"{v}%"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
        fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(time_r, use_container_width=True, hide_index=True)

    # Trend 4
    sec("🛒", "Trend 4 — Cart Additions vs Purchase Rate")
    cart_r = df.groupby("Cart_Additions")["Purchase_Status"].apply(
        lambda x: round((x=="Purchased").mean()*100,2)
    ).reset_index(); cart_r.columns = ["Cart Additions","Purchase Rate (%)"]

    col_l, col_r = st.columns([1.6,1])
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cart_r["Cart Additions"], y=cart_r["Purchase Rate (%)"],
            mode="lines+markers",
            line=dict(color=C_BLUE, width=2.8),
            marker=dict(size=8, color=C_BLUE),
            fill="tozeroy", fillcolor="rgba(20,110,180,0.12)",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False,
                          xaxis_title="Cart Additions", yaxis_title="Purchase Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(cart_r, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: ML TRAINING
# ═══════════════════════════════════════════════════════════
elif page == "🤖 ML Training":
    page_pill(9, "Prepare Data for Machine Learning")

    df = df_raw.copy()

    # Feature table
    sec("🔢", "Feature Set")
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Type": ["Numeric"]*10 + ["Encoded Categorical"]*5,
        "Description": [
            "Customer age","Product price (₹)","Quantity ordered",
            "Discount applied (%)","Customer rating (1–5)",
            "No. of prior purchases","Website visits",
            "Cart additions","Time on site (min)","Delivery days",
            "Gender (encoded)","Product Category (encoded)",
            "Payment Method (encoded)","Device (encoded)","Return Status (encoded)"
        ]
    })
    c_feat, c_split = st.columns([2,1])
    with c_feat:
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    with c_split:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(kpi("🏋️", "800", "Training Samples", C_ORANGE), unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown(kpi("🧪", "200", "Test Samples", C_BLUE), unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown(kpi("🔀", "80/20", "Train/Test Split", "#AB47BC"), unsafe_allow_html=True)

    page_pill(10, "Train Classification Models")

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_pre = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1  = f1_score(y_test, rf_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_pre = precision_score(y_test, lr_pred)
    lr_rec = recall_score(y_test, lr_pred)
    lr_f1  = f1_score(y_test, lr_pred)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("RF Accuracy",  f"{rf_acc:.3f}")
    with m2: st.metric("RF Precision", f"{rf_pre:.3f}")
    with m3: st.metric("RF Recall",    f"{rf_rec:.3f}")
    with m4: st.metric("RF F1-Score",  f"{rf_f1:.3f}")

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    m5, m6, m7, m8 = st.columns(4)
    with m5: st.metric("LR Accuracy",  f"{lr_acc:.3f}", delta=f"{lr_acc-rf_acc:+.3f}")
    with m6: st.metric("LR Precision", f"{lr_pre:.3f}", delta=f"{lr_pre-rf_pre:+.3f}")
    with m7: st.metric("LR Recall",    f"{lr_rec:.3f}", delta=f"{lr_rec-rf_rec:+.3f}")
    with m8: st.metric("LR F1-Score",  f"{lr_f1:.3f}",  delta=f"{lr_f1-rf_f1:+.3f}")

    st.success("✅ Random Forest & Logistic Regression trained on 800 samples, evaluated on 200.")

    sec("📊", "Feature Importance — Random Forest")
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=[[0,"#1E2130"],[1,C_ORANGE]],
                 text=fi["Importance"].apply(lambda v: f"{v:.3f}"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textposition="outside", textfont_color="#E8EAF0")
    fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False, xaxis_title="Importance Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Feature Importance Table"):
        st.dataframe(fi.sort_values("Importance", ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: PREDICTION & EVALUATION
# ═══════════════════════════════════════════════════════════
elif page == "🎯 Prediction & Evaluation":
    page_pill(11, "Evaluate the Model")

    # Comparison table
    sec("📊", "Model Performance Comparison")
    comp = pd.DataFrame({
        "Model":     ["Random Forest", "Logistic Regression"],
        "Accuracy":  [accuracy_score(y_test, rf_pred), accuracy_score(y_test, lr_pred)],
        "Precision": [precision_score(y_test, rf_pred), precision_score(y_test, lr_pred)],
        "Recall":    [recall_score(y_test, rf_pred), recall_score(y_test, lr_pred)],
        "F1-Score":  [f1_score(y_test, rf_pred), f1_score(y_test, lr_pred)],
    }).round(4)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # Confusion matrices
    sec("🧩", "Confusion Matrices")
    col1, col2 = st.columns(2)

    for col_w, pred, title, cmap in [
        (col1, rf_pred, "Random Forest",    "Oranges"),
        (col2, lr_pred, "Logistic Regression", "Blues")
    ]:
        with col_w:
            cm = confusion_matrix(y_test, pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
            fig_cm.patch.set_facecolor("#0F1117"); ax_cm.set_facecolor("#161929")
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                        xticklabels=["Not Purchased","Purchased"],
                        yticklabels=["Not Purchased","Purchased"],
                        linewidths=0.5, ax=ax_cm, cbar=False,
                        annot_kws={"size":14,"weight":"bold","color":"white"})
            ax_cm.set_xlabel("Predicted", color="#CBD5E1", fontsize=10)
            ax_cm.set_ylabel("Actual",    color="#CBD5E1", fontsize=10)
            ax_cm.set_title(f"{title}\nConfusion Matrix", color="#E8EAF0", fontsize=10, fontweight="bold")
            ax_cm.tick_params(colors="#CBD5E1")
            plt.setp(ax_cm.get_xticklabels(), color="#CBD5E1")
            plt.setp(ax_cm.get_yticklabels(), color="#CBD5E1")
            plt.tight_layout()
            st.pyplot(fig_cm); plt.close()

    # ROC Curve (Plotly)
    sec("📈", "ROC Curves")
    fig_roc = go.Figure()
    for label, prob, color in [
        ("Random Forest", rf_prob, C_ORANGE),
        ("Logistic Regression", lr_prob, C_BLUE)
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{label}  (AUC = {roc_auc:.3f})",
                                     line=dict(color=color, width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  name="Baseline", line=dict(color="#555", dash="dash", width=1.2)))
    fig_roc.update_layout(**PLOTLY_LAYOUT, height=360,
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           title_text="ROC Curve Comparison")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Classification reports
    sec("📋", "Classification Reports")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Random Forest**")
        rpt = classification_report(y_test, rf_pred, target_names=["Not Purchased","Purchased"], output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
    with col4:
        st.markdown("**Logistic Regression**")
        rpt2 = classification_report(y_test, lr_pred, target_names=["Not Purchased","Purchased"], output_dict=True)
        st.dataframe(pd.DataFrame(rpt2).T.round(3), use_container_width=True)

    # ── Live Predictor ──
    st.markdown("---")
    page_pill(12, "Live Customer Purchase Prediction")
    st.markdown(
        "<p style='color:#7B8DB0;font-size:0.9rem;margin-bottom:1rem;'>"
        "Enter customer details below and click <b>Predict</b> to score purchase probability in real time.</p>",
        unsafe_allow_html=True
    )

    with st.form("pred_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown("**🧑 Customer Profile**")
            age       = st.slider("Age", 18, 70, 30)
            gender    = st.selectbox("Gender", ["Male", "Female", "Other"])
            prev_pur  = st.number_input("Previous Purchases", 0, 50, 3, step=1)
            rating    = st.slider("Customer Rating", 1.0, 5.0, 3.5, 0.1)
        with r1c2:
            st.markdown("**🛍️ Product & Order**")
            category  = st.selectbox("Product Category", sorted(df_raw["Product_Category"].unique()))
            price     = st.number_input("Price (₹)", 100.0, 50000.0, 2000.0, step=100.0)
            quantity  = st.number_input("Quantity", 1, 10, 1)
            discount  = st.slider("Discount (%)", 0, 60, 10)
        with r1c3:
            st.markdown("**🌐 Browsing & Delivery**")
            device    = st.selectbox("Device", sorted(df_raw["Device"].unique()))
            payment   = st.selectbox("Payment Method", sorted(df_raw["Payment_Method"].unique()))
            visits    = st.number_input("Website Visits", 0, 20, 4, step=1)
            cart_add  = st.number_input("Cart Additions", 0, 15, 2, step=1)
            time_site = st.number_input("Time on Site (min)", 0.0, 60.0, 10.0, step=0.5)
            delivery  = st.number_input("Delivery Days", 1, 15, 4, step=1)
            ret_stat  = st.selectbox("Return Status", sorted(df_raw["Return_Status"].unique()))

        submitted = st.form_submit_button("🔮  Predict Purchase Likelihood", use_container_width=True)

    if submitted:
        enc_map = {}
        for col in ["Gender","Product_Category","Payment_Method","Device","Return_Status"]:
            le2 = LabelEncoder()
            le2.fit(df_raw[col].astype(str))
            enc_map[col] = le2

        def safe_enc(le_obj, val):
            return le_obj.transform([val])[0] if val in le_obj.classes_ else 0

        inp = np.array([[
            age, price, quantity, discount, rating,
            prev_pur, visits, cart_add, time_site, delivery,
            safe_enc(enc_map["Gender"], gender),
            safe_enc(enc_map["Product_Category"], category),
            safe_enc(enc_map["Payment_Method"], payment),
            safe_enc(enc_map["Device"], device),
            safe_enc(enc_map["Return_Status"], ret_stat),
        ]])
        inp_s    = scaler.transform(inp)
        pred_val = rf_model.predict(inp_s)[0]
        prob_buy = rf_model.predict_proba(inp_s)[0][1]
        prob_no  = 1 - prob_buy

        if pred_val == 1:
            st.markdown(f"""
            <div class="pred-yes">
              <div class="pred-label" style="color:#1DB954;">✅ WILL PURCHASE</div>
              <div class="pred-sub">Purchase Probability: <strong style="color:#1DB954;font-size:1.25rem;">{prob_buy*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-no">
              <div class="pred-label" style="color:{C_RED};">❌ UNLIKELY TO PURCHASE</div>
              <div class="pred-sub">No-Purchase Probability: <strong style="color:{C_RED};font-size:1.25rem;">{prob_no*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.metric("Purchase Probability",    f"{prob_buy*100:.1f}%")
        with mc2: st.metric("No-Purchase Probability", f"{prob_no*100:.1f}%")
        with mc3: st.metric("Prediction Confidence",   f"{max(prob_buy, prob_no)*100:.1f}%")

        # Visual probability bar
        st.markdown(f"""
        <div style="margin-top:0.8rem;">
          <div style="font-size:0.78rem;color:#7B8DB0;margin-bottom:0.3rem;">Purchase Probability Bar</div>
          <div style="display:flex;height:14px;border-radius:99px;overflow:hidden;border:1px solid {C_BORDER};">
            <div style="width:{prob_buy*100:.1f}%;background:{C_GREEN};"></div>
            <div style="width:{prob_no*100:.1f}%;background:{C_RED};opacity:0.7;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#7B8DB0;margin-top:0.3rem;">
            <span>🟢 Purchase {prob_buy*100:.1f}%</span>
            <span>🔴 No Purchase {prob_no*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    page_pill(13, "Business Insights & Strategic Recommendations")

    df = df_raw.copy()
    total_rev  = df["Sales_Amount"].sum()
    conv_rate  = (df["Purchase_Status"] == "Purchased").mean() * 100
    avg_rat    = df["Customer_Rating"].mean()
    ret_rate   = (df["Return_Status"] == "Returned").mean() * 100
    top_cat    = df.groupby("Product_Category")["Sales_Amount"].sum().idxmax()
    top_city   = df.groupby("City")["Sales_Amount"].sum().idxmax()

    c1,c2,c3 = st.columns(3)
    c4,c5,c6 = st.columns(3)
    with c1: st.markdown(kpi("💵", f"₹{total_rev:,.0f}", "Total Revenue"),            unsafe_allow_html=True)
    with c2: st.markdown(kpi("🎯", f"{conv_rate:.1f}%",  "Conversion Rate", C_BLUE),  unsafe_allow_html=True)
    with c3: st.markdown(kpi("⭐", f"{avg_rat:.2f}",     "Avg Rating",       C_GREEN), unsafe_allow_html=True)
    with c4: st.markdown(kpi("↩️", f"{ret_rate:.1f}%",  "Return Rate",      C_RED),   unsafe_allow_html=True)
    with c5: st.markdown(kpi("🏆", top_cat,              "Top Category",     "#AB47BC"), unsafe_allow_html=True)
    with c6: st.markdown(kpi("🌆", top_city,             "Top City",         "#26C6DA"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Key findings
    sec("📌", "Key Findings")

    findings = [
        ("#FF9900", "🏆 Revenue & Conversion",
         ["Electronics and Home Appliances generate the highest revenue per order.",
          "Mobile devices drive the largest share of traffic and purchases — mobile-first UX is critical.",
          "Overall conversion rate is strong, indicating healthy engagement across categories."]),
        ("#146EB4", "🎯 Customer Loyalty",
         ["Champion-tier customers (11+ purchases) convert at significantly higher rates.",
          "New customers (0–2 purchases) have lower conversion — targeted onboarding incentives can help.",
          "Sessions >15 min on site show markedly higher purchase intent."]),
        ("#1DB954", "💸 Discounts & Pricing",
         ["Discounts of 20–40% maximise purchase rate without deeply eroding margins.",
          "Very high discounts (50%+) do NOT proportionally increase conversion.",
          "Cart additions are one of the strongest predictors of purchase intent."]),
        ("#AB47BC", "🤖 ML Model Results",
         ["Random Forest outperforms Logistic Regression on all metrics.",
          "Top predictors: Previous Purchases, Customer Rating, Cart Additions, Website Visits.",
          "The live predictor tool can score any customer in real time for targeting."]),
    ]
    for accent, title, bullets in findings:
        st.markdown(f"""
        <div class="insight-card" style="--accent:{accent}">
          <h4>{title}</h4>
          <ul>{''.join(f"<li>{b}</li>" for b in bullets)}</ul>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    sec("🚀", "Strategic Recommendations")

    recs = [
        ("📱", "#FF9900", "Mobile-First Optimization",
         "Most purchases happen on mobile. Invest in responsive UX, fast load times, and one-tap checkout."),
        ("🔁", "#146EB4", "Loyalty Programme",
         "Reward repeat customers with exclusive discounts. Champion-tier customers have the highest LTV."),
        ("🎯", "#1DB954", "Precision Discounting",
         "Use the ML model to identify high-probability customers — offer targeted 20–35% discounts instead of blanket deals."),
        ("🛒", "#AB47BC", "Cart Abandonment Recovery",
         "Trigger automated recovery emails/push notifications within 30 min of cart addition without checkout."),
        ("⭐", "#26C6DA", "Rating & Quality Improvement",
         "Products rated below 3.5 convert poorly. Prioritise post-delivery feedback and resolve complaints fast."),
        ("🌆", "#EF5350", "Regional Expansion",
         "Top states drive disproportionate revenue. Increase warehouse proximity & improve delivery SLAs there."),
        ("📦", "#FFA726", "Inventory Optimisation",
         "Use category-level trends to forecast demand and prevent stockouts in high-revenue product lines."),
        ("🔬", "#66BB6A", "A/B Testing",
         "Test checkout flow variants for lowest-converting payment methods (UPI / Net Banking)."),
    ]
    col_a, col_b = st.columns(2)
    for i, (icon, accent, title, desc) in enumerate(recs):
        target = col_a if i % 2 == 0 else col_b
        with target:
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-icon">{icon}</div>
              <div>
                <div class="rec-title" style="color:{accent};">{title}</div>
                <div class="rec-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Mini conclusion chart
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    sec("📊", "Metric Comparison at a Glance")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest":    [accuracy_score(y_test, rf_pred), precision_score(y_test, rf_pred),
                             recall_score(y_test, rf_pred), f1_score(y_test, rf_pred)],
        "Logistic Regression": [accuracy_score(y_test, lr_pred), precision_score(y_test, lr_pred),
                                recall_score(y_test, lr_pred), f1_score(y_test, lr_pred)],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Random Forest",
                         x=metrics_df["Metric"], y=metrics_df["Random Forest"],
                         marker_color=C_ORANGE,
                         text=[f"{v:.3f}" for v in metrics_df["Random Forest"]],
                         textposition="outside", textfont_color="#E8EAF0"))
    fig.add_trace(go.Bar(name="Logistic Regression",
                         x=metrics_df["Metric"], y=metrics_df["Logistic Regression"],
                         marker_color=C_BLUE,
                         text=[f"{v:.3f}" for v in metrics_df["Logistic Regression"]],
                         textposition="outside", textfont_color="#E8EAF0"))
    fig.update_layout(**PLOTLY_LAYOUT, height=310, barmode="group",
                      yaxis=dict(range=[0, 1.05], gridcolor="#1E2130"),
                      yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
