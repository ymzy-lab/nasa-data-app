import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------
# SkyFusion Explorer : Solar AI
# NASA POWER data analysis app for mobile-first outreach.
# Comments are written in English as requested.
# ------------------------------------------------------------

APP_TITLE = "SkyFusion Explorer"
APP_MODULE = "Solar AI"
APP_SUBTITLE = "NASAの本物のデータで、太陽エネルギーをAI予測する。"

st.set_page_config(
    page_title=f"{APP_TITLE} : {APP_MODULE}",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------

DEFAULT_STATE = {
    "analysis_done": False,
    "df": None,
    "analysis_result": None,
    "last_request_key": None,
    "error_message": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# Premium CSS
# ------------------------------------------------------------

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 16% 0%, rgba(126,232,255,0.25), rgba(126,232,255,0) 28%),
            radial-gradient(circle at 90% 8%, rgba(255,134,200,0.22), rgba(255,134,200,0) 30%),
            radial-gradient(circle at 50% 0%, rgba(62,116,255,0.36), rgba(62,116,255,0) 48%),
            linear-gradient(180deg, #061632 0%, #040916 48%, #02040b 100%);
        color: #f8fbff;
    }

    .block-container { max-width: 1180px; padding-top: 1rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background: #050b16; }
    h1, h2, h3, h4 { color: #ffffff; letter-spacing: 0.01em; }

    .hero-shell {
        position: relative; overflow: hidden; border-radius: 34px; padding: 1px;
        background: linear-gradient(135deg, rgba(126,232,255,0.95), rgba(255,134,200,0.58), rgba(255,214,107,0.82));
        box-shadow: 0 24px 70px rgba(0,0,0,0.42); margin-bottom: 1.1rem;
    }
    .hero {
        position: relative; border-radius: 33px; padding: 1.55rem 1.35rem 1.25rem;
        background: radial-gradient(circle at 80% 15%, rgba(255,255,255,0.26), rgba(255,255,255,0) 28%),
                    linear-gradient(135deg, rgba(16,66,148,0.98), rgba(18,23,58,0.96) 55%, rgba(5,9,22,0.96));
        min-height: 250px;
    }
    .hero::after {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background-image: radial-gradient(circle, rgba(255,255,255,0.50) 1px, transparent 1.5px), radial-gradient(circle, rgba(255,255,255,0.24) 1px, transparent 1.5px);
        background-size: 38px 38px, 64px 64px; background-position: 0 0, 17px 19px; opacity: 0.28;
    }
    .eyebrow {
        position: relative; z-index: 1; display: inline-flex; align-items: center; gap: .45rem;
        padding: .36rem .72rem; border-radius: 999px; background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.20); color: rgba(255,255,255,.90); font-size: .86rem; font-weight: 700; margin-bottom: .9rem;
    }
    .hero-title {
        position: relative; z-index: 1; font-size: clamp(2.2rem, 8.5vw, 5.3rem); font-weight: 900;
        line-height: .96; letter-spacing: -.045em; color: #ffffff; text-shadow: 0 10px 34px rgba(0,0,0,.30); margin: 0 0 .65rem;
    }
    .hero-title span { background: linear-gradient(90deg, #ffffff 0%, #aef2ff 42%, #ffd66b 78%, #ff9bd1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { position: relative; z-index: 1; max-width: 760px; color: rgba(255,255,255,.91); font-size: clamp(1.02rem,3.7vw,1.34rem); font-weight: 700; line-height: 1.7; }
    .hero-chips { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: .48rem; margin-top: .9rem; }
    .chip { display: inline-flex; align-items: center; gap: .35rem; padding: .44rem .72rem; border-radius: 999px; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.18); color: rgba(255,255,255,.94); font-size: .88rem; font-weight: 700; }
    .orb { position:absolute; right:-55px; top:-45px; width:185px; height:185px; border-radius:999px; background: radial-gradient(circle at 35% 35%, #fff 0%, #f6fdff 16%, #69d9ff 34%, #2870e5 62%, #102d78 100%); box-shadow: inset -24px -22px 48px rgba(0,0,0,.24), 0 0 80px rgba(126,232,255,.42); opacity:.92; }
    .sun-dot { position:absolute; right:95px; bottom:34px; width:46px; height:46px; border-radius:999px; background: radial-gradient(circle, #fff8b8 0%, #ffd66b 45%, #ff8a3d 100%); box-shadow: 0 0 45px rgba(255,214,107,.8); }

    .panel, .glass-card {
        border-radius: 28px; padding: 1.05rem; background: rgba(255,255,255,.092); border: 1px solid rgba(255,255,255,.15);
        box-shadow: 0 18px 48px rgba(0,0,0,.24); backdrop-filter: blur(16px); margin-bottom: 1rem;
    }
    .panel-title { font-size: 1.05rem; font-weight: 900; color: #fff; margin-bottom: .25rem; }
    .panel-text { color: rgba(255,255,255,.78); font-size: .94rem; line-height: 1.72; }

    .mission-card {
        padding: 1.08rem; border-radius: 26px; background: linear-gradient(135deg, rgba(255,255,255,.97), rgba(239,250,255,.93));
        color: #07111f; box-shadow: 0 18px 48px rgba(0,0,0,.25); border: 1px solid rgba(255,255,255,.72); margin-bottom: 1rem;
    }
    .mission-card h3 { color: #07111f; margin-bottom: .35rem; }
    .mission-meta { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .65rem; margin-top: .7rem; }
    .mini-box { border-radius: 18px; padding: .75rem; background: rgba(7,17,31,.055); border: 1px solid rgba(7,17,31,.075); }
    .mini-label { color: rgba(7,17,31,.64); font-size: .78rem; font-weight: 800; margin-bottom: .2rem; }
    .mini-value { color: #07111f; font-size: .98rem; font-weight: 900; }

    .step-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .72rem; margin: .8rem 0 1rem; }
    .step { border-radius: 22px; padding: .92rem; background: rgba(255,255,255,.095); border: 1px solid rgba(255,255,255,.14); min-height: 118px; }
    .step-icon { font-size: 1.45rem; margin-bottom: .35rem; }
    .step-title { font-weight: 900; color: #fff; margin-bottom: .25rem; }
    .step-text { color: rgba(255,255,255,.73); font-size: .84rem; line-height: 1.55; }

    div.stButton > button {
        width: 100%; min-height: 4rem; border-radius: 999px; font-size: 1.12rem; font-weight: 900; letter-spacing: .01em;
        background: linear-gradient(90deg, #ffe07a 0%, #ff9b48 56%, #ff73bd 100%); color: #101827; border: none;
        box-shadow: 0 18px 38px rgba(255,138,61,.34); transition: transform .12s ease, filter .12s ease;
    }
    div.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.04); }
    div.stDownloadButton > button { width: 100%; min-height: 3.2rem; border-radius: 999px; font-weight: 900; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,.18), rgba(255,255,255,.09)); border: 1px solid rgba(255,255,255,.17);
        border-radius: 24px; padding: 1rem; box-shadow: 0 16px 38px rgba(0,0,0,.20); backdrop-filter: blur(14px);
    }
    div[data-testid="metric-container"] label { color: rgba(255,255,255,.78) !important; font-weight: 800; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #fff !important; font-weight: 900; }

    .stTabs [data-baseweb="tab-list"] { gap: .45rem; background: rgba(255,255,255,.08); padding: .35rem; border-radius: 999px; border: 1px solid rgba(255,255,255,.12); }
    .stTabs [data-baseweb="tab"] { border-radius: 999px; padding: .55rem .9rem; background: transparent; color: rgba(255,255,255,.78); font-weight: 900; }
    .stTabs [aria-selected="true"] { background: rgba(255,255,255,.18) !important; color: #fff !important; }

    .chart-note { border-left: 4px solid #ffd66b; padding: .85rem .95rem; border-radius: 16px; background: rgba(255,214,107,.11); color: rgba(255,255,255,.84); font-size: .92rem; line-height: 1.68; margin-bottom: .9rem; }
    .final-cta { border-radius: 30px; padding: 1.25rem; background: radial-gradient(circle at 92% 20%, rgba(255,214,107,.32), rgba(255,214,107,0) 32%), linear-gradient(135deg, rgba(126,232,255,.22), rgba(255,134,200,.16)); border: 1px solid rgba(255,255,255,.16); box-shadow: 0 18px 48px rgba(0,0,0,.24); margin-top: 1.1rem; }
    .final-cta-title { font-size: 1.45rem; font-weight: 900; color: #fff; }
    .final-cta-text { margin-top: .35rem; color: rgba(255,255,255,.78); line-height: 1.7; }
    .small-caption { font-size: .88rem; opacity: .72; }

    .stSelectbox label, .stNumberInput label, .stDateInput label { color: rgba(255,255,255,.78) !important; font-weight: 800; }
    div[data-baseweb="select"] > div { border-radius: 18px; background: rgba(255,255,255,.11); border-color: rgba(255,255,255,.15); }
    input { border-radius: 16px !important; }

    @media (max-width: 768px) {
        .block-container { padding-top: .85rem; padding-left: .82rem; padding-right: .82rem; }
        .hero-shell { border-radius: 28px; }
        .hero { border-radius: 27px; padding: 1.2rem 1rem; min-height: 270px; }
        .orb { width: 135px; height: 135px; right: -48px; top: -32px; opacity: .72; }
        .sun-dot { right: 35px; bottom: 20px; width: 38px; height: 38px; }
        .hero-title { font-size: clamp(2.05rem, 12vw, 3.2rem); max-width: 310px; }
        .hero-subtitle { max-width: 310px; line-height: 1.62; }
        .panel, .glass-card { border-radius: 24px; padding: .92rem; }
        .mission-card { border-radius: 24px; padding: .95rem; }
        .mission-meta { grid-template-columns: 1fr; gap: .5rem; }
        .step-row { grid-template-columns: 1fr 1fr; gap: .6rem; }
        .step { min-height: 116px; border-radius: 20px; padding: .82rem; }
        div[data-testid="metric-container"] { border-radius: 20px; padding: .85rem; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; justify-content: flex-start; }
        .stTabs [data-baseweb="tab"] { min-width: fit-content; padding: .52rem .82rem; }
    }
</style>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    f"""
<div class="hero-shell">
  <div class="hero">
    <div class="orb"></div>
    <div class="sun-dot"></div>
    <div class="eyebrow">✨ Open Campus Data Experience</div>
    <div class="hero-title">{APP_TITLE}<br><span>{APP_MODULE}</span></div>
    <div class="hero-subtitle">{APP_SUBTITLE}<br>宇宙と地球の未来を、データから探検する。</div>
    <div class="hero-chips">
      <div class="chip">🌏 NASA POWER</div>
      <div class="chip">🤖 Machine Learning</div>
      <div class="chip">☀️ Solar Energy</div>
      <div class="chip">📱 Mobile First</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="panel">
  <div class="panel-title">体験ミッション</div>
  <div class="panel-text">
    場所を選ぶ → NASAデータを取得 → AIが日射量を予測 → 太陽光ミッション適性を判定する。
    スマホではグラフ下のスライダー、右上の＋−ボタン、リセットボタンで安定して拡大縮小できる。
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Location and period settings
# ------------------------------------------------------------

locations = {
    "大阪電気通信大学": (34.766, 135.628),
    "京都": (35.0116, 135.7681),
    "東京": (35.6762, 139.6503),
    "沖縄": (26.2124, 127.6809),
    "南極・昭和基地付近": (-69.006, 39.590),
    "サハラ砂漠": (23.4162, 25.6628),
}

st.markdown("### 1. ミッション地点を選ぶ")
col_a, col_b = st.columns([1.2, 1])

with col_a:
    preset = st.selectbox(
        "解析する場所",
        list(locations.keys()) + ["自由入力"],
        index=0,
        label_visibility="collapsed"
    )

with col_b:
    period_mode = st.selectbox(
        "期間",
        ["直近1年", "直近180日", "直近90日", "自分で選ぶ"],
        index=0,
        label_visibility="collapsed"
    )

if preset == "自由入力":
    st.markdown("#### 緯度・経度")
    lat_col, lon_col = st.columns(2)
    with lat_col:
        lat = st.number_input("緯度", value=34.766, min_value=-90.0, max_value=90.0)
    with lon_col:
        lon = st.number_input("経度", value=135.628, min_value=-180.0, max_value=180.0)
else:
    lat, lon = locations[preset]

end_day = date.today() - timedelta(days=10)

if period_mode == "直近1年":
    start_date = end_day - timedelta(days=365)
    end_date = end_day
elif period_mode == "直近180日":
    start_date = end_day - timedelta(days=180)
    end_date = end_day
elif period_mode == "直近90日":
    start_date = end_day - timedelta(days=90)
    end_date = end_day
else:
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("開始日", end_day - timedelta(days=365))
    with date_col2:
        end_date = st.date_input("終了日", end_day)

request_key = (
    preset,
    round(float(lat), 6),
    round(float(lon), 6),
    str(start_date),
    str(end_date),
)

st.markdown(
    f"""
<div class="mission-card">
  <h3>📍 現在のミッション</h3>
  <div>解析条件を確認して、下のボタンからNASAデータ解析を開始する。</div>
  <div class="mission-meta">
    <div class="mini-box"><div class="mini-label">Location</div><div class="mini-value">{preset}</div></div>
    <div class="mini-box"><div class="mini-label">Latitude / Longitude</div><div class="mini-value">{lat:.3f} / {lon:.3f}</div></div>
    <div class="mini-box"><div class="mini-label">Period</div><div class="mini-value">{start_date} → {end_date}</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

run_button = st.button("🚀 NASAデータを解析する")

# ------------------------------------------------------------
# NASA POWER API function
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_nasa_power_data(latitude, longitude, start, end):
    """Fetch daily meteorological data from NASA POWER API."""

    parameters = [
        "T2M",              # Temperature at 2 meters
        "RH2M",             # Relative humidity at 2 meters
        "WS2M",             # Wind speed at 2 meters
        "PRECTOTCORR",      # Precipitation
        "ALLSKY_SFC_SW_DWN" # Solar radiation
    ]

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    raw = data["properties"]["parameter"]
    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index()
    df = df.replace(-999, np.nan).dropna()

    df = df.rename(columns={
        "T2M": "temperature_C",
        "RH2M": "humidity_percent",
        "WS2M": "wind_speed_m_s",
        "PRECTOTCORR": "precipitation_mm_day",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_kWh_m2_day"
    })

    return df

# ------------------------------------------------------------
# Analysis function
# ------------------------------------------------------------

def run_analysis(df):
    """Train a Random Forest model and return analysis artifacts."""

    features = [
        "temperature_C",
        "humidity_percent",
        "wind_speed_m_s",
        "precipitation_mm_day"
    ]
    target = "solar_radiation_kWh_m2_day"

    if len(df) < 30:
        raise ValueError("AI予測には30日以上のデータが必要です。期間を長くしてください。")

    X = df[features]
    y = df[target]

    # Time-series safe split:
    # shuffle=False prevents future data from leaking into the training set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    avg_solar = df[target].mean()
    max_solar = df[target].max()
    min_solar = df[target].min()

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    pred_df = pd.DataFrame({
        "date": y_test.index,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Error": y_pred - y_test.values
    })

    return {
        "features": features,
        "target": target,
        "model": model,
        "mae": mae,
        "r2": r2,
        "avg_solar": avg_solar,
        "max_solar": max_solar,
        "min_solar": min_solar,
        "importance": importance,
        "pred_df": pred_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

# ------------------------------------------------------------
# Plotly helper functions
# ------------------------------------------------------------

def apply_mobile_timeseries_layout(fig, height=580):
    """Apply a dark, mobile-friendly layout for time-series charts."""

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        dragmode="pan",
        hovermode="x unified",
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False,
        rangeslider=dict(visible=True, thickness=0.11),
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(count=180, label="180d", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="rgba(255,255,255,0.14)",
            activecolor="rgba(255,218,92,0.85)",
            font=dict(color="#ffffff")
        )
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False,
        fixedrange=False
    )

    return fig


def apply_mobile_scatter_layout(fig, height=620):
    """Apply a dark, mobile-friendly layout for scatter charts."""

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        dragmode="zoom",
        hovermode="closest",
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False,
        fixedrange=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False,
        fixedrange=False
    )

    return fig


def apply_mobile_bar_layout(fig, height=440):
    """Apply a dark, mobile-friendly layout for bar charts."""

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        dragmode="pan",
        hovermode="closest",
        font=dict(size=13)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False,
        fixedrange=False
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        fixedrange=False
    )

    return fig


def show_interactive_chart(fig):
    """Render a Plotly chart with stable mobile controls."""

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "responsive": True,
            "doubleClick": "reset",
            "modeBarButtonsToAdd": [
                "zoom2d",
                "pan2d",
                "zoomIn2d",
                "zoomOut2d",
                "resetScale2d"
            ],
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
                "autoScale2d"
            ]
        }
    )

# ------------------------------------------------------------
# Intro screen
# ------------------------------------------------------------

def show_intro_screen():
    """Show the initial guidance screen."""

    st.markdown("### 2. 体験の流れ")

    st.markdown(
        """
<div class="step-row">
  <div class="step"><div class="step-icon">🌏</div><div class="step-title">地点を選ぶ</div><div class="step-text">大阪・沖縄・南極・サハラ砂漠などから選ぶ。</div></div>
  <div class="step"><div class="step-icon">🛰️</div><div class="step-title">NASAデータ取得</div><div class="step-text">気温・湿度・風速・雨・日射量を読み込む。</div></div>
  <div class="step"><div class="step-icon">🤖</div><div class="step-title">AI予測</div><div class="step-text">機械学習で太陽エネルギーを予測する。</div></div>
  <div class="step"><div class="step-icon">☀️</div><div class="step-title">ミッション判定</div><div class="step-text">太陽光発電に向く場所かを判定する。</div></div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="mission-card">
<h3>おすすめ体験</h3>
<strong>大阪電気通信大学</strong> と <strong>サハラ砂漠</strong> を比べると、データの違いが一発で見える。<br>
次に <strong>南極</strong> を選ぶと、太陽光だけに頼る難しさが見えてくる。
</div>
""",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Run button behavior with session_state
# ------------------------------------------------------------

if run_button:
    st.session_state.error_message = None

    if start_date >= end_date:
        st.session_state.error_message = "開始日は終了日より前にしてください。"
        st.session_state.analysis_done = False
        st.session_state.df = None
        st.session_state.analysis_result = None
    else:
        with st.spinner("NASA POWER APIからデータを取得中..."):
            try:
                df_new = fetch_nasa_power_data(lat, lon, start_date, end_date)

                if df_new.empty:
                    raise ValueError("有効なデータが取得できませんでした。地点や期間を変更してください。")

                result_new = run_analysis(df_new)

                st.session_state.df = df_new
                st.session_state.analysis_result = result_new
                st.session_state.analysis_done = True
                st.session_state.last_request_key = request_key

            except Exception as e:
                st.session_state.analysis_done = False
                st.session_state.df = None
                st.session_state.analysis_result = None
                st.session_state.error_message = (
                    "データ取得または解析に失敗しました。期間を短くするか、通信状況を確認してください。"
                    f"\n\n詳細: {str(e)}"
                )

# ------------------------------------------------------------
# Display error or intro when no analysis exists
# ------------------------------------------------------------

if st.session_state.error_message:
    st.error(st.session_state.error_message)

if not st.session_state.analysis_done:
    show_intro_screen()
    st.stop()

# ------------------------------------------------------------
# Use stored analysis result
# ------------------------------------------------------------

df = st.session_state.df
result = st.session_state.analysis_result

if df is None or result is None:
    st.warning("解析結果が見つかりません。もう一度「NASAデータを解析する」を押してください。")
    show_intro_screen()
    st.stop()

if st.session_state.last_request_key != request_key:
    st.warning("地点または期間が変更されています。新しい条件で解析するには、もう一度「NASAデータを解析する」を押してください。")

features = result["features"]
target = result["target"]
mae = result["mae"]
r2 = result["r2"]
avg_solar = result["avg_solar"]
max_solar = result["max_solar"]
min_solar = result["min_solar"]
importance = result["importance"]
pred_df = result["pred_df"]
train_size = result["train_size"]
test_size = result["test_size"]

if avg_solar >= 5.0:
    verdict_title = "🟢 太陽光ミッション向き"
    verdict_text = "日射条件はかなり良好。太陽エネルギーを活用しやすい地点だ。"
elif avg_solar >= 3.0:
    verdict_title = "🟡 条件は中程度"
    verdict_text = "太陽光は使えるが、蓄電池やバックアップ電源が重要になる。"
else:
    verdict_title = "🔴 太陽光だけでは厳しい"
    verdict_text = "日射条件は弱め。太陽光だけに頼るミッション設計は危険だ。"

st.success("解析完了。NASAデータをAIで読み解いた。")

# ------------------------------------------------------------
# Result summary
# ------------------------------------------------------------

st.markdown("### 2. ミッション結果")

m1, m2, m3 = st.columns(3)
m1.metric("平均日射量", f"{avg_solar:.2f}", "kWh/m²/day")
m2.metric("AI平均誤差", f"{mae:.2f}", "kWh/m²/day")
m3.metric("AIスコア R²", f"{r2:.2f}")

st.markdown(
    f"""
<div class="mission-card">
<h3>{verdict_title}</h3>
{verdict_text}
<div class="mission-meta">
  <div class="mini-box"><div class="mini-label">Max / Min</div><div class="mini-value">{max_solar:.2f} / {min_solar:.2f}</div></div>
  <div class="mini-box"><div class="mini-label">Train / Test</div><div class="mini-value">{train_size}日 / {test_size}日</div></div>
  <div class="mini-box"><div class="mini-label">Method</div><div class="mini-value">Random Forest</div></div>
</div>
<span class="small-caption">時系列順に分割し、未来データの混入を避けている。</span>
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["📈 日射", "🤖 予測", "🔍 理由", "📄 データ"])

with tab1:
    st.markdown("#### 日射量の変化")
    st.markdown(
        """
<div class="chart-note">スマホではグラフ下の範囲スライダー、30d/90d/180dボタン、右上の＋−ボタンで拡大縮小できる。</div>
""",
        unsafe_allow_html=True
    )

    chart_df = df.reset_index().rename(columns={"index": "date"})

    fig1 = px.line(
        chart_df,
        x="date",
        y="solar_radiation_kWh_m2_day",
        title="Daily Solar Radiation from NASA POWER",
        labels={
            "date": "Date",
            "solar_radiation_kWh_m2_day": "Solar radiation [kWh/m²/day]"
        },
        hover_data={
            "temperature_C": ":.1f",
            "humidity_percent": ":.1f",
            "wind_speed_m_s": ":.2f",
            "precipitation_mm_day": ":.2f",
            "solar_radiation_kWh_m2_day": ":.2f"
        }
    )
    fig1.update_traces(line=dict(width=3))
    fig1 = apply_mobile_timeseries_layout(fig1, height=590)
    show_interactive_chart(fig1)

    st.markdown(
        """
<div class="glass-card">
山が高い日は、地表に届く太陽エネルギーが多い日。雲・雨・季節によって波形が変わる。
</div>
""",
        unsafe_allow_html=True
    )

with tab2:
    st.markdown("#### AI予測 vs 実測")
    st.markdown(
        """
<div class="chart-note">スマホでは範囲をなぞって拡大、右上の＋−ボタンで拡大縮小、家アイコンで全体表示に戻る。点が斜め線に近いほどAI予測が正確だ。</div>
""",
        unsafe_allow_html=True
    )

    min_v = float(min(pred_df["Actual"].min(), pred_df["Predicted"].min()))
    max_v = float(max(pred_df["Actual"].max(), pred_df["Predicted"].max()))
    pad = (max_v - min_v) * 0.06 if max_v > min_v else 0.5
    axis_min = min_v - pad
    axis_max = max_v + pad

    fig2 = px.scatter(
        pred_df,
        x="Actual",
        y="Predicted",
        title="AI Prediction Result: Actual vs Predicted",
        labels={
            "Actual": "Actual solar radiation [kWh/m²/day]",
            "Predicted": "Predicted solar radiation [kWh/m²/day]",
            "Error": "Prediction error"
        },
        hover_data={
            "date": True,
            "Actual": ":.2f",
            "Predicted": ":.2f",
            "Error": ":.2f"
        },
        opacity=0.78
    )

    fig2.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="Perfect prediction",
            line=dict(width=2, dash="dash")
        )
    )

    fig2.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
    fig2.update_xaxes(range=[axis_min, axis_max])
    fig2.update_yaxes(range=[axis_min, axis_max])
    fig2 = apply_mobile_scatter_layout(fig2, height=640)
    show_interactive_chart(fig2)

    st.markdown(
        """
<div class="glass-card">
AIは未来を魔法で当てているのではなく、過去から現在へ向かう時系列データのパターンを学習している。
今回の版では、未来のデータを学習に混ぜないよう、時系列順に学習用とテスト用を分けている。
</div>
""",
        unsafe_allow_html=True
    )

with tab3:
    st.markdown("#### 何が予測に効いたか")

    label_map = {
        "temperature_C": "気温",
        "humidity_percent": "湿度",
        "wind_speed_m_s": "風速",
        "precipitation_mm_day": "降水量"
    }
    importance = importance.copy()
    importance["label"] = importance["feature"].map(label_map)

    fig3 = px.bar(
        importance,
        x="importance",
        y="label",
        orientation="h",
        title="Feature Importance",
        labels={
            "importance": "Importance",
            "label": "Feature"
        },
        hover_data={"importance": ":.3f"}
    )
    fig3.update_traces(marker_line_width=0)
    fig3 = apply_mobile_bar_layout(fig3, height=450)
    show_interactive_chart(fig3)

    top_feature = importance.iloc[-1]["label"]
    st.markdown(
        f"""
<div class="mission-card">
<h3>今回もっとも効いた特徴量：{top_feature}</h3>
AIはこの地点では、{top_feature}の変化を強く見て日射量を予測している。
</div>
""",
        unsafe_allow_html=True
    )

with tab4:
    st.markdown("#### 取得データ")
    st.dataframe(df.tail(30), use_container_width=True)

    csv = df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        label="📥 CSVをダウンロード",
        data=csv,
        file_name="skyfusion_solar_ai_nasa_power_data.csv",
        mime="text/csv"
    )

st.markdown(
    """
<div class="final-cta">
  <div class="final-cta-title">君もNASAデータを解析する側へ。</div>
  <div class="final-cta-text">宇宙・気象・AI・データサイエンスを、スマホから体験しよう。次のアプリでは、系外惑星・電波天文学・宇宙線解析へ拡張できる。</div>
</div>
""",
    unsafe_allow_html=True
)
