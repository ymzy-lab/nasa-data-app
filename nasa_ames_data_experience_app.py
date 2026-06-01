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
# NASA Data Experience App
# Mobile-first UX/UI version with interactive Plotly charts.
# Comments are written in English as requested.
# ------------------------------------------------------------

st.set_page_config(
    page_title="NASA AI Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# Custom CSS for mobile-friendly visual design
# ------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top, #193b78 0%, #08111f 42%, #02040a 100%);
        color: #f5f7ff;
    }

    section[data-testid="stSidebar"] {
        background: #07111f;
    }

    h1, h2, h3 {
        color: #ffffff;
        letter-spacing: 0.02em;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.0rem;
    }

    .hero-card {
        padding: 1.4rem 1.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(61, 124, 255, 0.95), rgba(92, 220, 255, 0.72));
        box-shadow: 0 18px 48px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: clamp(2.0rem, 8vw, 4.2rem);
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: clamp(1.0rem, 4vw, 1.4rem);
        font-weight: 650;
        opacity: 0.98;
    }

    .glass-card {
        padding: 1.0rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 12px 32px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }

    .mission-card {
        padding: 1.0rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.92);
        color: #091321;
        margin-bottom: 0.8rem;
    }

    .mission-card h3 {
        color: #091321;
        margin-bottom: 0.2rem;
    }

    .small-caption {
        font-size: 0.9rem;
        opacity: 0.82;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3.8rem;
        border-radius: 999px;
        font-size: 1.15rem;
        font-weight: 850;
        background: linear-gradient(90deg, #ffda5c, #ff8a3d);
        color: #111827;
        border: none;
        box-shadow: 0 12px 28px rgba(255, 138, 61, 0.28);
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 999px;
        font-weight: 800;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        background: rgba(255,255,255,0.10);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .hero-card {
            border-radius: 24px;
            padding: 1.2rem 1.0rem;
        }
        .glass-card {
            border-radius: 20px;
            padding: 0.9rem;
        }
        div[data-testid="metric-container"] {
            padding: 0.85rem;
        }
    }
</style>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
<div class="hero-card">
  <div class="hero-title">🚀 NASA AI Lab</div>
  <div class="hero-subtitle">本物のNASAデータで、太陽エネルギーをAI予測する。</div>
  <p style="margin-top:0.8rem; font-size:1.02rem;">スマホで体験できる、宇宙・気象データサイエンス入門。</p>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="glass-card">
<strong>体験ミッション</strong><br>
場所を選ぶ → NASAデータ取得 → AIが日射量を予測 → ミッション判定。<br>
<span class="small-caption">グラフはスマホでピンチ拡大・ドラッグ移動できる。</span>
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

st.markdown(
    f"""
<div class="mission-card">
<h3>📍 現在のミッション</h3>
解析地点：<strong>{preset}</strong><br>
緯度：<strong>{lat:.3f}</strong> ／ 経度：<strong>{lon:.3f}</strong><br>
期間：<strong>{start_date}</strong> 〜 <strong>{end_date}</strong>
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
# Plotly helper functions
# ------------------------------------------------------------

def apply_mobile_plotly_layout(fig, height=560):
    """Apply a dark, mobile-friendly Plotly layout."""

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
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.14)",
        zeroline=False
    )
    return fig


def show_interactive_chart(fig):
    """Render a Plotly chart with mobile zoom enabled."""

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
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

if not run_button:
    st.markdown("### 2. 何ができる？")
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            """
<div class="glass-card">
<h3>🌏 NASAデータ</h3>
気温・湿度・風速・雨・日射量を取得する。
</div>
""",
            unsafe_allow_html=True
        )

    with card2:
        st.markdown(
            """
<div class="glass-card">
<h3>🤖 AI予測</h3>
気象条件から太陽エネルギーを予測する。
</div>
""",
            unsafe_allow_html=True
        )

    with card3:
        st.markdown(
            """
<div class="glass-card">
<h3>⚡ ミッション判定</h3>
太陽光発電に向く場所かを判定する。
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("### おすすめ体験")
    st.markdown(
        """
<div class="mission-card">
<strong>大阪電気通信大学</strong> と <strong>サハラ砂漠</strong> を比べると、データの違いが一発で見える。<br>
次に <strong>南極</strong> を選ぶと、太陽光だけに頼る難しさが見えてくる。
</div>
""",
        unsafe_allow_html=True
    )
    st.stop()

# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

if start_date >= end_date:
    st.error("開始日は終了日より前にしてください。")
    st.stop()

with st.spinner("NASA POWER APIからデータを取得中..."):
    try:
        df = fetch_nasa_power_data(lat, lon, start_date, end_date)
    except Exception as e:
        st.error("データ取得に失敗しました。期間を短くするか、ネットワークを確認してください。")
        st.exception(e)
        st.stop()

if df.empty:
    st.warning("有効なデータが取得できませんでした。")
    st.stop()

features = [
    "temperature_C",
    "humidity_percent",
    "wind_speed_m_s",
    "precipitation_mm_day"
]
target = "solar_radiation_kWh_m2_day"

if len(df) < 30:
    st.warning("AI予測には30日以上のデータを推奨します。")
    st.stop()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=8
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
avg_solar = df[target].mean()
max_solar = df[target].max()
min_solar = df[target].min()

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

st.markdown("### 2. ミッション結果")

m1, m2, m3 = st.columns(3)
m1.metric("平均日射量", f"{avg_solar:.2f}", "kWh/m²/day")
m2.metric("AI平均誤差", f"{mae:.2f}", "kWh/m²/day")
m3.metric("AIスコア R²", f"{r2:.2f}")

st.markdown(
    f"""
<div class="mission-card">
<h3>{verdict_title}</h3>
{verdict_text}<br>
<span class="small-caption">最高日射量：{max_solar:.2f} ／ 最低日射量：{min_solar:.2f} kWh/m²/day</span>
</div>
""",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["📈 日射", "🤖 予測", "🔍 理由", "📄 データ"])

with tab1:
    st.markdown("#### 日射量の変化")
    st.caption("スマホでは2本指でピンチ拡大、ドラッグで移動できる。")

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
    fig1 = apply_mobile_plotly_layout(fig1, height=560)
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
    st.caption("この散布図もピンチ拡大・ドラッグ移動できる。点が斜め線に近いほどAI予測が正確だ。")

    pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Error": y_pred - y_test.values
    })

    min_v = float(min(pred_df["Actual"].min(), pred_df["Predicted"].min()))
    max_v = float(max(pred_df["Actual"].max(), pred_df["Predicted"].max()))

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
            "Actual": ":.2f",
            "Predicted": ":.2f",
            "Error": ":.2f"
        },
        opacity=0.78
    )

    fig2.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Perfect prediction",
            line=dict(width=2, dash="dash")
        )
    )

    fig2.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
    fig2.update_xaxes(range=[min_v, max_v])
    fig2.update_yaxes(range=[min_v, max_v])
    fig2 = apply_mobile_plotly_layout(fig2, height=620)
    show_interactive_chart(fig2)

    st.markdown(
        """
<div class="glass-card">
AIは未来を魔法で当てているのではなく、気象データのパターンを読んでいる。実測値と予測値のズレを見ることで、AIの限界も見える。
</div>
""",
        unsafe_allow_html=True
    )

with tab3:
    st.markdown("#### 何が予測に効いたか")

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    label_map = {
        "temperature_C": "気温",
        "humidity_percent": "湿度",
        "wind_speed_m_s": "風速",
        "precipitation_mm_day": "降水量"
    }
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
    fig3 = apply_mobile_plotly_layout(fig3, height=440)
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
        file_name="nasa_power_data.csv",
        mime="text/csv"
    )

st.markdown(
    """
<div class="hero-card">
  <div style="font-size:1.5rem; font-weight:900;">君もNASAデータを解析する側へ。</div>
  <div style="font-size:1.0rem; margin-top:0.4rem;">宇宙・気象・AI・データサイエンスを、スマホから体験しよう。</div>
</div>
""",
    unsafe_allow_html=True
)
