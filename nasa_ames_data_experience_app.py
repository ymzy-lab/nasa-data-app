import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------
# NASA Data Experience App
# Theme: Analyze NASA POWER climate/solar data and predict solar energy potential.
# Comments are written in English as requested.
# ------------------------------------------------------------

st.set_page_config(
    page_title="NASA Data Experience App",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NASAデータ解析体験アプリ")
st.subheader("本物のNASA公開データを使って、AIで太陽光エネルギーを予測する")

st.markdown(
    """
このアプリでは、NASA POWER API の気象・日射データを取得し、  
気温・湿度・風速などから **日射量＝太陽光発電に関係する量** をAIで予測する。

オープンキャンパスでは、  
**「NASAの本物のデータをAIで読み解く」** 体験として使える。
"""
)

# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------

st.sidebar.header("解析地点を選ぶ")

preset = st.sidebar.selectbox(
    "場所",
    [
        "大阪電気通信大学 寝屋川キャンパス",
        "京都",
        "東京",
        "沖縄",
        "南極・昭和基地付近",
        "サハラ砂漠",
        "自由入力"
    ]
)

locations = {
    "大阪電気通信大学 寝屋川キャンパス": (34.766, 135.628),
    "京都": (35.0116, 135.7681),
    "東京": (35.6762, 139.6503),
    "沖縄": (26.2124, 127.6809),
    "南極・昭和基地付近": (-69.006, 39.590),
    "サハラ砂漠": (23.4162, 25.6628),
}

if preset == "自由入力":
    lat = st.sidebar.number_input("緯度", value=34.766, min_value=-90.0, max_value=90.0)
    lon = st.sidebar.number_input("経度", value=135.628, min_value=-180.0, max_value=180.0)
else:
    lat, lon = locations[preset]

st.sidebar.write(f"緯度: `{lat}`")
st.sidebar.write(f"経度: `{lon}`")

st.sidebar.header("期間")
end_day = date.today() - timedelta(days=10)
start_day = end_day - timedelta(days=365)

start_date = st.sidebar.date_input("開始日", start_day)
end_date = st.sidebar.date_input("終了日", end_day)

run_button = st.sidebar.button("NASAデータを取得して解析")

# ------------------------------------------------------------
# NASA POWER API function
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_nasa_power_data(latitude, longitude, start, end):
    """Fetch daily meteorological data from NASA POWER API."""

    parameters = [
        "T2M",        # Temperature at 2 meters
        "RH2M",       # Relative humidity at 2 meters
        "WS2M",       # Wind speed at 2 meters
        "PRECTOTCORR",# Precipitation
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
# Main analysis
# ------------------------------------------------------------

if run_button:
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

    st.success("NASAデータの取得に成功しました。")

    st.subheader("1. 取得したNASAデータ")
    st.dataframe(df.head(20), use_container_width=True)

    csv = df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        label="CSVをダウンロード",
        data=csv,
        file_name="nasa_power_data.csv",
        mime="text/csv"
    )

    st.subheader("2. 日射量の変化を見る")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df["solar_radiation_kWh_m2_day"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Solar radiation [kWh/m²/day]")
    ax1.set_title("Daily Solar Radiation from NASA POWER")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("3. AIで日射量を予測する")

    features = [
        "temperature_C",
        "humidity_percent",
        "wind_speed_m_s",
        "precipitation_mm_day"
    ]
    target = "solar_radiation_kWh_m2_day"

    X = df[features]
    y = df[target]

    if len(df) < 30:
        st.warning("AI予測には30日以上のデータを推奨します。")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
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

    col1, col2, col3 = st.columns(3)
    col1.metric("データ数", f"{len(df)} 日")
    col2.metric("平均誤差 MAE", f"{mae:.2f} kWh/m²/day")
    col3.metric("決定係数 R²", f"{r2:.2f}")

    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred
    }).sort_index()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.7)
    min_v = min(pred_df["actual"].min(), pred_df["predicted"].min())
    max_v = max(pred_df["actual"].max(), pred_df["predicted"].max())
    ax2.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax2.set_xlabel("Actual solar radiation")
    ax2.set_ylabel("Predicted solar radiation")
    ax2.set_title("AI Prediction Result")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("4. 何が予測に効いたか")

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.barh(importance["feature"], importance["importance"])
    ax3.set_xlabel("Importance")
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)

    st.subheader("5. ミッション判定")

    avg_solar = df[target].mean()

    if avg_solar >= 5.0:
        verdict = "🟢 太陽エネルギー条件は良好。太陽光発電ミッションに向いている。"
    elif avg_solar >= 3.0:
        verdict = "🟡 条件は中程度。蓄電池やバックアップ電源が重要。"
    else:
        verdict = "🔴 日射条件は厳しい。太陽光だけに頼るミッションは危険。"

    st.info(verdict)

    st.markdown(
        f"""
### 学びのポイント
- NASAの公開データは、宇宙・気象・エネルギーの研究に使える。
- AIは魔法ではなく、過去データのパターンから予測している。
- 場所によって、太陽エネルギーの使いやすさは大きく変わる。
- 宇宙探査でも地球環境でも、**データを読む力** が重要になる。

平均日射量： **{avg_solar:.2f} kWh/m²/day**
"""
    )

else:
    st.info("左のサイドバーで場所と期間を選び、解析ボタンを押してください。")

    st.markdown(
        """
### 体験シナリオ例
1. 大阪とサハラ砂漠を比べる  
2. 南極では太陽光発電が難しい時期があることを確認する  
3. AIがどの気象条件を見て日射量を予測しているか考える  
4. 「月面基地ならどんなデータが必要か？」を議論する  
"""
)
