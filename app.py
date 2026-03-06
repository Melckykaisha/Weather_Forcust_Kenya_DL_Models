"""
Kenya Local Weather Forecasting — Streamlit App
================================================
Victor Cheruiyot | Meru University of Science and Technology
BSc Data Science Research Project, 2026

Pages:
  1. 🗺️  Interactive Kenya Map
  2. 📍  County Forecast (7-day table + charts)
  3. 📊  Model Comparison (MAE, RMSE, R²)
  4. 📈  Predicted vs Actual

Run locally:
    pip install streamlit tensorflow scikit-learn folium streamlit-folium
               pandas numpy matplotlib plotly joblib
    streamlit run app.py

Deploy to Streamlit Cloud:
    1. Push this file + requirements.txt to GitHub
    2. Go to share.streamlit.io → Connect repo → Deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, joblib, warnings
from datetime import date, timedelta
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Weather Forecast",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .main { background: #0d1117; }

  .hero {
    background: linear-gradient(135deg, #0d3b2e 0%, #1a6b3c 50%, #0d3b2e 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    border: 1px solid #2ecc7133;
  }
  .hero h1 { color: #2ecc71; font-size: 2rem; margin: 0; font-weight: 700; }
  .hero p  { color: #a8d5b5; margin: 0.4rem 0 0; font-size: 0.95rem; }

  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-card .val  { font-size: 1.8rem; font-weight: 700; color: #2ecc71; }
  .metric-card .lbl  { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

  .forecast-row {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 0.8rem 1.2rem;
    margin: 0.4rem 0; display: flex; align-items: center;
    justify-content: space-between;
  }
  .day-badge {
    background: #1f2d3d; color: #58a6ff; font-weight: 600;
    padding: 4px 10px; border-radius: 6px; font-size: 0.85rem;
    min-width: 90px; text-align: center;
  }
  .weather-val { color: #e6edf3; font-size: 0.9rem; text-align: center; min-width: 80px; }
  .weather-lbl { color: #8b949e; font-size: 0.7rem; }

  .section-title {
    color: #e6edf3; font-size: 1.15rem; font-weight: 600;
    border-left: 3px solid #2ecc71; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
  }

  div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
  div[data-testid="stSidebar"] * { color: #e6edf3 !important; }

  .stSelectbox > div > div { background: #161b22 !important; color: #e6edf3 !important; }
  .stButton > button {
    background: #2ecc71; color: #0d1117; font-weight: 700;
    border: none; border-radius: 8px; padding: 0.5rem 1.5rem;
  }
  .stButton > button:hover { background: #27ae60; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update these paths to match your Google Drive mount or local folder
MODEL_DIR = "kenya_models"   # folder containing your saved models
DATA_PATH = "kenya_weather_data/kenya_all_counties.csv"

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model_and_meta():
    import tensorflow as tf
    with open(f"{MODEL_DIR}/model_metadata.json") as f:
        meta = json.load(f)
    # Try .keras format first (new), fall back to .h5 (legacy)
    keras_path = f"{MODEL_DIR}/best_model_{meta['best_model']}.keras"
    h5_path    = f"{MODEL_DIR}/best_model_{meta['best_model']}.h5"
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path)
    else:
        model = tf.keras.models.load_model(h5_path, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scalers = joblib.load(f"{MODEL_DIR}/scalers.pkl")
    return model, meta, scalers

@st.cache_data(ttl=86400)  # refreshes every 24 hours
def load_data():
    import requests, time
    from datetime import date, timedelta

    df        = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df        = df.sort_values(["county","date"]).reset_index(drop=True)
    last_date = df["date"].max().date()
    today     = date.today()
    gap_days  = (today - last_date).days

    if gap_days <= 1:
        return df   # already up to date

    gap_start     = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    gap_end       = today.strftime("%Y-%m-%d")
    county_coords = df.groupby("county")[["latitude","longitude"]].first().to_dict("index")
    new_rows      = []

    for county, coords in county_coords.items():
        try:
            params = {
                "latitude": coords["latitude"], "longitude": coords["longitude"],
                "start_date": gap_start, "end_date": gap_end,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                         "precipitation_sum,windspeed_10m_max,shortwave_radiation_sum,"
                         "et0_fao_evapotranspiration",
                "hourly": "relativehumidity_2m,surface_pressure",
                "timezone": "Africa/Nairobi",
            }
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                params=params, timeout=30)
            if resp.status_code != 200:
                continue
            data     = resp.json()
            daily_df = pd.DataFrame(data["daily"])
            daily_df.rename(columns={"time":"date"}, inplace=True)
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            hourly_df = pd.DataFrame(data["hourly"])
            hourly_df["time"] = pd.to_datetime(hourly_df["time"])
            hourly_df["date"] = hourly_df["time"].dt.normalize()
            hourly_daily = hourly_df.groupby("date")[
                ["relativehumidity_2m","surface_pressure"]].mean().reset_index()
            merged = pd.merge(daily_df, hourly_daily, on="date", how="left")
            merged.rename(columns={
                "temperature_2m_max":"temp_max_c","temperature_2m_min":"temp_min_c",
                "temperature_2m_mean":"temp_mean_c","precipitation_sum":"rainfall_mm",
                "windspeed_10m_max":"windspeed_kmh","shortwave_radiation_sum":"solar_radiation_mjm2",
                "et0_fao_evapotranspiration":"evapotranspiration_mm",
                "relativehumidity_2m":"humidity_pct","surface_pressure":"pressure_hpa",
            }, inplace=True)
            merged["county"]    = county
            merged["latitude"]  = coords["latitude"]
            merged["longitude"] = coords["longitude"]
            new_rows.append(merged)
            time.sleep(2)
        except Exception:
            continue

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        df     = pd.concat([df, new_df], ignore_index=True)
        df.drop_duplicates(subset=["county","date"], keep="last", inplace=True)
        df     = df.sort_values(["county","date"]).reset_index(drop=True)
        df.to_csv(DATA_PATH, index=False)  # saves so next startup is instant

    return df

@st.cache_data
def load_comparison():
    return pd.read_csv(f"{MODEL_DIR}/model_comparison.csv")


# ── Feature engineering (must match training) ─────────────────────────────────
def engineer_features(cdf):
    cdf = cdf.copy()
    TARGETS = ["temp_mean_c","rainfall_mm","humidity_pct","windspeed_kmh","pressure_hpa"]
    for col in TARGETS:
        if col in cdf.columns:
            cdf[col] = cdf[col].interpolate().ffill().bfill()
    cdf["day_of_year"] = cdf["date"].dt.dayofyear
    cdf["month"]       = cdf["date"].dt.month
    cdf["week"]        = cdf["date"].dt.isocalendar().week.astype(int)
    cdf["sin_doy"]     = np.sin(2 * np.pi * cdf["day_of_year"] / 365)
    cdf["cos_doy"]     = np.cos(2 * np.pi * cdf["day_of_year"] / 365)
    cdf["sin_month"]   = np.sin(2 * np.pi * cdf["month"] / 12)
    cdf["cos_month"]   = np.cos(2 * np.pi * cdf["month"] / 12)
    for col in TARGETS:
        if col in cdf.columns:
            for lag in [1,3,7]:
                cdf[f"{col}_lag{lag}"] = cdf[col].shift(lag)
            for win in [7,14]:
                cdf[f"{col}_roll{win}mean"] = cdf[col].rolling(win, min_periods=1).mean()
    cdf.dropna(inplace=True)
    return cdf


def predict_county(county, df, model, meta, scalers):
    FEATURE_COLS  = meta["feature_cols"]
    TARGET_COLS   = meta["target_cols"]
    LOOKBACK      = meta["lookback"]
    FORECAST_DAYS = meta["forecast_days"]

    cdf  = df[df["county"] == county].copy().reset_index(drop=True)
    cdf  = engineer_features(cdf)
    for c in FEATURE_COLS:
        if c not in cdf.columns:
            cdf[c] = 0.0

    scaler     = scalers[county]
    scaled     = scaler.transform(cdf[FEATURE_COLS])
    X_input    = scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURE_COLS))
    pred_norm  = model.predict(X_input, verbose=0)[0]

    target_idx = [FEATURE_COLS.index(t) for t in TARGET_COLS]
    dummy      = np.zeros((FORECAST_DAYS, len(FEATURE_COLS)))
    dummy[:, target_idx] = pred_norm
    inv        = scaler.inverse_transform(dummy)
    pred_real  = inv[:, target_idx]

    last_date = cdf["date"].max()
    dates     = pd.date_range(last_date + timedelta(days=1), periods=FORECAST_DAYS)
    fc = pd.DataFrame(pred_real, columns=TARGET_COLS)
    fc.insert(0, "date", dates)
    fc["temp_mean_c"]   = fc["temp_mean_c"].clip(5, 45)
    fc["rainfall_mm"]   = fc["rainfall_mm"].clip(0, 200)
    fc["humidity_pct"]  = fc["humidity_pct"].clip(0, 100)
    fc["windspeed_kmh"] = fc["windspeed_kmh"].clip(0, 120)
    fc["pressure_hpa"]  = fc["pressure_hpa"].clip(850, 1050)
    return fc


def rain_emoji(mm):
    if mm < 1:    return "☀️"
    elif mm < 5:  return "🌤️"
    elif mm < 15: return "🌧️"
    else:         return "⛈️"


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌦️ Kenya Weather")
    st.markdown("*Deep Learning Forecasting System*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🗺️  Kenya Map",
        "📍  County Forecast",
        "📊  Model Comparison",
        "📈  Predicted vs Actual",
    ])
    st.markdown("---")
    st.markdown("**Model:** GRU Neural Network")
    st.markdown("**Coverage:** 47 Counties")
    st.markdown("**Horizon:** 7 Days")
    st.markdown("**Variables:** Temp · Rain · Humidity · Wind · Pressure")
    st.markdown("---")
    st.caption("Meru University of Science & Technology  \nBSc Data Science, 2026")


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    model, meta, scalers = load_model_and_meta()
    df = load_data()
    comparison_df = load_comparison()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Could not load model/data: {e}")
    st.info("Make sure `kenya_models/` and `kenya_weather_data/` folders are in the same directory as app.py")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — KENYA MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️  Kenya Map":
    st.markdown("""
    <div class="hero">
      <h1>🌦️ Kenya Local Weather Forecasting</h1>
      <p>Deep Learning (GRU) · All 47 Counties · 7-Day Predictions · BSc Data Science Research</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Interactive County Map — Click any marker for forecast</div>',
                unsafe_allow_html=True)

    import folium
    from streamlit_folium import st_folium

    @st.cache_data
    def build_all_forecasts():
        forecasts = {}
        for county in df["county"].unique():
            try:
                forecasts[county] = predict_county(county, df, model, meta, scalers)
            except:
                pass
        return forecasts

    with st.spinner("🔮 Generating forecasts for all 47 counties..."):
        all_forecasts = build_all_forecasts()

    county_coords = df.groupby("county")[["latitude","longitude"]].first().to_dict("index")

    def temp_color(t):
        if t < 15:   return "#4fc3f7"
        elif t < 20: return "#81c784"
        elif t < 25: return "#aed581"
        elif t < 30: return "#ffb74d"
        else:        return "#ef5350"

    kenya_map = folium.Map(location=[0.023, 37.906], zoom_start=6,
                           tiles="CartoDB dark_matter")

    for county, fc in all_forecasts.items():
        coords   = county_coords.get(county, {})
        lat, lon = coords.get("latitude", 0), coords.get("longitude", 0)
        avg_temp = fc["temp_mean_c"].mean()
        color    = temp_color(avg_temp)

        rows = "".join([
            f"""<tr style='border-bottom:1px solid #333;'>
              <td style='padding:5px 8px;color:#aaa;'><b>{r['date'].strftime('%a %d %b')}</b></td>
              <td style='padding:5px 8px;'>{rain_emoji(r['rainfall_mm'])} {r['temp_mean_c']:.1f}°C</td>
              <td style='padding:5px 8px;color:#64b5f6;'>💧 {r['rainfall_mm']:.1f}mm</td>
              <td style='padding:5px 8px;color:#80cbc4;'>💦 {r['humidity_pct']:.0f}%</td>
              <td style='padding:5px 8px;color:#ffcc02;'>💨 {r['windspeed_kmh']:.1f}km/h</td>
            </tr>"""
            for _, r in fc.iterrows()
        ])

        popup_html = f"""
        <div style='font-family:Arial;background:#1a1a2e;color:#e0e0e0;width:380px;border-radius:10px;overflow:hidden;'>
          <div style='background:linear-gradient(135deg,#1a6b3c,#2ecc71);padding:10px 14px;'>
            <b style='font-size:15px;'>📍 {county} County</b><br>
            <span style='font-size:11px;opacity:0.85;'>7-Day Forecast · Avg {avg_temp:.1f}°C · Rain {fc['rainfall_mm'].sum():.0f}mm total</span>
          </div>
          <table style='width:100%;border-collapse:collapse;font-size:12px;'>
            <tr style='background:#0d2a1e;color:#2ecc71;font-size:11px;'>
              <th style='padding:5px 8px;'>Day</th><th>Temp</th>
              <th>Rain</th><th>Humidity</th><th>Wind</th>
            </tr>
            {rows}
          </table>
          <p style='font-size:10px;color:#555;margin:5px 8px;'>GRU Model · Victor Cheruiyot · MUST 2026</p>
        </div>"""

        folium.CircleMarker(
            location=[lat, lon], radius=13,
            color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=folium.Tooltip(f"<b>{county}</b> · {avg_temp:.1f}°C", sticky=True),
            popup=folium.Popup(folium.IFrame(popup_html, width=400, height=250), max_width=410)
        ).add_to(kenya_map)

    map_data = st_folium(kenya_map, width="100%", height=600)

    # Legend
    cols = st.columns(5)
    legend = [("< 15°C","#4fc3f7","Cool"), ("15–20°C","#81c784","Mild"),
              ("20–25°C","#aed581","Warm"), ("25–30°C","#ffb74d","Hot"), ("> 30°C","#ef5350","Very Hot")]
    for col, (rng, clr, lbl) in zip(cols, legend):
        col.markdown(f"""
        <div style='text-align:center;background:#161b22;border-radius:8px;padding:8px;border:1px solid #30363d;'>
          <div style='width:20px;height:20px;border-radius:50%;background:{clr};margin:0 auto 4px;'></div>
          <div style='font-size:11px;color:#e6edf3;font-weight:600;'>{rng}</div>
          <div style='font-size:10px;color:#8b949e;'>{lbl}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COUNTY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📍  County Forecast":
    st.markdown("""
    <div class="hero">
      <h1>📍 County 7-Day Forecast</h1>
      <p>Select any of Kenya's 47 counties to view the detailed weather forecast</p>
    </div>""", unsafe_allow_html=True)

    counties = sorted(df["county"].unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Select County", counties, index=counties.index("Nairobi"))

    with st.spinner(f"🔮 Predicting weather for {selected}..."):
        fc = predict_county(selected, df, model, meta, scalers)

    # Summary metrics
    st.markdown('<div class="section-title">7-Day Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, f"{fc['temp_mean_c'].mean():.1f}°C",  "Avg Temperature"),
        (m2, f"{fc['rainfall_mm'].sum():.1f}mm",    "Total Rainfall"),
        (m3, f"{fc['humidity_pct'].mean():.0f}%",   "Avg Humidity"),
        (m4, f"{fc['windspeed_kmh'].mean():.1f}km/h","Avg Wind Speed"),
        (m5, f"{fc['pressure_hpa'].mean():.0f}hPa", "Avg Pressure"),
    ]
    for col, val, lbl in metrics:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Daily forecast cards
    st.markdown('<div class="section-title">Daily Breakdown</div>', unsafe_allow_html=True)
    for _, row in fc.iterrows():
        emoji = rain_emoji(row["rainfall_mm"])
        st.markdown(f"""
        <div class="forecast-row">
          <span class="day-badge">{row['date'].strftime('%a %d %b')}</span>
          <div class="weather-val">{emoji} {row['temp_mean_c']:.1f}°C<br><span class="weather-lbl">Temperature</span></div>
          <div class="weather-val">💧 {row['rainfall_mm']:.1f}mm<br><span class="weather-lbl">Rainfall</span></div>
          <div class="weather-val">💦 {row['humidity_pct']:.0f}%<br><span class="weather-lbl">Humidity</span></div>
          <div class="weather-val">💨 {row['windspeed_kmh']:.1f}km/h<br><span class="weather-lbl">Wind</span></div>
          <div class="weather-val">🔵 {row['pressure_hpa']:.0f}hPa<br><span class="weather-lbl">Pressure</span></div>
        </div>""", unsafe_allow_html=True)

    # Line charts
    st.markdown('<div class="section-title">7-Day Trend Charts</div>', unsafe_allow_html=True)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=["Temperature (°C)","Rainfall (mm)","Humidity (%)",
                                        "Wind Speed (km/h)","Pressure (hPa)",""],
                        vertical_spacing=0.15)
    plot_data = [
        (fc["temp_mean_c"],   "#ef5350", 1, 1),
        (fc["rainfall_mm"],   "#42a5f5", 1, 2),
        (fc["humidity_pct"],  "#26c6da", 1, 3),
        (fc["windspeed_kmh"], "#ffa726", 2, 1),
        (fc["pressure_hpa"],  "#ab47bc", 2, 2),
    ]
    days = [d.strftime("%a %d") for d in fc["date"]]
    for vals, color, row, col in plot_data:
        fig.add_trace(go.Scatter(
            x=days, y=vals, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            showlegend=False
        ), row=row, col=col)

    fig.update_layout(
        height=420, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", size=11),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    fig.update_xaxes(gridcolor="#30363d", tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#30363d")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Comparison":
    st.markdown("""
    <div class="hero">
      <h1>📊 Model Performance Comparison</h1>
      <p>LSTM vs GRU vs ConvLSTM — Evaluated on MAE, RMSE, MAPE and R²</p>
    </div>""", unsafe_allow_html=True)

    import plotly.graph_objects as go

    # Overall metrics
    st.markdown('<div class="section-title">Overall Performance</div>', unsafe_allow_html=True)
    models = comparison_df["Model"].tolist()
    colors = {"LSTM": "#42a5f5", "GRU": "#2ecc71", "ConvLSTM": "#ffa726"}

    col1, col2, col3 = st.columns(3)
    for col, metric, title, lower_better in [
        (col1, "Overall_MAE",  "Mean Absolute Error (MAE)",  True),
        (col2, "Overall_RMSE", "Root Mean Square Error (RMSE)", True),
        (col3, "Overall_R2",   "R² Score", False),
    ]:
        vals    = comparison_df[metric].tolist()
        best_i  = vals.index(min(vals)) if lower_better else vals.index(max(vals))
        bar_colors = [colors.get(m, "#666") for m in models]
        fig = go.Figure(go.Bar(
            x=models, y=vals,
            marker_color=bar_colors,
            marker_line_color=["gold" if i == best_i else "rgba(0,0,0,0)" for i in range(len(models))],
            marker_line_width=3,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11)
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#e6edf3", size=13)),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            height=280, margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(gridcolor="#30363d"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        col.plotly_chart(fig, use_container_width=True)

    # Per-variable breakdown
    st.markdown('<div class="section-title">Per-Variable MAE Breakdown</div>',
                unsafe_allow_html=True)
    TARGET_COLS = meta["target_cols"]
    var_labels  = {"temp_mean_c":"Temperature","rainfall_mm":"Rainfall",
                   "humidity_pct":"Humidity","windspeed_kmh":"Wind Speed","pressure_hpa":"Pressure"}
    fig2 = go.Figure()
    for m in models:
        row  = comparison_df[comparison_df["Model"] == m].iloc[0]
        maes = [row.get(f"{v}_MAE", 0) for v in TARGET_COLS]
        fig2.add_trace(go.Bar(
            name=m, x=[var_labels.get(v, v) for v in TARGET_COLS],
            y=maes, marker_color=colors.get(m, "#666"),
            text=[f"{v:.4f}" for v in maes], textposition="outside",
            textfont=dict(size=9)
        ))
    fig2.update_layout(
        barmode="group", height=380,
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
        yaxis=dict(gridcolor="#30363d", title="MAE"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw metrics table
    st.markdown('<div class="section-title">Full Metrics Table</div>', unsafe_allow_html=True)
    display_cols = ["Model","Overall_MAE","Overall_RMSE","Overall_R2"] + \
                   [f"{v}_MAE" for v in TARGET_COLS]
    styled = comparison_df[display_cols].copy()
    styled.columns = ["Model","MAE","RMSE","R²"] + \
                     [f"{var_labels.get(v,v)} MAE" for v in TARGET_COLS]
    st.dataframe(styled.style.highlight_min(subset=["MAE","RMSE"], color="#1a3d2b")
                              .highlight_max(subset=["R²"], color="#1a3d2b")
                              .format({c: "{:.4f}" for c in styled.columns if c != "Model"}),
                 use_container_width=True)

    # Winner callout
    best = comparison_df.loc[comparison_df["Overall_RMSE"].idxmin(), "Model"]
    best_r2 = comparison_df.loc[comparison_df["Overall_RMSE"].idxmin(), "Overall_R2"]
    st.success(f"🏆 **Best Model: {best}** — Lowest RMSE with R² = {best_r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTED VS ACTUAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Predicted vs Actual":
    st.markdown("""
    <div class="hero">
      <h1>📈 Predicted vs Actual</h1>
      <p>Validate model accuracy by comparing predictions against real historical observations</p>
    </div>""", unsafe_allow_html=True)

    import plotly.graph_objects as go

    counties  = sorted(df["county"].unique().tolist())
    col1, col2, col3 = st.columns(3)
    county_sel = col1.selectbox("County", counties, index=counties.index("Nairobi"))
    var_map    = {"Temperature":"temp_mean_c","Rainfall":"rainfall_mm",
                  "Humidity":"humidity_pct","Wind Speed":"windspeed_kmh","Pressure":"pressure_hpa"}
    var_sel    = col2.selectbox("Variable", list(var_map.keys()))
    n_points   = col3.slider("Points to show", 30, 200, 90)

    var_col = var_map[var_sel]

    with st.spinner("Loading validation data..."):
        cdf  = df[df["county"] == county_sel].copy().reset_index(drop=True)
        cdf  = engineer_features(cdf)

        FEATURE_COLS  = meta["feature_cols"]
        TARGET_COLS   = meta["target_cols"]
        LOOKBACK      = meta["lookback"]

        for c in FEATURE_COLS:
            if c not in cdf.columns:
                cdf[c] = 0.0

        scaler    = scalers[county_sel]
        scaled    = scaler.transform(cdf[FEATURE_COLS])
        target_idx = [FEATURE_COLS.index(t) for t in TARGET_COLS]
        var_i      = TARGET_COLS.index(var_col)

        # Build test sequences (last 20% of data)
        n         = len(scaled)
        test_start = int(n * 0.85)
        actuals, preds = [], []

        for i in range(test_start, min(test_start + n_points, n - 1)):
            X_in   = scaled[i - LOOKBACK : i].reshape(1, LOOKBACK, len(FEATURE_COLS))
            p_norm = model.predict(X_in, verbose=0)[0]   # (7, 5)
            dummy  = np.zeros((7, len(FEATURE_COLS)))
            dummy[:, target_idx] = p_norm
            inv    = scaler.inverse_transform(dummy)
            preds.append(inv[0, target_idx[var_i]])      # day-1 prediction
            actuals.append(cdf.iloc[i][var_col])

        dates_plot = cdf.iloc[test_start : test_start + len(actuals)]["date"].tolist()

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)

    m1, m2, m3 = st.columns(3)
    for col, val, lbl in [(m1,f"{mae:.4f}","MAE"),(m2,f"{rmse:.4f}","RMSE"),(m3,f"{r2:.4f}","R²")]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{county_sel} · {var_sel} · {lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Predicted vs Actual Over Time</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_plot, y=actuals, name="Actual",
        line=dict(color="#42a5f5", width=2), mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=dates_plot, y=preds, name="Predicted",
        line=dict(color="#2ecc71", width=2, dash="dash"), mode="lines"
    ))
    fig.update_layout(
        height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", title=f"{var_sel}"),
        margin=dict(t=20, b=20, l=20, r=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.markdown('<div class="section-title">Scatter: Actual vs Predicted</div>',
                unsafe_allow_html=True)
    mn, mx = min(actuals + preds), max(actuals + preds)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=actuals, y=preds, mode="markers",
        marker=dict(color="#2ecc71", size=6, opacity=0.6),
        name="Samples"
    ))
    fig2.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color="#ef5350", dash="dash", width=1.5),
        name="Perfect fit"
    ))
    fig2.update_layout(
        height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#30363d", title="Actual"),
        yaxis=dict(gridcolor="#30363d", title="Predicted"),
        legend=dict(bgcolor="#0d1117"),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Points close to the red dashed line = accurate predictions · R² = {r2:.4f}")
