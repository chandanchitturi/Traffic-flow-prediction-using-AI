import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide"
)

# =========================
# HELPER: Base64 embed for map
# =========================
def get_map_base64(img_path="map_base.png"):
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# =========================
# STYLE
# =========================
st.markdown("""
<style>

/* global page bg */
body {
    background-color: #0f172a;
}

/* Streamlit container cleanup */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    background: #0f172a !important;
    max-width: 1400px !important;
}

/* kill default borders/padding around blocks */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: transparent !important;
    box-shadow: none !important;
    border: 0 !important;
    padding: 0 !important;
}

/* =========================
   HEADER FIX
   ========================= */

/* force header full width and not clipped */
.header-outer {
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    background: #0f172a;
    padding-top: 1rem;
    padding-bottom: 1.5rem;
    display: flex;
    justify-content: center;
    margin-top: 40px;
}

.header-inner {
    width: 100%;
    max-width: 1400px;
    box-sizing: border-box;

    background: linear-gradient(90deg, #111827 0%, #1f2937 100%);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 16px 40px rgba(0,0,0,0.6);
    color: #fff;
}

.header-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.header-subtitle {
    font-size: 0.9rem;
    color: #9ca3af;
    line-height: 1.4;
    margin-top: 0.4rem;
}

/* section headers like "Control Panel", "Junction Map" */
.section-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* =========================
   GLASS PANEL + INPUTS
   ========================= */

/* glassy card around all controls on left */
.glass-panel {
    background: rgba(31,41,55,0.35);
    border-radius: 16px;
    padding: 16px 20px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 30px 80px rgba(0,0,0,0.8);
    color: #fff;
    width: 100%;
    margin-bottom: 16px;
    backdrop-filter: blur(12px) saturate(140%);
    -webkit-backdrop-filter: blur(12px) saturate(140%);
}

/* label text for widgets */
label {
    color: #fff !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    margin-bottom: 0.25rem !important;
}

/* caption like "Weekend: No" */
small, .stCaption, .stMarkdown p, .st-emotion-cache-12w0qpk {
    color: #9ca3af !important;
    font-size: 0.8rem !important;
}

/* ---- FORCE glass style for individual Streamlit widgets ---- */

/* selectbox wrapper */
[data-testid="stSelectbox"] > div {
    background: rgba(15,23,42,0.45) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.8) !important;
    backdrop-filter: blur(10px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(140%) !important;
}
[data-testid="stSelectbox"] * {
    color: #fff !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] svg {
    color: #fff !important;
    fill: #fff !important;
}

/* number input wrapper */
[data-testid="stNumberInput"] > div > div {
    background: rgba(15,23,42,0.45) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.8) !important;
    backdrop-filter: blur(10px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(140%) !important;
}
[data-testid="stNumberInput"] input {
    color: #fff !important;
    font-size: 0.9rem !important;
}

/* slider track / handle */
[data-testid="stSlider"] div[data-baseweb="slider"] {
    /* the whole slider block container */
    background: rgba(15,23,42,0.45) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.8) !important;
    backdrop-filter: blur(10px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(140%) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #ef4444 !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 10px rgba(239,68,68,0.8) !important;
}
[data-testid="stSlider"] .st-af {
    background-color: rgba(255,255,255,0.2) !important;
}
[data-testid="stSlider"] .st-ah {
    background: linear-gradient(90deg,#ef4444,#ef4444) !important;
    height: 4px !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] * {
    color: #fff !important;
}

/* Predict button */
.stButton > button {
    background: rgba(15,23,42,0.6) !important;
    color: #fff !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.8) !important;
    backdrop-filter: blur(10px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(140%) !important;
    font-size: 0.9rem !important;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(31,41,55,0.8) !important;
}

/* =========================
   MAP AREA (already blended)
   ========================= */

.map-shell {
    position: relative;
    width: 100%;
    max-width: 520px;
    margin-top: 8px;

    /* subtle spotlight behind it */
    background: radial-gradient(
        circle at 40% 30%,
        rgba(255,255,200,0.10) 0%,
        rgba(15,23,42,0.0) 70%
    );
    border: none !important;
    box-shadow: none !important;
    border-radius: 0;
    padding: 0;
}

.map-wrapper {
    position: relative;
    width: 100%;
    border-radius: 0;
    overflow: visible;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0;
    margin: 0;
}

.map-img {
    width: 100%;
    display: block;
    border-radius: 0 !important;
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;

    filter: brightness(1.28) saturate(1.15) contrast(1.05);
    mix-blend-mode: screen;
    opacity: 0.9;

    -webkit-mask-image: radial-gradient(
        circle at 50% 45%,
        rgba(0,0,0,1) 0%,
        rgba(0,0,0,1) 60%,
        rgba(0,0,0,0) 90%
    );
    mask-image: radial-gradient(
        circle at 50% 45%,
        rgba(0,0,0,1) 0%,
        rgba(0,0,0,1) 60%,
        rgba(0,0,0,0) 90%
    );
}

/* Junction beacons */
.beacon {
    position: absolute;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(255,255,200,0.4) 0%,
        rgba(255,255,0,0.0) 70%
    );
    box-shadow:
        0 0 16px rgba(255,255,120,0.3),
        0 0 40px rgba(255,255,120,0.15);
    opacity: 0.25;
    transform: translate(-50%, -50%) scale(1);
    transition: all 0.22s ease-in-out;
    pointer-events: none;
}
.beacon.active {
    opacity: 1;
    background: radial-gradient(
        circle,
        rgba(255,255,220,0.95) 0%,
        rgba(255,255,0,0.0) 70%
    );
    box-shadow:
        0 0 30px rgba(255,255,170,1),
        0 0 80px rgba(255,255,120,0.7),
        0 0 160px rgba(255,255,80,0.4);
    transform: translate(-50%, -50%) scale(1.6);
}

/* =========================
   METRIC + PREDICTION CARDS
   ========================= */

.section-label-bottom {
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.metric-card {
    background: #111827;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #fff;
    box-shadow: 0 16px 40px rgba(0,0,0,0.7);
}
.metric-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: #fff;
}
.metric-sub {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 4px;
    line-height: 1.2;
}

.prediction-card {
    background: #111827;
    border-radius: 16px;
    padding: 16px 20px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #fff;
    box-shadow: 0 16px 40px rgba(0,0,0,0.7);
    width: 100%;
}
.prediction-value {
    font-size: 2rem;
    font-weight: 600;
    color: #fff;
}
.badge {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 8px;
    display: inline-block;
    margin-bottom: 10px;
}
.prediction-extra {
    background: #1f2937;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.07);
    color: #fff;
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
    font-size: 0.9rem;
    line-height: 1.5;
    margin-top: 16px;
}

.footer-note {
    font-size: 0.7rem;
    text-align: center;
    color: #6b7280;
    padding-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL STATS
# =========================
MODEL_STATS = {
    1: {"MAE": 16.05, "R2": 0.2476},
    2: {"MAE": 5.56,  "R2": 0.0136},
    3: {"MAE": 6.60,  "R2": -0.125},
    4: {"MAE": 2.09,  "R2": 0.3935},
}

# =========================
# HEADER SECTION
# =========================
st.markdown("""
<div class="header-outer">
    <div class="header-inner">
        <div class="header-title">🚦 Traffic Volume Predictor Dashboard</div>
        <div class="header-subtitle">
            Predict real-time hourly vehicle load at monitored city junctions using trained AI models.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# CONTROL PANEL + MAP ROW
# =========================
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="section-label">🧩 Control Panel</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    junction = st.selectbox("Select Junction for Prediction", [1, 2, 3, 4])
    day_name = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    hour = st.slider("Select Hour (0–23)", 0, 23, 17)
    day = st.number_input("Day of Month", 1, 31, 28)

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    dayofweek = day_map[day_name]
    is_weekend = 1 if dayofweek >= 5 else 0
    st.caption(f"Weekend: {'Yes' if is_weekend else 'No'}")

    run_prediction = st.button("🔮 Predict Traffic Volume", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-label">📍 Junction Map</div>', unsafe_allow_html=True)

    active_j = junction
    b1 = "beacon active" if active_j == 1 else "beacon"
    b2 = "beacon active" if active_j == 2 else "beacon"
    b3 = "beacon active" if active_j == 3 else "beacon"
    b4 = "beacon active" if active_j == 4 else "beacon"

    encoded_src = get_map_base64("map_base.png")

    if encoded_src is None:
        st.warning("⚠️ map_base.png not found in this folder. Put it next to traffic_app.py.")
    else:
        html_map = f"""
        <div class="map-shell">
            <div class="map-wrapper">
                <img class="map-img" src="{encoded_src}">
                <div class="{b1}" style="top:30%; left:32%;"></div>
                <div class="{b2}" style="top:32%; left:72%;"></div>
                <div class="{b3}" style="top:70%; left:28%;"></div>
                <div class="{b4}" style="top:75%; left:72%;"></div>
            </div>
        </div>
        """
        st.markdown(html_map, unsafe_allow_html=True)

# =========================
# PERFORMANCE + PREDICTION
# =========================
st.markdown(
    '<div class="section-label-bottom">📊 Model Performance & Forecast</div>',
    unsafe_allow_html=True
)

perf_col, pred_col = st.columns([1, 1])
stats = MODEL_STATS[junction]

with perf_col:
    mc1, mc2 = st.columns(2)

    mc1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Absolute Error (MAE)</div>
        <div class="metric-value">± {stats['MAE']:.2f} veh/hr</div>
        <div class="metric-sub">
            Average deviation: {stats['MAE']:.0f} vehicles/hour at Junction {junction}.
        </div>
    </div>
    """, unsafe_allow_html=True)

    stability = "More stable traffic." if stats["R2"] > 0.2 else "Noisy / unpredictable."
    mc2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{stats['R2']:.3f}</div>
        <div class="metric-sub">{stability}</div>
    </div>
    """, unsafe_allow_html=True)

with pred_col:
    if run_prediction:
        model = joblib.load(f"model_junction_{junction}.pkl")

        row = pd.DataFrame([{
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "day": day
        }])

        pred = model.predict(row)[0]

        if pred <= 10:
            level = "Low"
            color = "#10B981"
        elif pred <= 25:
            level = "Moderate"
            color = "#FACC15"
        else:
            level = "High"
            color = "#EF4444"

        full_day = pd.DataFrame({
            "hour": range(24),
            "dayofweek": [dayofweek]*24,
            "is_weekend": [is_weekend]*24,
            "day": [day]*24
        })
        full_day["Predicted"] = model.predict(full_day)
        busiest = full_day.loc[full_day["Predicted"].idxmax()]

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-value">{round(pred)} vehicles/hr</div>
            <span class="badge" style="background:{color};">{level} Traffic</span>
            <div style="font-size:0.9rem; line-height:1.4; color:#fff;">
                At <b>{hour}:00</b> on <b>{day_name}</b> (day {day}),
                Junction <b>{junction}</b> expects <b>{round(pred)} vehicles/hour</b>.
            </div>
            <div style="font-size:0.75rem; color:#9ca3af; margin-top:8px;">
                MAE ±{stats['MAE']:.1f} veh/hr • R² {stats['R2']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prediction-extra">
            <div style="font-weight:600; margin-bottom:6px;">🧠 Summary</div>
            • Busiest hour: {int(busiest['hour'])}:00<br>
            • Peak load: ~{int(busiest['Predicted'])} vehicles/hr<br>
            • Status: {level}<br>
            • Weekend: {"Yes" if is_weekend else "No"}
        </div>
        """, unsafe_allow_html=True)

        st.line_chart(
            full_day.set_index("hour")["Predicted"],
            height=200
        )
    else:
        st.info("Press **Predict Traffic Volume** to generate congestion level, summary, and 24-hour outlook.")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer-note">
Model: Random Forest Regressor trained on 48k+ hourly traffic records across 4 monitored junctions.
</div>
""", unsafe_allow_html=True)