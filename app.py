import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Solar & Wind Energy Predictor",
    page_icon="⚡",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 2rem 2rem 1.5rem;
    text-align: center; margin-bottom: 1.5rem;
}
.hero h1 { color: #fff; font-size: 2rem; margin: 0 0 0.3rem; }
.hero p  { color: #a0aec0; font-size: 0.95rem; margin: 0; }

.result-card {
    border-radius: 14px; padding: 1.8rem;
    text-align: center; margin-top: 1rem;
}
.result-solar { background: linear-gradient(135deg, #f6d365, #fda085); }
.result-wind  { background: linear-gradient(135deg, #89f7fe, #66a6ff); }
.result-card .value { font-size: 3rem; font-weight: 700; color: #1a1a2e; }
.result-card .label { font-size: 1rem; color: #2d3748; margin-top: 0.3rem; }
.result-card .sub   { font-size: 0.85rem; color: #4a5568; margin-top: 0.5rem; }

.tip-box {
    background: #f0fff4; border-left: 4px solid #38a169;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.88rem; color: #276749; margin-top: 1rem;
}
.warn-box {
    background: #fffbeb; border-left: 4px solid #d69e2e;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.88rem; color: #7b5e00; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚡ Solar & Wind Energy Predictor</h1>
    <p>Germany Renewable Energy Model · Trained on 2015–2021 data · Random Forest + SVM</p>
</div>
""", unsafe_allow_html=True)

# ── Load models.pkl ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models.pkl")

@st.cache_resource(show_spinner="Loading prediction models…")
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ `models.pkl` not found in the app folder.")
    st.markdown("""
**To fix this:**
1. Run `train_and_save.py` at the end of your Colab notebook
2. Download the generated `models.pkl`
3. Place it in the same folder as `app.py`
4. Push to GitHub and redeploy
""")
    st.stop()

bundle = load_bundle(MODEL_PATH)
solar_model   = bundle["solar_model"]
solar_scaler  = bundle["solar_scaler"]
wind_model    = bundle["wind_model"]
wind_scaler   = bundle["wind_scaler"]
stats         = bundle["stats"]

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_solar, tab_wind = st.tabs(["☀️ Solar Energy", "💨 Wind Energy"])


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLAR TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_solar:
    st.markdown("### ☀️ Predict Solar Generation")
    st.caption("Enter the solar irradiance profile value (0 = no sun, 1 = maximum capacity factor)")

    sp = stats["solar_profile"]

    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            solar_profile = st.slider(
                "Solar Profile (Capacity Factor)",
                min_value=0.0,
                max_value=round(sp["max"], 3),
                value=round(sp["mean"], 3),
                step=0.001,
                help="Ratio of actual solar irradiance to maximum possible. 0 = night/cloudy, ~0.7–0.9 = bright noon."
            )
        with col2:
            st.metric("You entered", f"{solar_profile:.3f}")
            level = "🌑 Night" if solar_profile < 0.05 else \
                    "🌥️ Cloudy" if solar_profile < 0.2 else \
                    "⛅ Partly cloudy" if solar_profile < 0.4 else \
                    "🌤️ Mostly sunny" if solar_profile < 0.65 else "☀️ Bright sun"
            st.caption(level)

    if st.button("🔮 Predict Solar Output", use_container_width=True, type="primary"):
        X = np.array([[solar_profile]])
        X_scaled = solar_scaler.transform(X)
        pred_mw  = float(solar_model.predict(X_scaled)[0])
        pred_mw  = max(0.0, pred_mw)   # can't be negative

        pct = (pred_mw / stats["solar_generation_max"]) * 100

        st.markdown(f"""
        <div class="result-card result-solar">
            <div class="value">{pred_mw:,.0f} MW</div>
            <div class="label">Predicted Solar Generation</div>
            <div class="sub">{pct:.1f}% of historical peak capacity ({stats['solar_generation_max']:,.0f} MW)</div>
        </div>
        """, unsafe_allow_html=True)

        avg = stats["solar_generation_mean"]
        if pred_mw > avg * 1.3:
            st.markdown('<div class="tip-box">✅ <b>Above average</b> solar output — excellent conditions for solar harvesting.</div>', unsafe_allow_html=True)
        elif pred_mw < avg * 0.3:
            st.markdown('<div class="warn-box">⚠️ <b>Low output</b> — cloudy or night-time conditions. Storage/backup may be needed.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-box">📊 <b>Average</b> solar output expected for these conditions.</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("ℹ️ What is Solar Profile?"):
        st.markdown("""
- **Solar Profile** = capacity factor = actual output ÷ maximum possible output
- Ranges from **0** (night or fully overcast) to ~**0.85** (peak summer noon)
- Germany average during daylight hours: ~**0.10 – 0.35**
- Source: [Open Power System Data](https://open-power-system-data.org/)
- Model: **Random Forest** (best performer for solar in this dataset)
        """)


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wind:
    st.markdown("### 💨 Predict Wind Generation")
    st.caption("Enter the wind capacity factor profiles for onshore and offshore wind farms")

    wp   = stats["wind_profile"]
    won  = stats["wind_onshore_profile"]
    woff = stats["wind_offshore_profile"]

    col1, col2 = st.columns(2)
    with col1:
        wind_profile = st.slider(
            "Overall Wind Profile",
            min_value=0.0, max_value=round(wp["max"], 3),
            value=round(wp["mean"], 3), step=0.001,
            help="Overall wind capacity factor across Germany"
        )
        wind_onshore = st.slider(
            "Onshore Wind Profile",
            min_value=0.0, max_value=round(won["max"], 3),
            value=round(won["mean"], 3), step=0.001,
            help="Capacity factor for onshore wind farms"
        )
    with col2:
        wind_offshore = st.slider(
            "Offshore Wind Profile",
            min_value=0.0, max_value=round(woff["max"], 3),
            value=round(woff["mean"], 3), step=0.001,
            help="Capacity factor for offshore wind farms (North/Baltic Sea)"
        )

        # Wind strength indicator
        avg_wind = (wind_onshore + wind_offshore) / 2
        strength = "🌬️ Calm"       if avg_wind < 0.1  else \
                   "💨 Light"      if avg_wind < 0.25 else \
                   "🌀 Moderate"   if avg_wind < 0.45 else \
                   "⛈️ Strong"     if avg_wind < 0.65 else \
                   "🌪️ Very strong"
        st.metric("Wind Strength", strength)
        st.metric("Avg Capacity", f"{avg_wind:.3f}")

    if st.button("🔮 Predict Wind Output", use_container_width=True, type="primary"):
        # Use only the features the model was trained on
        wind_feats = bundle["wind_features"]
        feat_map   = {
            "DE_wind_profile":           wind_profile,
            "DE_wind_onshore_profile":   wind_onshore,
            "DE_wind_offshore_profile":  wind_offshore,
        }
        X       = np.array([[feat_map[f] for f in wind_feats]])
        X_scaled = wind_scaler.transform(X)
        pred_mw  = float(wind_model.predict(X_scaled)[0])
        pred_mw  = max(0.0, pred_mw)

        pct = (pred_mw / stats["wind_generation_max"]) * 100

        st.markdown(f"""
        <div class="result-card result-wind">
            <div class="value">{pred_mw:,.0f} MW</div>
            <div class="label">Predicted Wind Generation</div>
            <div class="sub">{pct:.1f}% of historical peak capacity ({stats['wind_generation_max']:,.0f} MW)</div>
        </div>
        """, unsafe_allow_html=True)

        avg = stats["wind_generation_mean"]
        if pred_mw > avg * 1.3:
            st.markdown('<div class="tip-box">✅ <b>High wind output</b> — excellent generation conditions.</div>', unsafe_allow_html=True)
        elif pred_mw < avg * 0.4:
            st.markdown('<div class="warn-box">⚠️ <b>Low wind output</b> — calm conditions. Grid may need backup sources.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-box">📊 <b>Average wind output</b> expected for these conditions.</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("ℹ️ What are Wind Profiles?"):
        st.markdown("""
- **Wind Profile** = capacity factor = actual output ÷ installed capacity
- **Onshore**: land-based turbines (more numerous but lower capacity factor ~0.25–0.35)
- **Offshore**: sea-based turbines (fewer but higher capacity factor ~0.35–0.55, steadier wind)
- Germany has ~60 GW onshore + ~8 GW offshore installed capacity
- Model: **SVM with RBF kernel** (best performer for wind in this dataset)
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · Models: Random Forest (Solar) & SVM (Wind) · Data: Open Power System Data · Germany 2015–2021")
