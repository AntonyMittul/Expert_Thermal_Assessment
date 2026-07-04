# -*- coding: utf-8 -*-
"""
Streamlit demo for the heat sink ML surrogate assessment.

Run locally:   streamlit run app.py
The app trains the Random Forest surrogate once at startup (cached) from
data/heat_sink_dataset.csv, then compares physics vs surrogate live as the
user moves the input sliders.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from heat_sink_model import compute_thermals

FEATURES = ["TDP_W", "V_air_m_s", "k_tim_W_mK"]
TARGETS = ["R_total_C_W", "T_junction_C"]

st.set_page_config(page_title="Heat Sink ML Surrogate", page_icon="🌡️", layout="wide")


@st.cache_data
def load_dataset():
    return pd.read_csv(os.path.join("data", "heat_sink_dataset.csv"))


@st.cache_resource
def train_surrogate():
    df = load_dataset()
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(df[FEATURES], df[TARGETS])
    return model


df = load_dataset()
surrogate = train_surrogate()

st.title("Heat Sink Thermal Model — ML Surrogate Demo")
st.caption(
    "Physics-based thermal resistance network vs. a Random Forest surrogate "
    "trained on an 8,280-point parameter sweep. Expert Thermal / XThermal assessment, Task 1."
)

# ---- sidebar inputs ----
st.sidebar.header("Operating point")
tdp = st.sidebar.slider("TDP — Thermal Design Power (W)", 30, 250, 150, step=5)
v_air = st.sidebar.slider("Air velocity (m/s)", 0.5, 15.0, 1.0, step=0.5)
k_tim = st.sidebar.slider("TIM thermal conductivity (W/m·K)", 1.0, 12.0, 4.0, step=0.5)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fixed geometry:** 90×116×27 mm Al heat sink, 60 fins × 0.8 mm, "
    "52.5×45 mm die, 0.1 mm TIM layer, 25 °C ambient."
)

# ---- live comparison ----
physics = compute_thermals(TDP=tdp, V_air=v_air, k_tim=k_tim)
pred = surrogate.predict(pd.DataFrame([[tdp, v_air, k_tim]], columns=FEATURES))[0]

col_physics, col_ml = st.columns(2)
with col_physics:
    st.subheader("Physics model")
    st.metric("Total thermal resistance", f"{physics['R_total_C_W']:.4f} °C/W")
    st.metric("Junction temperature", f"{physics['T_junction_C']:.2f} °C")
with col_ml:
    st.subheader("ML surrogate (Random Forest)")
    st.metric(
        "Total thermal resistance",
        f"{pred[0]:.4f} °C/W",
        delta=f"{pred[0] - physics['R_total_C_W']:+.4f} vs physics",
        delta_color="off",
    )
    st.metric(
        "Junction temperature",
        f"{pred[1]:.2f} °C",
        delta=f"{pred[1] - physics['T_junction_C']:+.2f} °C vs physics",
        delta_color="off",
    )

# simple thermal verdict against a typical max junction temperature
T_J_MAX = 100.0
if physics["T_junction_C"] > T_J_MAX:
    st.error(
        f"Junction temperature exceeds the typical {T_J_MAX:.0f} °C limit — "
        "this design would throttle or fail at this operating point."
    )
else:
    margin = T_J_MAX - physics["T_junction_C"]
    st.success(f"Within thermal limits — {margin:.1f} °C of margin to {T_J_MAX:.0f} °C.")

with st.expander("Flow details at this operating point"):
    st.write(
        f"Fin spacing: {physics['fin_spacing_m']*1000:.2f} mm · "
        f"Reynolds number: {physics['Re']:.0f} "
        f"({'laminar' if physics['Re'] < 2300 else 'turbulent'}) · "
        f"Nusselt number: {physics['Nu']:.2f} · "
        f"h = {physics['h_W_m2K']:.1f} W/m²K"
    )

# ---- tabs ----
tab_perf, tab_sens, tab_data = st.tabs(
    ["Model performance", "Sensitivity analysis", "Dataset"]
)

with tab_perf:
    st.subheader("Surrogate accuracy (held-out test set)")
    metrics = pd.read_csv(os.path.join("results", "model_metrics.csv"))
    st.dataframe(metrics.round(4), use_container_width=True)
    st.markdown(
        "Linear Regression underfits total resistance (R² ≈ 0.71) because the "
        "physics is nonlinear (h ∝ V^⅓, R_TIM ∝ 1/k). The Random Forest captures "
        "both targets to better than 0.1 °C mean error."
    )
    st.image(os.path.join("results", "predicted_vs_actual.png"))

with tab_sens:
    st.subheader("What drives the outputs?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Spearman correlation (inputs vs outputs)**")
        spearman = pd.read_csv(
            os.path.join("results", "correlation_spearman.csv"), index_col=0
        )
        st.dataframe(spearman.round(3), use_container_width=True)
    with col2:
        st.markdown(
            "- Junction temperature is driven almost entirely by **TDP**.\n"
            "- Thermal resistance is driven by **air velocity**, with "
            "diminishing returns (V^⅓ law).\n"
            "- **TIM conductivity** only matters when the TIM is poor "
            "(< ~4 W/m·K).\n"
            "- TDP has exactly zero effect on resistance — as the physics demands."
        )
    st.image(os.path.join("results", "sensitivity_trends.png"))
    st.image(os.path.join("results", "permutation_importance.png"))

with tab_data:
    st.subheader("Parameter sweep dataset")
    st.write(f"{len(df):,} design points — full factorial over TDP, velocity, k_tim.")
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        file_name="heat_sink_dataset.csv",
        mime="text/csv",
    )
