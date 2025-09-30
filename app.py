import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, gumbel_r
import matplotlib.patheffects as pe

st.set_page_config(page_title="Extreme Precipitation Analysis", layout="wide")
st.markdown(
    """
    <div style="
        background-color:#461D7C;
        color:white;
        text-align:center;
        padding:12px;
        font-size:28px;
        font-weight:bold;
        border-radius:5px;">
        🌧️ Extreme Precipitation Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Upload a CSV file containing precipitation values (one column of numbers). "
    "Click **Run Analysis** to estimate return levels."
)

# File upload
uploaded_file = st.file_uploader("Upload Precipitation CSV", type=["csv"])

# Return periods selector
default_rp = [2, 5, 10, 25, 50, 100]
rp = st.multiselect("Select Return Periods (years):", default_rp, default=default_rp)
rp = np.array(rp)
p = 1 - 1/rp

# Load data
if uploaded_file is None:
    st.info("No file uploaded. Please upload a CSV.")
    st.stop()

df = pd.read_csv(uploaded_file)
x = pd.to_numeric(df.values.flatten(), errors='coerce')
x = x[~np.isnan(x)]

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #461D7C;
        color: white;
        font-weight: bold;
        border-radius:5px;
        border: none;
        padding: 0.5em 1em;
    }
    div.stButton > button:first-child:hover {
        background-color: #5A2DA8; /* slightly lighter purple on hover */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if st.button("Run Analysis"):
    # --- Fit parameters ---
    shape, loc, scale = genextreme.fit(x)
    loc_g, scale_g = gumbel_r.fit(x)

    # --- Point estimates of return levels ---
    gev_levels = genextreme.ppf(p, shape, loc=loc, scale=scale)
    gumbel_levels = gumbel_r.ppf(p, loc=loc_g, scale=scale_g)

    # --- Bootstrap 95% CI ---
    nboot = 200  # keep modest for speed
    rng = np.random.default_rng()

    gev_boot = np.zeros((nboot, len(rp)))
    gumbel_boot = np.zeros((nboot, len(rp)))

    for i in range(nboot):
        sample = rng.choice(x, size=len(x), replace=True)
        s, l, sc = genextreme.fit(sample)
        locb, scaleb = gumbel_r.fit(sample)
        gev_boot[i] = genextreme.ppf(p, s, loc=l, scale=sc)
        gumbel_boot[i] = gumbel_r.ppf(p, loc=locb, scale=scaleb)

    gev_lower, gev_upper = np.percentile(gev_boot, [2.5, 97.5], axis=0)
    gumbel_lower, gumbel_upper = np.percentile(gumbel_boot, [2.5, 97.5], axis=0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(rp, gev_levels, 'o-', color='red', lw=2.0, label='GEV')
    ax.plot(rp, gev_lower, ':', color='red', lw=1)
    ax.plot(rp, gev_upper, ':', color='red', lw=1)

    ax.plot(rp, gumbel_levels, '-', color='black', lw=2.0, label='Gumbel')
    ax.scatter(rp, gumbel_levels, marker='^', color='black', s=40)
    ax.plot(rp, gumbel_lower, ':', color='black', lw=1)
    ax.plot(rp, gumbel_upper, ':', color='black', lw=1)

    # Annotate values
    for x_rp, y_val in zip(rp, gev_levels):
        ax.text(x_rp, y_val + 0.3, f'{y_val:.1f}', color='red',
                ha='center', fontsize=8)
    for x_rp, y_val in zip(rp, gumbel_levels):
        ax.text(x_rp, y_val - 0.5, f'{y_val:.1f}', color='black',
                ha='center', fontsize=8,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('Precip (in)')
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    plt.xticks(rp, labels=[str(r) for r in rp])  # exact ticks

    st.pyplot(fig)

st.markdown(
    """
    <div style="
        text-align:center;
        font-size:0.9em;
        margin-top:15px;
        background-color:#461D7C;  /* LSU purple */
        color:white;
        padding:8px;
        border-radius:5px;">
        Developed and deployed by <b>Md Shahinoor Rahman, PhD</b> — 
        School of Public Health, LSUHSC New Orleans
    </div>
    """,
    unsafe_allow_html=True
)
