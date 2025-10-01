import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import genextreme, gumbel_r
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Extreme Precipitation Analysis", layout="wide")

# ---------- CUSTOM STYLES ----------
st.markdown(
    """
    <style>
    /* LSU purple header */
    .header {
        background-color: #461D7C;
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 28px;
        font-weight: bold;
        border-radius: 6px;
        margin-bottom: 24px;
    }

    /* LSU button styling */
    div.stButton > button:first-child {
        background-color: #461D7C;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    div.stButton > button:hover {
        background-color: #5e2ea7;
        color: white;
    }

    /* Footer credit */
    .footer {
        background-color: #461D7C;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        margin-top: 15px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ----------
st.markdown("<div class='header'>🌧️ Extreme Precipitation Analysis</div>", unsafe_allow_html=True)

st.write("Upload a CSV file containing precipitation values (one column of numbers). "
         "Click **Run Analysis** to estimate return levels.")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload Precipitation CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default sample dataset (NOLA_Precip.csv in repo).")
    df = pd.read_csv("NOLA_Precip.csv")  # keep a sample file in repo

x = pd.to_numeric(df.values.flatten(), errors='coerce')
x = x[~np.isnan(x)]

return_periods = st.multiselect(
    "Select Return Periods (years):",
    [2, 5, 10, 25, 50, 100],
    default=[2, 5, 10, 25, 50, 100]
)

# ---------- BUTTON ----------
if st.button("Run Analysis"):
    rp = np.array(return_periods)
    p = 1 - 1 / rp

    # --- Fit parameters ---
    shape, loc, scale = genextreme.fit(x)
    loc_g, scale_g = gumbel_r.fit(x)

    # --- Point estimates ---
    gev_levels = genextreme.ppf(p, shape, loc=loc, scale=scale)
    gumbel_levels = gumbel_r.ppf(p, loc=loc_g, scale=scale_g)

    # --- Bootstrap 95% CI ---
    nboot = 200  # modest for speed
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
    fig, ax = plt.subplots(figsize=(8, 5))  # slightly smaller

    ax.grid(True, which='both', linestyle=':')

    # GEV line
    ax.plot(rp, gev_levels, 'o-', color='red', lw=1.5, label='GEV',
            markersize=4, markerfacecolor='red')
    ax.plot(rp, gev_lower, ':', color='red', lw=0.7)
    ax.plot(rp, gev_upper, ':', color='red', lw=0.7)

    # Gumbel line
    ax.plot(rp, gumbel_levels, '-', color='blue', lw=1.5, label='Gumbel')
    ax.scatter(rp, gumbel_levels, marker='^', color='blue', s=25)
    ax.plot(rp, gumbel_lower, ':', color='blue', lw=0.7)
    ax.plot(rp, gumbel_upper, ':', color='blue', lw=0.7)

    # Annotations
    for x_rp, y_val in zip(rp, gev_levels):
        ax.text(x_rp, y_val + 0.5, f'{y_val:.1f}', color='red', ha='center', fontsize=8)
    for x_rp, y_val in zip(rp, gumbel_levels):
        ax.text(x_rp, y_val - 1.0, f'{y_val:.1f}', color='blue', ha='center', fontsize=8)

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Precip (in)")
    ax.set_xscale('log')
    ax.set_xticks(rp)
    ax.set_xticklabels(rp)
    ax.legend()

    # ---- show figure at 50% width and centered ----
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False, clear_figure=False)
    st.markdown("</div>", unsafe_allow_html=True)


    # ---------- FOOTER ----------
    st.markdown(
        "<div class='footer'>Developed and deployed by <b>Md Shahinoor Rahman, PhD</b> — School of Public Health, LSUHSC New Orleans</div>",
        unsafe_allow_html=True
)
