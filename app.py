"""
app.py
------
Frontend / UI layer only. This file is responsible for:
    - collecting user input (file uploads, filters, toggles)
    - calling into backend.analysis for number-crunching
    - calling into backend.hardware_interface for (future) live acquisition
    - drawing Plotly figures

It deliberately contains NO data-manipulation logic of its own (that all
lives in backend/analysis.py), NO filename-parsing logic of its own (that
lives in backend/data_loader.py), and NO instrument I/O of its own (that
lives in backend/hardware_interface.py). This separation is what lets
features like multi-file classification or live hardware control evolve
without turning this file into a tangle of UI code mixed with parsing/I/O.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import analysis
from backend.data_loader import load_multiple_excel
from backend.hardware_interface import MockVNA, SweepConfig

st.set_page_config(page_title="Photodiode Sweep Analysis", layout="wide")

# ---------------------------------------------------------------------------
# Session state (holds the hardware handle across reruns, once wired up)
# ---------------------------------------------------------------------------
if "hw" not in st.session_state:
    # Swap MockVNA() for PyVISAVNA("GPIB0::16::INSTR") when real hardware is
    # available - see backend/hardware_interface.py for the full recipe.
    st.session_state.hw = MockVNA()
if "pending_sweep_future" not in st.session_state:
    st.session_state.pending_sweep_future = None

st.title("🔬 Photodiode VNA Sweep Analysis")
st.caption(
    "Replaces the MATLAB PhotodiodeAnalysisTool. Multi-file upload with "
    "Date/Modulation classification, Mean±STD / Select-Trial views, phase "
    "normalization, and dedicated A/B condition comparison."
)

# ---------------------------------------------------------------------------
# Data source: multi-file upload & classification
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded_files = st.file_uploader(
        "Upload tidy-format measurement file(s) (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Filenames are parsed for Date and Modulation, e.g. "
             "'Data_2026-07-06_Bm.xlsx' -> Date=2026-07-06, Modulation=Bm.",
    )

if uploaded_files:
    df, manifest = load_multiple_excel(uploaded_files)
    with st.sidebar:
        with st.expander(f"📄 {len(uploaded_files)} file(s) loaded - parsed as", expanded=False):
            st.dataframe(manifest, width="stretch", hide_index=True)
elif os.path.exists("AI_Ready_Measurements.xlsx"):
    # Convenience fallback for local/demo use - a single default file whose
    # name may not follow the Date/Modulation convention, so those columns
    # will show up as "Unknown" until real multi-file data is uploaded.
    df, manifest = load_multiple_excel(["AI_Ready_Measurements.xlsx"])
    st.sidebar.caption("Using default local file AI_Ready_Measurements.xlsx (no files uploaded).")
else:
    st.info("Upload one or more tidy-format .xlsx files in the sidebar to begin.")
    st.stop()

if df.empty:
    st.warning("No data could be loaded from the provided file(s).")
    st.stop()

all_wl = sorted(df["Wavelength_nm"].unique().tolist())
all_cyl = sorted(df["Gas_Cylinder"].unique().tolist())
all_vbias = sorted(df["Vbias_V"].unique().tolist())
all_trials = sorted(df["Trial"].unique().tolist())
all_dates = sorted(df["Date"].unique().tolist())
all_mods = sorted(df["Modulation"].unique().tolist())

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    sel_date = st.multiselect("Date", all_dates, default=all_dates)
    sel_mod = st.multiselect("Modulation", all_mods, default=all_mods)
    sel_wl = st.multiselect("Wavelength (nm)", all_wl, default=all_wl)
    sel_cyl = st.multiselect("Gas Cylinder", all_cyl, default=all_cyl)
    sel_vbias = st.multiselect("Vbias (V)", all_vbias, default=all_vbias)

    dark_mode = st.radio(
        "Measurement type",
        ["Illuminated only", "Dark only", "Both"],
        index=0,
        help="Is_Dark column: illuminated (0) vs dark reference (1) sweeps.",
    )
    is_dark_filter = {"Illuminated only": 0, "Dark only": 1, "Both": None}[dark_mode]

    ref_cyl = st.selectbox("Reference cylinder (phase normalization)", all_cyl, index=0)

    st.header("Trial Display")
    trial_mode = st.radio("Mode", ["Mean + STD", "Select Trial"], index=0)
    selected_trial = None
    if trial_mode == "Select Trial":
        selected_trial = st.selectbox("Trial #", all_trials, index=0)
    show_band = st.checkbox("Show ± STD shaded band", value=True)

    st.header("Smoothing")
    use_smoothing = st.checkbox("Smooth traces", value=True)
    smooth_window = st.slider("Smoothing window", min_value=1, max_value=20, value=4)

if not sel_wl or not sel_cyl or not sel_vbias or not sel_date or not sel_mod:
    st.warning("Select at least one Date, Modulation, Wavelength, Gas Cylinder, and Vbias in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Filtered data (shared across the main tabs)
# ---------------------------------------------------------------------------
df_filtered = analysis.filter_data(
    df, sel_wl, sel_cyl, sel_vbias, is_dark_filter, modulations=sel_mod, dates=sel_date
)

if df_filtered.empty:
    st.warning("No rows match the current filter selection.")
    st.stop()


def make_label(wl, cyl, vb, mod=None, date=None) -> str:
    return analysis.condition_label(wl, cyl, vb, mod, date)


tabs = st.tabs(
    [
        "Magnitude vs Frequency",
        "Phase vs Frequency",
        "vs Gas Cylinder",
        "A/B Comparison",
        "Live Acquisition (future)",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Magnitude vs Frequency
# ---------------------------------------------------------------------------
with tabs[0]:
    agg = analysis.aggregate_traces(
        df_filtered, trial_mode, selected_trial, "Magnitude_dB", use_smoothing, smooth_window
    )
    fig = go.Figure()
    for (wl, cyl, vb, mod, date), g in agg.groupby(analysis.CONDITION_COLS):
        g = g.sort_values("Frequency_Hz")
        label = make_label(wl, cyl, vb, mod, date)
        fig.add_trace(go.Scatter(x=g["Frequency_Hz"], y=g["mean"], mode="lines", name=label))
        if trial_mode == "Mean + STD" and show_band and g["std"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([g["Frequency_Hz"], g["Frequency_Hz"][::-1]]),
                    y=pd.concat([g["mean"] + g["std"], (g["mean"] - g["std"])[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.08)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="|S21| [dB]",
        height=600,
        legend_title="Condition",
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: Phase vs Frequency (normalized to reference cylinder)
# ---------------------------------------------------------------------------
with tabs[1]:
    agg_phase = analysis.aggregate_traces(
        df_filtered, trial_mode, selected_trial, "Phase_deg", use_smoothing, smooth_window
    )
    agg_phase_norm = analysis.normalize_phase_to_reference(agg_phase, ref_cyl)

    fig2 = go.Figure()
    for (wl, cyl, vb, mod, date), g in agg_phase_norm.groupby(analysis.CONDITION_COLS):
        g = g.sort_values("Frequency_Hz")
        label = make_label(wl, cyl, vb, mod, date)
        fig2.add_trace(go.Scatter(x=g["Frequency_Hz"], y=g["mean_norm"], mode="lines", name=label))
    fig2.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title=f"Δ Phase [deg] (normalized to Cylinder {ref_cyl})",
        height=600,
        legend_title="Condition",
    )
    st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: Amplitude/Phase vs Gas Cylinder, with frequency slider
# ---------------------------------------------------------------------------
with tabs[2]:
    freqs_available = sorted(df_filtered["Frequency_Hz"].unique().tolist())
    freq_idx = st.slider(
        "Frequency index", 0, len(freqs_available) - 1, len(freqs_available) // 2, key="freq_slider_tab3"
    )
    freq_pick = freqs_available[freq_idx]
    st.caption(f"Frequency: {freq_pick / 1e6:.3f} MHz")

    d_at_f = df_filtered[np.isclose(df_filtered["Frequency_Hz"], freq_pick)]

    col_a, col_b = st.columns(2)
    fig3a = go.Figure()
    fig3b = go.Figure()
    for (wl, vb, mod, date), g in d_at_f.groupby(["Wavelength_nm", "Vbias_V", "Modulation", "Date"]):
        if trial_mode == "Select Trial":
            g_use = g[g["Trial"] == selected_trial]
            stat = g_use.groupby("Gas_Cylinder").agg(
                amp=("Magnitude_dB", "mean"), phase=("Phase_deg", "mean")
            ).reset_index()
            amp_err = None
            phase_err = None
        else:
            stat = g.groupby("Gas_Cylinder").agg(
                amp=("Magnitude_dB", "mean"),
                amp_std=("Magnitude_dB", "std"),
                phase=("Phase_deg", "mean"),
                phase_std=("Phase_deg", "std"),
            ).reset_index()
            amp_err = stat["amp_std"]
            phase_err = stat["phase_std"]

        stat = stat.sort_values("Gas_Cylinder")
        if stat.empty:
            continue
        ref_phase = stat["phase"].iloc[0]
        stat["phase_rel"] = stat["phase"] - ref_phase
        label = make_label(wl, "*", vb, mod, date).replace("Cyl *, ", "")

        fig3a.add_trace(
            go.Scatter(
                x=stat["Gas_Cylinder"], y=stat["amp"], mode="lines+markers", name=label,
                error_y=dict(type="data", array=amp_err) if amp_err is not None else None,
            )
        )
        fig3b.add_trace(
            go.Scatter(
                x=stat["Gas_Cylinder"], y=stat["phase_rel"], mode="lines+markers", name=label,
                error_y=dict(type="data", array=phase_err) if phase_err is not None else None,
            )
        )
    fig3a.update_layout(xaxis_title="Gas Cylinder", yaxis_title="Amplitude |S21| [dB]", height=500)
    fig3b.update_layout(xaxis_title="Gas Cylinder", yaxis_title="Δ Phase [deg] (norm. to first cylinder)", height=500)
    col_a.plotly_chart(fig3a, width="stretch")
    col_b.plotly_chart(fig3b, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: A/B Comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    st.write(
        "**Shared condition:** Wavelength, Vbias, and Modulation apply to both "
        "groups. Only **Date** and **Trial** differ between Group A (solid "
        "lines) and Group B (dashed lines) - one line per Gas Cylinder, on "
        "the same axes."
    )

    value_choice = st.radio(
        "Value to compare", ["Magnitude_dB", "Phase_deg"], horizontal=True, key="ab_value_choice"
    )

    # --- Shared (global) filters -------------------------------------------------
    s1, s2, s3 = st.columns(3)
    shared_wl = s1.selectbox("Wavelength (nm)", all_wl, key="ab_shared_wl")
    shared_vbias = s2.selectbox("Vbias (V)", all_vbias, key="ab_shared_vbias")
    shared_mod = s3.selectbox("Modulation", all_mods, key="ab_shared_mod")

    # --- Cylinder selection is hardcoded by graph type, not user-selected -------
    if value_choice == "Magnitude_dB":
        ab_cylinders = sorted(df["Gas_Cylinder"].unique().tolist())
        st.caption(f"Plotting all available cylinders for Magnitude: {ab_cylinders}")
    else:
        ab_cylinders = [c for c in analysis.PHASE_AB_CYLINDERS if c in set(df["Gas_Cylinder"].unique())]
        st.caption(f"Plotting hardcoded cylinder set for Phase (Normalized): {analysis.PHASE_AB_CYLINDERS}")

    ab_phase_ref_cyl = None
    if value_choice == "Phase_deg":
        ab_phase_ref_cyl = st.selectbox(
            "Reference cylinder (phase normalization)",
            all_cyl,
            index=0,
            key="ab_phase_ref_cyl",
            help="Every cylinder's phase trace is normalized point-by-point "
                 "(matched by Frequency, Wavelength, Date, Modulation, and "
                 "Trial) against this cylinder.",
        )

    # --- Per-group Date / Trial -------------------------------------------------
    shared_mask = (
        (df["Wavelength_nm"] == shared_wl)
        & (df["Vbias_V"] == shared_vbias)
        & (df["Modulation"] == shared_mod)
    )
    df_shared = df.loc[shared_mask]

    def group_date_trial_ui(group_name: str, key_prefix: str) -> dict:
        st.markdown(f"**Group {group_name}**")
        dates_for_cond = sorted(df_shared["Date"].unique().tolist())
        if not dates_for_cond:
            st.warning(f"No data for this shared condition (Group {group_name}).")
            return dict(date=None, trial=None)
        g_date = st.selectbox(f"Date {group_name}", dates_for_cond, key=f"{key_prefix}_date")
        trials_for_cond = sorted(
            df_shared.loc[df_shared["Date"] == g_date, "Trial"].unique().tolist()
        )
        if not trials_for_cond:
            st.warning(f"No trials for {g_date} (Group {group_name}).")
            return dict(date=g_date, trial=None)
        g_trial = st.selectbox(f"Trial {group_name}", trials_for_cond, key=f"{key_prefix}_trial")
        return dict(date=g_date, trial=g_trial)

    col_left, col_right = st.columns(2)
    with col_left:
        group_a_cfg = group_date_trial_ui("A", "ab_a")
    with col_right:
        group_b_cfg = group_date_trial_ui("B", "ab_b")

    traces_a = analysis.aggregate_ab_traces(
        df, wavelength=shared_wl, vbias=shared_vbias, modulation=shared_mod,
        cylinders=ab_cylinders, value_col=value_choice,
        smoothing=use_smoothing, smooth_window=smooth_window,
        phase_ref_cylinder=ab_phase_ref_cyl, **group_a_cfg,
    )
    traces_b = analysis.aggregate_ab_traces(
        df, wavelength=shared_wl, vbias=shared_vbias, modulation=shared_mod,
        cylinders=ab_cylinders, value_col=value_choice,
        smoothing=use_smoothing, smooth_window=smooth_window,
        phase_ref_cylinder=ab_phase_ref_cyl, **group_b_cfg,
    )

    # Consistent color per cylinder, shared between Group A and Group B traces
    # of that cylinder - only line style (solid vs dashed) differs.
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    color_by_cyl = {cyl: palette[i % len(palette)] for i, cyl in enumerate(ab_cylinders)}

    fig_ab = go.Figure()
    any_trace = False
    for cyl in ab_cylinders:
        if cyl in traces_a:
            any_trace = True
            t = traces_a[cyl]
            label = f"Cyl {cyl} - {group_a_cfg['date']} / Trial {group_a_cfg['trial']} (Solid)"
            fig_ab.add_trace(
                go.Scatter(
                    x=t["freq"], y=t["values"], mode="lines",
                    line=dict(dash="solid", width=2.5, color=color_by_cyl[cyl]),
                    name=label,
                )
            )
        if cyl in traces_b:
            any_trace = True
            t = traces_b[cyl]
            label = f"Cyl {cyl} - {group_b_cfg['date']} / Trial {group_b_cfg['trial']} (Dashed)"
            fig_ab.add_trace(
                go.Scatter(
                    x=t["freq"], y=t["values"], mode="lines",
                    line=dict(dash="dash", width=2.5, color=color_by_cyl[cyl]),
                    name=label,
                )
            )

    if not any_trace:
        st.warning("No data for the selected shared condition / Date / Trial combination.")

    if value_choice == "Magnitude_dB":
        y_title = "|S21| [dB]"
    elif ab_phase_ref_cyl is not None:
        y_title = f"Δ Phase [deg] (normalized to Cylinder {ab_phase_ref_cyl})"
    else:
        y_title = "Phase [deg]"
    fig_ab.update_layout(
        xaxis_title="Frequency [Hz]", yaxis_title=y_title, height=600, legend_title="Cylinder / Group"
    )
    st.plotly_chart(fig_ab, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5: Live Acquisition (placeholder demonstrating the hardware hook)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.write(
        "This tab demonstrates the architecture for **future direct VNA control** "
        "via `backend/hardware_interface.py`. It currently talks to a `MockVNA` "
        "that fabricates a plausible sweep instead of a real instrument, so you "
        "can build/test the workflow before hardware is connected."
    )
    hw = st.session_state.hw

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect to VNA" if not hw.connected else "Reconnect"):
            hw.connect()
            st.success("Connected (mock).")
    with col2:
        if st.button("Disconnect"):
            hw.disconnect()
            st.info("Disconnected.")

    st.write(f"Status: {'🟢 Connected' if hw.connected else '🔴 Not connected'}")

    with st.form("sweep_form"):
        st.write("Configure and run a new sweep (non-blocking):")
        c1, c2, c3, c4 = st.columns(4)
        cfg_wl = c1.number_input("Wavelength (nm)", value=int(all_wl[0]))
        cfg_cyl = c2.number_input("Gas Cylinder", value=int(all_cyl[0]))
        cfg_vb = c3.number_input("Vbias (V)", value=float(all_vbias[0]))
        cfg_mod = c4.selectbox("Modulation", ["Bm", "Lm"])
        cfg_trial = st.number_input("Trial #", value=1, min_value=1)
        submitted = st.form_submit_button("Run Sweep")

    if submitted:
        if not hw.connected:
            st.error("Connect to the instrument first.")
        else:
            config = SweepConfig(
                wavelength_nm=int(cfg_wl), gas_cylinder=int(cfg_cyl),
                vbias_v=float(cfg_vb), trial=int(cfg_trial), modulation=cfg_mod,
            )
            # Non-blocking: submitted to a background thread. The Streamlit
            # UI thread returns immediately instead of freezing for the
            # duration of the (simulated, here 0.5s / real, could be much
            # longer) sweep.
            st.session_state.pending_sweep_future = hw.run_sweep_async(config)
            st.info("Sweep submitted in the background...")

    future = st.session_state.pending_sweep_future
    if future is not None:
        if future.done():
            new_df = future.result()
            st.success("Sweep complete.")
            st.dataframe(new_df.head(10), width="stretch")
            st.caption(
                "In a full implementation, this new_df would be appended to the "
                "master dataset (and/or written back to disk) so it immediately "
                "shows up in the analysis tabs above - no code changes needed "
                "there, since they only depend on the tidy schema plus "
                "Date/Modulation."
            )
        else:
            st.warning("Sweep still running in the background... click again to check.")
