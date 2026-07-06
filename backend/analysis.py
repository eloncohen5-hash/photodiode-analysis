"""
backend/analysis.py
--------------------
Pure data-manipulation logic: filtering, aggregation, smoothing, and phase
normalization. Nothing in this file touches Streamlit or Plotly - it only
takes/returns pandas DataFrames and numpy arrays. That separation means
these functions can be unit-tested on their own, and reused unchanged if
the UI layer is ever swapped (e.g. Streamlit -> Dash).

A "condition" is a unique combination of Wavelength, Gas Cylinder, Vbias,
Modulation, and Date. Trial is treated separately: it's the dimension that
gets collapsed by "Mean + STD" (averaged across trials) or picked out by
"Select Trial" (one specific trial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Every dimension except Trial defines a distinct measurement "condition".
# Trial is the repeat axis that Mean+STD / Select-Trial collapses or picks from.
CONDITION_COLS = ["Wavelength_nm", "Gas_Cylinder", "Vbias_V", "Modulation", "Date"]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_data(
    df: pd.DataFrame,
    wavelengths: list[int],
    cylinders: list[int],
    vbias: list[float],
    is_dark: int | None,
    modulations: list[str] | None = None,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Slice the master tidy DataFrame down to the user's current filter
    selection. `modulations`/`dates` are optional so this still works on
    data that hasn't been tagged with those columns.

    is_dark: 0 -> illuminated only, 1 -> dark only, None -> both.

    IMPORTANT - Dark measurements are physically independent of Gas_Cylinder
    (and Vbias): a dark sweep isn't "for" a particular cylinder, so it must
    stay visible no matter which cylinders/Vbias the user has checked in the
    sidebar. Illuminated rows are still filtered by Gas_Cylinder/Vbias as
    before. Wavelength, Modulation, and Date still apply to both.
    """
    base_mask = df["Wavelength_nm"].isin(wavelengths)
    if modulations is not None and "Modulation" in df.columns:
        base_mask &= df["Modulation"].isin(modulations)
    if dates is not None and "Date" in df.columns:
        base_mask &= df["Date"].isin(dates)

    is_dark_bool = df["Is_Dark"].astype(bool)

    # Illuminated rows: full filter, including Gas_Cylinder and Vbias.
    illuminated_mask = (
        base_mask
        & ~is_dark_bool
        & df["Gas_Cylinder"].isin(cylinders)
        & df["Vbias_V"].isin(vbias)
    )
    # Dark rows: Gas_Cylinder/Vbias filters are deliberately NOT applied -
    # a dark sweep remains visible regardless of which cylinders are checked.
    dark_mask = base_mask & is_dark_bool

    if is_dark == 0:
        mask = illuminated_mask
    elif is_dark == 1:
        mask = dark_mask
    else:  # None -> both illuminated and dark
        mask = illuminated_mask | dark_mask

    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing, equivalent to MATLAB's smoothdata('movmean')."""
    if window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------
# Mean+STD / Select-Trial aggregation
# ---------------------------------------------------------------------------
def aggregate_traces(
    df: pd.DataFrame,
    trial_mode: str,
    selected_trial: int | None,
    value_col: str,
    smoothing: bool,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Collapse the Trial dimension for each condition (Wavelength, Gas_Cylinder,
    Vbias, Modulation, Date) into one plottable trace.

    Returns a long DataFrame with columns:
        Wavelength_nm, Gas_Cylinder, Vbias_V, Modulation, Date,
        Frequency_Hz, mean, std, n_trials
    'std' is NaN when trial_mode == 'Select Trial' (single trace, no spread).
    """
    out_rows = []
    for cond, g in df.groupby(CONDITION_COLS):
        wl, cyl, vb, mod, date = cond
        if trial_mode == "Select Trial":
            g_trial = g[g["Trial"] == selected_trial]
            if g_trial.empty:
                continue
            g_trial = g_trial.sort_values("Frequency_Hz")
            y = g_trial[value_col].to_numpy()
            if smoothing:
                y = smooth_series(y, smooth_window)
            out_rows.append(
                pd.DataFrame(
                    {
                        "Wavelength_nm": wl,
                        "Gas_Cylinder": cyl,
                        "Vbias_V": vb,
                        "Modulation": mod,
                        "Date": date,
                        "Frequency_Hz": g_trial["Frequency_Hz"].to_numpy(),
                        "mean": y,
                        "std": np.nan,
                        "n_trials": 1,
                    }
                )
            )
        else:  # Mean + STD
            pivot = g.pivot_table(
                index="Frequency_Hz", columns="Trial", values=value_col
            ).sort_index()
            y_mean = pivot.mean(axis=1).to_numpy()
            y_std = pivot.std(axis=1).to_numpy()
            if smoothing:
                y_mean = smooth_series(y_mean, smooth_window)
                y_std = smooth_series(y_std, smooth_window)
            out_rows.append(
                pd.DataFrame(
                    {
                        "Wavelength_nm": wl,
                        "Gas_Cylinder": cyl,
                        "Vbias_V": vb,
                        "Modulation": mod,
                        "Date": date,
                        "Frequency_Hz": pivot.index.to_numpy(),
                        "mean": y_mean,
                        "std": y_std,
                        "n_trials": pivot.shape[1],
                    }
                )
            )
    if not out_rows:
        return pd.DataFrame(
            columns=CONDITION_COLS + ["Frequency_Hz", "mean", "std", "n_trials"]
        )
    return pd.concat(out_rows, ignore_index=True)


def condition_label(wl, cyl, vb, mod=None, date=None) -> str:
    """Human-readable label for a condition tuple, used in plot legends."""
    parts = [f"{wl}nm", f"Cyl {cyl}", f"{vb:g}V"]
    if mod is not None and mod != "":
        parts.append(str(mod))
    if date is not None and date != "":
        parts.append(str(date))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Phase normalization against a reference cylinder
# ---------------------------------------------------------------------------
def normalize_phase_to_reference(agg_df: pd.DataFrame, ref_cylinder: int) -> pd.DataFrame:
    """
    Subtract the reference cylinder's phase (matched by Frequency_Hz, within
    the same Wavelength/Vbias/Modulation/Date) from every trace's phase.
    Mirrors the MATLAB tool's ddRefCyl behavior, extended to the new
    Modulation/Date dimensions so traces from different modulation types or
    dates aren't normalized against each other's reference cylinder.
    """
    group_keys = ["Wavelength_nm", "Vbias_V", "Modulation", "Date"]
    result = []
    for key, g in agg_df.groupby(group_keys):
        ref = g[g["Gas_Cylinder"] == ref_cylinder][["Frequency_Hz", "mean"]].rename(
            columns={"mean": "ref_mean"}
        )
        if ref.empty:
            merged = g.copy()
            merged["mean_norm"] = merged["mean"]
        else:
            merged = g.merge(ref, on="Frequency_Hz", how="left")
            merged["mean_norm"] = merged["mean"] - merged["ref_mean"]
        result.append(merged)
    return pd.concat(result, ignore_index=True)


# ---------------------------------------------------------------------------
# A/B Comparison support
# ---------------------------------------------------------------------------
def normalize_phase_raw_to_reference(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    ref_cylinder: int,
) -> pd.DataFrame:
    """
    Point-by-point phase normalization on RAW (non-aggregated) rows, for use
    before averaging across trials (unlike `normalize_phase_to_reference`,
    which operates on already-aggregated Mean+STD traces).

    For every row in `target_df`, subtracts the Phase_deg of the row in
    `source_df` that shares the same Wavelength_nm, Vbias_V, Modulation,
    Date, Trial, and Frequency_Hz but has Gas_Cylinder == ref_cylinder.
    `source_df` is passed separately (rather than reusing target_df) because
    the reference cylinder's rows may live outside whatever single-condition
    slice is being normalized (e.g. Group A/B in the A/B Comparison tab,
    where the group's own cylinder is different from the reference
    cylinder).

    Note Gas_Cylinder is intentionally NOT part of the match key (it
    obviously differs by construction: target rows are whatever cylinder is
    being plotted, the reference row is always `ref_cylinder`). Vbias_V IS
    included, since phase depends strongly on reverse bias - normalizing
    across different Vbias would corrupt the result.

    Adds a 'Phase_deg_norm' column; rows with no matching reference get NaN.
    """
    match_keys = ["Wavelength_nm", "Vbias_V", "Modulation", "Date", "Trial", "Frequency_Hz"]
    ref = (
        source_df.loc[source_df["Gas_Cylinder"] == ref_cylinder, match_keys + ["Phase_deg"]]
        .rename(columns={"Phase_deg": "ref_phase"})
        .drop_duplicates(subset=match_keys)
    )
    merged = target_df.merge(ref, on=match_keys, how="left")
    merged["Phase_deg_norm"] = merged["Phase_deg"] - merged["ref_phase"]
    return merged


# Hardcoded default cylinder set for Phase (Normalized) plots in the A/B tab,
# per current lab convention - Magnitude plots still show every cylinder in
# the dataset for the chosen shared condition.
PHASE_AB_CYLINDERS = [3, 4, 5]


def aggregate_ab_traces(
    df: pd.DataFrame,
    wavelength: int,
    vbias: float,
    modulation: str | None,
    date: str | None,
    trial: int | None,
    cylinders: list[int],
    value_col: str,
    smoothing: bool,
    smooth_window: int,
    phase_ref_cylinder: int | None = None,
) -> dict[int, dict]:
    """
    Build one trace per Gas_Cylinder for a single shared condition (one side
    - "A" or "B" - of the A/B Comparison tab): fixed Wavelength/Vbias/
    Modulation, plus that side's own Date and Trial pick. Cylinder is no
    longer a UI filter here; the caller passes in whichever cylinder list
    applies for the value being plotted (all cylinders for Magnitude_dB,
    PHASE_AB_CYLINDERS for Phase_deg).

    phase_ref_cylinder : int | None
        Only used when value_col == "Phase_deg". If given, rows are
        normalized point-by-point against this cylinder (matched by
        Wavelength/Modulation/Date/Trial/Frequency, via
        `normalize_phase_raw_to_reference`) before being split out per
        cylinder, so each returned trace is Normalized Phase.

    Returns {cylinder: {"freq": np.ndarray, "values": np.ndarray}},
    silently skipping any cylinder with no data for this exact condition
    (single Trial, not averaged - this tab compares one trial against
    another, not trial-averaged means). Only illuminated rows (Is_Dark == 0)
    are considered: mixing in dark-sweep rows at the same frequencies would
    zig-zag the line between two different physical measurements.
    """
    mask = (
        (df["Wavelength_nm"] == wavelength)
        & (df["Vbias_V"] == vbias)
        & (df["Is_Dark"] == 0)
    )
    if modulation is not None and "Modulation" in df.columns:
        mask &= df["Modulation"] == modulation
    if date is not None and "Date" in df.columns:
        mask &= df["Date"] == date
    if trial is not None:
        mask &= df["Trial"] == trial

    d = df.loc[mask]

    working_value_col = value_col
    if value_col == "Phase_deg" and phase_ref_cylinder is not None:
        d = normalize_phase_raw_to_reference(d, df, phase_ref_cylinder)
        working_value_col = "Phase_deg_norm"

    traces: dict[int, dict] = {}
    for cyl in cylinders:
        g = d.loc[d["Gas_Cylinder"] == cyl].sort_values("Frequency_Hz")
        if g.empty:
            continue
        y = g[working_value_col].to_numpy()
        if smoothing:
            y = smooth_series(y, smooth_window)
        traces[cyl] = {"freq": g["Frequency_Hz"].to_numpy(), "values": y}
    return traces
