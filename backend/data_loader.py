"""
backend/data_loader.py
-----------------------
Everything related to GETTING data into the app lives here.

Today that means: reading one or more tidy-format Excel exports and
classifying each by Date and Modulation type based on its filename.
Tomorrow that will also mean: pulling a live sweep back from a VNA over
GPIB/USB (PyVISA) and handing it to the rest of the app in the exact same
shape. Because every other module (analysis.py, app.py) only ever consumes
a pandas DataFrame with the columns below, swapping "read Excel" for
"read live instrument" later requires touching ONLY this file.

Expected tidy schema, per file (one row per frequency point per measurement):
    Wavelength_nm : int      (0 or negative convention allowed for "dark", but
                               here Is_Dark is the authoritative flag)
    Is_Dark       : int/bool (0 = illuminated, 1 = dark reference sweep)
    Gas_Cylinder  : int      (cylinder / gas condition ID)
    Vbias_V       : float    (reverse bias voltage applied to the photodiode)
    Trial         : int      (repeat index, 1..N, for the same conditions)
    Frequency_Hz  : float
    Magnitude_dB  : float    (S21 magnitude)
    Phase_deg     : float    (S21 phase)

After loading, two extra columns are attached from the filename itself:
    Date          : str  (e.g. "2026-07-06")
    Modulation    : str  ("Bm" or "Lm")

Filename convention assumed: Data_<YYYY-MM-DD>_<Bm|Lm>.xlsx
e.g. "Data_2026-07-06_Bm.xlsx" -> Date="2026-07-06", Modulation="Bm"
The parser is tolerant of extra prefixes/suffixes and case, and falls back
to "Unknown" for whichever piece it can't find, rather than failing the
whole upload.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "Wavelength_nm",
    "Is_Dark",
    "Gas_Cylinder",
    "Vbias_V",
    "Trial",
    "Frequency_Hz",
    "Magnitude_dB",
    "Phase_deg",
]

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_MODULATION_PATTERN = re.compile(r"(Bm|Lm)", re.IGNORECASE)
_MODULATION_CANONICAL = {"bm": "Bm", "lm": "Lm"}


def parse_filename_metadata(filename: str) -> tuple[str, str]:
    """
    Extract (date, modulation) from a filename like 'Data_2026-07-06_Bm.xlsx'.
    Returns "Unknown" for either piece that can't be found, so a file with a
    non-conforming name can still be loaded (just won't filter usefully by
    that dimension).
    """
    date_match = _DATE_PATTERN.search(filename)
    date_str = date_match.group(1) if date_match else "Unknown"

    mod_match = _MODULATION_PATTERN.search(filename)
    modulation = _MODULATION_CANONICAL[mod_match.group(1).lower()] if mod_match else "Unknown"

    return date_str, modulation


@st.cache_data(show_spinner=False)
def load_excel(file_path_or_buffer) -> pd.DataFrame:
    """
    Load one tidy-format Excel export produced by the lab's Auto_Save_Data
    pipeline (the same data that used to be indexed via index.mat + loose
    .mat sweep files for the MATLAB tool).

    Parameters
    ----------
    file_path_or_buffer : str | Path | UploadedFile
        Path to the .xlsx file, or a Streamlit-uploaded file object.

    Returns
    -------
    pd.DataFrame validated against REQUIRED_COLUMNS (Date/Modulation not
    yet attached - that happens in load_multiple_excel, since it needs the
    filename which this cached function also uses as its cache key).
    """
    df = pd.read_excel(file_path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File is missing required column(s): {missing}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    # Normalize dtypes so downstream comparisons (e.g. multiselect filters)
    # behave consistently regardless of how Excel typed the cells.
    df["Wavelength_nm"] = df["Wavelength_nm"].astype(int)
    df["Is_Dark"] = df["Is_Dark"].astype(int)
    df["Gas_Cylinder"] = df["Gas_Cylinder"].astype(int)
    df["Vbias_V"] = df["Vbias_V"].astype(float)
    df["Trial"] = df["Trial"].astype(int)
    df["Frequency_Hz"] = df["Frequency_Hz"].astype(float)

    return df


def load_multiple_excel(files) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate several tidy-format Excel files into one master
    DataFrame, tagging every row with the Date and Modulation parsed from
    its own filename.

    Parameters
    ----------
    files : list of (str path | Streamlit UploadedFile)
        Each item must expose a filename either as a plain string path or
        via a `.name` attribute (as Streamlit's UploadedFile does).

    Returns
    -------
    (master_df, manifest_df)
        master_df   : concatenated tidy data + Date/Modulation/SourceFile columns
        manifest_df : one row per input file, showing what was parsed from its
                      name and how many rows it contributed - handy for a
                      "did my files load correctly" sanity check in the UI.
    """
    frames = []
    manifest_rows = []

    for f in files:
        filename = getattr(f, "name", str(f))
        date_str, modulation = parse_filename_metadata(filename)

        df = load_excel(f).copy()
        df["Date"] = date_str
        df["Modulation"] = modulation
        df["SourceFile"] = filename
        frames.append(df)

        manifest_rows.append(
            {
                "SourceFile": filename,
                "Parsed Date": date_str,
                "Parsed Modulation": modulation,
                "Rows": len(df),
            }
        )

    if not frames:
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS + ["Date", "Modulation", "SourceFile"])
        return empty, pd.DataFrame(columns=["SourceFile", "Parsed Date", "Parsed Modulation", "Rows"])

    master = pd.concat(frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    return master, manifest


# ---------------------------------------------------------------------------
# FUTURE HARDWARE HOOK
# ---------------------------------------------------------------------------
# When live VNA control is wired up (see backend/hardware_interface.py), add
# a sibling function here, e.g.:
#
#     def load_from_hardware(hw: "VNAHardwareInterface", sweep_config) -> pd.DataFrame:
#         raw = hw.run_sweep(sweep_config)      # returns tidy DataFrame already
#         raw["Date"] = datetime.now().strftime("%Y-%m-%d")
#         raw["Modulation"] = sweep_config.modulation
#         return raw
#
# app.py would then offer a toggle: "Load from file(s)" vs "Acquire from VNA",
# and everything downstream (filters, plots, analysis) needs zero changes
# because both paths return the same tidy schema plus Date/Modulation.
# ---------------------------------------------------------------------------
