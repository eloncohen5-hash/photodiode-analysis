# Photodiode VNA Sweep Analysis (Python / Streamlit)

Replaces `PhotodiodeAnalysisTool.m`. Reads one or more tidy-format Excel
exports and reproduces (and extends) the MATLAB tool's functionality in an
interactive browser app.

## Project layout

```
photodiode_app/
├── app.py                       # Streamlit UI (frontend only)
├── backend/
│   ├── data_loader.py           # Multi-file loading + filename Date/Modulation parsing
│   ├── analysis.py              # All pandas/numpy number-crunching (framework-agnostic)
│   └── hardware_interface.py    # Abstract VNA interface + MockVNA + PyVISA integration guide
├── requirements.txt
└── README.md
```

**Why it's split this way:** `app.py` never manipulates data, parses
filenames, or talks to instruments directly — it only calls
`backend.data_loader`, `backend.analysis`, and `backend.hardware_interface`.
When you're ready to control a real VNA, you write one new class in
`hardware_interface.py` (see the `PyVISAVNA` recipe in that file's
docstring) and swap one line in `app.py`. Hardware calls run through
`run_sweep_async()`, which executes on a background thread so a slow sweep
never freezes the UI.

## Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Upload one or more `.xlsx` files via the sidebar, or (for local/demo use)
place `AI_Ready_Measurements.xlsx` next to `app.py` as a fallback.

## Multi-file upload & classification

Upload as many files as you like in one go. Each filename is parsed for a
**Date** and a **Modulation** type:

```
Data_2026-07-06_Bm.xlsx  ->  Date = 2026-07-06,  Modulation = Bm
Data_2026-07-07_Lm.xlsx  ->  Date = 2026-07-07,  Modulation = Lm
```

The parser (`backend.data_loader.parse_filename_metadata`) looks for a
`YYYY-MM-DD` date and a `Bm`/`Lm` token anywhere in the filename (case
insensitive), so it tolerates extra prefixes/suffixes. Anything it can't
find falls back to `"Unknown"` rather than rejecting the file. After
upload, a small expander in the sidebar shows exactly what was parsed from
each filename, so you can catch a misnamed file immediately.

All uploaded files are concatenated into one master DataFrame with `Date`
and `Modulation` added as first-class columns, filterable from the sidebar
alongside Wavelength, Gas Cylinder, and Vbias.

## Features

**Core (replicating the MATLAB tool)**
- Multi-select filters: Date, Modulation, Wavelength, Gas Cylinder, Vbias
- "Mean + STD" (with shaded band) vs "Select Trial" display modes
- Tabs: Magnitude vs Frequency, Phase vs Frequency (normalized to a chosen
  reference cylinder, per Wavelength/Vbias/Modulation/Date), Amplitude/Phase
  vs Gas Cylinder with a frequency slider
- `Is_Dark` handled explicitly via a sidebar radio (Illuminated / Dark / Both)
- Optional moving-average smoothing (window adjustable), matching the
  MATLAB tool's `smoothdata('movmean', N)`

**New**
- **Multi-file upload & classification** (see above)
- **A/B Comparison tab:** pick one full condition (Wavelength, Cylinder,
  Vbias, Modulation, Date, and which Trial(s) to average) for Group A and
  another for Group B. Group A is drawn as a solid line, Group B as a
  dashed line, on the same axes, for either Magnitude or Phase. Use this to
  compare e.g. Trial 1 vs Trial 2 of the same condition, or the same
  condition across two different Dates.
- **Live Acquisition (future) tab:** working demo of the async
  hardware-control architecture using a `MockVNA` stand-in — connect,
  configure (including Modulation), and run a simulated sweep without
  freezing the page.

**Removed in this version** (previously present, dropped per current scope):
Baseline Subtraction, Nyquist plot, Heatmap/contour plot. The underlying
`analysis.py` module was trimmed accordingly to keep the codebase lean.

## Wiring up real VNA hardware later

See the docstring at the top of `backend/hardware_interface.py`. In short:
implement a `PyVISAVNA(VNAHardwareInterface)` class with `connect`,
`disconnect`, and `run_sweep`, then in `app.py` change:

```python
st.session_state.hw = MockVNA()
```
to
```python
st.session_state.hw = PyVISAVNA(resource_name="GPIB0::16::INSTR")
```

No other file needs to change.
