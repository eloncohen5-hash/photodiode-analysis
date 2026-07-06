"""
backend/hardware_interface.py
------------------------------
THIS IS THE FILE YOU WILL EDIT WHEN YOU CONNECT A REAL VNA.

Nothing in app.py or analysis.py needs to change when you go from
"file-based analysis" to "live instrument control" - they only ever talk
to the VNAHardwareInterface contract defined below, and they only ever
receive/produce the same tidy DataFrame schema used everywhere else
(see backend/data_loader.py).

Design principles used here, and why:

1. Abstract base class (VNAHardwareInterface)
   Defines *what* a sweep source can do (connect, disconnect, run a sweep)
   without saying *how*. Today `MockVNA` fakes it with numpy. Later,
   `PyVISAVNA` will implement the same methods using real GPIB/USB/Ethernet
   calls. app.py doesn't care which one it's holding.

2. Background execution, not blocking calls
   A real VNA sweep can take seconds. Calling it directly from a Streamlit
   button handler would freeze the whole page for every user until it
   returns. `run_sweep_async` submits the (potentially slow) hardware call
   to a background thread pool and hands back a `concurrent.futures.Future`
   immediately, so the UI thread stays responsive (spinners, other widgets,
   etc. keep working). app.py polls the Future (or awaits it) instead of
   blocking.

3. Config as data (SweepConfig dataclass)
   Whatever the UI collects (wavelength, cylinder, Vbias, number of trials,
   frequency range) is packaged into one plain object. This keeps the
   hardware call signature stable even as the UI evolves.

------------------------------------------------------------------------
HOW TO WIRE UP A REAL INSTRUMENT LATER (rough outline):

    pip install pyvisa pyvisa-py

    import pyvisa

    class PyVISAVNA(VNAHardwareInterface):
        def __init__(self, resource_name: str):
            self._rm = pyvisa.ResourceManager()
            self._inst = None
            self._resource_name = resource_name

        def connect(self):
            self._inst = self._rm.open_resource(self._resource_name)
            self._inst.write("*IDN?")   # sanity check
            self.connected = True

        def disconnect(self):
            if self._inst:
                self._inst.close()
            self.connected = False

        def run_sweep(self, config: SweepConfig) -> pd.DataFrame:
            # Example SCPI-ish exchange - replace with your VNA's actual command set
            self._inst.write(f"SENS:FREQ:START {config.freq_start_hz}")
            self._inst.write(f"SENS:FREQ:STOP {config.freq_stop_hz}")
            self._inst.write(f"SENS:SWE:POIN {config.n_points}")
            self._inst.write("INIT:IMM; *WAI")
            raw = self._inst.query_ascii_values("CALC:DATA:SDATA?")
            # ... reshape `raw` into freq/mag/phase, then build the tidy
            # DataFrame with the SweepConfig's condition columns attached.
            return build_tidy_frame_from_raw(raw, config)

Then in app.py, swap:
    hw = MockVNA()
for:
    hw = PyVISAVNA(resource_name="GPIB0::16::INSTR")
and everything else (the async wrapper, the UI hook) keeps working as-is.
------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

# One shared background worker pool for hardware calls. Kept small since a
# VNA is a single physical instrument - you generally want sweeps to queue,
# not run concurrently against the same box.
_HARDWARE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vna-hw")


@dataclass
class SweepConfig:
    """Everything needed to request one sweep from the instrument."""

    wavelength_nm: int
    gas_cylinder: int
    vbias_v: float
    trial: int
    is_dark: int = 0
    modulation: str = "Bm"  # "Bm" or "Lm" - kept alongside Date so a live
    #                         sweep tags itself the same way an uploaded
    #                         Data_<date>_<Bm|Lm>.xlsx file would.
    freq_start_hz: float = 1e5
    freq_stop_hz: float = 1e8
    n_points: int = 100


class VNAHardwareInterface(ABC):
    """Contract every VNA backend (mock or real) must satisfy."""

    connected: bool = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def run_sweep(self, config: SweepConfig) -> pd.DataFrame:
        """Blocking call. Must return a tidy DataFrame (see data_loader.py schema)."""
        ...

    def run_sweep_async(self, config: SweepConfig) -> Future:
        """
        Non-blocking wrapper around run_sweep(). Returns a Future immediately;
        the UI thread is free to keep rendering while the sweep runs.
        Usage in app.py:
            future = hw.run_sweep_async(config)
            # ... later, e.g. on a rerun/poll ...
            if future.done():
                new_df = future.result()
        """
        return _HARDWARE_EXECUTOR.submit(self.run_sweep, config)


class MockVNA(VNAHardwareInterface):
    """
    Software stand-in for a real instrument. Lets the rest of the app (and
    a future "Live Acquisition" tab) be built and tested today, with zero
    hardware attached. Simulates realistic latency and a plausible S21
    curve with noise so the UI/plots behave like they will against a real
    photodiode.
    """

    def connect(self) -> None:
        time.sleep(0.2)  # simulate handshake latency
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def run_sweep(self, config: SweepConfig) -> pd.DataFrame:
        if not self.connected:
            raise RuntimeError("MockVNA is not connected. Call connect() first.")

        time.sleep(0.5)  # simulate sweep acquisition time

        freq = np.linspace(config.freq_start_hz, config.freq_stop_hz, config.n_points)
        # A crude single-pole roll-off + noise, just for a believable shape.
        f3db = 5e6 + config.gas_cylinder * 2e5 + config.vbias_v * 1e4
        mag_db = -3 - 20 * np.log10(np.sqrt(1 + (freq / f3db) ** 2))
        mag_db += np.random.normal(0, 0.3, size=freq.shape)
        phase_deg = -np.rad2deg(np.arctan(freq / f3db))
        phase_deg += np.random.normal(0, 1.0, size=freq.shape)

        return pd.DataFrame(
            {
                "Wavelength_nm": config.wavelength_nm,
                "Is_Dark": config.is_dark,
                "Gas_Cylinder": config.gas_cylinder,
                "Vbias_V": config.vbias_v,
                "Trial": config.trial,
                "Frequency_Hz": freq,
                "Magnitude_dB": mag_db,
                "Phase_deg": phase_deg,
                "Modulation": config.modulation,
                "Date": time.strftime("%Y-%m-%d"),
            }
        )
