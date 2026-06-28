#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_filter_data_estimate_Z_amp.py

Filter SEISAN-Hutton station ML results, then measure early P-wave Pd, tau_c,
and tau_p for passed event-station records.

Workflow
--------
1. Read station ML table from:
       output/02_report_ML_compare_Huton_none_filter/
       station_ML_compare_Seisan_Hutton_nm_HZ.csv

2. Apply QC:
       residual = ML_station_mean - header_ML
       keep if |residual - mean(residual)| <= 2 * std(residual)

   plus optional:
       ML_H_std <= MAX_H_STD
       |ML_Z_minus_H_mean| <= MAX_ABS_Z_MINUS_H

3. For passed event-station rows, read SAC files by:
       event directory + network + station + component

   No SAC file name/path is written to output tables.

4. Measure from vertical component HHZ by default:
       Pd      = zero-to-peak high-pass-filtered displacement amplitude
                 saved in both nm and cm
       tau_c   = 2*pi*sqrt(integral(u^2 dt) / integral(v^2 dt))
       tau_p   = recursive dominant-period proxy

   for P window:
       P+3 s only

   Pd follows the EEW-style convention:
       displacement high-pass filtered at 0.075 Hz before peak picking
       Pd_cm = Pd_nm * 1e-7

5. Save tables and figures to:
       output/03_report_P_amp_filter_2sdt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import read, Trace, UTCDateTime


# =====================================================================
# USER PARAMETERS
# =====================================================================

# Input station ML table from previous SEISAN-Hutton H/Z script.
STATION_ML_CSV = Path(
    "output/02_report_ML_compare_Huton_none_filter/"
    "station_ML_compare_Seisan_Hutton_nm_HZ.csv"
)

# SAC roots.
INPUT_DISP_ROOT = Path("output/02_disp_data")          # displacement SAC, unit nm
INPUT_VEL_ROOT = Path("output/01_aligned_vel_data")    # velocity SAC, unit nm/s

# Output folder requested by user.
OUTPUT_REPORT_ROOT = Path("output/03_report_P_amp_filter_2sdt")
OUTPUT_FIG_ROOT = OUTPUT_REPORT_ROOT / "figures"

# Individual waveform check figures for each accepted event-station-component.
# Requested output pattern: figures/03_pd_pick_peak_check/*/*.png
PD_PEAK_CHECK_FIG_ROOT = Path("figures/03_pd_pick_peak_check")

# Component for P-wave Pd/tau measurement.
# For EEW-style Pd, vertical P displacement is commonly used first.
P_COMPONENTS = ["HHZ"]

# Required SAC components for accepted station-event.
REQUIRED_COMPONENTS_IN_SAC = ["HHE", "HHN", "HHZ"]

# P-window lengths in seconds.
P_WINDOWS_SEC = [3.0]

# Fallback theoretical P pick if SAC P pick is missing.
VP = 6.5  # km/s

# Unit convention.
# Previous script stores displacement in nm and velocity input in nm/s.
DISP_UNIT = "nm"
VEL_UNIT = "nm/s"
DISPLACEMENT_INPUT_TO_NM = 1.0
VELOCITY_INPUT_TO_NM_PER_SEC = 1.0

# QC filtering.
USE_BIAS_CENTERED_2STD_FILTER = True
RESIDUAL_STD_MULTIPLIER = 2.0

APPLY_H_COMPONENT_STD_FILTER = True
MAX_H_STD = 0.30

APPLY_Z_MINUS_H_FILTER = True
MAX_ABS_Z_MINUS_H = 0.70

REQUIRE_HEADER_ML = True

# If False, only station rows with s_source == header are kept.
KEEP_THEORY_S_SOURCE = True

# Noise window for baseline and SNR.
NOISE_PRE_P_SEC = 10.0
NOISE_GAP_SEC = 0.5
MIN_PREP_NOISE_SEC = 2.0
MIN_P_WINDOW_SEC = 0.5

# Tau-p recursive smoothing time.
TAUP_SMOOTHING_TAU_SEC = 1.0

# Baseline options.
REMOVE_PREP_MEAN_FOR_PD = True
REMOVE_PREP_MEAN_FOR_VELOCITY = True

# ---------------------------------------------------------------------
# Pd high-pass filter and output unit.
# ---------------------------------------------------------------------
# Literature-style Pd is usually measured after removing long-period drift.
# Here we filter the displacement trace before picking Pd.
APPLY_PD_HIGHPASS_FILTER = True
PD_HIGHPASS_FREQ_HZ = 0.075
PD_HIGHPASS_CORNERS = 2
PD_HIGHPASS_ZEROPHASE = False       # False is closer to one-way real-time filtering.
PD_HIGHPASS_TAPER_MAX_PERCENTAGE = 0.05
PD_HIGHPASS_DETREND = True

# Unit conversion.
# Hutton/SEISAN ML check remains in nm.
# Pd regression output should use cm.
NM_TO_CM = 1.0e-6

# ---------------------------------------------------------------------
# Wu-equation Mpd.
# ---------------------------------------------------------------------
# Mpd is ALWAYS calculated in the output table.
# The plotting of Mpd diagnostics is optional.
WU_MPD_A = 4.748
WU_MPD_B = 1.371
WU_MPD_C = 1.883
WU_MPD_REFERENCE_COL = "header_ML"
PLOT_WU_MPD_RESULTS = True

# Plot controls.
MAKE_PLOTS = True
PLOT_USE_HEADER_ML = True       # True: x-axis = header_ML; False: ML_station_mean
PLOT_PD_UNIT = "cm"             # "cm" for EEW Pd regression plots, or "nm"
PLOT_LOG10_PD = True            # True: y-axis = log10(Pd)
PLOT_MIN_PD_NM = 1e-6           # avoid log10(0)
PLOT_MIN_PD_CM = PLOT_MIN_PD_NM * NM_TO_CM

# Individual waveform QC figure for Pd picking.
# This is the slow part because it creates one full-waveform + zoom figure
# for every station-event component. Turn it off for fast final report runs.
MAKE_PD_PEAK_CHECK_PLOTS = True
PLOT_REJECTED_WAVEFORM_CHECKS = True
PD_PEAK_CHECK_WINDOW_SEC = 3.0
PD_PEAK_PLOT_PRE_P_SEC = 1.0
PD_PEAK_PLOT_POST_P_SEC = 4.0
PD_PEAK_PLOT_DPI = 180

# SAC undefined value.
SAC_UNDEF = -12345.0

# Output files.
PASSED_CSV = OUTPUT_REPORT_ROOT / "station_event_passed_2sdt.csv"
REJECTED_CSV = OUTPUT_REPORT_ROOT / "station_event_rejected_2sdt.csv"
P_TABLE_CSV = OUTPUT_REPORT_ROOT / "P_amp_tau_Z_3s_passed_2sdt.csv"
EVENT_SUMMARY_CSV = OUTPUT_REPORT_ROOT / "event_summary_P_amp_tau_Z_3s.csv"
CONFIG_TXT = OUTPUT_REPORT_ROOT / "run_config_filter_P_amp_tau.txt"
FAIL_LOG = OUTPUT_REPORT_ROOT / "fail.log"
SUCCESS_LOG = OUTPUT_REPORT_ROOT / "success.log"


# =====================================================================
# SETUP
# =====================================================================

OUTPUT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_ROOT.mkdir(parents=True, exist_ok=True)
PD_PEAK_CHECK_FIG_ROOT.mkdir(parents=True, exist_ok=True)
FAIL_LOG.write_text("")
SUCCESS_LOG.write_text("")


def log_fail(msg: str) -> None:
    with open(FAIL_LOG, "a") as f:
        f.write(msg + "\n")


def log_success(msg: str) -> None:
    with open(SUCCESS_LOG, "a") as f:
        f.write(msg + "\n")


# =====================================================================
# SAC HEADER HELPERS
# =====================================================================

def is_defined(x) -> bool:
    if x is None:
        return False
    try:
        return float(x) != SAC_UNDEF
    except Exception:
        return False


def get_sac_header(tr: Trace, key: str, default=None):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return default
    return getattr(sac, key, default)


def get_reference_time_from_sac(tr: Trace) -> UTCDateTime:
    """
    Return absolute SAC reference time from NZ* headers.

    SAC relative headers such as O, A, T0-T9 are measured relative
    to this reference time.
    """
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return tr.stats.starttime

    nzyear = get_sac_header(tr, "nzyear")
    nzjday = get_sac_header(tr, "nzjday")
    nzhour = get_sac_header(tr, "nzhour", 0)
    nzmin = get_sac_header(tr, "nzmin", 0)
    nzsec = get_sac_header(tr, "nzsec", 0)
    nzmsec = get_sac_header(tr, "nzmsec", 0)

    vals = [nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec]
    if not all(is_defined(v) for v in vals):
        return tr.stats.starttime

    return UTCDateTime(
        year=int(nzyear),
        julday=int(nzjday),
        hour=int(nzhour),
        minute=int(nzmin),
        second=int(nzsec),
        microsecond=int(nzmsec) * 1000,
    )


def get_origin_time(tr: Trace) -> Optional[UTCDateTime]:
    o = get_sac_header(tr, "o")
    if not is_defined(o):
        return None
    return get_reference_time_from_sac(tr) + float(o)


def get_pick_abs(tr: Trace, rel_pick_sec: float) -> UTCDateTime:
    return get_reference_time_from_sac(tr) + float(rel_pick_sec)


def get_distance_km(tr: Trace) -> Optional[float]:
    dist = get_sac_header(tr, "dist")
    if is_defined(dist) and float(dist) > 0:
        return float(dist)

    gcarc = get_sac_header(tr, "gcarc")
    if is_defined(gcarc) and float(gcarc) > 0:
        return float(gcarc) * 111.19

    return None


def clean_phase_label(label) -> str:
    if label is None:
        return ""
    return str(label).strip().upper().replace(" ", "")


def classify_pick(label) -> Optional[str]:
    s = clean_phase_label(label)
    if not s:
        return None

    if "PG" in s:
        return "Pg"
    if "PN" in s:
        return "Pn"
    if s == "P" or s.startswith("P"):
        return "P"

    if "SG" in s:
        return "Sg"
    if "SN" in s:
        return "Sn"
    if s == "S" or s.startswith("S"):
        return "S"

    return None


def collect_p_pick_from_traces(
    traces: Dict[str, Trace],
) -> Tuple[Optional[UTCDateTime], str, str]:
    """
    Collect first P pick from SAC A/KA and T0-T9/KT0-KT9.

    Returns:
        p_pick_abs, p_label, p_source_component
    """
    picks: List[Tuple[UTCDateTime, str, str]] = []

    for comp, tr in traces.items():
        a = get_sac_header(tr, "a")
        ka = get_sac_header(tr, "ka")

        if is_defined(a):
            lab = classify_pick(ka)
            if lab is None:
                lab = "P"
            if lab.startswith("P"):
                picks.append((get_pick_abs(tr, float(a)), lab, comp))

        for i in range(10):
            t = get_sac_header(tr, f"t{i}")
            kt = get_sac_header(tr, f"kt{i}")
            if not is_defined(t):
                continue

            lab = classify_pick(kt)
            if lab is not None and lab.startswith("P"):
                picks.append((get_pick_abs(tr, float(t)), lab, comp))

    if not picks:
        return None, "", ""

    p_pick, p_label, p_comp = sorted(picks, key=lambda x: x[0])[0]
    return p_pick, p_label, p_comp


# =====================================================================
# SAC FILE SEARCH
# =====================================================================

def find_sac_files_for_station(
    root: Path,
    event: str,
    net: str,
    sta: str,
    components: List[str],
) -> Dict[str, Path]:
    """
    Find SAC files only through event directory + net + station + component.

    The output table does not save file paths.
    """
    event_dir = root / event
    out: Dict[str, Path] = {}

    if not event_dir.exists():
        return out

    needed = set(components)

    for f in sorted(event_dir.rglob("*.SAC")):
        if needed.issubset(out.keys()):
            break

        try:
            tr = read(str(f), headonly=True)[0]
            tr_net = str(
                getattr(tr.stats, "network", "") or get_sac_header(tr, "knetwk", "")
            ).strip()
            tr_sta = str(
                getattr(tr.stats, "station", "") or get_sac_header(tr, "kstnm", "")
            ).strip()
            tr_cha = str(
                getattr(tr.stats, "channel", "") or get_sac_header(tr, "kcmpnm", "")
            ).strip()

            if tr_net == str(net) and tr_sta == str(sta) and tr_cha in needed:
                out[tr_cha] = f

        except Exception as e:
            log_fail(f"[READ-HEAD-FAIL] {f} :: {e}")

    return out


def read_station_traces(
    root: Path,
    event: str,
    net: str,
    sta: str,
    components: List[str],
) -> Dict[str, Trace]:
    files = find_sac_files_for_station(root, event, net, sta, components)
    traces: Dict[str, Trace] = {}

    for comp in components:
        if comp not in files:
            continue

        try:
            traces[comp] = read(str(files[comp]))[0]
        except Exception as e:
            log_fail(f"[READ-FAIL] {event} {net}.{sta}.{comp} :: {e}")

    return traces


# =====================================================================
# FILTERING
# =====================================================================

def finite_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def apply_station_filters(station_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply bias-centered 2-sigma residual filter plus optional H/Z checks.
    """
    df = station_df.copy()

    required_cols = ["event", "net", "sta", "ML_station_minus_header", "header_ML"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["reject_reason"] = ""
    df["pass_filter"] = True

    header_ml = pd.to_numeric(df["header_ML"], errors="coerce")

    if REQUIRE_HEADER_ML:
        bad = ~np.isfinite(header_ml)
        df.loc[bad, "pass_filter"] = False
        df.loc[bad, "reject_reason"] += "missing_header_ML;"

    residual = pd.to_numeric(df["ML_station_minus_header"], errors="coerce")
    valid_res = residual[np.isfinite(residual)]

    if valid_res.empty:
        raise ValueError("No finite ML_station_minus_header values found.")

    res_mean = float(valid_res.mean())
    res_median = float(valid_res.median())
    res_std = float(valid_res.std(ddof=1))

    if USE_BIAS_CENTERED_2STD_FILTER:
        if not np.isfinite(res_std) or res_std <= 0:
            raise ValueError("Residual std is not positive; cannot use 2std filter.")

        limit = float(RESIDUAL_STD_MULTIPLIER) * res_std
        centered = residual - res_mean
        bad = ~np.isfinite(centered) | (centered.abs() > limit)

        df.loc[bad, "pass_filter"] = False
        df.loc[bad, "reject_reason"] += "residual_outside_bias_centered_2std;"
    else:
        limit = np.nan

    if APPLY_H_COMPONENT_STD_FILTER and "ML_H_std" in df.columns:
        hstd = pd.to_numeric(df["ML_H_std"], errors="coerce")
        bad = ~np.isfinite(hstd) | (hstd > MAX_H_STD)

        df.loc[bad, "pass_filter"] = False
        df.loc[bad, "reject_reason"] += "H_component_std_too_large;"

    if APPLY_Z_MINUS_H_FILTER and "ML_Z_minus_H_mean" in df.columns:
        zdh = pd.to_numeric(df["ML_Z_minus_H_mean"], errors="coerce")
        bad = ~np.isfinite(zdh) | (zdh.abs() > MAX_ABS_Z_MINUS_H)

        df.loc[bad, "pass_filter"] = False
        df.loc[bad, "reject_reason"] += "Z_minus_H_too_large;"

    if not KEEP_THEORY_S_SOURCE and "s_source" in df.columns:
        bad = df["s_source"].astype(str).str.lower() != "header"
        df.loc[bad, "pass_filter"] = False
        df.loc[bad, "reject_reason"] += "non_header_S_pick;"

    passed = df[df["pass_filter"]].copy()
    rejected = df[~df["pass_filter"]].copy()

    info = {
        "residual_mean": res_mean,
        "residual_median": res_median,
        "residual_std": res_std,
        "residual_lower_limit": res_mean - limit if np.isfinite(limit) else np.nan,
        "residual_upper_limit": res_mean + limit if np.isfinite(limit) else np.nan,
        "n_input": int(len(df)),
        "n_passed": int(len(passed)),
        "n_rejected": int(len(rejected)),
    }

    return passed, rejected, info


# =====================================================================
# WINDOW / TAU FUNCTIONS
# =====================================================================

def trim_data_array(tr: Trace, t0: UTCDateTime, t1: UTCDateTime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return data and relative time array for window t0-t1 without padding.
    """
    if t1 <= t0:
        return np.array([], dtype=float), np.array([], dtype=float)

    trw = tr.copy()
    trw.trim(t0, t1, pad=False)

    if trw.stats.npts < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    data = trw.data.astype(np.float64)
    t = np.arange(trw.stats.npts, dtype=float) * float(trw.stats.delta)

    return data, t


def window_mean(tr: Trace, t0: UTCDateTime, t1: UTCDateTime) -> Tuple[float, int]:
    data, _ = trim_data_array(tr, t0, t1)
    if data.size < 2 or not np.any(np.isfinite(data)):
        return np.nan, 0
    return float(np.nanmean(data)), int(data.size)


def compute_tau_c_sec(disp_nm: np.ndarray, vel_nmps: np.ndarray, dt: float) -> float:
    """
    tau_c = 2*pi*sqrt(integral(displacement^2 dt) / integral(velocity^2 dt)).
    """
    if disp_nm.size < 2 or vel_nmps.size < 2:
        return np.nan

    n = min(disp_nm.size, vel_nmps.size)
    u = disp_nm[:n].astype(np.float64)
    v = vel_nmps[:n].astype(np.float64)

    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 2:
        return np.nan

    u = u[ok]
    v = v[ok]

    num = float(np.sum(u * u) * dt)
    den = float(np.sum(v * v) * dt)

    if num <= 0 or den <= 0:
        return np.nan

    return float(2.0 * np.pi * np.sqrt(num / den))


def compute_tau_p_max_sec(
    disp_nm: np.ndarray,
    vel_nmps: np.ndarray,
    dt: float,
) -> Tuple[float, float]:
    """
    Recursive tau_p proxy.

        X_i = alpha * X_{i-1} + u_i^2
        D_i = alpha * D_{i-1} + v_i^2
        tau_p_i = 2*pi*sqrt(X_i / D_i)

    Returns:
        tau_p_max_sec, tau_p_end_sec
    """
    if disp_nm.size < 2 or vel_nmps.size < 2:
        return np.nan, np.nan

    n = min(disp_nm.size, vel_nmps.size)
    u = disp_nm[:n].astype(np.float64)
    v = vel_nmps[:n].astype(np.float64)

    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 2:
        return np.nan, np.nan

    u = u[ok]
    v = v[ok]

    tau0 = max(float(TAUP_SMOOTHING_TAU_SEC), dt)
    alpha = max(0.0, min(1.0, 1.0 - dt / tau0))

    X = 0.0
    D = 0.0
    tau_vals = []

    for uu, vv in zip(u, v):
        X = alpha * X + uu * uu
        D = alpha * D + vv * vv

        if D > 0 and X > 0:
            tau_vals.append(2.0 * np.pi * np.sqrt(X / D))

    if not tau_vals:
        return np.nan, np.nan

    arr = np.asarray(tau_vals, dtype=float)
    return float(np.nanmax(arr)), float(arr[-1])



# =====================================================================
# INDIVIDUAL PD WINDOW CHECK PLOTS
# =====================================================================

def make_scaled_trace(tr: Trace, scale: float = 1.0) -> Trace:
    """
    Return a trace copy with data converted by a scale factor.

    For displacement:
        scale = DISPLACEMENT_INPUT_TO_NM
        output data unit = nm

    For velocity:
        scale = VELOCITY_INPUT_TO_NM_PER_SEC
        output data unit = nm/s
    """
    out = tr.copy()
    out.data = out.data.astype(np.float64) * float(scale)
    return out


def highpass_trace_for_pd(tr: Trace, scale: float = 1.0) -> Trace:
    """
    Scale and high-pass filter a trace for Pd/tau measurement.

    Filtering is applied to the whole trace before the P window is cut.
    This reduces edge artifacts compared with filtering only the short window.
    """
    out = make_scaled_trace(tr, scale=scale)

    if not APPLY_PD_HIGHPASS_FILTER:
        return out

    try:
        if PD_HIGHPASS_DETREND:
            out.detrend("demean")
            out.detrend("linear")

        if PD_HIGHPASS_TAPER_MAX_PERCENTAGE and PD_HIGHPASS_TAPER_MAX_PERCENTAGE > 0:
            out.taper(max_percentage=float(PD_HIGHPASS_TAPER_MAX_PERCENTAGE))

        out.filter(
            "highpass",
            freq=float(PD_HIGHPASS_FREQ_HZ),
            corners=int(PD_HIGHPASS_CORNERS),
            zerophase=bool(PD_HIGHPASS_ZEROPHASE),
        )
    except Exception as e:
        raise RuntimeError(
            f"High-pass filter failed: freq={PD_HIGHPASS_FREQ_HZ}, "
            f"corners={PD_HIGHPASS_CORNERS}, zerophase={PD_HIGHPASS_ZEROPHASE}: {e}"
        )

    return out


def get_processed_disp_and_vel_for_pd(
    tr_disp: Trace,
    tr_vel: Trace,
) -> Tuple[Trace, Trace]:
    """
    Return displacement and velocity traces in nm / nm/s after applying
    the Pd high-pass filter.

    The same high-pass corner is applied to both displacement and velocity so
    tau_c and tau_p use consistent long-period content.
    """
    tr_disp_hp = highpass_trace_for_pd(tr_disp, scale=DISPLACEMENT_INPUT_TO_NM)
    tr_vel_hp = highpass_trace_for_pd(tr_vel, scale=VELOCITY_INPUT_TO_NM_PER_SEC)
    return tr_disp_hp, tr_vel_hp



def safe_name(x: object) -> str:
    """Make a safe string for file/folder names."""
    s = str(x).strip()
    for ch in ["/", "\\", " ", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s


def plot_pd_peak_window_check(
    tr_disp: Trace,
    event: str,
    net: str,
    sta: str,
    comp: str,
    p_pick: UTCDateTime,
    p_source: str,
    p_label: str,
    distance_km: float,
    header_ml: float,
    ml_station_mean: float,
    ml_filter_passed: bool = True,
    ml_filter_reason: str = "",
) -> None:
    """
    Plot processed displacement waveform with two stacked subfigures:
    - Top   : full processed waveform.
    - Bottom: zoom-in view around the P pick and 3-second Pd window.

    The plotted data are scaled to nm, detrended/demeaned, tapered,
    high-pass filtered, and baseline corrected. No raw/unfiltered trace is shown.
    Pd peak is picked from the same processed trace.
    """
    if not MAKE_PD_PEAK_CHECK_PLOTS:
        return

    w = float(PD_PEAK_CHECK_WINDOW_SEC)
    zoom_t0 = p_pick - float(PD_PEAK_PLOT_PRE_P_SEC)
    zoom_t1 = p_pick + float(PD_PEAK_PLOT_POST_P_SEC)

    try:
        tr_hp_full = highpass_trace_for_pd(tr_disp, scale=DISPLACEMENT_INPUT_TO_NM)
    except Exception as e:
        log_fail(f"[PLOT-FILTER-FAIL] {event} {net}.{sta}.{comp} :: {e}")
        return

    if tr_hp_full.stats.npts < 2:
        log_fail(f"[PLOT-FAIL] {event} {net}.{sta}.{comp} too few samples for full-waveform plot")
        return

    # Full processed waveform time axis.
    dt_full = float(tr_hp_full.stats.delta)
    t_rel_full = (
        np.arange(tr_hp_full.stats.npts, dtype=float) * dt_full
        + float(tr_hp_full.stats.starttime - p_pick)
    )
    hp_full = tr_hp_full.data.astype(np.float64)

    # Baseline correction on processed trace for cleaner Pd picking/plotting.
    noise_t0 = p_pick - NOISE_PRE_P_SEC
    noise_t1 = p_pick - NOISE_GAP_SEC
    hp_noise_mean, _ = window_mean(tr_hp_full, noise_t0, noise_t1)

    if REMOVE_PREP_MEAN_FOR_PD and np.isfinite(hp_noise_mean):
        hp_full_corr = hp_full - hp_noise_mean
    else:
        hp_full_corr = hp_full - hp_full[0]

    ml_filter_status = "ACCEPTED / USED" if bool(ml_filter_passed) else "REJECTED / NOT USED"
    ml_filter_color = "blue" if bool(ml_filter_passed) else "red"
    ml_filter_note = f"ML filter: {ml_filter_status}"

    # Zoom processed waveform.
    tr_hp_zoom = tr_hp_full.copy()
    tr_hp_zoom.trim(zoom_t0, zoom_t1, pad=False)

    if tr_hp_zoom.stats.npts < 2:
        log_fail(f"[PLOT-FAIL] {event} {net}.{sta}.{comp} too few samples for zoom plot")
        return

    dt_zoom = float(tr_hp_zoom.stats.delta)
    t_rel_zoom = (
        np.arange(tr_hp_zoom.stats.npts, dtype=float) * dt_zoom
        + float(tr_hp_zoom.stats.starttime - p_pick)
    )
    hp_zoom = tr_hp_zoom.data.astype(np.float64)

    if REMOVE_PREP_MEAN_FOR_PD and np.isfinite(hp_noise_mean):
        hp_zoom_corr = hp_zoom - hp_noise_mean
    else:
        hp_zoom_corr = hp_zoom - hp_zoom[0]

    # Pd peak from processed/high-pass displacement in P to P+3 s.
    mask_zoom = (t_rel_zoom >= 0.0) & (t_rel_zoom <= w) & np.isfinite(hp_zoom_corr)
    if mask_zoom.sum() < 2:
        log_fail(f"[PLOT-FAIL] {event} {net}.{sta}.{comp} no valid samples in P+{w:g}s window")
        return

    t_win = t_rel_zoom[mask_zoom]
    y_win = hp_zoom_corr[mask_zoom]
    i_peak = int(np.nanargmax(np.abs(y_win)))

    peak_t_rel = float(t_win[i_peak])
    peak_signed_nm = float(y_win[i_peak])
    pd_nm = abs(peak_signed_nm)
    pd_cm = pd_nm * NM_TO_CM
    peak_abs_time = p_pick + peak_t_rel

    # Find matching full-trace point for the peak marker.
    i_full_peak = int(np.nanargmin(np.abs(t_rel_full - peak_t_rel)))
    peak_full_y = float(hp_full_corr[i_full_peak])

    out_dir = PD_PEAK_CHECK_FIG_ROOT / safe_name(event)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{safe_name(event)}_{safe_name(net)}_{safe_name(sta)}_{safe_name(comp)}_P3s_peak.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.0, 7.4), sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.15]}
    )

    filter_label = (
        f"detrend+demean+taper+HP {PD_HIGHPASS_FREQ_HZ:g} Hz, "
        f"corners={PD_HIGHPASS_CORNERS}, zerophase={PD_HIGHPASS_ZEROPHASE}"
        if APPLY_PD_HIGHPASS_FILTER else "scaled + baseline corrected only"
    )

    # Top: full processed waveform.
    ax1.plot(t_rel_full, hp_full_corr, linewidth=0.95, label=f"processed displacement ({DISP_UNIT})")
    ax1.axvline(0.0, linewidth=1.2, linestyle="--", label=f"P pick ({p_source}, {p_label})")
    ax1.axvspan(0.0, w, color="lightyellow", alpha=0.6, label=f"Pd window: P to P+{w:g}s")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.scatter([peak_t_rel], [peak_full_y], s=40, color="red", zorder=5, label="Pd peak")
    ax1.set_title(f"Full processed waveform: {event} | {net}.{sta}.{comp} | {filter_label}", fontsize=10)
    ax1.set_xlabel("Time from P pick (s)")
    ax1.set_ylabel(f"Displacement ({DISP_UNIT})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # Bottom: zoom around P pick.
    ax2.plot(t_rel_zoom, hp_zoom_corr, linewidth=1.05, label=f"processed displacement ({DISP_UNIT})")
    ax2.axvline(0.0, linewidth=1.4, linestyle="--", label=f"P pick ({p_source}, {p_label})")
    ax2.axvspan(0.0, w, color="lightyellow", alpha=0.6, label=f"Pd window: P to P+{w:g}s")
    ax2.axhline(0.0, linewidth=0.8)
    ax2.scatter([peak_t_rel], [peak_signed_nm], s=75, color="red", zorder=5, label="Pd peak")
    ax2.annotate(
        f"Pd = {pd_cm:.4e} cm\n"
        f"t = P + {peak_t_rel:.3f} s",
        xy=(peak_t_rel, peak_signed_nm),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", linewidth=0.9),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.85),
    )
    ax2.set_xlim(float(t_rel_zoom.min()), float(t_rel_zoom.max()))
    ax2.set_title(
        f"Zoom-in: R={distance_km:.2f} km | header ML={header_ml:.2f} | "
        f"MLcal={ml_station_mean:.2f} | peak={peak_abs_time.isoformat()} | "
        f"{ml_filter_note}",
        fontsize=10,
        color=ml_filter_color,
    )
    ax2.set_xlabel("Time from P pick (s)")
    ax2.set_ylabel(f"Displacement ({DISP_UNIT})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(PD_PEAK_PLOT_DPI))
    plt.close(fig)



# =====================================================================
# P-WAVE MEASUREMENT
# =====================================================================

def measure_p_metrics_for_component(
    tr_disp: Trace,
    tr_vel: Trace,
    p_pick: UTCDateTime,
    windows_sec: List[float],
) -> dict:
    """
    Measure Pd, tau_c, tau_p, and SNR for all P windows.

    Pd is picked from high-pass-filtered displacement and saved in nm and cm.
    tau_c and tau_p use high-pass-filtered displacement and velocity.
    """
    out: dict = {}

    try:
        tr_disp_proc, tr_vel_proc = get_processed_disp_and_vel_for_pd(tr_disp, tr_vel)
    except Exception as e:
        raise RuntimeError(f"Could not prepare high-pass traces for Pd/tau: {e}")

    noise_t0 = p_pick - NOISE_PRE_P_SEC
    noise_t1 = p_pick - NOISE_GAP_SEC
    noise_len = float(noise_t1 - noise_t0)

    # Noise and baseline are computed on the processed/high-pass displacement.
    disp_noise_mean, disp_noise_npts = window_mean(tr_disp_proc, noise_t0, noise_t1)
    vel_noise_mean, vel_noise_npts = window_mean(tr_vel_proc, noise_t0, noise_t1)

    out["preP_noise_window_start"] = noise_t0.isoformat()
    out["preP_noise_window_end"] = noise_t1.isoformat()
    out["preP_noise_window_sec"] = noise_len
    out["preP_noise_npts_disp"] = int(disp_noise_npts)
    out["preP_noise_npts_vel"] = int(vel_noise_npts)
    out["preP_disp_mean_nm"] = disp_noise_mean
    out["preP_vel_mean_nmps"] = vel_noise_mean
    out["Pd_filter_applied"] = bool(APPLY_PD_HIGHPASS_FILTER)
    out["Pd_highpass_freq_Hz"] = float(PD_HIGHPASS_FREQ_HZ) if APPLY_PD_HIGHPASS_FILTER else np.nan
    out["Pd_highpass_corners"] = int(PD_HIGHPASS_CORNERS) if APPLY_PD_HIGHPASS_FILTER else np.nan
    out["Pd_highpass_zerophase"] = bool(PD_HIGHPASS_ZEROPHASE) if APPLY_PD_HIGHPASS_FILTER else False

    disp_noise_data, _ = trim_data_array(tr_disp_proc, noise_t0, noise_t1)
    if (
        noise_len >= MIN_PREP_NOISE_SEC
        and disp_noise_data.size >= 2
        and np.isfinite(disp_noise_mean)
    ):
        d_noise = disp_noise_data.astype(np.float64) - disp_noise_mean
        noise_absmax_nm = float(np.nanmax(np.abs(d_noise)))
        noise_rms_nm = float(np.sqrt(np.nanmean(d_noise ** 2)))
    else:
        noise_absmax_nm = np.nan
        noise_rms_nm = np.nan

    out["preP_noise_absmax_nm"] = noise_absmax_nm
    out["preP_noise_absmax_cm"] = noise_absmax_nm * NM_TO_CM if np.isfinite(noise_absmax_nm) else np.nan
    out["preP_noise_rms_nm"] = noise_rms_nm
    out["preP_noise_rms_cm"] = noise_rms_nm * NM_TO_CM if np.isfinite(noise_rms_nm) else np.nan

    for w in windows_sec:
        wtag = f"{int(w)}s" if float(w).is_integer() else f"{w:g}s"

        t0 = p_pick
        t1 = p_pick + float(w)
        actual_sec = float(t1 - t0)

        disp_data, _ = trim_data_array(tr_disp_proc, t0, t1)
        vel_data, _ = trim_data_array(tr_vel_proc, t0, t1)

        n = min(disp_data.size, vel_data.size)

        if n < 2 or actual_sec < MIN_P_WINDOW_SEC:
            out[f"Pd_{wtag}_cm"] = np.nan
            out[f"Pd_peak_signed_{wtag}_cm"] = np.nan
            out[f"Pd_peak_time_{wtag}"] = ""
            out[f"Pd_peak_t_rel_{wtag}_sec"] = np.nan
            out[f"tau_c_{wtag}_sec"] = np.nan
            out[f"tau_p_max_{wtag}_sec"] = np.nan
            out[f"tau_p_end_{wtag}_sec"] = np.nan
            out[f"snr_absmax_{wtag}"] = np.nan
            out[f"snr_rms_{wtag}"] = np.nan
            out[f"npts_{wtag}"] = int(n)
            continue

        disp = disp_data[:n].astype(np.float64)   # already nm
        vel = vel_data[:n].astype(np.float64)     # already nm/s

        if REMOVE_PREP_MEAN_FOR_PD and np.isfinite(disp_noise_mean):
            disp_corr = disp - disp_noise_mean
        else:
            disp_corr = disp - disp[0]

        if REMOVE_PREP_MEAN_FOR_VELOCITY and np.isfinite(vel_noise_mean):
            vel_corr = vel - vel_noise_mean
        else:
            vel_corr = vel - np.nanmean(vel)

        if not np.any(np.isfinite(disp_corr)):
            pd_nm = np.nan
            pd_cm = np.nan
            pd_um = np.nan
            pd_mm = np.nan
            peak_signed_nm = np.nan
            peak_signed_cm = np.nan
            peak_time = ""
            peak_t_rel = np.nan
        else:
            idx = int(np.nanargmax(np.abs(disp_corr)))
            peak_signed_nm = float(disp_corr[idx])
            pd_nm = float(abs(peak_signed_nm))
            pd_cm = pd_nm * NM_TO_CM
            pd_um = pd_nm * 1e-3
            pd_mm = pd_nm * 1e-6
            peak_signed_cm = peak_signed_nm * NM_TO_CM
            peak_t_rel = idx * float(tr_disp_proc.stats.delta)

            trw = tr_disp_proc.copy()
            trw.trim(t0, t1, pad=False)
            peak_abs_time = trw.stats.starttime + idx * float(trw.stats.delta)
            peak_time = peak_abs_time.isoformat()

        dt = float(tr_disp_proc.stats.delta)
        tau_c = compute_tau_c_sec(disp_corr, vel_corr, dt)
        tau_p_max, tau_p_end = compute_tau_p_max_sec(disp_corr, vel_corr, dt)

        if np.isfinite(pd_nm) and np.isfinite(noise_absmax_nm) and noise_absmax_nm > 0:
            snr_absmax = pd_nm / noise_absmax_nm
        else:
            snr_absmax = np.nan

        if np.isfinite(pd_nm) and np.isfinite(noise_rms_nm) and noise_rms_nm > 0:
            snr_rms = pd_nm / noise_rms_nm
        else:
            snr_rms = np.nan

        out[f"Pd_{wtag}_cm"] = pd_cm
        out[f"Pd_peak_signed_{wtag}_cm"] = peak_signed_cm
        out[f"Pd_peak_time_{wtag}"] = peak_time
        out[f"Pd_peak_t_rel_{wtag}_sec"] = peak_t_rel
        out[f"tau_c_{wtag}_sec"] = tau_c
        out[f"tau_p_max_{wtag}_sec"] = tau_p_max
        out[f"tau_p_end_{wtag}_sec"] = tau_p_end
        out[f"snr_absmax_{wtag}"] = snr_absmax
        out[f"snr_rms_{wtag}"] = snr_rms
        out[f"npts_{wtag}"] = int(n)

    return out


def calculate_wu_mpd(pd_cm: float, distance_km: float) -> float:
    """
    Calculate Mpd from Wu-style Pd equation.

        Mpd = A + B*log10(Pd_cm) + C*log10(R_km)

    Pd must be in cm.
    Distance must be in km.
    """
    if not np.isfinite(pd_cm) or not np.isfinite(distance_km):
        return np.nan
    if pd_cm <= 0 or distance_km <= 0:
        return np.nan

    return float(
        WU_MPD_A
        + WU_MPD_B * np.log10(float(pd_cm))
        + WU_MPD_C * np.log10(float(distance_km))
    )


def add_wu_mpd_metrics(out: dict) -> dict:
    """
    Add Wu-equation Mpd and Mpd-header difference to one output row.

    This is always calculated for accepted rows when Pd and distance are valid.
    """
    distance_km = finite_float(out.get("distance_km", np.nan))
    ref_mag = finite_float(out.get(WU_MPD_REFERENCE_COL, np.nan))

    for w in P_WINDOWS_SEC:
        wtag = f"{int(w)}s" if float(w).is_integer() else f"{w:g}s"
        pd_col = f"Pd_{wtag}_cm"
        mpd_col = f"Mpd_Wu_{wtag}"
        diff_col = f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}"
        inv_diff_col = f"{WU_MPD_REFERENCE_COL}_minus_Mpd_Wu_{wtag}"

        pd_cm = finite_float(out.get(pd_col, np.nan))
        mpd = calculate_wu_mpd(pd_cm=pd_cm, distance_km=distance_km)

        out[mpd_col] = mpd

        if np.isfinite(mpd) and np.isfinite(ref_mag):
            out[diff_col] = float(mpd - ref_mag)
            out[inv_diff_col] = float(ref_mag - mpd)
        else:
            out[diff_col] = np.nan
            out[inv_diff_col] = np.nan

    return out



def process_passed_row(row: pd.Series, collect_metrics: bool = True) -> List[dict]:
    """
    Process one station-event row.

    If collect_metrics is True, save the Pd/tau result row.
    If collect_metrics is False, only make the Pd peak-check figure.
    This allows rejected stations to be plotted in red without entering
    the final accepted Pd table.
    """
    event = str(row["event"])
    net = str(row["net"])
    sta = str(row["sta"])

    disp_all = read_station_traces(
        INPUT_DISP_ROOT, event, net, sta, REQUIRED_COMPONENTS_IN_SAC
    )
    vel_all = read_station_traces(
        INPUT_VEL_ROOT, event, net, sta, REQUIRED_COMPONENTS_IN_SAC
    )

    missing_disp = [c for c in REQUIRED_COMPONENTS_IN_SAC if c not in disp_all]
    missing_vel = [c for c in REQUIRED_COMPONENTS_IN_SAC if c not in vel_all]

    if missing_disp:
        log_fail(f"[MISS-DISP] {event} {net}.{sta} missing {missing_disp}")
        return []

    if missing_vel:
        log_fail(f"[MISS-VEL] {event} {net}.{sta} missing {missing_vel}")
        return []

    p_pick, p_label, p_comp = collect_p_pick_from_traces(disp_all)
    p_source = "header"

    ref_comp = P_COMPONENTS[0] if P_COMPONENTS[0] in disp_all else REQUIRED_COMPONENTS_IN_SAC[0]
    ref_tr = disp_all[ref_comp]

    origin = get_origin_time(ref_tr)

    distance_km = finite_float(row.get("distance_km", np.nan))
    if not np.isfinite(distance_km):
        d = get_distance_km(ref_tr)
        distance_km = float(d) if d is not None else np.nan

    if p_pick is None:
        if origin is None or not np.isfinite(distance_km) or distance_km <= 0:
            log_fail(f"[NO-P-PICK] {event} {net}.{sta} no header P and cannot use theory")
            return []

        p_pick = origin + float(distance_km) / float(VP)
        p_label = "P"
        p_comp = "theory"
        p_source = "theory"

    rows: List[dict] = []

    for comp in P_COMPONENTS:
        if comp not in disp_all or comp not in vel_all:
            log_fail(f"[MISS-P-COMP] {event} {net}.{sta}.{comp}")
            continue

        tr_disp = disp_all[comp]
        tr_vel = vel_all[comp]

        if collect_metrics:
            metrics = measure_p_metrics_for_component(
                tr_disp=tr_disp,
                tr_vel=tr_vel,
                p_pick=p_pick,
                windows_sec=P_WINDOWS_SEC,
            )
        else:
            metrics = {}

        ml_filter_passed = bool(row.get("pass_filter", True))
        ml_filter_reason = str(row.get("reject_reason", "") or "")

        if MAKE_PD_PEAK_CHECK_PLOTS:
            plot_pd_peak_window_check(
                tr_disp=tr_disp,
                event=event,
                net=net,
                sta=sta,
                comp=comp,
                p_pick=p_pick,
                p_source=p_source,
                p_label=p_label,
                distance_km=distance_km,
                header_ml=finite_float(row.get("header_ML", np.nan)),
                ml_station_mean=finite_float(row.get("ML_station_mean", np.nan)),
                ml_filter_passed=ml_filter_passed,
                ml_filter_reason=ml_filter_reason,
            )

        if not collect_metrics:
            continue

        out = {
            "event": event,
            "net": net,
            "sta": sta,
            "comp": comp,
            "distance_km": distance_km,

            "header_ML": finite_float(row.get("header_ML", np.nan)),
            "ML_station_mean": finite_float(row.get("ML_station_mean", np.nan)),
            "ML_H_mean": finite_float(row.get("ML_H_mean", np.nan)),
            "ML_H_std": finite_float(row.get("ML_H_std", np.nan)),
            "ML_Z": finite_float(row.get("ML_Z", np.nan)),
            "ML_Z_minus_H_mean": finite_float(row.get("ML_Z_minus_H_mean", np.nan)),
            "ML_station_minus_header": finite_float(row.get("ML_station_minus_header", np.nan)),

            "p_pick": p_pick.isoformat(),
            "p_label": p_label,
            "p_source": p_source,
            "p_source_component": p_comp,

            "disp_unit": DISP_UNIT,
            "vel_unit": VEL_UNIT,
            "remove_preP_mean_for_Pd": bool(REMOVE_PREP_MEAN_FOR_PD),
            "taup_smoothing_tau_sec": float(TAUP_SMOOTHING_TAU_SEC),
        }

        out.update(metrics)
        out = add_wu_mpd_metrics(out)
        rows.append(out)

    if rows:
        log_success(
            f"[OK] {event} {net}.{sta} P={p_pick.isoformat()} "
            f"source={p_source} comps={P_COMPONENTS}"
        )

    return rows


# =====================================================================
# SUMMARY TABLE
# =====================================================================

def make_event_summary(p_df: pd.DataFrame) -> pd.DataFrame:
    if p_df.empty:
        return pd.DataFrame()

    agg = {
        "header_ML": "mean",
        "ML_station_mean": ["mean", "median", "std", "count"],
        "distance_km": ["min", "max", "mean"],
    }

    for w in P_WINDOWS_SEC:
        wtag = f"{int(w)}s" if float(w).is_integer() else f"{w:g}s"

        for col in [
            f"Pd_{wtag}_cm",
            f"Pd_peak_t_rel_{wtag}_sec",
            f"Mpd_Wu_{wtag}",
            f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}",
            f"{WU_MPD_REFERENCE_COL}_minus_Mpd_Wu_{wtag}",
            f"tau_c_{wtag}_sec",
            f"tau_p_max_{wtag}_sec",
            f"tau_p_end_{wtag}_sec",
            f"snr_absmax_{wtag}",
            f"snr_rms_{wtag}",
        ]:
            if col in p_df.columns:
                agg[col] = ["mean", "median", "std"]

    event_df = p_df.groupby("event").agg(agg)
    event_df.columns = [
        "_".join([c for c in col if c]).strip("_")
        for col in event_df.columns.values
    ]
    event_df = event_df.reset_index()

    event_df = event_df.rename(
        columns={
            "ML_station_mean_mean": "ML_ref_mean",
            "ML_station_mean_median": "ML_ref_median",
            "ML_station_mean_std": "ML_ref_std",
            "ML_station_mean_count": "n_station",
        }
    )

    return event_df


# =====================================================================
# PLOT FUNCTIONS
# =====================================================================

def _format_window_tag(w: float) -> str:
    return f"{int(w)}s" if float(w).is_integer() else f"{w:g}s"


def get_pd_column_for_plot(df: pd.DataFrame, wtag: str) -> Tuple[Optional[str], str, float]:
    """
    Select Pd column and unit for report plots.

    Returns:
        pd_col, pd_unit, min_value
    """
    if f"Pd_{wtag}_cm" in df.columns:
        return f"Pd_{wtag}_cm", "cm", float(PLOT_MIN_PD_CM)

    return None, "cm", np.nan


def _prepare_pd_for_plot(df: pd.DataFrame, pd_col: str, min_pd_value: float) -> pd.DataFrame:
    """
    Prepare Pd column for plotting.
    """
    tmp = df.copy()
    tmp[pd_col] = pd.to_numeric(tmp[pd_col], errors="coerce")
    tmp = tmp[np.isfinite(tmp[pd_col])]
    tmp = tmp[tmp[pd_col] > 0]

    if PLOT_LOG10_PD:
        tmp = tmp[tmp[pd_col] >= float(min_pd_value)]
        tmp["Pd_plot"] = np.log10(tmp[pd_col].astype(float))
    else:
        tmp["Pd_plot"] = tmp[pd_col].astype(float)

    return tmp


def plot_pd_vs_ml(p_df: pd.DataFrame) -> None:
    """
    Plot Pd amplitude versus ML for each P window.
    """
    if p_df.empty:
        return

    ml_col = "header_ML" if PLOT_USE_HEADER_ML else "ML_station_mean"
    ml_label = "Header ML" if PLOT_USE_HEADER_ML else "Calculated station ML"

    if ml_col not in p_df.columns:
        return

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        pd_col, pd_unit, min_pd_value = get_pd_column_for_plot(p_df, wtag)

        if pd_col is None or pd_col not in p_df.columns:
            continue

        tmp = p_df[[ml_col, pd_col, "distance_km"]].copy()
        tmp[ml_col] = pd.to_numeric(tmp[ml_col], errors="coerce")
        tmp = tmp[np.isfinite(tmp[ml_col])]
        tmp = _prepare_pd_for_plot(tmp, pd_col, min_pd_value)

        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(tmp[ml_col], tmp["Pd_plot"], s=18, alpha=0.75)

        ax.set_xlabel(ml_label)
        if PLOT_LOG10_PD:
            ax.set_ylabel(f"log10(Pd {wtag}) [{pd_unit}]")
        else:
            ax.set_ylabel(f"Pd {wtag} [{pd_unit}]")

        ax.set_title(f"Pd {wtag} amplitude versus {ml_label}")
        ax.grid(True, alpha=0.3)

        # Optional simple trend line.
        x = tmp[ml_col].to_numpy(dtype=float)
        y = tmp["Pd_plot"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            coef = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, linewidth=1.2, label=f"fit: y={coef[0]:.3f}x+{coef[1]:.3f}")
            ax.legend()

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Pd_{wtag}_vs_ML.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_pd_vs_distance(p_df: pd.DataFrame) -> None:
    """
    Plot Pd amplitude versus distance for each P window.
    """
    if p_df.empty:
        return

    if "distance_km" not in p_df.columns:
        return

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        pd_col, pd_unit, min_pd_value = get_pd_column_for_plot(p_df, wtag)

        if pd_col is None or pd_col not in p_df.columns:
            continue

        tmp = p_df[["distance_km", pd_col, "header_ML"]].copy()
        tmp["distance_km"] = pd.to_numeric(tmp["distance_km"], errors="coerce")
        tmp = tmp[np.isfinite(tmp["distance_km"])]
        tmp = _prepare_pd_for_plot(tmp, pd_col, min_pd_value)

        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(tmp["distance_km"], tmp["Pd_plot"], s=18, alpha=0.75)

        ax.set_xlabel("Distance [km]")
        if PLOT_LOG10_PD:
            ax.set_ylabel(f"log10(Pd {wtag}) [{pd_unit}]")
        else:
            ax.set_ylabel(f"Pd {wtag} [{pd_unit}]")

        ax.set_title(f"Pd {wtag} amplitude versus distance")
        ax.grid(True, alpha=0.3)

        # Optional simple trend line.
        x = tmp["distance_km"].to_numpy(dtype=float)
        y = tmp["Pd_plot"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            coef = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, linewidth=1.2, label=f"fit: y={coef[0]:.4f}x+{coef[1]:.3f}")
            ax.legend()

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Pd_{wtag}_vs_distance.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_pd_vs_ml_by_distance_color(p_df: pd.DataFrame) -> None:
    """
    Plot Pd versus ML with distance as color.
    This helps see if Pd-ML relation is biased by distance.
    """
    if p_df.empty:
        return

    ml_col = "header_ML" if PLOT_USE_HEADER_ML else "ML_station_mean"
    ml_label = "Header ML" if PLOT_USE_HEADER_ML else "Calculated station ML"

    if ml_col not in p_df.columns or "distance_km" not in p_df.columns:
        return

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        pd_col, pd_unit, min_pd_value = get_pd_column_for_plot(p_df, wtag)

        if pd_col is None or pd_col not in p_df.columns:
            continue

        tmp = p_df[[ml_col, "distance_km", pd_col]].copy()
        tmp[ml_col] = pd.to_numeric(tmp[ml_col], errors="coerce")
        tmp["distance_km"] = pd.to_numeric(tmp["distance_km"], errors="coerce")
        tmp = tmp[np.isfinite(tmp[ml_col]) & np.isfinite(tmp["distance_km"])]
        tmp = _prepare_pd_for_plot(tmp, pd_col, min_pd_value)

        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sc = ax.scatter(
            tmp[ml_col],
            tmp["Pd_plot"],
            c=tmp["distance_km"],
            s=18,
            alpha=0.80,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Distance [km]")

        ax.set_xlabel(ml_label)
        if PLOT_LOG10_PD:
            ax.set_ylabel(f"log10(Pd {wtag}) [{pd_unit}]")
        else:
            ax.set_ylabel(f"Pd {wtag} [{pd_unit}]")

        ax.set_title(f"Pd {wtag} versus {ml_label}, colored by distance")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Pd_{wtag}_vs_ML_color_distance.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_residual_hist_passed_rejected(passed: pd.DataFrame, rejected: pd.DataFrame) -> None:
    """
    Plot residual histogram for passed and rejected station-event records.
    """
    if passed.empty and rejected.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if not passed.empty and "ML_station_minus_header" in passed.columns:
        vals = pd.to_numeric(passed["ML_station_minus_header"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.70, label="passed")

    if not rejected.empty and "ML_station_minus_header" in rejected.columns:
        vals = pd.to_numeric(rejected["ML_station_minus_header"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.45, label="rejected")

    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("ML_station_mean - header_ML")
    ax.set_ylabel("Count")
    ax.set_title("Station ML residual: passed versus rejected")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_FIG_ROOT / "ML_residual_hist_passed_rejected.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_wu_mpd_vs_reference(p_df: pd.DataFrame) -> None:
    """
    Plot Wu Mpd versus reference magnitude.
    """
    if p_df.empty or WU_MPD_REFERENCE_COL not in p_df.columns:
        return

    ref = pd.to_numeric(p_df[WU_MPD_REFERENCE_COL], errors="coerce")

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        mpd_col = f"Mpd_Wu_{wtag}"

        if mpd_col not in p_df.columns:
            continue

        mpd = pd.to_numeric(p_df[mpd_col], errors="coerce")
        ok = np.isfinite(ref) & np.isfinite(mpd)

        if ok.sum() < 3:
            continue

        x = ref[ok].to_numpy(dtype=float)
        y = mpd[ok].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.scatter(x, y, s=20, alpha=0.75)

        lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        pad = 0.25
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=1.2, label="1:1")

        residual = y - x
        med = float(np.nanmedian(residual))
        mean = float(np.nanmean(residual))
        std = float(np.nanstd(residual, ddof=1)) if residual.size > 1 else np.nan

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(WU_MPD_REFERENCE_COL)
        ax.set_ylabel(f"Wu Mpd {wtag}")
        ax.set_title(
            f"Wu Mpd {wtag} versus {WU_MPD_REFERENCE_COL}\n"
            f"median(Mpd-ref)={med:.3f}, mean={mean:.3f}, std={std:.3f}, n={ok.sum()}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Mpd_Wu_{wtag}_vs_{WU_MPD_REFERENCE_COL}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_wu_mpd_residual_hist(p_df: pd.DataFrame) -> None:
    """
    Plot histogram of Wu Mpd minus reference magnitude.
    """
    if p_df.empty:
        return

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        diff_col = f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}"

        if diff_col not in p_df.columns:
            continue

        vals = pd.to_numeric(p_df[diff_col], errors="coerce")
        vals = vals[np.isfinite(vals)]

        if len(vals) < 3:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        ax.hist(vals, bins=35, alpha=0.75)
        ax.axvline(0.0, linewidth=1.0)
        ax.axvline(float(np.nanmedian(vals)), linestyle="--", linewidth=1.1, label="median")
        ax.set_xlabel(f"Wu Mpd {wtag} - {WU_MPD_REFERENCE_COL}")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Wu Mpd residual histogram ({wtag})\n"
            f"median={np.nanmedian(vals):.3f}, mean={np.nanmean(vals):.3f}, "
            f"std={np.nanstd(vals, ddof=1):.3f}, n={len(vals)}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}_hist.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_wu_mpd_residual_vs_distance(p_df: pd.DataFrame) -> None:
    """
    Plot Wu Mpd minus reference magnitude versus distance.
    """
    if p_df.empty or "distance_km" not in p_df.columns:
        return

    dist = pd.to_numeric(p_df["distance_km"], errors="coerce")

    for w in P_WINDOWS_SEC:
        wtag = _format_window_tag(w)
        diff_col = f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}"

        if diff_col not in p_df.columns:
            continue

        diff = pd.to_numeric(p_df[diff_col], errors="coerce")
        ok = np.isfinite(dist) & np.isfinite(diff)

        if ok.sum() < 3:
            continue

        x = dist[ok].to_numpy(dtype=float)
        y = diff[ok].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        ax.scatter(x, y, s=20, alpha=0.75)
        ax.axhline(0.0, linewidth=1.0)

        ax.set_xlabel("Distance [km]")
        ax.set_ylabel(f"Wu Mpd {wtag} - {WU_MPD_REFERENCE_COL}")
        ax.set_title(f"Wu Mpd residual versus distance ({wtag})")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = OUTPUT_FIG_ROOT / f"Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}_vs_distance.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def plot_wu_mpd_results(p_df: pd.DataFrame) -> None:
    """
    Generate all Wu Mpd diagnostic plots.
    """
    if not PLOT_WU_MPD_RESULTS:
        return

    plot_wu_mpd_vs_reference(p_df)
    plot_wu_mpd_residual_hist(p_df)
    plot_wu_mpd_residual_vs_distance(p_df)



def make_all_plots(
    passed_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    p_df: pd.DataFrame,
) -> None:
    """
    Generate all report figures.
    """
    OUTPUT_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    PD_PEAK_CHECK_FIG_ROOT.mkdir(parents=True, exist_ok=True)

    plot_residual_hist_passed_rejected(passed_df, rejected_df)
    plot_pd_vs_ml(p_df)
    plot_pd_vs_distance(p_df)
    plot_pd_vs_ml_by_distance_color(p_df)
    plot_wu_mpd_results(p_df)


# =====================================================================
# CONFIG
# =====================================================================

def write_config(filter_info: dict) -> None:
    with open(CONFIG_TXT, "w") as f:
        f.write("Filter and estimate P-wave Pd/tau parameters\n")
        f.write(f"Station ML CSV: {STATION_ML_CSV}\n")
        f.write(f"Input displacement root: {INPUT_DISP_ROOT}\n")
        f.write(f"Input velocity root: {INPUT_VEL_ROOT}\n")
        f.write(f"Output report root: {OUTPUT_REPORT_ROOT}\n")
        f.write(f"Output figure root: {OUTPUT_FIG_ROOT}\n")
        f.write(f"Pd peak check figure root: {PD_PEAK_CHECK_FIG_ROOT}\n")
        f.write(f"P components: {P_COMPONENTS}\n")
        f.write(f"Required components in SAC: {REQUIRED_COMPONENTS_IN_SAC}\n")
        f.write(f"P windows sec: {P_WINDOWS_SEC}\n")
        f.write(f"MAKE_PD_PEAK_CHECK_PLOTS: {MAKE_PD_PEAK_CHECK_PLOTS}\n")
        f.write(f"PLOT_REJECTED_WAVEFORM_CHECKS: {PLOT_REJECTED_WAVEFORM_CHECKS}\n")
        f.write(f"PD_PEAK_CHECK_WINDOW_SEC: {PD_PEAK_CHECK_WINDOW_SEC}\n")
        f.write(f"PD_PEAK_PLOT_PRE_P_SEC: {PD_PEAK_PLOT_PRE_P_SEC}\n")
        f.write(f"PD_PEAK_PLOT_POST_P_SEC: {PD_PEAK_PLOT_POST_P_SEC}\n")
        f.write(f"VP fallback: {VP}\n")
        f.write(f"Displacement unit: {DISP_UNIT}\n")
        f.write(f"Velocity unit: {VEL_UNIT}\n")
        f.write(f"USE_BIAS_CENTERED_2STD_FILTER: {USE_BIAS_CENTERED_2STD_FILTER}\n")
        f.write(f"RESIDUAL_STD_MULTIPLIER: {RESIDUAL_STD_MULTIPLIER}\n")
        f.write(f"APPLY_H_COMPONENT_STD_FILTER: {APPLY_H_COMPONENT_STD_FILTER}\n")
        f.write(f"MAX_H_STD: {MAX_H_STD}\n")
        f.write(f"APPLY_Z_MINUS_H_FILTER: {APPLY_Z_MINUS_H_FILTER}\n")
        f.write(f"MAX_ABS_Z_MINUS_H: {MAX_ABS_Z_MINUS_H}\n")
        f.write(f"REQUIRE_HEADER_ML: {REQUIRE_HEADER_ML}\n")
        f.write(f"KEEP_THEORY_S_SOURCE: {KEEP_THEORY_S_SOURCE}\n")
        f.write(f"NOISE_PRE_P_SEC: {NOISE_PRE_P_SEC}\n")
        f.write(f"NOISE_GAP_SEC: {NOISE_GAP_SEC}\n")
        f.write(f"MIN_PREP_NOISE_SEC: {MIN_PREP_NOISE_SEC}\n")
        f.write(f"MIN_P_WINDOW_SEC: {MIN_P_WINDOW_SEC}\n")
        f.write(f"TAUP_SMOOTHING_TAU_SEC: {TAUP_SMOOTHING_TAU_SEC}\n")
        f.write(f"REMOVE_PREP_MEAN_FOR_PD: {REMOVE_PREP_MEAN_FOR_PD}\n")
        f.write(f"REMOVE_PREP_MEAN_FOR_VELOCITY: {REMOVE_PREP_MEAN_FOR_VELOCITY}\n")
        f.write(f"APPLY_PD_HIGHPASS_FILTER: {APPLY_PD_HIGHPASS_FILTER}\n")
        f.write(f"PD_HIGHPASS_FREQ_HZ: {PD_HIGHPASS_FREQ_HZ}\n")
        f.write(f"PD_HIGHPASS_CORNERS: {PD_HIGHPASS_CORNERS}\n")
        f.write(f"PD_HIGHPASS_ZEROPHASE: {PD_HIGHPASS_ZEROPHASE}\n")
        f.write(f"PD_HIGHPASS_TAPER_MAX_PERCENTAGE: {PD_HIGHPASS_TAPER_MAX_PERCENTAGE}\n")
        f.write(f"PD_HIGHPASS_DETREND: {PD_HIGHPASS_DETREND}\n")
        f.write(f"NM_TO_CM: {NM_TO_CM}\n")
        f.write(f"WU_MPD_A: {WU_MPD_A}\n")
        f.write(f"WU_MPD_B: {WU_MPD_B}\n")
        f.write(f"WU_MPD_C: {WU_MPD_C}\n")
        f.write(f"WU_MPD_REFERENCE_COL: {WU_MPD_REFERENCE_COL}\n")
        f.write(f"PLOT_WU_MPD_RESULTS: {PLOT_WU_MPD_RESULTS}\n")
        f.write(f"MAKE_PLOTS: {MAKE_PLOTS}\n")
        f.write(f"PLOT_USE_HEADER_ML: {PLOT_USE_HEADER_ML}\n")
        f.write(f"PLOT_PD_UNIT: {PLOT_PD_UNIT}\n")
        f.write(f"PLOT_LOG10_PD: {PLOT_LOG10_PD}\n")
        f.write(f"PLOT_MIN_PD_NM: {PLOT_MIN_PD_NM}\n")
        f.write(f"PLOT_MIN_PD_CM: {PLOT_MIN_PD_CM}\n")

        f.write("\nFilter statistics:\n")
        for k, v in filter_info.items():
            f.write(f"{k}: {v}\n")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    if not STATION_ML_CSV.exists():
        print(f"[ERROR] Missing station ML table: {STATION_ML_CSV}")
        return

    print(f"[INFO] Reading station ML table: {STATION_ML_CSV}")
    station_df = pd.read_csv(STATION_ML_CSV)

    passed_df, rejected_df, filter_info = apply_station_filters(station_df)

    passed_df.to_csv(PASSED_CSV, index=False)
    rejected_df.to_csv(REJECTED_CSV, index=False)
    write_config(filter_info)

    print("========== FILTER SUMMARY ==========")
    for k, v in filter_info.items():
        print(f"{k}: {v}")

    print(f"Passed table  : {PASSED_CSV}")
    print(f"Rejected table: {REJECTED_CSV}")

    if passed_df.empty:
        print("[ERROR] No passed rows after filtering.")
        return

    all_rows: List[dict] = []

    if MAKE_PD_PEAK_CHECK_PLOTS and PLOT_REJECTED_WAVEFORM_CHECKS and not rejected_df.empty:
        print("[INFO] Making Pd peak-check plots for rejected station-event rows...")
        for _, row in rejected_df.iterrows():
            process_passed_row(row, collect_metrics=False)

    print("[INFO] Measuring P-wave Pd, tau_c, and tau_p for accepted station-event rows...")

    for _, row in passed_df.iterrows():
        rows = process_passed_row(row, collect_metrics=True)
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No P-wave rows produced. Check fail log.")
        print(f"Fail log: {FAIL_LOG}")
        return

    p_df = pd.DataFrame(all_rows)
    p_df = p_df.replace([np.inf, -np.inf], np.nan)
    p_df.to_csv(P_TABLE_CSV, index=False)

    event_df = make_event_summary(p_df)
    if not event_df.empty:
        event_df.to_csv(EVENT_SUMMARY_CSV, index=False)

    if MAKE_PLOTS:
        print("[INFO] Making report plots...")
        make_all_plots(passed_df, rejected_df, p_df)

    print("========== DONE ==========")
    print(f"Output root      : {OUTPUT_REPORT_ROOT}")
    print(f"Figure root      : {OUTPUT_FIG_ROOT}")
    print(f"Pd peak figures  : {PD_PEAK_CHECK_FIG_ROOT}")
    print(f"P table          : {P_TABLE_CSV}")
    print(f"Event summary    : {EVENT_SUMMARY_CSV}")
    print(f"Success log      : {SUCCESS_LOG}")
    print(f"Fail log         : {FAIL_LOG}")
    print(f"Run config       : {CONFIG_TXT}")
    print(f"P rows produced  : {len(p_df)}")

    if not event_df.empty:
        print(f"Event rows       : {len(event_df)}")

    if MAKE_PLOTS:
        print("Figures:")
        print(f"  {OUTPUT_FIG_ROOT / 'ML_residual_hist_passed_rejected.png'}")
        for w in P_WINDOWS_SEC:
            wtag = _format_window_tag(w)
            print(f"  {OUTPUT_FIG_ROOT / f'Pd_{wtag}_vs_ML.png'}")
            print(f"  {OUTPUT_FIG_ROOT / f'Pd_{wtag}_vs_distance.png'}")
            print(f"  {OUTPUT_FIG_ROOT / f'Pd_{wtag}_vs_ML_color_distance.png'}")
            if PLOT_WU_MPD_RESULTS:
                print(f"  {OUTPUT_FIG_ROOT / f'Mpd_Wu_{wtag}_vs_{WU_MPD_REFERENCE_COL}.png'}")
                print(f"  {OUTPUT_FIG_ROOT / f'Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}_hist.png'}")
                print(f"  {OUTPUT_FIG_ROOT / f'Mpd_Wu_{wtag}_minus_{WU_MPD_REFERENCE_COL}_vs_distance.png'}")


if __name__ == "__main__":
    main()