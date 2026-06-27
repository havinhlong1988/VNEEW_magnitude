#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEISAN/Nordic-compatible Hutton ML check, no Wood-Anderson simulation.

Purpose
-------
Use the practical SEISAN/Nordic IAML convention for local magnitude:

    response-removed velocity SAC, assumed in nm/s
        -> integrate to ground displacement in nm
        -> save displacement SAC files in output/02_disp_data
        -> require station has HHE, HHN, and HHZ for ML checking
        -> measure zero-to-peak S-window ground displacement amplitude in nm
        -> calculate component ML with SEISAN Hutton form
        -> calculate station horizontal ML from HHE/HHN
        -> calculate event ML with standard deviation across stations
        -> compare calculated ML with header ML

Important convention
--------------------
This script does NOT use the Hutton & Boore Wood-Anderson-mm amplitude form.
It uses the SEISAN/Nordic IAML-equivalent form:

    ML = log10(A_nm) + 1.11 log10(R_km) + 0.00189 R_km - 2.09

where:
    A_nm = zero-to-peak ground displacement amplitude in nanometers
    R_km = distance in km from SAC dist or gcarc

The constant -2.09 includes the Wood-Anderson magnification / unit conversion
used by the SEISAN convention, so do NOT convert the measured amplitude to WA mm
before calling this formula.

Outputs
-------
Displacement SAC files:
    output/02_disp_data/<same_subdir>/<same_file_name>.SAC

Reports:
    output/02_report_ML_compare_Huton_none_filter/component_ML_compare_Seisan_Hutton_nm_HZ.csv
    output/02_report_ML_compare_Huton_none_filter/station_ML_compare_Seisan_Hutton_nm_HZ.csv
    output/02_report_ML_compare_Huton_none_filter/event_ML_compare_Seisan_Hutton_nm_HZ.csv
    output/02_report_ML_compare_Huton_none_filter/event_component_ML_compare_Seisan_Hutton_nm_HZ.csv

Figures:
    figures/02_report_ML_compare_Huton_none_filter/*.png
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import read, Trace, UTCDateTime


# ========================= USER PARAMETERS ========================= #

# Input: velocity SAC files from your previous cutting script.
INPUT_VEL_ROOT = Path("output/01_aligned_vel_data")

# Output: displacement SAC files, same subdirectory structure and same file names.
OUTPUT_DISP_ROOT = Path("output/02_disp_data")

# Output: ML comparison reports.
# Kept as requested, including original "Huton" spelling.
OUTPUT_REPORT_ROOT = Path("output/02_report_ML_compare_Huton_none_filter")
OUTPUT_FIG_ROOT = Path("figures/02_report_ML_compare_Huton_none_filter")

# A station is accepted only if it has all of these components.
# This gives both horizontal and vertical checks.
REQUIRED_STATION_COMPONENTS = ["HHE", "HHN", "HHZ"]

# Component-level ML rows to write.
COMPONENTS_FOR_ML = ["HHE", "HHN", "HHZ"]

# Standard station/event ML uses horizontal components only.
HORIZONTAL_COMPONENTS_FOR_STATION_ML = ["HHE", "HHN"]
VERTICAL_COMPONENT = "HHZ"

# Velocity model for fallback theoretical arrivals if header picks are missing.
VP = 6.5        # km/s
VS = 3.5        # km/s
VSURFACE = 1.5  # km/s

# Amplitude measurement window.
# Window = S pick + AMP_AFTER_S_SEC to surface arrival - AMP_BEFORE_SURF_SEC.
AMP_AFTER_S_SEC = 0.0
AMP_BEFORE_SURF_SEC = 2.0
MIN_AMP_WINDOW_SEC = 5.0

# ========================== UNIT CONTROL =========================== #

# SEISAN/Nordic IAML convention uses ground displacement amplitude in nm.
# Therefore this script saves integrated displacement as nanometers.
#
# If input velocity SAC data are already in nm/s, keep this as 1.0.
# If input velocity SAC data are in m/s, use 1e9.
# If input velocity SAC data are in micrometer/s, use 1000.0.
# If input velocity SAC data are in mm/s, use 1e6.
VELOCITY_INPUT_TO_NM_PER_SEC = 1.0

# Output displacement unit for SAC data.
DISPLACEMENT_OUTPUT_UNIT = "nm"

# SEISAN/Hutton settings.
ML_FORMULA_NAME = "SEISAN_HUTTON_NM"
WA_STATIC_MAGNIFICATION = 2080.0

# No extra filter is applied by default.
# Your report name says "none_filter"; keep APPLY_FILTER_FOR_ML=False.
APPLY_FILTER_FOR_ML = False
FILTER_FREQ_HZ = 0.05
FILTER_CORNERS = 4
FILTER_ZEROPHASE = True

# Integration stabilization.
# These are baseline operations, not bandpass/highpass filtering.
DEMEAN_VELOCITY_BEFORE_INTEGRATE = True
DETREND_VELOCITY_BEFORE_INTEGRATE = True
TAPER_VELOCITY_BEFORE_INTEGRATE = False
TAPER_MAX_PERCENTAGE = 0.05

# Usually keep these False so the saved displacement is a direct integral.
DEMEAN_DISPLACEMENT_AFTER_INTEGRATE = False
DETREND_DISPLACEMENT_AFTER_INTEGRATE = False

# For amplitude picking, subtract the local mean inside the amplitude window.
# This removes arbitrary integration offset without applying any frequency filter.
REMOVE_WINDOW_MEAN_FOR_AMPLITUDE = True

# SAC header magnitude source.
# Most SAC files use "mag". If your header ML is stored elsewhere, change this
# to e.g. "user0", "user1", etc.
HEADER_ML_KEY = "mag"

# Event directory pattern. Use "20*" if event folders all start with year.
EVENT_DIR_GLOB = "*"

# SAC undefined value.
SAC_UNDEF = -12345.0

# Output files.
SUCCESS_LOG = OUTPUT_REPORT_ROOT / "success.log"
FAIL_LOG = OUTPUT_REPORT_ROOT / "fail.log"
CONVERSION_LOG = OUTPUT_REPORT_ROOT / "conversion.log"

COMPONENT_CSV = OUTPUT_REPORT_ROOT / "component_ML_compare_Seisan_Hutton_nm_HZ.csv"
STATION_CSV = OUTPUT_REPORT_ROOT / "station_ML_compare_Seisan_Hutton_nm_HZ.csv"
EVENT_CSV = OUTPUT_REPORT_ROOT / "event_ML_compare_Seisan_Hutton_nm_HZ.csv"
EVENT_COMPONENT_CSV = OUTPUT_REPORT_ROOT / "event_component_ML_compare_Seisan_Hutton_nm_HZ.csv"

# Figures.
MAKE_FIGURES = True
FIG_DPI = 300

# ================================================================== #


for p in [OUTPUT_DISP_ROOT, OUTPUT_REPORT_ROOT, OUTPUT_FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

for p in [SUCCESS_LOG, FAIL_LOG, CONVERSION_LOG]:
    p.write_text("")


def log_success(msg: str) -> None:
    with open(SUCCESS_LOG, "a") as f:
        f.write(msg + "\n")


def log_fail(msg: str) -> None:
    with open(FAIL_LOG, "a") as f:
        f.write(msg + "\n")


def log_conversion(msg: str) -> None:
    with open(CONVERSION_LOG, "a") as f:
        f.write(msg + "\n")


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

    SAC picks A/T0/T1/... are relative to this reference time, not necessarily
    relative to trace starttime.
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

    if not all(is_defined(x) for x in [nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec]):
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
    """
    Get event origin time from SAC O header.

    Correct SAC logic:
        origin_time = SAC_reference_time + O
    """
    o = get_sac_header(tr, "o")
    if not is_defined(o):
        return None

    ref = get_reference_time_from_sac(tr)
    return ref + float(o)


def get_pick_abs(tr: Trace, rel_pick_sec: float) -> UTCDateTime:
    ref = get_reference_time_from_sac(tr)
    return ref + float(rel_pick_sec)


def get_distance_km(tr: Trace) -> Optional[float]:
    dist = get_sac_header(tr, "dist")
    if is_defined(dist) and float(dist) > 0:
        return float(dist)

    gcarc = get_sac_header(tr, "gcarc")
    if is_defined(gcarc) and float(gcarc) > 0:
        return float(gcarc) * 111.19

    return None


def get_header_ml(tr: Trace) -> float:
    val = get_sac_header(tr, HEADER_ML_KEY)
    if not is_defined(val):
        return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def clean_phase_label(label) -> str:
    if label is None:
        return ""
    return str(label).strip().upper().replace(" ", "")


def classify_pick(label) -> Optional[str]:
    """Classify SAC pick label into P/S phase groups."""
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


def collect_picks(traces: Dict[str, Trace]) -> Tuple[Optional[UTCDateTime], Optional[UTCDateTime], str, str]:
    """
    Collect P and S picks from SAC headers.

    Uses:
        A / KA
        T0-T9 / KT0-KT9
    """
    P: List[Tuple[UTCDateTime, str]] = []
    S: List[Tuple[UTCDateTime, str]] = []

    for comp, tr in traces.items():
        a = get_sac_header(tr, "a")
        ka = get_sac_header(tr, "ka")

        if is_defined(a):
            lab = classify_pick(ka)
            if lab is None:
                lab = "P"  # A is commonly P if label is missing.

            abs_t = get_pick_abs(tr, float(a))
            if lab.startswith("P"):
                P.append((abs_t, lab))
            elif lab.startswith("S"):
                S.append((abs_t, lab))

        for i in range(10):
            t = get_sac_header(tr, f"t{i}")
            k = get_sac_header(tr, f"kt{i}")

            if not is_defined(t):
                continue

            lab = classify_pick(k)
            if lab is None:
                continue

            abs_t = get_pick_abs(tr, float(t))
            if lab.startswith("P"):
                P.append((abs_t, lab))
            elif lab.startswith("S"):
                S.append((abs_t, lab))

    if P:
        p_pick, p_lab = sorted(P, key=lambda x: x[0])[0]
    else:
        p_pick, p_lab = None, ""

    if S:
        s_pick, s_lab = sorted(S, key=lambda x: x[0])[0]
    else:
        s_pick, s_lab = None, ""

    return p_pick, s_pick, p_lab, s_lab


def update_displacement_sac_header_nm(tr: Trace) -> Trace:
    """Update SAC headers after velocity -> displacement conversion in nm."""
    tr = tr.copy()
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return tr

    ref = get_reference_time_from_sac(tr)

    # Keep timing consistent with SAC reference time.
    sac.b = float(tr.stats.starttime - ref)
    sac.e = float(tr.stats.endtime - ref)

    # SAC dependent variable type: 6 = IDISP, 7 = IVEL, 8 = IACC.
    try:
        sac.idep = 6
    except Exception:
        pass

    data = tr.data.astype(np.float64)
    if data.size > 0 and np.any(np.isfinite(data)):
        sac.depmin = float(np.nanmin(data))
        sac.depmax = float(np.nanmax(data))
        sac.depmen = float(np.nanmean(data))

    # Short labels, SAC strings are limited.
    try:
        sac.kuser0 = "DISP_NM"
        sac.kuser1 = "SEISAN"
        sac.kuser2 = "NOFILTER" if not APPLY_FILTER_FOR_ML else "FILTERED"
    except Exception:
        pass

    return tr


def velocity_to_displacement_nm(tr_vel: Trace) -> Trace:
    """
    Convert response-removed velocity to displacement by time integration.

    Input velocity is converted to nm/s using VELOCITY_INPUT_TO_NM_PER_SEC.
    Output displacement data are in nm.
    """
    tr = tr_vel.copy()
    tr.data = tr.data.astype(np.float64) * float(VELOCITY_INPUT_TO_NM_PER_SEC)

    if DEMEAN_VELOCITY_BEFORE_INTEGRATE:
        tr.detrend("demean")

    if DETREND_VELOCITY_BEFORE_INTEGRATE:
        tr.detrend("linear")

    if TAPER_VELOCITY_BEFORE_INTEGRATE:
        tr.taper(max_percentage=TAPER_MAX_PERCENTAGE, type="hann")

    if APPLY_FILTER_FOR_ML:
        tr.filter(
            "highpass",
            freq=float(FILTER_FREQ_HZ),
            corners=FILTER_CORNERS,
            zerophase=FILTER_ZEROPHASE,
        )

    # Velocity nm/s -> displacement nm.
    tr.integrate(method="cumtrapz")

    if DEMEAN_DISPLACEMENT_AFTER_INTEGRATE:
        tr.detrend("demean")

    if DETREND_DISPLACEMENT_AFTER_INTEGRATE:
        tr.detrend("linear")

    tr = update_displacement_sac_header_nm(tr)
    return tr


def output_disp_path_for_velocity_file(vel_file: Path) -> Path:
    rel = vel_file.relative_to(INPUT_VEL_ROOT)
    return OUTPUT_DISP_ROOT / rel


def convert_velocity_tree_to_displacement(event_dir: Path) -> int:
    """Convert all SAC files in one event directory to displacement in nm."""
    n_ok = 0

    for vel_file in sorted(event_dir.rglob("*.SAC")):
        out_file = output_disp_path_for_velocity_file(vel_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            tr_vel = read(str(vel_file))[0]
            tr_disp = velocity_to_displacement_nm(tr_vel)
            tr_disp.write(str(out_file), format="SAC")
            n_ok += 1

            log_conversion(
                f"[OK] {vel_file} -> {out_file} "
                f"unit={DISPLACEMENT_OUTPUT_UNIT} "
                f"npts={tr_disp.stats.npts} "
                f"start={tr_disp.stats.starttime.isoformat()} "
                f"end={tr_disp.stats.endtime.isoformat()}"
            )

        except Exception as e:
            log_fail(f"[CONVERT-FAIL] {vel_file} :: {e}")

    return n_ok


def group_files(event_dir: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    """Group displacement SAC files by net.station for required components."""
    groups: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)

    for f in sorted(event_dir.rglob("*.SAC")):
        try:
            tr = read(str(f), headonly=True)[0]

            net = str(getattr(tr.stats, "network", "") or get_sac_header(tr, "knetwk", "")).strip()
            sta = str(getattr(tr.stats, "station", "") or get_sac_header(tr, "kstnm", "")).strip()
            cha = str(getattr(tr.stats, "channel", "") or get_sac_header(tr, "kcmpnm", "")).strip()

            if sta and cha in REQUIRED_STATION_COMPONENTS:
                groups[(net, sta)][cha] = f

        except Exception as e:
            log_fail(f"[READ-HEAD-FAIL] {f} :: {e}")

    return groups


def hutton_ml_seisan_nm(amplitude_nm: float, distance_km: float) -> float:
    """
    SEISAN/Nordic IAML-compatible Hutton local magnitude.

    amplitude_nm:
        Zero-to-peak ground displacement amplitude in nm.

    distance_km:
        Distance in km.
    """
    A = float(amplitude_nm)
    R = float(distance_km)

    if A <= 0 or R <= 0 or not np.isfinite(A) or not np.isfinite(R):
        return np.nan

    return (
        np.log10(A)
        + 1.11 * np.log10(R)
        + 0.00189 * R
        - 2.09
    )


def hutton_ml_wa_mm_equivalent(amplitude_nm: float, distance_km: float) -> float:
    """
    Diagnostic only: convert ground displacement nm to WA mm using 2080 gain,
    then use the Hutton & Boore WA-mm form. This should be nearly equivalent
    to hutton_ml_seisan_nm except for rounding of the constant.
    """
    A_nm = float(amplitude_nm)
    R = float(distance_km)
    if A_nm <= 0 or R <= 0 or not np.isfinite(A_nm) or not np.isfinite(R):
        return np.nan

    A_wa_mm = A_nm * WA_STATIC_MAGNIFICATION * 1e-6
    return (
        np.log10(A_wa_mm)
        + 1.11 * np.log10(R / 100.0)
        + 0.00189 * (R - 100.0)
        + 3.0
    )


def measure_zero_to_peak_amplitude_nm(
    tr_disp: Trace,
    t0: UTCDateTime,
    t1: UTCDateTime,
) -> Tuple[float, float, float, float, Optional[UTCDateTime], float, int]:
    """
    Measure maximum absolute displacement amplitude in nm.

    Returns:
        amp_nm, amp_um, ground_amp_mm, wa_equiv_mm,
        peak_time, peak_signed_nm, npts
    """
    tr_win = tr_disp.copy()
    tr_win.trim(t0, t1, pad=False)

    if tr_win.stats.npts < 2:
        return np.nan, np.nan, np.nan, np.nan, None, np.nan, 0

    data = tr_win.data.astype(np.float64)

    if data.size < 2:
        return np.nan, np.nan, np.nan, np.nan, None, np.nan, 0

    if not np.any(np.isfinite(data)):
        return np.nan, np.nan, np.nan, np.nan, None, np.nan, int(data.size)

    if REMOVE_WINDOW_MEAN_FOR_AMPLITUDE:
        data = data - np.nanmean(data)

    i = int(np.nanargmax(np.abs(data)))
    peak_signed_nm = float(data[i])
    amp_nm = abs(peak_signed_nm)
    amp_um = amp_nm * 1e-3
    ground_amp_mm = amp_nm * 1e-6
    wa_equiv_mm = amp_nm * WA_STATIC_MAGNIFICATION * 1e-6
    peak_time = tr_win.stats.starttime + i / tr_win.stats.sampling_rate

    return amp_nm, amp_um, ground_amp_mm, wa_equiv_mm, peak_time, peak_signed_nm, int(tr_win.stats.npts)


def horizontal_row_from_en_rows(rows: List[dict]) -> Optional[dict]:
    """
    Create derived HMAX and HMEAN rows from HHE/HHN rows.

    HMAX uses the larger horizontal displacement amplitude.
    HMEAN uses the mean of component ML values.
    These rows are diagnostic and make H-vs-Z checking easier.
    """
    by_comp = {r["comp"]: r for r in rows}
    if not all(c in by_comp for c in HORIZONTAL_COMPONENTS_FOR_STATION_ML):
        return None

    e = by_comp[HORIZONTAL_COMPONENTS_FOR_STATION_ML[0]]
    n = by_comp[HORIZONTAL_COMPONENTS_FOR_STATION_ML[1]]

    amp_e = float(e["amplitude_displacement_nm"])
    amp_n = float(n["amplitude_displacement_nm"])
    distance_km = float(e["distance_km"])

    if not np.isfinite(amp_e) or not np.isfinite(amp_n):
        return None

    amp_hmax = max(amp_e, amp_n)
    ml_hmax = hutton_ml_seisan_nm(amp_hmax, distance_km)
    ml_hmean = float(np.nanmean([e["ML_component"], n["ML_component"]]))

    out = dict(e)
    out.update({
        "comp": "HMAX",
        "ML_component": float(ml_hmax) if np.isfinite(ml_hmax) else np.nan,
        "ML_component_minus_header": (
            float(ml_hmax - e["header_ML"]) if np.isfinite(ml_hmax) and np.isfinite(e["header_ML"]) else np.nan
        ),
        "abs_ML_component_minus_header": (
            float(abs(ml_hmax - e["header_ML"])) if np.isfinite(ml_hmax) and np.isfinite(e["header_ML"]) else np.nan
        ),
        "ML_component_WA_mm_equivalent": hutton_ml_wa_mm_equivalent(amp_hmax, distance_km),
        "ML_seisan_minus_WA_equivalent": np.nan,
        "peak_time": "",
        "peak_signed_displacement_nm": np.nan,
        "amplitude_displacement_nm": float(amp_hmax),
        "amplitude_displacement_um": float(amp_hmax * 1e-3),
        "amplitude_ground_displacement_mm": float(amp_hmax * 1e-6),
        "amplitude_WA_equivalent_mm": float(amp_hmax * WA_STATIC_MAGNIFICATION * 1e-6),
        "derived_component": True,
        "derived_from": "+".join(HORIZONTAL_COMPONENTS_FOR_STATION_ML),
        "ML_HHE_HHN_mean": ml_hmean,
    })
    return out


def process_station(event_dir: Path, net: str, sta: str, files: Dict[str, Path]) -> List[dict]:
    missing = [c for c in REQUIRED_STATION_COMPONENTS if c not in files]
    if missing:
        log_fail(f"[MISS-COMP] {event_dir.name} {net}.{sta} missing {missing}")
        return []

    try:
        traces = {c: read(str(files[c]))[0] for c in REQUIRED_STATION_COMPONENTS}
    except Exception as e:
        log_fail(f"[READ-FAIL] {event_dir.name} {net}.{sta} :: {e}")
        return []

    ref_comp = HORIZONTAL_COMPONENTS_FOR_STATION_ML[0]
    tr_ref = traces[ref_comp]

    origin = get_origin_time(tr_ref)
    distance_km = get_distance_km(tr_ref)
    header_ml = get_header_ml(tr_ref)

    if origin is None:
        log_fail(f"[NO-ORIGIN] {event_dir.name} {net}.{sta}")
        return []

    if distance_km is None or distance_km <= 0:
        log_fail(f"[NO-DIST] {event_dir.name} {net}.{sta}")
        return []

    p_pick, s_pick, p_label, s_label = collect_picks(traces)

    p_source = "header"
    s_source = "header"

    if p_pick is None:
        p_pick = origin + distance_km / VP
        p_label = "P"
        p_source = "theory"

    if s_pick is None:
        s_pick = origin + distance_km / VS
        s_label = "S"
        s_source = "theory"

    surface_pick = origin + distance_km / VSURFACE

    amp_window_start = s_pick + AMP_AFTER_S_SEC
    amp_window_end = surface_pick - AMP_BEFORE_SURF_SEC
    amp_window_sec = float(amp_window_end - amp_window_start)

    if amp_window_sec < MIN_AMP_WINDOW_SEC:
        log_fail(
            f"[BAD-AMP-WINDOW] {event_dir.name} {net}.{sta} "
            f"S={s_pick.isoformat()} SURF={surface_pick.isoformat()} "
            f"WIN={amp_window_start.isoformat()}->{amp_window_end.isoformat()} "
            f"LEN={amp_window_sec:.2f}s"
        )
        return []

    rows: List[dict] = []

    for comp in COMPONENTS_FOR_ML:
        tr = traces[comp]

        (
            amp_nm,
            amp_um,
            ground_amp_mm,
            wa_equiv_mm,
            peak_time,
            peak_signed_nm,
            npts,
        ) = measure_zero_to_peak_amplitude_nm(tr, amp_window_start, amp_window_end)

        ml_component = hutton_ml_seisan_nm(amp_nm, distance_km)
        ml_wa_equiv = hutton_ml_wa_mm_equivalent(amp_nm, distance_km)

        if not np.isfinite(ml_component):
            log_fail(
                f"[BAD-ML] {event_dir.name} {net}.{sta}.{comp} "
                f"A_nm={amp_nm} R={distance_km}"
            )
            continue

        rows.append({
            "event": event_dir.name,
            "net": net,
            "sta": sta,
            "comp": comp,

            "distance_km": float(distance_km),
            "header_ML_key": HEADER_ML_KEY,
            "header_ML": float(header_ml) if np.isfinite(header_ml) else np.nan,

            "ML_formula": ML_FORMULA_NAME,
            "ML_component": float(ml_component),
            "ML_component_minus_header": (
                float(ml_component - header_ml) if np.isfinite(header_ml) else np.nan
            ),
            "abs_ML_component_minus_header": (
                float(abs(ml_component - header_ml)) if np.isfinite(header_ml) else np.nan
            ),

            # Diagnostic equivalence with Hutton & Boore WA-mm form.
            "ML_component_WA_mm_equivalent": float(ml_wa_equiv) if np.isfinite(ml_wa_equiv) else np.nan,
            "ML_seisan_minus_WA_equivalent": (
                float(ml_component - ml_wa_equiv) if np.isfinite(ml_wa_equiv) else np.nan
            ),

            "origin_time": origin.isoformat(),
            "p_pick": p_pick.isoformat(),
            "p_label": p_label,
            "p_source": p_source,
            "s_pick": s_pick.isoformat(),
            "s_label": s_label,
            "s_source": s_source,
            "surface_pick": surface_pick.isoformat(),

            "amp_window_start": amp_window_start.isoformat(),
            "amp_window_end": amp_window_end.isoformat(),
            "amp_window_sec": amp_window_sec,

            "peak_time": peak_time.isoformat() if peak_time else "",
            "peak_signed_displacement_nm": peak_signed_nm,
            "amplitude_displacement_nm": amp_nm,
            "amplitude_displacement_um": amp_um,
            "amplitude_ground_displacement_mm": ground_amp_mm,
            "amplitude_WA_equivalent_mm": wa_equiv_mm,
            "amplitude_unit_for_ML": "nm",
            "remove_window_mean_for_amplitude": bool(REMOVE_WINDOW_MEAN_FOR_AMPLITUDE),
            "npts_window": int(npts),
            "derived_component": False,
            "derived_from": "",

            "velocity_input_to_nm_per_sec": float(VELOCITY_INPUT_TO_NM_PER_SEC),
            "displacement_output_unit": DISPLACEMENT_OUTPUT_UNIT,
            "apply_filter_for_ml": bool(APPLY_FILTER_FOR_ML),
        })

    hrow = horizontal_row_from_en_rows(rows)
    if hrow is not None and np.isfinite(hrow.get("ML_component", np.nan)):
        rows.append(hrow)

    if rows:
        log_success(
            f"[OK] {event_dir.name} {net}.{sta} "
            f"dist={distance_km:.2f}km "
            f"header_ML={header_ml} "
            f"components={','.join(sorted(files.keys()))} "
            f"P={p_pick.isoformat()}({p_label},{p_source}) "
            f"S={s_pick.isoformat()}({s_label},{s_source}) "
            f"SURF={surface_pick.isoformat()} "
            f"WIN={amp_window_start.isoformat()}->{amp_window_end.isoformat()} "
            f"formula={ML_FORMULA_NAME} unit=nm filter={APPLY_FILTER_FOR_ML}"
        )

    return rows


def _nanstd_ddof1(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().values.astype(float)
    if arr.size < 2:
        return np.nan
    return float(np.std(arr, ddof=1))


def make_station_summary(component_df: pd.DataFrame) -> pd.DataFrame:
    """Make station-level summary. Standard station ML uses HHE/HHN only."""
    if component_df.empty:
        return pd.DataFrame()

    raw_df = component_df[component_df["comp"].isin(COMPONENTS_FOR_ML)].copy()
    h_df = component_df[component_df["comp"].isin(HORIZONTAL_COMPONENTS_FOR_STATION_ML)].copy()

    # Keep only stations that really have required HHE/HHN/HHZ valid rows.
    raw_counts = raw_df.groupby(["event", "net", "sta"])["comp"].nunique().reset_index(name="n_raw_component")
    valid_keys = raw_counts[raw_counts["n_raw_component"] >= len(REQUIRED_STATION_COMPONENTS)][["event", "net", "sta"]]

    h_station = h_df.groupby(["event", "net", "sta"], as_index=False).agg(
        ML_station_mean=("ML_component", "mean"),
        ML_station_median=("ML_component", "median"),
        ML_station_std=("ML_component", _nanstd_ddof1),
        n_horizontal_component=("ML_component", "count"),

        distance_km=("distance_km", "mean"),
        distance_min_km=("distance_km", "min"),
        distance_max_km=("distance_km", "max"),

        header_ML=("header_ML", "mean"),
        amplitude_h_nm_max=("amplitude_displacement_nm", "max"),
        amplitude_h_nm_mean=("amplitude_displacement_nm", "mean"),
        amp_window_start=("amp_window_start", "first"),
        amp_window_end=("amp_window_end", "first"),
        s_pick=("s_pick", "first"),
        s_source=("s_source", "first"),
        ML_formula=("ML_formula", "first"),
    )

    station_df = valid_keys.merge(h_station, on=["event", "net", "sta"], how="inner")
    if station_df.empty:
        return station_df

    ml_pivot = raw_df.pivot_table(
        index=["event", "net", "sta"],
        columns="comp",
        values="ML_component",
        aggfunc="first",
    ).reset_index()
    amp_pivot = raw_df.pivot_table(
        index=["event", "net", "sta"],
        columns="comp",
        values="amplitude_displacement_nm",
        aggfunc="first",
    ).reset_index()

    ml_rename = {c: f"ML_{c}" for c in COMPONENTS_FOR_ML if c in ml_pivot.columns}
    amp_rename = {c: f"amp_nm_{c}" for c in COMPONENTS_FOR_ML if c in amp_pivot.columns}
    ml_pivot = ml_pivot.rename(columns=ml_rename)
    amp_pivot = amp_pivot.rename(columns=amp_rename)

    station_df = station_df.merge(ml_pivot, on=["event", "net", "sta"], how="left")
    station_df = station_df.merge(amp_pivot, on=["event", "net", "sta"], how="left")

    # Useful H/Z diagnostic columns.
    if all(f"ML_{c}" in station_df.columns for c in HORIZONTAL_COMPONENTS_FOR_STATION_ML):
        station_df["ML_H_mean"] = station_df[[f"ML_{c}" for c in HORIZONTAL_COMPONENTS_FOR_STATION_ML]].mean(axis=1)
        station_df["ML_H_std"] = station_df[[f"ML_{c}" for c in HORIZONTAL_COMPONENTS_FOR_STATION_ML]].std(axis=1, ddof=1)
    if f"ML_{VERTICAL_COMPONENT}" in station_df.columns:
        station_df["ML_Z"] = station_df[f"ML_{VERTICAL_COMPONENT}"]
        station_df["ML_Z_minus_H_mean"] = station_df["ML_Z"] - station_df["ML_H_mean"]

    station_df["ML_station_minus_header"] = station_df["ML_station_mean"] - station_df["header_ML"]
    station_df["abs_ML_station_minus_header"] = station_df["ML_station_minus_header"].abs()

    return station_df


def make_event_summary(station_df: pd.DataFrame) -> pd.DataFrame:
    """Make event-level summary from station horizontal ML."""
    if station_df.empty:
        return pd.DataFrame()

    event_df = station_df.groupby("event", as_index=False).agg(
        ML_event_mean=("ML_station_mean", "mean"),
        ML_event_median=("ML_station_mean", "median"),
        ML_event_std=("ML_station_mean", _nanstd_ddof1),
        n_station=("ML_station_mean", "count"),

        distance_min_km=("distance_km", "min"),
        distance_max_km=("distance_km", "max"),
        distance_mean_km=("distance_km", "mean"),
        distance_median_km=("distance_km", "median"),

        header_ML_mean=("header_ML", "mean"),
        header_ML_median=("header_ML", "median"),
        amplitude_h_nm_mean=("amplitude_h_nm_mean", "mean"),
        ML_formula=("ML_formula", "first"),
    )

    event_df["ML_event_std_filled0"] = event_df["ML_event_std"].fillna(0.0)
    event_df["ML_event_std_error"] = event_df["ML_event_std"] / np.sqrt(event_df["n_station"].where(event_df["n_station"] > 0, np.nan))
    event_df["ML_event_mean_minus_header"] = event_df["ML_event_mean"] - event_df["header_ML_mean"]
    event_df["ML_event_median_minus_header"] = event_df["ML_event_median"] - event_df["header_ML_median"]
    event_df["abs_ML_event_mean_minus_header"] = event_df["ML_event_mean_minus_header"].abs()

    return event_df


def make_event_component_summary(component_df: pd.DataFrame) -> pd.DataFrame:
    """Make event-level summary for each component, including H and Z diagnostics."""
    if component_df.empty:
        return pd.DataFrame()

    tmp = component_df[component_df["comp"].isin(COMPONENTS_FOR_ML + ["HMAX"])].copy()
    if tmp.empty:
        return pd.DataFrame()

    out = tmp.groupby(["event", "comp"], as_index=False).agg(
        ML_component_event_mean=("ML_component", "mean"),
        ML_component_event_median=("ML_component", "median"),
        ML_component_event_std=("ML_component", _nanstd_ddof1),
        n_station_component=("ML_component", "count"),
        header_ML_mean=("header_ML", "mean"),
        distance_mean_km=("distance_km", "mean"),
        distance_min_km=("distance_km", "min"),
        distance_max_km=("distance_km", "max"),
        amplitude_nm_mean=("amplitude_displacement_nm", "mean"),
    )
    out["ML_component_event_std_filled0"] = out["ML_component_event_std"].fillna(0.0)
    out["ML_component_event_mean_minus_header"] = out["ML_component_event_mean"] - out["header_ML_mean"]
    return out


def _savefig(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_FIG_ROOT / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_component_residual_vs_distance(component_df: pd.DataFrame) -> None:
    if component_df.empty:
        return

    tmp = component_df.dropna(subset=["distance_km", "ML_component_minus_header"]).copy()
    tmp = tmp[tmp["comp"].isin(COMPONENTS_FOR_ML + ["HMAX"])]
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for comp in COMPONENTS_FOR_ML + ["HMAX"]:
        sub = tmp[tmp["comp"] == comp]
        if sub.empty:
            continue
        ax.scatter(sub["distance_km"], sub["ML_component_minus_header"], s=18, label=comp, alpha=0.75)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("MLcal - header ML")
    ax.set_title("SEISAN Hutton NM: component residual versus distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "component_MLcal_minus_header_vs_distance_HZ.png")


def plot_station_header_vs_residual(station_df: pd.DataFrame) -> None:
    """Requested diagnostic: header ML against MLcal - header ML."""
    if station_df.empty:
        return

    tmp = station_df.dropna(subset=["header_ML", "ML_station_minus_header"]).copy()
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(tmp["header_ML"], tmp["ML_station_minus_header"], s=24, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Header ML")
    ax.set_ylabel("Station MLcal - header ML")
    ax.set_title("Station residual versus header ML")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "station_headerML_vs_MLcal_minus_header.png")


def plot_station_ml_vs_distance_errorbar(station_df: pd.DataFrame) -> None:
    """Requested diagnostic: MLcal with component std error bar versus distance."""
    if station_df.empty:
        return

    tmp = station_df.dropna(subset=["distance_km", "ML_station_mean"]).copy()
    if tmp.empty:
        return

    yerr = tmp["ML_station_std"].fillna(0.0).values

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        tmp["distance_km"].values,
        tmp["ML_station_mean"].values,
        yerr=yerr,
        fmt="o",
        markersize=4,
        capsize=2,
        alpha=0.75,
        linewidth=0.8,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Station MLcal from HHE/HHN")
    ax.set_title("Station MLcal with H-component std versus distance")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "station_MLcal_errorbar_componentSTD_vs_distance.png")


def plot_event_ml_vs_header_errorbar(event_df: pd.DataFrame) -> None:
    """Event MLcal versus header ML with event station std as y error bar."""
    if event_df.empty:
        return

    tmp = event_df.dropna(subset=["header_ML_mean", "ML_event_mean"]).copy()
    if tmp.empty:
        return

    vmin = float(min(tmp["header_ML_mean"].min(), tmp["ML_event_mean"].min()))
    vmax = float(max(tmp["header_ML_mean"].max(), tmp["ML_event_mean"].max()))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    yerr = tmp["ML_event_std_filled0"].values

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.errorbar(
        tmp["header_ML_mean"].values,
        tmp["ML_event_mean"].values,
        yerr=yerr,
        fmt="o",
        markersize=4,
        capsize=2,
        alpha=0.75,
        linewidth=0.8,
    )
    ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.0)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Header ML")
    ax.set_ylabel("Event MLcal mean")
    ax.set_title("Event MLcal versus header ML with station std")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "event_MLcal_vs_header_errorbar_eventSTD.png")


def plot_event_residual_vs_header(event_df: pd.DataFrame) -> None:
    if event_df.empty:
        return

    tmp = event_df.dropna(subset=["header_ML_mean", "ML_event_mean_minus_header"]).copy()
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        tmp["header_ML_mean"].values,
        tmp["ML_event_mean_minus_header"].values,
        yerr=tmp["ML_event_std_filled0"].values,
        fmt="o",
        markersize=4,
        capsize=2,
        alpha=0.75,
        linewidth=0.8,
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Header ML")
    ax.set_ylabel("Event MLcal mean - header ML")
    ax.set_title("Event residual versus header ML with station std")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "event_headerML_vs_MLcal_minus_header_errorbar_eventSTD.png")


def plot_z_minus_h_vs_distance(station_df: pd.DataFrame) -> None:
    if station_df.empty or "ML_Z_minus_H_mean" not in station_df.columns:
        return

    tmp = station_df.dropna(subset=["distance_km", "ML_Z_minus_H_mean"]).copy()
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(tmp["distance_km"], tmp["ML_Z_minus_H_mean"], s=24, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("ML_Z - ML_H_mean")
    ax.set_title("Vertical-horizontal ML difference versus distance")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "station_MLZ_minus_MLH_vs_distance.png")


def write_run_config() -> None:
    config_file = OUTPUT_REPORT_ROOT / "run_config_Seisan_Hutton_nm_HZ.txt"
    with open(config_file, "w") as f:
        f.write("Option A: velocity -> ground displacement -> SEISAN Hutton ML\n")
        f.write("Wood-Anderson simulation: False\n")
        f.write("Additional filtering: False by default\n")
        f.write(f"Input velocity root: {INPUT_VEL_ROOT}\n")
        f.write(f"Output displacement root: {OUTPUT_DISP_ROOT}\n")
        f.write(f"Output report root: {OUTPUT_REPORT_ROOT}\n")
        f.write(f"ML formula: {ML_FORMULA_NAME}\n")
        f.write("ML equation: ML = log10(A_nm) + 1.11*log10(R_km) + 0.00189*R_km - 2.09\n")
        f.write(f"Required station components: {REQUIRED_STATION_COMPONENTS}\n")
        f.write(f"Component rows: {COMPONENTS_FOR_ML}\n")
        f.write(f"Station/event ML horizontal components: {HORIZONTAL_COMPONENTS_FOR_STATION_ML}\n")
        f.write(f"Vertical diagnostic component: {VERTICAL_COMPONENT}\n")
        f.write(f"VP: {VP}\n")
        f.write(f"VS: {VS}\n")
        f.write(f"VSURFACE: {VSURFACE}\n")
        f.write(f"AMP_AFTER_S_SEC: {AMP_AFTER_S_SEC}\n")
        f.write(f"AMP_BEFORE_SURF_SEC: {AMP_BEFORE_SURF_SEC}\n")
        f.write(f"MIN_AMP_WINDOW_SEC: {MIN_AMP_WINDOW_SEC}\n")
        f.write(f"VELOCITY_INPUT_TO_NM_PER_SEC: {VELOCITY_INPUT_TO_NM_PER_SEC}\n")
        f.write(f"DISPLACEMENT_OUTPUT_UNIT: {DISPLACEMENT_OUTPUT_UNIT}\n")
        f.write(f"WA_STATIC_MAGNIFICATION: {WA_STATIC_MAGNIFICATION}\n")
        f.write(f"APPLY_FILTER_FOR_ML: {APPLY_FILTER_FOR_ML}\n")
        f.write(f"FILTER_FREQ_HZ: {FILTER_FREQ_HZ}\n")
        f.write(f"DEMEAN_VELOCITY_BEFORE_INTEGRATE: {DEMEAN_VELOCITY_BEFORE_INTEGRATE}\n")
        f.write(f"DETREND_VELOCITY_BEFORE_INTEGRATE: {DETREND_VELOCITY_BEFORE_INTEGRATE}\n")
        f.write(f"TAPER_VELOCITY_BEFORE_INTEGRATE: {TAPER_VELOCITY_BEFORE_INTEGRATE}\n")
        f.write(f"DEMEAN_DISPLACEMENT_AFTER_INTEGRATE: {DEMEAN_DISPLACEMENT_AFTER_INTEGRATE}\n")
        f.write(f"DETREND_DISPLACEMENT_AFTER_INTEGRATE: {DETREND_DISPLACEMENT_AFTER_INTEGRATE}\n")
        f.write(f"REMOVE_WINDOW_MEAN_FOR_AMPLITUDE: {REMOVE_WINDOW_MEAN_FOR_AMPLITUDE}\n")
        f.write(f"HEADER_ML_KEY: {HEADER_ML_KEY}\n")


def main() -> None:
    write_run_config()

    event_dirs = sorted([d for d in INPUT_VEL_ROOT.glob(EVENT_DIR_GLOB) if d.is_dir()])

    if not event_dirs:
        print(f"[ERROR] No event directories found: {INPUT_VEL_ROOT}/{EVENT_DIR_GLOB}")
        return

    print(f"[INFO] Found {len(event_dirs)} event directories")
    print(f"[INFO] Converting velocity SAC to displacement SAC in nm: {OUTPUT_DISP_ROOT}")
    print(f"[INFO] ML formula: {ML_FORMULA_NAME}")
    print(f"[INFO] Required station components: {REQUIRED_STATION_COMPONENTS}")

    total_converted = 0
    all_component_rows: List[dict] = []

    for event_dir in event_dirs:
        n_conv = convert_velocity_tree_to_displacement(event_dir)
        total_converted += n_conv

        disp_event_dir = OUTPUT_DISP_ROOT / event_dir.relative_to(INPUT_VEL_ROOT)
        groups = group_files(disp_event_dir)

        if not groups:
            log_fail(f"[NO-ML-GROUPS] {disp_event_dir}")
            continue

        for (net, sta), files in sorted(groups.items()):
            rows = process_station(disp_event_dir, net, sta, files)
            all_component_rows.extend(rows)

    if not all_component_rows:
        print("[ERROR] No valid component ML rows created.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    component_df = pd.DataFrame(all_component_rows)
    component_df = component_df.replace([np.inf, -np.inf], np.nan)
    component_df = component_df.dropna(subset=["ML_component", "distance_km", "amplitude_displacement_nm"]).copy()

    if component_df.empty:
        print("[ERROR] Component DataFrame is empty after dropping NaN values.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    station_df = make_station_summary(component_df)
    if station_df.empty:
        print("[ERROR] Station DataFrame is empty.")
        print("This usually means no station has valid HHE, HHN, and HHZ ML rows.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    event_df = make_event_summary(station_df)
    event_component_df = make_event_component_summary(component_df)

    # Remove large path/name columns from component output. This keeps the table smaller.
    drop_cols = [c for c in ["file", "disp_file"] if c in component_df.columns]
    if drop_cols:
        component_df = component_df.drop(columns=drop_cols)

    component_df.to_csv(COMPONENT_CSV, index=False)
    station_df.to_csv(STATION_CSV, index=False)
    event_df.to_csv(EVENT_CSV, index=False)
    event_component_df.to_csv(EVENT_COMPONENT_CSV, index=False)

    if MAKE_FIGURES:
        plot_component_residual_vs_distance(component_df)
        plot_station_header_vs_residual(station_df)
        plot_station_ml_vs_distance_errorbar(station_df)
        plot_event_ml_vs_header_errorbar(event_df)
        plot_event_residual_vs_header(event_df)
        plot_z_minus_h_vs_distance(station_df)

    print("========== DONE ==========")
    print(f"Velocity input root       : {INPUT_VEL_ROOT}")
    print(f"Velocity input scale      : {VELOCITY_INPUT_TO_NM_PER_SEC} -> nm/s")
    print(f"Displacement SAC root     : {OUTPUT_DISP_ROOT}")
    print(f"Displacement SAC unit     : {DISPLACEMENT_OUTPUT_UNIT}")
    print(f"Report root               : {OUTPUT_REPORT_ROOT}")
    print(f"ML formula                : {ML_FORMULA_NAME}")
    print("ML equation               : log10(A_nm) + 1.11log10(R_km) + 0.00189R_km - 2.09")
    print(f"Required station comps    : {REQUIRED_STATION_COMPONENTS}")
    print(f"Converted SAC files       : {total_converted}")
    print(f"Component ML rows         : {len(component_df)}")
    print(f"Station ML rows           : {len(station_df)}")
    print(f"Event ML rows             : {len(event_df)}")
    print(f"Event-component rows      : {len(event_component_df)}")
    print(f"Component CSV             : {COMPONENT_CSV}")
    print(f"Station CSV               : {STATION_CSV}")
    print(f"Event CSV                 : {EVENT_CSV}")
    print(f"Event component CSV       : {EVENT_COMPONENT_CSV}")
    print(f"Success log               : {SUCCESS_LOG}")
    print(f"Fail log                  : {FAIL_LOG}")
    print(f"Conversion log            : {CONVERSION_LOG}")
    if MAKE_FIGURES:
        print(f"Figure root               : {OUTPUT_FIG_ROOT}")


if __name__ == "__main__":
    main()
