#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import re
import csv
import warnings

import numpy as np
import pygmt
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy.geodetics import gps2dist_azimuth


# =============================================================================
# USER PARAMETERS
# =============================================================================
INPUT_ROOT = Path("input/data_vres_fixed_name")
OUTPUT_CSV = Path("output/Vung_vres_ml_check/event_station_ml_vres.csv")

# Figures
FIG_ROOT = Path("figures/vres_ml_check")
SAVE_FIGURES = False   # True = save QC figures, False = do not plot/save

# Unit of INPUT VELOCITY trace
# choose one of: "m", "cm", "mm", "um", "nm", or None
DATA_UNIT = None
AUTO_UNIT_FALLBACK = "m"

# Distance service limit of provided equations
MAX_SERVICE_DISTANCE_KM = 600.0

# =============================================================================
# WINDOW LOGIC PARAMETERS
# =============================================================================
# Assumed velocities (km/s)
VP_KM_S = 8.0
VS_KM_S = 3.0
VSURF_KM_S = 1.0

# Final main-signal window:
#   start = P - PRE_P_SEC
#   end   = surface-wave arrival + POST_SURF_SEC
PRE_P_SEC = 10.0
POST_SURF_SEC = 10.0

# If all phase/model methods fail
STALTA_AFTER_ON_SEC = 300.0
FALLBACK_SIGNAL_LEN_SEC = 300.0

# STA/LTA settings
STA_SEC = 0.5
LTA_SEC = 5.0
TRIG_ON = 3.0
TRIG_OFF = 1.2

# Preprocessing before integration
DO_DEMEAN = True
DO_DETREND = True
DO_TAPER = True
TAPER_MAX_PERCENTAGE = 0.05

# High-pass before integration to reduce tilt
HIGHPASS_FREQ = 0.075
HIGHPASS_CORNERS = 2
HIGHPASS_ZEROPHASE = True

# Plot settings
PLOT_WIDTH = "16c"
PLOT_HEIGHT = "7c"
WINDOW_FILL = "gray90"
TRACE_PEN = "1p,black"
PEAK_STYLE = "x0.32c"
PEAK_PEN = "1.2p,red"
PHASE_P_PEN = "0.8p,blue,-"
PHASE_S_PEN = "0.8p,darkgreen,-"
FONT_LABEL = "11p,Helvetica,black"
FONT_ANNOT = "9p,Helvetica,black"
FONT_TITLE = "12p,Helvetica-Bold,black"

# Logs
SUCCESS_LOG = OUTPUT_CSV.parent / "success.log"
FAIL_LOG = OUTPUT_CSV.parent / "fail.log"


# =============================================================================
# ML FUNCTIONS
# =============================================================================
def hutton83(amplitude, distance, correction):
    import numpy as np
    """
    calculate the ML from Hutton and Boore 1983
    input:
        + amplitude: displacement amplitude data (in mm)
        + distance: hypocenter distance (in km)
    output:
        + logA0: distance correction value
        + ML: Local magnitude value
    """
    if distance > 600:
        print("The distance exceeded the services, return 0", distance)
        ML = 0
        logA0 = 0
    else:
        if correction == []:
            correction = 0
        logA0 = -(1.110 * np.log10(distance / 100) + 0.00189 * (distance - 100) + 3.0)
        ML = np.log10(amplitude) + 1.11 * np.log10(distance) + 0.00189 * distance - 2.09

    return ML, logA0


def le08(amplitude, distance, correction):
    """
    calculate the ML from Le et al 2008
    input:
        + amplitude: displacement amplitude data (in nm)
        + distance: hypocenter distance (in km)
    output:
        + logA0: distance correction value
        + ML: Local magnitude value
    """
    if distance > 600:
        print("The distance exceeded the services, return 0", distance)
        ML = 0
        logA0 = 0
    else:
        logA0 = -(1.018 * np.log10(distance / 100) + 0.00232 * (distance - 100) + 3.00)
        ML = np.log10(amplitude) + 1.018 * np.log10(distance) + 0.00232 * distance - 2.09 + correction
    return ML, logA0


def nguyen11(amplitude, distance, correction):
    import numpy as np
    """
    Current user-provided implementation.
    NOTE: this function is kept as provided by the user.
    """
    if distance > 600:
        print("The distance exceeded the services, return 0", distance)
        ML = 0
        logA0 = 0
    else:
        if correction == []:
            correction = 0
        logA0 = -(1.74 * np.log10(distance / 100) + 0.00048 * (distance - 100) + 3.0)
        ML = np.log10(amplitude) + 1.74 * np.log10(distance) + 0.00048 * distance - 0.528 + correction
    return ML, logA0


# =============================================================================
# HELPERS
# =============================================================================
def log_success(msg: str):
    with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_fail(msg: str):
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def reset_logs():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_FIGURES:
        FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SUCCESS_LOG.write_text("", encoding="utf-8")
    FAIL_LOG.write_text("", encoding="utf-8")


def safe_float(x):
    try:
        v = float(x)
        if v == -12345:
            return None
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def safe_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s == "-12345":
        return ""
    return s


def get_sac_header(tr, key, default=None):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return default
    return getattr(sac, key, default)


def get_event_magnitude(tr):
    """
    Read event magnitude from SAC header.
    Prefer 'mag'. If missing, return None.
    """
    mag = safe_float(get_sac_header(tr, "mag"))
    return mag


def length_factor_to_meter(unit: str) -> float:
    unit = unit.lower()
    factors = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "μm": 1e-6,
        "nm": 1e-9,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported unit: {unit}")
    return factors[unit]


def convert_length_value(value, from_unit: str, to_unit: str) -> float:
    f_from = length_factor_to_meter(from_unit)
    f_to = length_factor_to_meter(to_unit)
    return value * f_from / f_to


def guess_data_unit_from_sac(tr):
    if DATA_UNIT is not None:
        return DATA_UNIT

    candidate_text = []
    sac = getattr(tr.stats, "sac", None)
    if sac is not None:
        for key in ["kuser0", "kuser1", "kuser2", "kinst", "kevnm", "kstnm", "kcmpnm"]:
            candidate_text.append(safe_str(getattr(sac, key, "")))

    blob = " | ".join(candidate_text).lower()
    patterns = [
        (r"\bnm/s\b|\bnm\b", "nm"),
        (r"\bum/s\b|\bμm/s\b|\bum\b|\bμm\b", "um"),
        (r"\bmm/s\b|\bmm\b", "mm"),
        (r"\bcm/s\b|\bcm\b", "cm"),
        (r"\bm/s\b|\bm\b", "m"),
    ]
    for pat, unit in patterns:
        if re.search(pat, blob):
            return unit

    return AUTO_UNIT_FALLBACK


def get_origin_time_offset_rel(tr):
    """
    Return SAC header 'o' in seconds relative to SAC reference.
    If unavailable, assume origin = 0 in SAC-relative frame.
    """
    o = safe_float(get_sac_header(tr, "o"))
    if o is None:
        return 0.0
    return o


def get_distances_km(tr):
    """
    Return (epi_km, hypo_km).

    Prefer SAC 'dist' as epicentral distance if present.
    If coordinates exist, compute epicentral and hypocentral distances.
    """
    dist = safe_float(get_sac_header(tr, "dist"))
    evdp = safe_float(get_sac_header(tr, "evdp"))

    if dist is not None and dist > 0:
        epi_km = dist
        if evdp is not None and evdp >= 0:
            hypo_km = math.sqrt(epi_km ** 2 + evdp ** 2)
        else:
            hypo_km = epi_km
        return epi_km, hypo_km

    stla = safe_float(get_sac_header(tr, "stla"))
    stlo = safe_float(get_sac_header(tr, "stlo"))
    evla = safe_float(get_sac_header(tr, "evla"))
    evlo = safe_float(get_sac_header(tr, "evlo"))

    if None in (stla, stlo, evla, evlo):
        return None, None

    epi_m, _, _ = gps2dist_azimuth(stla, stlo, evla, evlo)
    epi_km = epi_m / 1000.0

    if evdp is None or evdp < 0:
        hypo_km = epi_km
    else:
        hypo_km = math.sqrt(epi_km ** 2 + evdp ** 2)

    return epi_km, hypo_km


def extract_phase_picks_from_header(tr):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return None, None

    picks = []

    a_val = safe_float(getattr(sac, "a", None))
    a_lab = safe_str(getattr(sac, "ka", ""))
    if a_val is not None:
        picks.append(("a", a_val, a_lab))

    for i in range(10):
        t_val = safe_float(getattr(sac, f"t{i}", None))
        t_lab = safe_str(getattr(sac, f"kt{i}", ""))
        if t_val is not None:
            picks.append((f"t{i}", t_val, t_lab))

    if not picks:
        return None, None

    def is_p_label(lbl: str) -> bool:
        lbl = lbl.strip().upper()
        return lbl.startswith("P") or lbl in {"PG", "PN", "P1", "P2"}

    def is_s_label(lbl: str) -> bool:
        lbl = lbl.strip().upper()
        return lbl.startswith("S") or lbl in {"SG", "SN", "S1", "S2"}

    p_candidates = [v for _, v, lbl in picks if is_p_label(lbl)]
    s_candidates = [v for _, v, lbl in picks if is_s_label(lbl)]

    p_pick = min(p_candidates) if p_candidates else None
    s_pick = min(s_candidates) if s_candidates else None

    if p_pick is not None and s_pick is not None and s_pick <= p_pick:
        s_pick = None

    return p_pick, s_pick


def sac_relative_time_to_index(tr, t_rel):
    if t_rel is None:
        return None
    b = safe_float(get_sac_header(tr, "b"))
    if b is None:
        b = 0.0
    dt = tr.stats.delta
    idx = int(round((t_rel - b) / dt))
    idx = max(0, min(tr.stats.npts - 1, idx))
    return idx


def index_to_sac_relative_time(tr, idx):
    b = safe_float(get_sac_header(tr, "b"))
    if b is None:
        b = 0.0
    return b + idx * tr.stats.delta


def detect_signal_onset_stalta(tr):
    """
    Return STA/LTA trigger onset index.
    If no trigger is found, fallback to index of global absolute maximum.
    """
    data = np.asarray(tr.data, dtype=float)
    data = np.nan_to_num(data)

    if data.size < 10:
        return None

    x = data.copy()
    x -= np.mean(x)

    nsta = max(1, int(round(STA_SEC / tr.stats.delta)))
    nlta = max(nsta + 1, int(round(LTA_SEC / tr.stats.delta)))

    if len(x) <= nlta + 5:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cft = recursive_sta_lta(x, nsta, nlta)

    on_off = trigger_onset(cft, TRIG_ON, TRIG_OFF)

    if len(on_off) > 0:
        best_on = None
        best_peak = -np.inf
        absx = np.abs(x)

        for on, off in on_off:
            on = max(0, int(on))
            off = min(len(x) - 1, int(off))
            if off <= on:
                continue
            peak = float(absx[on:off + 1].max())
            if peak > best_peak:
                best_peak = peak
                best_on = on

        if best_on is not None:
            return best_on

    return int(np.argmax(np.abs(x)))


def estimate_theoretical_arrivals(tr, epi_km):
    """
    Return theoretical P, S, and surface-wave arrival times in SAC-relative seconds.

    Travel times are origin-relative:
        tp = distance / Vp
        ts = distance / Vs
        tsurf = epicentral_distance / Vsurf

    Convert to SAC-relative using header 'o':
        t_rel = o + travel_time
    """
    if epi_km is None or epi_km <= 0:
        return None, None, None

    o_rel = get_origin_time_offset_rel(tr)

    p_rel = o_rel + (epi_km / VP_KM_S) if VP_KM_S > 0 else None
    s_rel = o_rel + (epi_km / VS_KM_S) if VS_KM_S > 0 else None
    surf_rel = o_rel + (epi_km / VSURF_KM_S) if VSURF_KM_S > 0 else None

    return p_rel, s_rel, surf_rel


def estimate_p_from_s(hypo_km, s_rel):
    """
    If only S is known, estimate P by tracing back using:
        S - P = dist * (1/Vs - 1/Vp)
    """
    if hypo_km is None or hypo_km <= 0:
        return None
    if s_rel is None:
        return None
    if VP_KM_S <= 0 or VS_KM_S <= 0:
        return None

    sp_time = hypo_km * (1.0 / VS_KM_S - 1.0 / VP_KM_S)
    p_rel = s_rel - sp_time
    return p_rel


def build_measurement_window(tr, epi_km, hypo_km, p_rel_hdr, s_rel_hdr):
    """
    Priority:
      1) header P and S
      2) only S -> back-calc P from S using hypo distance
      3) only P
      4) no picks -> theoretical P/S/surface
      5) STA/LTA fallback

    Final window:
      start = P - PRE_P_SEC
      end   = surface_wave + POST_SURF_SEC
    """
    p_theo, s_theo, surf_theo = estimate_theoretical_arrivals(tr, epi_km)

    used_p = None
    used_s = None
    used_surf = surf_theo
    phase_source = None

    if p_rel_hdr is not None and s_rel_hdr is not None and s_rel_hdr > p_rel_hdr:
        used_p = p_rel_hdr
        used_s = s_rel_hdr
        phase_source = "header_PS"

    elif p_rel_hdr is None and s_rel_hdr is not None:
        p_from_s = estimate_p_from_s(hypo_km, s_rel_hdr)
        if p_from_s is not None:
            used_p = p_from_s
            used_s = s_rel_hdr
            phase_source = "header_S_only_backcalc_P"

    elif p_rel_hdr is not None:
        used_p = p_rel_hdr
        used_s = s_theo
        phase_source = "header_P_only"

    elif p_theo is not None and surf_theo is not None:
        used_p = p_theo
        used_s = s_theo
        phase_source = "model_PSVsurf"

    if used_p is not None and used_surf is not None:
        start_rel = used_p - PRE_P_SEC
        end_rel = used_surf + POST_SURF_SEC

        i1 = sac_relative_time_to_index(tr, start_rel)
        i2 = sac_relative_time_to_index(tr, end_rel)

        if i1 is not None and i2 is not None and i2 > i1:
            return i1, i2, phase_source, used_p, used_s, used_surf

    i_on = detect_signal_onset_stalta(tr)
    if i_on is not None:
        i1 = max(0, i_on)
        i2 = min(tr.stats.npts - 1, i1 + int(round(STALTA_AFTER_ON_SEC / tr.stats.delta)))
        if i2 > i1:
            used_p = index_to_sac_relative_time(tr, i1)
            return i1, i2, "stalta_fallback", used_p, None, None

    i_peak = int(np.argmax(np.abs(np.asarray(tr.data, dtype=float))))
    i1 = max(0, i_peak)
    i2 = min(tr.stats.npts - 1, i1 + int(round(FALLBACK_SIGNAL_LEN_SEC / tr.stats.delta)))
    used_p = index_to_sac_relative_time(tr, i1)
    return i1, i2, "full_trace_fallback", used_p, None, None


def preprocess_for_integration(tr):
    tr2 = tr.copy()
    tr2.data = np.asarray(tr2.data, dtype=float)

    if DO_DEMEAN:
        tr2.detrend("demean")
    if DO_DETREND:
        tr2.detrend("linear")
    if DO_TAPER:
        tr2.taper(max_percentage=TAPER_MAX_PERCENTAGE, type="cosine")

    tr2.filter(
        "highpass",
        freq=HIGHPASS_FREQ,
        corners=HIGHPASS_CORNERS,
        zerophase=HIGHPASS_ZEROPHASE,
    )

    return tr2


def compute_amplitudes_and_peak(tr, i1, i2, data_unit):
    """
    Input trace is velocity.

    Peak rule:
      peak amplitude = maximum absolute displacement amplitude
      within the final main signal window
    """
    i1 = max(0, i1)
    i2 = min(tr.stats.npts - 1, i2)
    if i2 <= i1:
        raise ValueError("Invalid measurement window")

    vel = np.asarray(tr.data[i1:i2 + 1], dtype=float)
    vel = np.nan_to_num(vel)
    max_vel_native = float(np.max(np.abs(vel)))

    tr_int = preprocess_for_integration(tr)
    tr_disp = tr_int.copy()
    tr_disp.integrate()

    disp_win = np.asarray(tr_disp.data[i1:i2 + 1], dtype=float)
    disp_win = np.nan_to_num(disp_win)

    local_peak_idx = int(np.argmax(np.abs(disp_win)))
    peak_idx = i1 + local_peak_idx
    peak_val_disp_native = float(tr_disp.data[peak_idx])
    max_disp_native = abs(peak_val_disp_native)

    max_disp_mm = convert_length_value(max_disp_native, data_unit, "mm")
    max_vel_mm = convert_length_value(max_vel_native, data_unit, "mm")

    return max_disp_mm, max_vel_mm, peak_idx, peak_val_disp_native, tr_disp


def compute_ml_values(max_disp_mm, max_vel_mm, distance_km):
    if distance_km is None or distance_km <= 0:
        return (
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan
        )

    max_disp_nm = convert_length_value(max_disp_mm, "mm", "nm")
    max_disp_um = convert_length_value(max_disp_mm, "mm", "um")

    ML_hut, logA0_hut = hutton83(max_disp_mm, distance_km, 0)
    ML_le, logA0_le = le08(max_disp_nm, distance_km, 0)
    ML_ng, logA0_ng = nguyen11(max_disp_mm, distance_km, 0)

    return (
        logA0_hut, logA0_le, logA0_ng,
        ML_hut, ML_le, ML_ng
    )


def get_sta_chan_from_filename_or_trace(sacfile: Path, tr):
    sta = safe_str(getattr(tr.stats, "station", "")) or safe_str(get_sac_header(tr, "kstnm", ""))
    chan = safe_str(getattr(tr.stats, "channel", "")) or safe_str(get_sac_header(tr, "kcmpnm", ""))

    if sta and chan:
        return sta, chan

    parts = sacfile.name.split(".")
    if len(parts) >= 2:
        sta = sta or parts[0]
        chan = chan or parts[1]

    sta = sta or sacfile.stem.replace(".", "_")
    chan = chan or "UNK"
    return sta, chan


def iter_sac_files(root: Path):
    for evdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sacfile in sorted(evdir.glob("*.SAC")):
            yield evdir, sacfile


def nice_region(x, y):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    if xmin == xmax:
        xmax += 1.0
    if ymin == ymax:
        ymax += 1.0

    yr = ymax - ymin
    xr = xmax - xmin
    ypad = 0.08 * yr if yr > 0 else 1.0
    xpad = 0.02 * xr if xr > 0 else 1.0

    return [xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad]


def make_qc_figure(
    event_name,
    fig_path,
    tr_disp,
    i1,
    i2,
    peak_idx,
    peak_val_native,
    p_hdr_rel=None,
    s_hdr_rel=None,
    data_unit="m",
    distance_km=None,
    phase_source="",
    p_used_rel=None,
    s_used_rel=None,
    surf_used_rel=None,
):
    """
    Plot displacement trace because maxdisp is measured on displacement here.
    Signal trace in black, main window filled light gray, peak by red cross.

    IMPORTANT:
    - P and S markers plotted on figure are header-only.
    - Estimated P/S/surface are only reported in text, not plotted as markers.
    """
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    n = tr_disp.stats.npts
    x = np.array([index_to_sac_relative_time(tr_disp, i) for i in range(n)], dtype=float)
    y_native = np.asarray(tr_disp.data, dtype=float)
    y_mm = convert_length_value(y_native, data_unit, "mm")

    ywin = y_mm[i1:i2 + 1]

    peak_x = x[peak_idx]
    peak_y = convert_length_value(peak_val_native, data_unit, "mm")

    region = nice_region(x, y_mm)

    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL=FONT_LABEL,
        FONT_ANNOT_PRIMARY=FONT_ANNOT,
        FONT_TITLE=FONT_TITLE,
        MAP_FRAME_TYPE="plain",
    )

    fig.basemap(
        region=region,
        projection=f"X{PLOT_WIDTH}/{PLOT_HEIGHT}",
        frame=[
            "WSen+tDisplacement waveform used for ML",
            'xaf+l"Time relative to SAC reference (s)"',
            'yaf+l"Displacement (mm)"',
        ],
    )

    maxabs = float(np.max(np.abs(ywin)))

    x1 = x[i1]
    x2 = x[i2]

    rect_x = [x1, x2, x2, x1, x1]
    rect_y = [-maxabs, -maxabs, maxabs, maxabs, -maxabs]

    fig.plot(x=rect_x, y=rect_y, fill=WINDOW_FILL, pen="0.2p,gray70")
    fig.plot(x=x, y=y_mm, pen=TRACE_PEN)
    fig.plot(x=[peak_x], y=[peak_y], style=PEAK_STYLE, pen=PEAK_PEN)

    # Plot ONLY header P and S markers
    if p_hdr_rel is not None:
        fig.plot(x=[p_hdr_rel, p_hdr_rel], y=[region[2], region[3]], pen=PHASE_P_PEN)
    if s_hdr_rel is not None:
        fig.plot(x=[s_hdr_rel, s_hdr_rel], y=[region[2], region[3]], pen=PHASE_S_PEN)

    text_lines = [
        f"event = {event_name}",
        f"phase_source = {phase_source}",
        f"window = {x[i1]:.2f} to {x[i2]:.2f} s",
        f"P_hdr = {p_hdr_rel:.2f} s" if p_hdr_rel is not None else "P_hdr = None",
        f"S_hdr = {s_hdr_rel:.2f} s" if s_hdr_rel is not None else "S_hdr = None",
        f"P_used = {p_used_rel:.2f} s" if p_used_rel is not None else "P_used = None",
        f"S_used = {s_used_rel:.2f} s" if s_used_rel is not None else "S_used = None",
        f"Surf_used = {surf_used_rel:.2f} s" if surf_used_rel is not None else "Surf_used = None",
        f"peak(signed) = {peak_y:.4e} mm at {peak_x:.2f} s",
        f"peak(abs) = {abs(peak_y):.4e} mm",
    ]
    if distance_km is not None:
        text_lines.append(f"distance = {distance_km:.2f} km")

    tx = region[1] - 0.02 * (region[1] - region[0])
    ty = region[3] - 0.05 * (region[3] - region[2])
    dy = 0.06 * (region[3] - region[2])

    for i, line in enumerate(text_lines):
        fig.text(
            x=tx,
            y=ty - i * dy,
            text=line,
            font="9p,Helvetica,black",
            justify="RM",
        )

    fig.savefig(str(fig_path))


# =============================================================================
# MAIN
# =============================================================================
def main():
    reset_logs()

    rows = []

    for evdir, sacfile in iter_sac_files(INPUT_ROOT):
        event_name = evdir.name
        fname = sacfile.name

        try:
            st = read(str(sacfile))
            if len(st) == 0:
                raise RuntimeError("Empty stream")
            tr = st[0]

            data_unit = guess_data_unit_from_sac(tr)

            epi_km, hypo_km = get_distances_km(tr)
            distance_km = hypo_km

            p_hdr_rel, s_hdr_rel = extract_phase_picks_from_header(tr)

            i1, i2, phase_source, p_used_rel, s_used_rel, surf_used_rel = build_measurement_window(
                tr=tr,
                epi_km=epi_km,
                hypo_km=hypo_km,
                p_rel_hdr=p_hdr_rel,
                s_rel_hdr=s_hdr_rel,
            )

            max_disp_mm, max_vel_mm, peak_idx, peak_val_disp_native, tr_disp = compute_amplitudes_and_peak(
                tr, i1, i2, data_unit
            )

            peak_time_rel = index_to_sac_relative_time(tr, peak_idx)
            window_start_rel = index_to_sac_relative_time(tr, i1)
            window_end_rel = index_to_sac_relative_time(tr, i2)

            (
                logA0_hut, logA0_le, logA0_ng,
                ML_hut, ML_le, ML_ng
            ) = compute_ml_values(max_disp_mm, max_vel_mm, distance_km)

            sta, chan = get_sta_chan_from_filename_or_trace(sacfile, tr)
            component = chan
            event_magnitude = get_event_magnitude(tr)

            if SAVE_FIGURES:
                fig_path = FIG_ROOT / event_name / f"{sta}.{chan}.png"
                make_qc_figure(
                    event_name=event_name,
                    fig_path=fig_path,
                    tr_disp=tr_disp,
                    i1=i1,
                    i2=i2,
                    peak_idx=peak_idx,
                    peak_val_native=peak_val_disp_native,
                    p_hdr_rel=p_hdr_rel,
                    s_hdr_rel=s_hdr_rel,
                    data_unit=data_unit,
                    distance_km=distance_km,
                    phase_source=phase_source,
                    p_used_rel=p_used_rel,
                    s_used_rel=s_used_rel,
                    surf_used_rel=surf_used_rel,
                )
                fig_msg = str(fig_path)
            else:
                fig_msg = "SKIPPED"

            row = {
                "EVENT": event_name,
                "filename": fname,
                "component": component,
                "distance": distance_km,
                "magnitude": event_magnitude,
                "epi_km": epi_km,
                "hypo_km": hypo_km,
                "maxdisp": max_disp_mm,
                "maxvel_mm": max_vel_mm,
                "window_start_rel": window_start_rel,
                "window_end_rel": window_end_rel,
                "peak_time_rel": peak_time_rel,
                "p_pick_rel_header": p_hdr_rel,
                "s_pick_rel_header": s_hdr_rel,
                "p_used_rel": p_used_rel,
                "s_used_rel": s_used_rel,
                "surf_used_rel": surf_used_rel,
                "phase_source": phase_source,
                "LogA0Huton83": logA0_hut,
                "LogA0Le08": logA0_le,
                "LogA0Nguyen11": logA0_ng,
                "MLHuton83": ML_hut,
                "MLLe08": ML_le,
                "MLNguyen11": ML_ng,
            }
            rows.append(row)

            dist_str = f"{distance_km:.3f}" if distance_km is not None else "None"
            log_success(
                f"[OK] {event_name} | {fname} | comp={component} | mag={event_magnitude} | unit={data_unit} | "
                f"phase_source={phase_source} | epi={epi_km} km | hypo={hypo_km} km | "
                f"P_hdr={p_hdr_rel} | S_hdr={s_hdr_rel} | "
                f"P_used={p_used_rel} | S_used={s_used_rel} | Surf_used={surf_used_rel} | "
                f"window=({window_start_rel:.2f},{window_end_rel:.2f}) | "
                f"maxdisp={max_disp_mm:.6e} mm | maxvel={max_vel_mm:.6e} mm | "
                f"fig={fig_msg}"
            )

        except Exception as e:
            log_fail(f"[FAIL] {event_name} | {fname} | {type(e).__name__}: {e}")

    fieldnames = [
        "EVENT",
        "filename",
        "component",
        "distance",
        "magnitude",
        "epi_km",
        "hypo_km",
        "maxdisp",
        "maxvel_mm",
        "window_start_rel",
        "window_end_rel",
        "peak_time_rel",
        "p_pick_rel_header",
        "s_pick_rel_header",
        "p_used_rel",
        "s_used_rel",
        "surf_used_rel",
        "phase_source",
        "LogA0Huton83",
        "LogA0Le08",
        "LogA0Nguyen11",
        "MLHuton83",
        "MLLe08",
        "MLNguyen11",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] CSV saved to: {OUTPUT_CSV}")
    if SAVE_FIGURES:
        print(f"[DONE] Figures saved under: {FIG_ROOT}")
    else:
        print("[DONE] Figures skipped (SAVE_FIGURES=False)")
    print(f"[DONE] Success log: {SUCCESS_LOG}")
    print(f"[DONE] Fail log:    {FAIL_LOG}")
    print(f"[DONE] Total rows:  {len(rows)}")


if __name__ == "__main__":
    main()