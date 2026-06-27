#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import csv
import warnings
import re
import traceback

import numpy as np
import pygmt
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy.geodetics import gps2dist_azimuth


# =============================================================================
# USER PARAMETERS
# =============================================================================
INPUT_ROOT = Path("output/02_cut_main_signal_windows_amp")
OUTPUT_CSV = Path("output/03_ml_check_amp/event_station_ml.csv")

# Figures
FIG_ROOT = Path("figures/03_ml_check_amp")
SAVE_FIGURES = False # Plot the figure for main signal window and peak location 

# Input file pattern: already WA displacement
INPUT_GLOB = "*.cut.SAC"

# Unit of WA displacement trace
# choose one of: "m", "cm", "mm", "um", "nm", or None
DATA_UNIT = "nm"
AUTO_UNIT_FALLBACK = "nm"

# Distance service limit of provided equations
MAX_SERVICE_DISTANCE_KM = 600.0

# -------------------------------------------------------------------------
# Window settings
# Measure maximum amplitude AFTER S arrival
# -------------------------------------------------------------------------
POST_S_SEC = 200.0
MIN_AFTER_S_SEC = 5.0
USE_PICKS_FROM_ALL_COMPONENTS = True

# If S is missing:
# estimate S from P using assumed velocities
ESTIMATE_S_IF_MISSING = True
ASSUME_VP_KM_S = 6.0
ASSUME_VS_KM_S = 3.5

# If both P and S missing, fallback to STA/LTA
USE_STALTA_IF_NO_PHASE = True
STALTA_AFTER_ON_SEC = 300.0

# fallback if STA/LTA fails completely
FALLBACK_SIGNAL_LEN_SEC = 300.0

# STA/LTA settings
STA_SEC = 0.5
LTA_SEC = 5.0
TRIG_ON = 3.0
TRIG_OFF = 1.2

# Optional preprocessing before measurement
DO_DEMEAN = True
DO_DETREND = True
DO_TAPER = True
TAPER_MAX_PERCENTAGE = 0.05

DO_HIGHPASS = True
HIGHPASS_FREQ = 0.02   # Hz for WA displacement baseline correction

# Plot settings
PLOT_WIDTH = "16c"
PLOT_HEIGHT = "7c"
WINDOW_FILL = "gray90"
TRACE_PEN = "1p,dodgerblue3"
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
    if distance > MAX_SERVICE_DISTANCE_KM:
        ML = 0
        logA0 = 0
    else:
        if correction == []:
            correction = 0
        logA0 = -(1.110 * np.log10(distance / 100) + 0.00189 * (distance - 100) + 3.0)
        ML = np.log10(amplitude) + 1.11 * np.log10(distance) + 0.00189 * distance - 2.09
    return ML, logA0


def le08(amplitude, distance, correction):
    if distance > MAX_SERVICE_DISTANCE_KM:
        ML = 0
        logA0 = 0
    else:
        logA0 = -(1.018 * np.log10(distance / 100) + 0.00232 * (distance - 100) + 3.00)
        ML = np.log10(amplitude) + 1.018 * np.log10(distance) + 0.00232 * distance - 2.09 + correction
    return ML, logA0


def nguyen11(amplitude, distance, correction):
    if distance > MAX_SERVICE_DISTANCE_KM:
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
        (r"\bnm\b", "nm"),
        (r"\bum\b|\bμm\b", "um"),
        (r"\bmm\b", "mm"),
        (r"\bcm\b", "cm"),
        (r"\bm\b", "m"),
    ]
    for pat, unit in patterns:
        if re.search(pat, blob):
            return unit

    return AUTO_UNIT_FALLBACK


def get_distance_km(tr):
    dist = safe_float(get_sac_header(tr, "dist"))
    if dist is not None and dist > 0:
        return dist

    stla = safe_float(get_sac_header(tr, "stla"))
    stlo = safe_float(get_sac_header(tr, "stlo"))
    evla = safe_float(get_sac_header(tr, "evla"))
    evlo = safe_float(get_sac_header(tr, "evlo"))
    evdp = safe_float(get_sac_header(tr, "evdp"))

    if None in (stla, stlo, evla, evlo):
        return None

    epi_m, _, _ = gps2dist_azimuth(stla, stlo, evla, evlo)
    epi_km = epi_m / 1000.0

    if evdp is None:
        return epi_km

    return math.sqrt(epi_km ** 2 + evdp ** 2)


def get_event_magnitude(tr):
    mag = safe_float(get_sac_header(tr, "mag"))
    return mag


def get_component(tr, sacfile=None):
    comp = safe_str(getattr(tr.stats, "channel", ""))
    if comp:
        return comp

    comp = safe_str(get_sac_header(tr, "kcmpnm", ""))
    if comp:
        return comp

    if sacfile is not None:
        parts = sacfile.name.split(".")
        if len(parts) >= 2:
            return parts[1]

    return "UNK"


def get_all_phase_picks_from_header(tr):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return []

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

    return picks


def is_p_label(lbl: str) -> bool:
    lbl = lbl.strip().upper()
    return lbl.startswith("P") or lbl in {"PG", "PN", "P1", "P2"}


def is_s_label(lbl: str) -> bool:
    lbl = lbl.strip().upper()
    return lbl.startswith("S") or lbl in {"SG", "SN", "S1", "S2"}


def extract_phase_picks_all_components(files):
    p_candidates = []
    s_candidates = []

    for sacfile in files:
        try:
            st = read(str(sacfile), headonly=True)
            if len(st) == 0:
                continue
            tr = st[0]
            picks = get_all_phase_picks_from_header(tr)

            for key, val, lab in picks:
                labu = safe_str(lab).upper()
                if is_p_label(labu):
                    p_candidates.append((val, labu, sacfile.name))
                if is_s_label(labu):
                    s_candidates.append((val, labu, sacfile.name))
        except Exception:
            continue

    p_pick = min([x[0] for x in p_candidates]) if p_candidates else None
    s_pick = min([x[0] for x in s_candidates]) if s_candidates else None

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


def estimate_s_from_p(p_pick, distance_km):
    if p_pick is None or distance_km is None or distance_km <= 0:
        return None
    dt_sp = distance_km * (1.0 / ASSUME_VS_KM_S - 1.0 / ASSUME_VP_KM_S)
    return p_pick + dt_sp


def preprocess_trace(tr):
    tr2 = tr.copy()
    tr2.data = np.asarray(tr2.data, dtype=float)

    if DO_DEMEAN:
        tr2.detrend("demean")
    if DO_DETREND:
        tr2.detrend("linear")
    if DO_TAPER:
        tr2.taper(max_percentage=TAPER_MAX_PERCENTAGE, type="cosine")
    if DO_HIGHPASS:
        tr2.filter("highpass", freq=HIGHPASS_FREQ, corners=2, zerophase=True)

    return tr2


def build_measurement_window_after_s(tr, p_rel, s_rel):
    if s_rel is not None:
        i1 = sac_relative_time_to_index(tr, s_rel)
        i2 = sac_relative_time_to_index(tr, s_rel + POST_S_SEC)
        if i1 is not None and i2 is not None and i2 > i1:
            return i1, i2, "after_S"

    if USE_STALTA_IF_NO_PHASE:
        i_on = detect_signal_onset_stalta(tr)
        if i_on is not None:
            i1 = max(0, i_on)
            i2 = min(tr.stats.npts - 1, i1 + int(round(STALTA_AFTER_ON_SEC / tr.stats.delta)))
            if i2 > i1:
                return i1, i2, "stalta"

    i_peak = int(np.argmax(np.abs(np.asarray(tr.data, dtype=float))))
    i1 = max(0, i_peak)
    i2 = min(tr.stats.npts - 1, i1 + int(round(FALLBACK_SIGNAL_LEN_SEC / tr.stats.delta)))
    return i1, i2, "fallback_peak"


def compute_wa_amplitude_mm(tr, i1, i2, data_unit):
    i1 = max(0, i1)
    i2 = min(tr.stats.npts - 1, i2)
    if i2 <= i1:
        raise ValueError("Invalid measurement window")

    tr_wa = preprocess_trace(tr)
    win = np.asarray(tr_wa.data[i1:i2 + 1], dtype=float)
    win = np.nan_to_num(win)

    local_peak_idx = int(np.argmax(np.abs(win)))
    peak_idx = i1 + local_peak_idx
    peak_val_native = float(tr_wa.data[peak_idx])
    wa_amp_native = abs(peak_val_native)
    wa_amp_mm = convert_length_value(wa_amp_native, data_unit, "mm")

    return wa_amp_mm, peak_idx, peak_val_native, tr_wa


def compute_ml_values(amplitude_mm, distance_km):
    if distance_km is None or distance_km <= 0 or not np.isfinite(amplitude_mm) or amplitude_mm <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    amplitude_nm = convert_length_value(amplitude_mm, "mm", "nm")
    amplitude_um = convert_length_value(amplitude_mm, "mm", "um")

    ML_hut, logA0_hut = hutton83(amplitude_nm, distance_km, 0) # Hutton83 does not have correction term, always set to 0
    ML_le, logA0_le = le08(amplitude_nm, distance_km, 0) # Le08 uses nm unit and does not have correction term, always set to 0
    # ML_le, logA0_le = le08(amplitude_nm, distance_km, 0) # Le08 uses nm unit and does not have correction term, always set to 0
    ML_ng, logA0_ng = nguyen11(amplitude_um, distance_km, 0) # Nguyen11 does not have correction term, always set to 0  

    return logA0_hut, logA0_le, logA0_ng, ML_hut, ML_le, ML_ng


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


def iter_event_dirs(root: Path):
    for evdir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield evdir


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
    tr_wa,
    i1,
    i2,
    peak_idx,
    peak_val_native,
    p_rel=None,
    s_rel=None,
    data_unit="m",
    distance_km=None,
    phase_source="",
    wa_amp_mm=np.nan,
):
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    n = tr_wa.stats.npts
    x = np.array([index_to_sac_relative_time(tr_wa, i) for i in range(n)], dtype=float)

    y_native = np.asarray(tr_wa.data, dtype=float)
    y_mm = convert_length_value(y_native, data_unit, "mm")
    ywin = y_mm[i1:i2 + 1]

    peak_x = x[peak_idx]
    peak_y = convert_length_value(peak_val_native, data_unit, "mm")
    region = nice_region(x, y_mm)
    maxabs = float(np.max(np.abs(ywin)))

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
            'WSen+tML check (WA displacement input)',
            'xaf+l"Time relative to SAC reference (s)"',
            'yaf+l"Amplitude (mm)"',
        ],
    )

    x1 = x[i1]
    x2 = x[i2]
    rect_x = [x1, x2, x2, x1, x1]
    rect_y = [-maxabs, -maxabs, maxabs, maxabs, -maxabs]

    fig.plot(x=rect_x, y=rect_y, fill=WINDOW_FILL, pen="0.2p,gray70")
    fig.plot(x=x, y=y_mm, pen=TRACE_PEN)
    fig.plot(x=[peak_x], y=[peak_y], style=PEAK_STYLE, pen=PEAK_PEN)

    if p_rel is not None:
        fig.plot(x=[p_rel, p_rel], y=[region[2], region[3]], pen=PHASE_P_PEN)
    if s_rel is not None:
        fig.plot(x=[s_rel, s_rel], y=[region[2], region[3]], pen=PHASE_S_PEN)

    text_lines = [
        f"event = {event_name}",
        f"phase_source = {phase_source}",
        f"window = {x[i1]:.2f} to {x[i2]:.2f} s",
        f"WAdisp = {wa_amp_mm:.4e} mm",
    ]
    if distance_km is not None:
        text_lines.append(f"distance = {distance_km:.2f} km")

    tx = region[0] + 0.02 * (region[1] - region[0])
    ty = region[3] - 0.05 * (region[3] - region[2])
    dy = 0.06 * (region[3] - region[2])

    for i, line in enumerate(text_lines):
        fig.text(
            x=tx,
            y=ty - i * dy,
            text=line,
            font="9p,Helvetica,black",
            justify="LM",
        )

    fig.savefig(str(fig_path))


# =============================================================================
# MAIN
# =============================================================================
def main():
    reset_logs()
    rows = []

    for evdir in iter_event_dirs(INPUT_ROOT):
        event_name = evdir.name
        sacfiles = sorted(evdir.glob(INPUT_GLOB))
        if not sacfiles:
            continue

        p_all, s_all = extract_phase_picks_all_components(sacfiles)

        for sacfile in sacfiles:
            fname = sacfile.name

            try:
                st = read(str(sacfile))
                if len(st) == 0:
                    raise RuntimeError("Empty stream")
                tr = st[0]

                data_unit = guess_data_unit_from_sac(tr)
                distance_km = get_distance_km(tr)
                magnitude = get_event_magnitude(tr)
                component = get_component(tr, sacfile)

                p_rel = p_all
                s_rel = s_all
                phase_source = "header_all_components"

                if s_rel is None and p_rel is not None and ESTIMATE_S_IF_MISSING:
                    s_rel = estimate_s_from_p(p_rel, distance_km)
                    if s_rel is not None:
                        phase_source = "P_header_S_estimated"

                i1, i2, window_source = build_measurement_window_after_s(tr, p_rel, s_rel)
                phase_source = f"{phase_source}|{window_source}"

                wa_amp_mm, peak_idx, peak_val_native, tr_wa = compute_wa_amplitude_mm(
                    tr, i1, i2, data_unit
                )

                peak_time_rel = index_to_sac_relative_time(tr, peak_idx)
                window_start_rel = index_to_sac_relative_time(tr, i1)
                window_end_rel = index_to_sac_relative_time(tr, i2)

                (
                    LogA0Huton83,
                    LogA0Le08,
                    LogA0Nguyen11,
                    MLHuton83,
                    MLLe08,
                    MLNguyen11,
                ) = compute_ml_values(wa_amp_mm, distance_km)

                sta, chan = get_sta_chan_from_filename_or_trace(sacfile, tr)

                if SAVE_FIGURES:
                    fig_path = FIG_ROOT / event_name / f"{sta}.{chan}.png"
                    make_qc_figure(
                        event_name=event_name,
                        fig_path=fig_path,
                        tr_wa=tr_wa,
                        i1=i1,
                        i2=i2,
                        peak_idx=peak_idx,
                        peak_val_native=peak_val_native,
                        p_rel=p_rel,
                        s_rel=s_rel,
                        data_unit=data_unit,
                        distance_km=distance_km,
                        phase_source=phase_source,
                        wa_amp_mm=wa_amp_mm,
                    )
                    print(f"[FIG] saved: {fig_path}")
                    fig_msg = str(fig_path)
                else:
                    fig_msg = "SKIPPED"

                row = {
                    "EVENT": event_name,
                    "filename": fname,
                    "component": component,
                    "distance": distance_km,
                    "magnitude": np.round(magnitude,2),
                    "WAdisp": wa_amp_mm,
                    "window_start_rel": window_start_rel,
                    "window_end_rel": window_end_rel,
                    "peak_time_rel": peak_time_rel,
                    "p_pick_rel": p_rel,
                    "s_pick_rel": s_rel,
                    "phase_source": phase_source,

                    "LogA0Huton83": LogA0Huton83,
                    "LogA0Le08": LogA0Le08,
                    "LogA0Nguyen11": LogA0Nguyen11,
                    "MLHuton83": MLHuton83,
                    "MLLe08": MLLe08,
                    "MLNguyen11": MLNguyen11,
                }
                rows.append(row)

                dist_str = f"{distance_km:.3f}" if distance_km is not None else "None"
                mag_str = f"{magnitude:.2f}" if magnitude is not None else "None"
                log_success(
                    f"[OK] {event_name} | {fname} | comp={component} | mag={mag_str} | unit={data_unit} | "
                    f"phase_source={phase_source} | dist={dist_str} km | "
                    f"window=({window_start_rel:.2f},{window_end_rel:.2f}) | "
                    f"WAdisp={wa_amp_mm:.6e} mm | fig={fig_msg}"
                )

            except Exception as e:
                msg = f"[FAIL] {event_name} | {fname} | {type(e).__name__}: {e}"
                print(msg)
                traceback.print_exc()
                log_fail(msg)

    fieldnames = [
        "EVENT",
        "filename",
        "component",
        "distance",
        "magnitude",
        "WAdisp",
        "window_start_rel",
        "window_end_rel",
        "peak_time_rel",
        "p_pick_rel",
        "s_pick_rel",
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