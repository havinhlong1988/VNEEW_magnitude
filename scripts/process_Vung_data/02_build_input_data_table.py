#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import csv
import numpy as np
from obspy import read
from obspy.io.sac import SACTrace
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("/Users/vinhlongha/Research/EEW/input/2025_ML_VN/input_data/")   # root: year/event_id/*.disp.SAC
OUT_STATION_CSV = Path("output/station_magnitude_detail.csv")
OUT_EVENT_CSV   = Path("output/event_magnitude_summary.csv")

OUT_STATION_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_EVENT_CSV.parent.mkdir(parents=True, exist_ok=True)

# surface-wave start ~ distance / velocity
SURFACE_VEL_KM_S = 3.5

# fallback if distance missing
DEFAULT_SIGNAL_WINDOW_SEC = 5.0
MIN_SIGNAL_WINDOW_SEC = 1.0

# detrend window before taking amplitude
DETREND_WINDOW = True

# magnitude formula:
# M = log10(A_um) + ML_A * log10(R_km) + ML_B * R_km + ML_C
ML_A = 1.11
ML_B = 0.00189
ML_C = -2.09

# displacement unit of input *.disp.SAC
# supported:
#   "m"   = meter
#   "mm"  = millimeter
#   "um"  = micrometer
#   "mcm" = alias of micrometer
#   "nm"  = nanometer
DISP_UNIT = "m"

DEBUG_T1 = True

# ---------- STA/LTA fallback parameters ----------
USE_STALTA_IF_NO_T1 = True
STALTA_TRACE_ORDER = ["HHZ", "HHE", "HHN"]   # preferred trace for picking
STALTA_PREPROCESS_DEMEAN = True
STALTA_PREPROCESS_DETREND = True
STALTA_PREPROCESS_TAPER = True
STALTA_HIGHPASS_HZ = 1.0     # set None to disable
STALTA_STA_SEC = 0.2
STALTA_LTA_SEC = 2.0
STALTA_ON = 3.0
STALTA_OFF = 1.5
STALTA_SEARCH_START_SEC = 0.0
STALTA_SEARCH_END_SEC = None   # None = whole trace
# ================================================================ #


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def normalize_disp_unit(unit_str):
    """
    Normalize displacement unit string.
    Supported:
      m, mm, um, mcm, nm
    """
    if unit_str is None:
        raise ValueError("DISP_UNIT is None")

    u = str(unit_str).strip().lower()

    aliases = {
        "m": "m",
        "meter": "m",
        "meters": "m",

        "mm": "mm",
        "millimeter": "mm",
        "millimeters": "mm",

        "um": "um",
        "μm": "um",
        "micrometer": "um",
        "micrometers": "um",
        "micron": "um",
        "microns": "um",
        "mcm": "um",

        "nm": "nm",
        "nanometer": "nm",
        "nanometers": "nm",
    }

    if u not in aliases:
        raise ValueError(
            f"Unsupported DISP_UNIT = {unit_str!r}. "
            f"Use one of: m, mm, um, mcm, nm"
        )

    return aliases[u]

def get_dist_km_header_only(sac):
    if sac is None:
        return np.nan
    try:
        dist = float(sac.dist)
        if np.isfinite(dist) and dist > 0 and dist > -12344:
            return dist
    except Exception:
        pass
    return np.nan

def amplitude_to_um(amplitude_value, disp_unit):
    """
    Convert displacement amplitude from input unit to micrometers (um).
    """
    if not np.isfinite(amplitude_value):
        return np.nan

    u = normalize_disp_unit(disp_unit)

    factors_to_um = {
        "m":  1.0e6,   # 1 m  = 1e6 um
        "mm": 1.0e3,   # 1 mm = 1e3 um
        "um": 1.0,     # 1 um = 1 um
        "nm": 1.0e-3,  # 1 nm = 1e-3 um
    }

    return float(amplitude_value) * factors_to_um[u]


def linear_detrend(y):
    if y.size < 2:
        return y.copy()
    x = np.arange(y.size, dtype=float)
    p = np.polyfit(x, y, 1)
    return y - (p[0] * x + p[1])


def get_filename_station_channel(path):
    # expected clean filename: STA.CHAN.disp.SAC
    parts = path.name.split(".")
    if len(parts) < 4:
        return "", ""
    sta = parts[0].strip()
    chan = parts[1].strip().upper().replace("_", "").replace(" ", "").replace("-", "")
    return sta, chan


def get_sac_header(path):
    try:
        return SACTrace.read(str(path), headonly=True)
    except Exception:
        return None


def get_t1_from_sac(sac):
    if sac is None:
        return np.nan
    try:
        t1 = sac.t1
        if t1 is None:
            return np.nan
        t1 = float(t1)
        if np.isfinite(t1) and t1 > -12344 and abs(t1) < 1.0e20:
            return t1
    except Exception:
        pass
    return np.nan


def get_mag_from_sac(sac):
    if sac is None:
        return np.nan
    try:
        mag = float(sac.mag)
        if np.isfinite(mag) and mag > -12344 and abs(mag) < 100:
            return mag
    except Exception:
        pass
    return np.nan


def get_dist_km_from_sac(sac):
    if sac is None:
        return np.nan

    try:
        dist = float(sac.dist)
        if np.isfinite(dist) and dist > 0 and dist > -12344:
            return dist
    except Exception:
        pass

    try:
        evla = float(sac.evla)
        evlo = float(sac.evlo)
        stla = float(sac.stla)
        stlo = float(sac.stlo)
        if all(np.isfinite(v) for v in [evla, evlo, stla, stlo]):
            dist_m, _, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
            return dist_m / 1000.0
    except Exception:
        pass

    return np.nan


def get_relative_time_vector(tr):
    b = 0.0
    if hasattr(tr.stats, "sac"):
        try:
            b = float(tr.stats.sac.b)
            if not np.isfinite(b):
                b = 0.0
        except Exception:
            b = 0.0
    n = tr.stats.npts
    dt = tr.stats.delta
    return b + np.arange(n, dtype=float) * dt


def get_window_data(tr, t_start, t_end):
    t = get_relative_time_vector(tr)
    mask = (t >= t_start) & (t <= t_end)
    if np.count_nonzero(mask) < 2:
        return np.array([], dtype=float)
    return tr.data[mask].astype(float)


def get_window_amplitude(tr, t_start, t_end):
    y = get_window_data(tr, t_start, t_end)
    if y.size < 2:
        return np.nan

    if DETREND_WINDOW:
        y = linear_detrend(y)
    else:
        y = y - np.mean(y)

    amp = np.max(np.abs(y))
    return float(amp) if np.isfinite(amp) else np.nan


def local_magnitude_from_amp_um(A_um, R_km):
    if not np.isfinite(A_um) or A_um <= 0:
        return np.nan
    if not np.isfinite(R_km) or R_km <= 0:
        return np.nan
    return math.log10(A_um) + ML_A * math.log10(R_km) + ML_B * R_km + ML_C


def safe_mean(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def safe_std(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.std(vals, ddof=0))


def count_finite(values):
    vals = np.array(values, dtype=float)
    return int(np.sum(np.isfinite(vals)))


def pick_onset_stalta(tr):
    """
    Return onset time in SAC-relative seconds (same reference as b/t1),
    or np.nan if no onset found.
    """
    try:
        trp = tr.copy()

        if STALTA_PREPROCESS_DEMEAN:
            trp.detrend("demean")
        if STALTA_PREPROCESS_DETREND:
            trp.detrend("linear")
        if STALTA_PREPROCESS_TAPER:
            trp.taper(max_percentage=0.05, type="hann")
        if STALTA_HIGHPASS_HZ is not None:
            trp.filter("highpass", freq=STALTA_HIGHPASS_HZ, corners=2, zerophase=True)

        dt = trp.stats.delta
        nsta = max(1, int(round(STALTA_STA_SEC / dt)))
        nlta = max(nsta + 1, int(round(STALTA_LTA_SEC / dt)))

        t = get_relative_time_vector(trp)
        y = trp.data.astype(float)

        i0 = 0
        i1 = len(y)

        if STALTA_SEARCH_START_SEC is not None:
            idx = np.where(t >= STALTA_SEARCH_START_SEC)[0]
            if len(idx) == 0:
                return np.nan
            i0 = idx[0]

        if STALTA_SEARCH_END_SEC is not None:
            idx = np.where(t <= STALTA_SEARCH_END_SEC)[0]
            if len(idx) == 0:
                return np.nan
            i1 = idx[-1] + 1

        if i1 - i0 < nlta + 2:
            return np.nan

        ycut = y[i0:i1]
        tcut = t[i0:i1]

        cft = classic_sta_lta(ycut, nsta, nlta)
        on_off = trigger_onset(cft, STALTA_ON, STALTA_OFF)

        if len(on_off) == 0:
            return np.nan

        i_on = int(on_off[0][0])
        if i_on < 0 or i_on >= len(tcut):
            return np.nan

        return float(tcut[i_on])

    except Exception:
        return np.nan


def get_pick_time(tr_ref, sac_ref, event_id, sta, path_ref):
    """
    Priority:
      1) T1 from SAC header
      2) STA/LTA fallback
    Returns:
      pick_time_sec, pick_source
    """
    t1 = get_t1_from_sac(sac_ref)
    if np.isfinite(t1):
        return t1, "T1"

    if DEBUG_T1:
        raw_t1 = getattr(sac_ref, "t1", None) if sac_ref is not None else None
        raw_kt1 = getattr(sac_ref, "kt1", None) if sac_ref is not None else None
        print(f"[DEBUG] event={event_id}, station={sta}, raw_t1={raw_t1}, raw_kt1={raw_kt1}, sacfile={path_ref}")

    if USE_STALTA_IF_NO_T1:
        t_pick = pick_onset_stalta(tr_ref)
        if np.isfinite(t_pick):
            print(f"[PICK] Using STA/LTA for event={event_id}, station={sta}, t_pick={t_pick:.3f}, sacfile={path_ref}")
            return t_pick, "STA_LTA"

    print(f"[SKIP] No T1 and no STA/LTA pick for event={event_id}, station={sta}")
    return np.nan, "NONE"


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if not INPUT_ROOT.exists():
    raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT.resolve()}")

disp_unit_norm = normalize_disp_unit(DISP_UNIT)

event_dirs = []
for year_dir in sorted(INPUT_ROOT.iterdir()):
    if not year_dir.is_dir():
        continue
    for event_dir in sorted(year_dir.iterdir()):
        if event_dir.is_dir():
            event_dirs.append(event_dir)

print(f"INPUT_ROOT  : {INPUT_ROOT.resolve()}")
print(f"N_EVENTS    : {len(event_dirs)}")
print(f"OUT_STATION : {OUT_STATION_CSV}")
print(f"OUT_EVENT   : {OUT_EVENT_CSV}")
print(f"DISP_UNIT   : {disp_unit_norm}")

station_rows = []
event_rows = []

for event_dir in event_dirs:
    year = event_dir.parent.name
    event_id = event_dir.name

    sac_files = sorted([p for p in event_dir.glob("*.disp.SAC") if p.is_file()])
    if len(sac_files) == 0:
        continue

    station_dict = {}
    event_mag_headers = []
    event_dist_headers = []

    for sac_path in sac_files:
        try:
            st = read(str(sac_path))
            if len(st) == 0:
                continue
            tr = st[0]
            sac = get_sac_header(sac_path)

            sta, ch = get_filename_station_channel(sac_path)
            if not sta or not ch:
                print(f"[SKIP] Bad cleaned filename: {sac_path}")
                continue

            mag_hdr = get_mag_from_sac(sac)
            if np.isfinite(mag_hdr):
                event_mag_headers.append(mag_hdr)

            # dist_hdr = get_dist_km_from_sac(sac)
            # if np.isfinite(dist_hdr):
            #     event_dist_headers.append(dist_hdr)
            dist_hdr = get_dist_km_header_only(sac)
            if np.isfinite(dist_hdr):
                event_dist_headers.append(dist_hdr)

            if sta not in station_dict:
                station_dict[sta] = {}

            station_dict[sta][ch] = {
                "trace": tr,
                "path": sac_path,
                "sac": sac,
            }

        except Exception as e:
            print(f"[WARN] Cannot read {sac_path}: {e}")

    event_mag_mean = safe_mean(event_mag_headers)
    event_hypdist_mean = safe_mean(event_dist_headers)

    event_ml_z = []
    event_ml_h = []

    for sta, comp_dict in sorted(station_dict.items()):
        trZ = comp_dict.get("HHZ", {}).get("trace", None)
        sacZ = comp_dict.get("HHZ", {}).get("sac", None)

        trE = comp_dict.get("HHE", {}).get("trace", None)
        sacE = comp_dict.get("HHE", {}).get("sac", None)

        trN = comp_dict.get("HHN", {}).get("trace", None)
        sacN = comp_dict.get("HHN", {}).get("sac", None)

        if trZ is None:
            for k in comp_dict:
                if k.endswith("Z"):
                    trZ = comp_dict[k]["trace"]
                    sacZ = comp_dict[k]["sac"]
                    break

        if trE is None:
            for k in comp_dict:
                if k.endswith("E"):
                    trE = comp_dict[k]["trace"]
                    sacE = comp_dict[k]["sac"]
                    break

        if trN is None:
            for k in comp_dict:
                if k.endswith("N"):
                    trN = comp_dict[k]["trace"]
                    sacN = comp_dict[k]["sac"]
                    break

        # choose preferred trace for picking / metadata
        tr_ref = None
        sac_ref = None
        path_ref = None

        for want in STALTA_TRACE_ORDER:
            if want in comp_dict:
                tr_ref = comp_dict[want]["trace"]
                sac_ref = comp_dict[want]["sac"]
                path_ref = comp_dict[want]["path"]
                break

        if tr_ref is None:
            tr_ref = trZ if trZ is not None else (trE if trE is not None else trN)
            sac_ref = sacZ if sacZ is not None else (sacE if sacE is not None else sacN)
            if trZ is not None:
                path_ref = comp_dict.get("HHZ", {}).get("path", None)
            elif trE is not None:
                path_ref = comp_dict.get("HHE", {}).get("path", None)
            elif trN is not None:
                path_ref = comp_dict.get("HHN", {}).get("path", None)

        if tr_ref is None:
            continue

        dist_km = get_dist_km_from_sac(sac_ref)
        t_pick, pick_source = get_pick_time(tr_ref, sac_ref, event_id, sta, path_ref)

        if not np.isfinite(t_pick):
            continue

        if np.isfinite(dist_km) and dist_km > 0:
            surf_start = dist_km / SURFACE_VEL_KM_S
        else:
            surf_start = t_pick + DEFAULT_SIGNAL_WINDOW_SEC

        if not np.isfinite(surf_start) or surf_start <= t_pick:
            surf_start = t_pick + DEFAULT_SIGNAL_WINDOW_SEC

        if (surf_start - t_pick) < MIN_SIGNAL_WINDOW_SEC:
            surf_start = t_pick + MIN_SIGNAL_WINDOW_SEC

        amp_z_raw = np.nan
        amp_z_um = np.nan
        ml_z = np.nan

        if trZ is not None:
            amp_z_raw = get_window_amplitude(trZ, t_pick, surf_start)
            if np.isfinite(amp_z_raw):
                amp_z_um = amplitude_to_um(amp_z_raw, disp_unit_norm)
                ml_z = local_magnitude_from_amp_um(amp_z_um, dist_km)
                if np.isfinite(ml_z):
                    event_ml_z.append(ml_z)

        amp_h_raw = np.nan
        amp_h_um = np.nan
        ml_h = np.nan
        horiz_used = ""
        n_horiz = 0

        if trE is not None and trN is not None:
            yE = get_window_data(trE, t_pick, surf_start)
            yN = get_window_data(trN, t_pick, surf_start)

            if yE.size >= 2 and yN.size >= 2:
                n = min(len(yE), len(yN))
                yE = yE[:n]
                yN = yN[:n]

                if DETREND_WINDOW:
                    yE = linear_detrend(yE)
                    yN = linear_detrend(yN)
                else:
                    yE = yE - np.mean(yE)
                    yN = yN - np.mean(yN)

                yH = np.sqrt(yE**2 + yN**2)
                if yH.size > 0:
                    amp_h_raw = float(np.max(np.abs(yH)))
                    amp_h_um = amplitude_to_um(amp_h_raw, disp_unit_norm)
                    ml_h = local_magnitude_from_amp_um(amp_h_um, dist_km)
                    horiz_used = "EN_vector"
                    n_horiz = 2
                    if np.isfinite(ml_h):
                        event_ml_h.append(ml_h)

        elif trE is not None:
            amp_h_raw = get_window_amplitude(trE, t_pick, surf_start)
            if np.isfinite(amp_h_raw):
                amp_h_um = amplitude_to_um(amp_h_raw, disp_unit_norm)
                ml_h = local_magnitude_from_amp_um(amp_h_um, dist_km)
                horiz_used = "E_only"
                n_horiz = 1
                if np.isfinite(ml_h):
                    event_ml_h.append(ml_h)

        elif trN is not None:
            amp_h_raw = get_window_amplitude(trN, t_pick, surf_start)
            if np.isfinite(amp_h_raw):
                amp_h_um = amplitude_to_um(amp_h_raw, disp_unit_norm)
                ml_h = local_magnitude_from_amp_um(amp_h_um, dist_km)
                horiz_used = "N_only"
                n_horiz = 1
                if np.isfinite(ml_h):
                    event_ml_h.append(ml_h)

        station_rows.append({
            "year": year,
            "event_id": event_id,
            "event_mag_header_mean": event_mag_mean,
            "station": sta,
            "dist_km": dist_km,
            "pick_sec": t_pick,
            "pick_source": pick_source,
            "signal_end_sec": surf_start,
            "disp_unit": disp_unit_norm,
            "amp_z_raw": amp_z_raw,
            "amp_z_um": amp_z_um,
            "ml_z": ml_z,
            "amp_h_raw": amp_h_raw,
            "amp_h_um": amp_h_um,
            "ml_h": ml_h,
            "horiz_used": horiz_used,
            "n_horiz": n_horiz,
        })

    event_rows.append({
        "ID": event_id,
        "mean_header_mag": event_mag_mean,
        "mean_header_hypdist_km": event_hypdist_mean,
        "magnitudez": safe_mean(event_ml_z),
        "magnitudezstd": safe_std(event_ml_z),
        "magnitudeh": safe_mean(event_ml_h),
        "magnitudehstd": safe_std(event_ml_h),
        "nsta_z": count_finite(event_ml_z),
        "nsta_h": count_finite(event_ml_h),
    })

station_fieldnames = [
    "year",
    "event_id",
    "event_mag_header_mean",
    "station",
    "dist_km",
    "pick_sec",
    "pick_source",
    "signal_end_sec",
    "disp_unit",
    "amp_z_raw",
    "amp_z_um",
    "ml_z",
    "amp_h_raw",
    "amp_h_um",
    "ml_h",
    "horiz_used",
    "n_horiz",
]

with open(OUT_STATION_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=station_fieldnames)
    writer.writeheader()
    writer.writerows(station_rows)

event_fieldnames = [
    "ID",
    "mean_header_mag",
    "mean_header_hypdist_km",
    "magnitudez",
    "magnitudezstd",
    "magnitudeh",
    "magnitudehstd",
    "nsta_z",
    "nsta_h",
]

with open(OUT_EVENT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=event_fieldnames)
    writer.writeheader()
    writer.writerows(event_rows)

print("-" * 60)
print(f"[OK] Station detail saved : {OUT_STATION_CSV}")
print(f"[OK] Event summary saved  : {OUT_EVENT_CSV}")
print(f"[OK] N_station_rows       : {len(station_rows)}")
print(f"[OK] N_event_rows         : {len(event_rows)}")