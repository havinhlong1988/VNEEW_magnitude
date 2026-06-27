#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_magnitude_statistic.py

Collect all cut SAC data from step-2 and calculate station/event magnitude statistics.

Input structure:
    output/02_cut_main_signal_windows/{event_id}/*.disp.SAC

Output:
    output/03_magnitude_statistic/station_magnitude_detail.csv
    output/03_magnitude_statistic/event_magnitude_summary_Z.csv
    output/03_magnitude_statistic/event_magnitude_summary_H.csv
    output/03_magnitude_statistic/event_magnitude_summary.csv
    output/03_magnitude_statistic/magnitude_success.log
    output/03_magnitude_statistic/magnitude_fail.log

Main idea follows the pasted script, but adapted to:
- use step-2 cut files
- compute magnitude from the cut main-signal window
- compute each horizontal component individually
- also compute EN vector / RT vector if both horizontals are available
"""

from pathlib import Path
from datetime import datetime
import math
import csv
import traceback
import numpy as np
from obspy import read
from obspy.io.sac import SACTrace
from obspy.geodetics import gps2dist_azimuth


# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("output/02_cut_main_signal_windows")
OUTPUT_ROOT = Path("output/03_magnitude_statistic")

OUT_STATION_CSV = OUTPUT_ROOT / "station_magnitude_detail.csv"
OUT_EVENT_Z_CSV = OUTPUT_ROOT / "event_magnitude_summary_Z.csv"
OUT_EVENT_H_CSV = OUTPUT_ROOT / "event_magnitude_summary_H.csv"
OUT_EVENT_ALL_CSV = OUTPUT_ROOT / "event_magnitude_summary.csv"
SUCCESS_LOG = OUTPUT_ROOT / "magnitude_success.log"
FAIL_LOG = OUTPUT_ROOT / "magnitude_fail.log"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

INPUT_GLOB = "*.disp.SAC"

# displacement unit of input *.disp.SAC
# supported:
#   "m"   = meter
#   "mm"  = millimeter
#   "um"  = micrometer
#   "mcm" = alias of micrometer
#   "nm"  = nanometer
DISP_UNIT = "um"

# detrend cut window before taking amplitude
DETREND_WINDOW = True

# ----- Magnitude formula -----
# Default: local magnitude ML
# M = log10(A_um) + MAG_A * log10(R_km) + MAG_B * R_km + MAG_C
MAG_TYPE = "ML"
MAG_A = 1.11
MAG_B = 0.00189
MAG_C = -2.09

# event folder: first 14 chars digits
ONLY_EVENT_DIR_14DIGIT = True
# ================================================================ #


# ================================================================
# HELPERS
# ================================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def reset_logs():
    for p in [SUCCESS_LOG, FAIL_LOG]:
        if p.exists():
            p.unlink()
    write_log(SUCCESS_LOG, f"# magnitude success log started: {now_str()}")
    write_log(FAIL_LOG, f"# magnitude fail log started: {now_str()}")


def is_event_dir_name(name: str) -> bool:
    if not ONLY_EVENT_DIR_14DIGIT:
        return True
    return len(name) >= 14 and name[:14].isdigit()


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def normalize_disp_unit(unit_str):
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


def amplitude_to_um(amplitude_value, disp_unit):
    if not np.isfinite(amplitude_value):
        return np.nan

    u = normalize_disp_unit(disp_unit)
    factors_to_um = {
        "m":  1.0e6,
        "mm": 1.0e3,
        "um": 1.0,
        "nm": 1.0e-3,
    }
    return float(amplitude_value) * factors_to_um[u]


def linear_detrend(y):
    if y.size < 2:
        return y.copy()
    x = np.arange(y.size, dtype=float)
    p = np.polyfit(x, y, 1)
    return y - (p[0] * x + p[1])


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


# ================================================================
# SAC / FILENAME HELPERS
# ================================================================
def get_sac_header(path):
    try:
        return SACTrace.read(str(path), headonly=True)
    except Exception:
        return None


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


def get_filename_station_channel_network(path):
    """
    Support names like:
        STA.CHAN.disp.SAC
        STA.CHAN.main.SAC
        eventid_NET_STA_CHAN.disp.SAC
    """
    name = path.name
    stem = path.stem  # remove .SAC

    parts_us = stem.split("_")
    if len(parts_us) >= 4:
        evt = parts_us[0].strip()
        net = parts_us[1].strip().upper()
        sta = parts_us[2].strip().upper()
        chan = parts_us[3].split(".")[0].strip().upper().replace("-", "").replace(" ", "")
        if len(evt) >= 14 and sta and chan:
            return net, sta, chan

    parts_dot = name.split(".")
    if len(parts_dot) >= 4:
        sta = parts_dot[0].strip().upper()
        chan = parts_dot[1].strip().upper().replace("_", "").replace(" ", "").replace("-", "")
        return "", sta, chan

    return "", "", ""


# ================================================================
# AMPLITUDE / MAGNITUDE
# ================================================================
def get_trace_amplitude(tr):
    """
    Use the whole cut main-signal window.
    """
    y = tr.data.astype(float)
    if y.size < 2:
        return np.nan

    if DETREND_WINDOW:
        y = linear_detrend(y)
    else:
        y = y - np.mean(y)

    amp = np.max(np.abs(y))
    return float(amp) if np.isfinite(amp) else np.nan


def get_vector_amplitude(tr1, tr2):
    y1 = tr1.data.astype(float)
    y2 = tr2.data.astype(float)
    if y1.size < 2 or y2.size < 2:
        return np.nan

    n = min(len(y1), len(y2))
    y1 = y1[:n]
    y2 = y2[:n]

    if DETREND_WINDOW:
        y1 = linear_detrend(y1)
        y2 = linear_detrend(y2)
    else:
        y1 = y1 - np.mean(y1)
        y2 = y2 - np.mean(y2)

    yv = np.sqrt(y1**2 + y2**2)
    amp = np.max(np.abs(yv))
    return float(amp) if np.isfinite(amp) else np.nan


def magnitude_from_amplitude_um(A_um, R_km):
    """
    Default input equation:
        M = log10(A_um) + MAG_A * log10(R_km) + MAG_B * R_km + MAG_C
    """
    if not np.isfinite(A_um) or A_um <= 0:
        return np.nan
    if not np.isfinite(R_km) or R_km <= 0:
        return np.nan
    return math.log10(A_um) + MAG_A * math.log10(R_km) + MAG_B * R_km + MAG_C


# ================================================================
# MAIN
# ================================================================
def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT.resolve()}")

    reset_logs()
    disp_unit_norm = normalize_disp_unit(DISP_UNIT)

    event_dirs = [p for p in sorted(INPUT_ROOT.iterdir())
                  if p.is_dir() and is_event_dir_name(p.name)]

    print(f"INPUT_ROOT          : {INPUT_ROOT.resolve()}")
    print(f"N_EVENTS            : {len(event_dirs)}")
    print(f"OUT_STATION         : {OUT_STATION_CSV}")
    print(f"OUT_EVENT_Z_CSV     : {OUT_EVENT_Z_CSV}")
    print(f"OUT_EVENT_H_CSV     : {OUT_EVENT_H_CSV}")
    print(f"OUT_EVENT_ALL_CSV   : {OUT_EVENT_ALL_CSV}")
    print(f"DISP_UNIT           : {disp_unit_norm}")
    print(f"MAG_TYPE            : {MAG_TYPE}")
    print(f"MAG_EQ              : M = log10(A_um) + {MAG_A}*log10(R_km) + {MAG_B}*R_km + {MAG_C}")

    station_rows = []
    event_rows_z = []
    event_rows_h = []
    event_rows_all = []

    for event_dir in event_dirs:
        event_id = event_dir.name
        sac_files = sorted([p for p in event_dir.glob(INPUT_GLOB) if p.is_file()])
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

                net, sta, ch = get_filename_station_channel_network(sac_path)
                if not sta or not ch:
                    msg = f"[SKIP] Bad SAC filename: {sac_path}"
                    print(msg)
                    write_log(FAIL_LOG, msg)
                    continue

                mag_hdr = get_mag_from_sac(sac)
                if np.isfinite(mag_hdr):
                    event_mag_headers.append(mag_hdr)

                dist_hdr = get_dist_km_from_sac(sac)
                if np.isfinite(dist_hdr):
                    event_dist_headers.append(dist_hdr)

                if sta not in station_dict:
                    station_dict[sta] = {}

                station_dict[sta][ch] = {
                    "trace": tr,
                    "path": sac_path,
                    "sac": sac,
                    "net": net,
                }

            except Exception as e:
                err = f"{sac_path} | ERROR reading: {e}"
                print(f"[WARN] {err}")
                write_log(FAIL_LOG, err)
                write_log(FAIL_LOG, traceback.format_exc())

        event_mag_mean = safe_mean(event_mag_headers)
        event_hypdist_mean = safe_mean(event_dist_headers)

        event_ml_z = []
        event_ml_h = []
        event_ml_h_comp = []

        for sta, comp_dict in sorted(station_dict.items()):
            trZ = sacZ = None
            trE = sacE = None
            trN = sacN = None
            trR = sacR = None
            trT = sacT = None

            for ch, item in comp_dict.items():
                if ch.endswith("Z") and trZ is None:
                    trZ, sacZ = item["trace"], item["sac"]
                elif ch.endswith("E") and trE is None:
                    trE, sacE = item["trace"], item["sac"]
                elif ch.endswith("N") and trN is None:
                    trN, sacN = item["trace"], item["sac"]
                elif ch.endswith("R") and trR is None:
                    trR, sacR = item["trace"], item["sac"]
                elif ch.endswith("T") and trT is None:
                    trT, sacT = item["trace"], item["sac"]

            sac_ref = None
            net_ref = ""
            for ch, item in comp_dict.items():
                sac_ref = item["sac"]
                net_ref = item.get("net", "")
                if sac_ref is not None:
                    break

            dist_km = get_dist_km_from_sac(sac_ref)

            amp_z_raw = np.nan
            amp_z_um = np.nan
            ml_z = np.nan

            if trZ is not None:
                amp_z_raw = get_trace_amplitude(trZ)
                if np.isfinite(amp_z_raw):
                    amp_z_um = amplitude_to_um(amp_z_raw, disp_unit_norm)
                    ml_z = magnitude_from_amplitude_um(amp_z_um, dist_km)
                    if np.isfinite(ml_z):
                        event_ml_z.append(ml_z)

            amp_e_raw = amp_e_um = ml_e = np.nan
            amp_n_raw = amp_n_um = ml_n = np.nan
            amp_r_raw = amp_r_um = ml_r = np.nan
            amp_t_raw = amp_t_um = ml_t = np.nan

            if trE is not None:
                amp_e_raw = get_trace_amplitude(trE)
                if np.isfinite(amp_e_raw):
                    amp_e_um = amplitude_to_um(amp_e_raw, disp_unit_norm)
                    ml_e = magnitude_from_amplitude_um(amp_e_um, dist_km)
                    if np.isfinite(ml_e):
                        event_ml_h_comp.append(ml_e)

            if trN is not None:
                amp_n_raw = get_trace_amplitude(trN)
                if np.isfinite(amp_n_raw):
                    amp_n_um = amplitude_to_um(amp_n_raw, disp_unit_norm)
                    ml_n = magnitude_from_amplitude_um(amp_n_um, dist_km)
                    if np.isfinite(ml_n):
                        event_ml_h_comp.append(ml_n)

            if trR is not None:
                amp_r_raw = get_trace_amplitude(trR)
                if np.isfinite(amp_r_raw):
                    amp_r_um = amplitude_to_um(amp_r_raw, disp_unit_norm)
                    ml_r = magnitude_from_amplitude_um(amp_r_um, dist_km)
                    if np.isfinite(ml_r):
                        event_ml_h_comp.append(ml_r)

            if trT is not None:
                amp_t_raw = get_trace_amplitude(trT)
                if np.isfinite(amp_t_raw):
                    amp_t_um = amplitude_to_um(amp_t_raw, disp_unit_norm)
                    ml_t = magnitude_from_amplitude_um(amp_t_um, dist_km)
                    if np.isfinite(ml_t):
                        event_ml_h_comp.append(ml_t)

            amp_hvec_raw = np.nan
            amp_hvec_um = np.nan
            ml_hvec = np.nan
            horiz_vector_used = ""

            if trE is not None and trN is not None:
                amp_hvec_raw = get_vector_amplitude(trE, trN)
                horiz_vector_used = "EN_vector"
            elif trR is not None and trT is not None:
                amp_hvec_raw = get_vector_amplitude(trR, trT)
                horiz_vector_used = "RT_vector"

            if np.isfinite(amp_hvec_raw):
                amp_hvec_um = amplitude_to_um(amp_hvec_raw, disp_unit_norm)
                ml_hvec = magnitude_from_amplitude_um(amp_hvec_um, dist_km)
                if np.isfinite(ml_hvec):
                    event_ml_h.append(ml_hvec)

            n_horiz = int(sum([
                trE is not None,
                trN is not None,
                trR is not None,
                trT is not None,
            ]))

            station_rows.append({
                "event_id": event_id,
                "event_mag_header_mean": event_mag_mean,
                "network": net_ref,
                "station": sta,
                "dist_km": dist_km,
                "disp_unit": disp_unit_norm,
                "mag_type": MAG_TYPE,

                "amp_z_raw": amp_z_raw,
                "amp_z_um": amp_z_um,
                "mag_z": ml_z,

                "amp_e_raw": amp_e_raw,
                "amp_e_um": amp_e_um,
                "mag_e": ml_e,

                "amp_n_raw": amp_n_raw,
                "amp_n_um": amp_n_um,
                "mag_n": ml_n,

                "amp_r_raw": amp_r_raw,
                "amp_r_um": amp_r_um,
                "mag_r": ml_r,

                "amp_t_raw": amp_t_raw,
                "amp_t_um": amp_t_um,
                "mag_t": ml_t,

                "amp_hvec_raw": amp_hvec_raw,
                "amp_hvec_um": amp_hvec_um,
                "mag_hvec": ml_hvec,
                "horiz_vector_used": horiz_vector_used,
                "n_horiz": n_horiz,
            })

            # summary row for Z
            event_rows_z.append({
                "ID": event_id,
                "network": net_ref,
                "station": sta,
                "Ml_on_this_station": ml_z,
                "mean_header_mag": event_mag_mean,
                "mean_header_hypdist_km": event_hypdist_mean,
                "dist_km": dist_km,
                "magnitude_z_mean": ml_z,
                "magnitude_z_std": np.nan,
                "nsta_z": 1 if np.isfinite(ml_z) else 0,
            })

            # summary row for H
            ml_h_station = ml_hvec if np.isfinite(ml_hvec) else safe_mean([ml_e, ml_n, ml_r, ml_t])
            event_rows_h.append({
                "ID": event_id,
                "network": net_ref,
                "station": sta,
                "Ml_on_this_station": ml_h_station,
                "mean_header_mag": event_mag_mean,
                "mean_header_hypdist_km": event_hypdist_mean,
                "dist_km": dist_km,
                "magnitude_h_vector_mean": ml_hvec,
                "magnitude_h_vector_std": np.nan,
                "nsta_h_vector": 1 if np.isfinite(ml_hvec) else 0,
                "magnitude_h_component_mean": safe_mean([ml_e, ml_n, ml_r, ml_t]),
                "magnitude_h_component_std": safe_std([ml_e, ml_n, ml_r, ml_t]),
                "nsta_h_component": count_finite([ml_e, ml_n, ml_r, ml_t]),
            })

            # combined summary row
            event_rows_all.append({
                "ID": event_id,
                "network": net_ref,
                "station": sta,
                "mean_header_mag": event_mag_mean,
                "mean_header_hypdist_km": event_hypdist_mean,
                "dist_km": dist_km,
                "Ml_Z_on_this_station": ml_z,
                "Ml_H_on_this_station": ml_h_station,
                "Ml_H_vector_on_this_station": ml_hvec,
                "Ml_H_component_mean_on_this_station": safe_mean([ml_e, ml_n, ml_r, ml_t]),
                "mag_e": ml_e,
                "mag_n": ml_n,
                "mag_r": ml_r,
                "mag_t": ml_t,
            })

            ok_msg = (
                f"event={event_id} | sta={sta} | net={net_ref or 'NONE'} | "
                f"dist_km={dist_km if np.isfinite(dist_km) else 'nan'} | "
                f"mag_z={ml_z if np.isfinite(ml_z) else 'nan'} | "
                f"mag_e={ml_e if np.isfinite(ml_e) else 'nan'} | "
                f"mag_n={ml_n if np.isfinite(ml_n) else 'nan'} | "
                f"mag_r={ml_r if np.isfinite(ml_r) else 'nan'} | "
                f"mag_t={ml_t if np.isfinite(ml_t) else 'nan'} | "
                f"mag_hvec={ml_hvec if np.isfinite(ml_hvec) else 'nan'}"
            )
            write_log(SUCCESS_LOG, ok_msg)

        # event-level aggregated rows
        event_rows_z.append({
            "ID": event_id,
            "network": "ALL",
            "station": "ALL",
            "Ml_on_this_station": safe_mean(event_ml_z),
            "mean_header_mag": event_mag_mean,
            "mean_header_hypdist_km": event_hypdist_mean,
            "dist_km": event_hypdist_mean,
            "magnitude_z_mean": safe_mean(event_ml_z),
            "magnitude_z_std": safe_std(event_ml_z),
            "nsta_z": count_finite(event_ml_z),
        })

        event_rows_h.append({
            "ID": event_id,
            "network": "ALL",
            "station": "ALL",
            "Ml_on_this_station": safe_mean(event_ml_h),
            "mean_header_mag": event_mag_mean,
            "mean_header_hypdist_km": event_hypdist_mean,
            "dist_km": event_hypdist_mean,
            "magnitude_h_vector_mean": safe_mean(event_ml_h),
            "magnitude_h_vector_std": safe_std(event_ml_h),
            "nsta_h_vector": count_finite(event_ml_h),
            "magnitude_h_component_mean": safe_mean(event_ml_h_comp),
            "magnitude_h_component_std": safe_std(event_ml_h_comp),
            "nsta_h_component": count_finite(event_ml_h_comp),
        })

        event_rows_all.append({
            "ID": event_id,
            "network": "ALL",
            "station": "ALL",
            "mean_header_mag": event_mag_mean,
            "mean_header_hypdist_km": event_hypdist_mean,
            "dist_km": event_hypdist_mean,
            "Ml_Z_on_this_station": safe_mean(event_ml_z),
            "Ml_H_on_this_station": safe_mean(event_ml_h),
            "Ml_H_vector_on_this_station": safe_mean(event_ml_h),
            "Ml_H_component_mean_on_this_station": safe_mean(event_ml_h_comp),
            "mag_e": np.nan,
            "mag_n": np.nan,
            "mag_r": np.nan,
            "mag_t": np.nan,
        })

    station_fieldnames = [
        "event_id",
        "event_mag_header_mean",
        "network",
        "station",
        "dist_km",
        "disp_unit",
        "mag_type",

        "amp_z_raw",
        "amp_z_um",
        "mag_z",

        "amp_e_raw",
        "amp_e_um",
        "mag_e",

        "amp_n_raw",
        "amp_n_um",
        "mag_n",

        "amp_r_raw",
        "amp_r_um",
        "mag_r",

        "amp_t_raw",
        "amp_t_um",
        "mag_t",

        "amp_hvec_raw",
        "amp_hvec_um",
        "mag_hvec",
        "horiz_vector_used",
        "n_horiz",
    ]

    with open(OUT_STATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=station_fieldnames)
        writer.writeheader()
        writer.writerows(station_rows)

    event_fieldnames_z = [
        "ID",
        "network",
        "station",
        "Ml_on_this_station",
        "mean_header_mag",
        "mean_header_hypdist_km",
        "dist_km",
        "magnitude_z_mean",
        "magnitude_z_std",
        "nsta_z",
    ]

    with open(OUT_EVENT_Z_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames_z)
        writer.writeheader()
        writer.writerows(event_rows_z)

    event_fieldnames_h = [
        "ID",
        "network",
        "station",
        "Ml_on_this_station",
        "mean_header_mag",
        "mean_header_hypdist_km",
        "dist_km",
        "magnitude_h_vector_mean",
        "magnitude_h_vector_std",
        "nsta_h_vector",
        "magnitude_h_component_mean",
        "magnitude_h_component_std",
        "nsta_h_component",
    ]

    with open(OUT_EVENT_H_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames_h)
        writer.writeheader()
        writer.writerows(event_rows_h)

    event_fieldnames_all = [
        "ID",
        "network",
        "station",
        "mean_header_mag",
        "mean_header_hypdist_km",
        "dist_km",
        "Ml_Z_on_this_station",
        "Ml_H_on_this_station",
        "Ml_H_vector_on_this_station",
        "Ml_H_component_mean_on_this_station",
        "mag_e",
        "mag_n",
        "mag_r",
        "mag_t",
    ]

    with open(OUT_EVENT_ALL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames_all)
        writer.writeheader()
        writer.writerows(event_rows_all)

    summary = (
        f"SUMMARY | N_station_rows={len(station_rows)} | "
        f"N_event_rows_Z={len(event_rows_z)} | "
        f"N_event_rows_H={len(event_rows_h)} | "
        f"N_event_rows_ALL={len(event_rows_all)} | time={now_str()}"
    )
    print("-" * 60)
    print(f"[OK] Station detail saved   : {OUT_STATION_CSV}")
    print(f"[OK] Event summary Z saved  : {OUT_EVENT_Z_CSV}")
    print(f"[OK] Event summary H saved  : {OUT_EVENT_H_CSV}")
    print(f"[OK] Event summary ALL saved: {OUT_EVENT_ALL_CSV}")
    print(f"[OK] Success log            : {SUCCESS_LOG}")
    print(f"[OK] Fail log               : {FAIL_LOG}")
    print(f"[OK] N_station_rows         : {len(station_rows)}")
    print(f"[OK] N_event_rows_Z         : {len(event_rows_z)}")
    print(f"[OK] N_event_rows_H         : {len(event_rows_h)}")
    print(f"[OK] N_event_rows_ALL       : {len(event_rows_all)}")

    write_log(SUCCESS_LOG, summary)
    write_log(FAIL_LOG, summary)


if __name__ == "__main__":
    main()