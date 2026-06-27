#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_cut_wa_main_signal_windows.py

Cut main signal windows from WA displacement SAC files.

Input structure:
    output/IES_data/{event_id}/*.wa_disp.SAC

Output structure:
    output/02_cut_main_signal_windows/{event_id}/{sta}.{chan}.wa_cut.SAC

Window definition:
    start = P - PRE_P_SEC
    end   = surface_arrival + POST_SURFACE_SEC

where:
    surface_arrival = dist_km / ASSUME_SURFACE_VEL_KM_S

Rules:
- Check ALL available components for header P and S picks.
- Preferred header priority:
    P: Pn > Pg > P > any phase starting with P
    S: Sn > Sg > S > any phase starting with S
- If only S exists, estimate:
    P = S - dist_km * (1/Vs - 1/Vp)
- If only P exists, estimate:
    S = P + dist_km * (1/Vs - 1/Vp)
- If neither P nor S exists, optionally fallback to STA/LTA for P/S.
- The final cutting window is always controlled by:
    [P - PRE_P_SEC, surface_arrival + POST_SURFACE_SEC]

Notes:
- dist is read preferably from SAC header 'dist' (km).
- If dist is missing, try gcarc * 111.19 km/deg.
"""

from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
from obspy import read
from obspy.io.sac import SACTrace
from obspy.signal.trigger import classic_sta_lta, trigger_onset


# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_ROOT = Path("output/IES_data")
OUTPUT_ROOT = Path("output/02_cut_main_signal_windows_amp")


INPUT_GLOB = "*.disp.SAC"
OUTPUT_SUFFIX = "cut"

# ---------------- window ----------------
PRE_P_SEC = 10.0
POST_SURFACE_SEC = 10.0
MIN_WINDOW_SEC = 5.0

# ---------------- assumed velocities (km/s) ----------------
ASSUME_P_VEL_KM_S = 8.0
ASSUME_S_VEL_KM_S = 3.0
ASSUME_SURFACE_VEL_KM_S = 1.0

# ---------------- phase search from SAC headers ----------------
USE_HEADER_PHASES = True

# ---------------- STA/LTA fallback ----------------
USE_STALTA_FALLBACK = True

P_STALTA_TRACE_ORDER = ["HHZ", "EHZ", "BHZ", "SHZ", "HNZ", "ENZ"]
S_STALTA_TRACE_ORDER = [
    "HHE", "HHN", "HHR", "HHT", "HH1", "HH2",
    "EHE", "EHN", "EHR", "EHT", "EH1", "EH2",
    "BHE", "BHN", "BHR", "BHT", "BH1", "BH2",
    "SHE", "SHN", "SHR", "SHT", "SH1", "SH2",
]

P_STA_SEC = 0.2
P_LTA_SEC = 2.0
P_ON = 3.0
P_OFF = 1.0

S_STA_SEC = 0.3
S_LTA_SEC = 3.0
S_ON = 2.5
S_OFF = 1.2

MIN_S_AFTER_P_SEC = 0.5

# optional preprocessing before STA/LTA
STALTA_DEMEAN = True
STALTA_DETREND = True
STALTA_TAPER = True
STALTA_HIGHPASS_HZ = 0.02   # set None to disable

# ---------------- folder selection ----------------
ONLY_EVENT_DIR_14DIGIT = True

# ---------------- logs ----------------
SUCCESS_LOG = OUTPUT_ROOT / "cut_success.log"
FAIL_LOG = OUTPUT_ROOT / "cut_fail.log"


# ============================================================
# HELPERS
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_log(path: Path, text: str):
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def reset_logs():
    for logf in [SUCCESS_LOG, FAIL_LOG]:
        if logf.exists():
            logf.unlink()
    write_log(SUCCESS_LOG, f"# cut success log started: {now_str()}")
    write_log(FAIL_LOG, f"# cut fail log started: {now_str()}")


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


def clean_phase_label(label):
    if label is None:
        return ""
    s = str(label).strip().upper()
    if not s or s == "-12345":
        return ""
    return s


def get_sac_header(path):
    try:
        return SACTrace.read(str(path), headonly=True)
    except Exception:
        return None


def get_relative_time_vector(tr):
    b = 0.0
    if hasattr(tr.stats, "sac") and tr.stats.sac is not None:
        try:
            b = float(tr.stats.sac.b)
            if not np.isfinite(b):
                b = 0.0
        except Exception:
            b = 0.0
    n = tr.stats.npts
    dt = tr.stats.delta
    return b + np.arange(n, dtype=float) * dt


def get_trace_b_e(tr):
    t = get_relative_time_vector(tr)
    if len(t) == 0:
        return np.nan, np.nan
    return float(t[0]), float(t[-1])


def infer_channel(tr, path):
    cha = getattr(tr.stats, "channel", None)
    if cha:
        return str(cha).strip().upper()

    sac = getattr(tr.stats, "sac", None)
    if sac is not None:
        cha = getattr(sac, "kcmpnm", None)
        if cha:
            return str(cha).strip().upper()

    stem = path.stem
    parts_us = stem.split("_")
    if len(parts_us) >= 2:
        cand = parts_us[-2].strip().upper()
        if len(cand) >= 3:
            return cand

    parts_dot = path.name.split(".")
    if len(parts_dot) >= 2:
        cand = parts_dot[1].strip().upper()
        if len(cand) >= 3:
            return cand

    return "UNK"


def infer_station(tr, path):
    sta = getattr(tr.stats, "station", None)
    if sta:
        return str(sta).strip().upper()

    sac = getattr(tr.stats, "sac", None)
    if sac is not None:
        sta = getattr(sac, "kstnm", None)
        if sta and str(sta).strip() != "-12345":
            return str(sta).strip().upper()

    stem = path.stem
    parts_us = stem.split("_")
    if len(parts_us) >= 1:
        sta = parts_us[0].strip().upper()
        if sta:
            return sta

    parts_dot = path.name.split(".")
    if len(parts_dot) >= 1:
        sta = parts_dot[0].strip().upper()
        if sta:
            return sta

    return "UNKNOWN"


def build_output_path(out_event_dir: Path, infile: Path, tr) -> Path:
    sta = infer_station(tr, infile)
    chan = infer_channel(tr, infile)
    outname = f"{sta}.{chan}.{OUTPUT_SUFFIX}.SAC"
    return out_event_dir / outname


def get_dist_km_from_sac(sac):
    if sac is None:
        return np.nan

    dist = safe_float(getattr(sac, "dist", np.nan))
    if np.isfinite(dist) and dist > 0:
        return float(dist)

    gcarc = safe_float(getattr(sac, "gcarc", np.nan))
    if np.isfinite(gcarc) and gcarc > 0:
        return float(gcarc) * 111.19

    return np.nan


# ============================================================
# PHASE PICK FROM SAC HEADERS
# ============================================================
def get_header_phases(sac):
    """
    Read SAC t0..t9 with labels kt0..kt9.
    Return dict: {LABEL: time_sec}
    """
    phases = {}
    if sac is None:
        return phases

    for i in range(10):
        tkey = f"t{i}"
        kkey = f"kt{i}"

        tval = getattr(sac, tkey, None)
        kval = getattr(sac, kkey, None)

        label = clean_phase_label(kval)
        tsec = safe_float(tval)

        if label and np.isfinite(tsec) and tsec > -12344:
            phases[label] = float(tsec)

    return phases


def choose_best_p_from_phases(phases):
    if not phases:
        return np.nan, ""

    for want in ["PN", "PG", "P"]:
        if want in phases:
            return phases[want], want

    for label, t in phases.items():
        if label.startswith("PN"):
            return t, label
    for label, t in phases.items():
        if label.startswith("PG"):
            return t, label
    for label, t in phases.items():
        if label.startswith("P"):
            return t, label

    return np.nan, ""


def choose_best_s_from_phases(phases):
    if not phases:
        return np.nan, ""

    for want in ["SN", "SG", "S"]:
        if want in phases:
            return phases[want], want

    for label, t in phases.items():
        if label.startswith("SN"):
            return t, label
    for label, t in phases.items():
        if label.startswith("SG"):
            return t, label
    for label, t in phases.items():
        if label.startswith("S"):
            return t, label

    return np.nan, ""


def collect_best_header_p_s(all_items):
    """
    Check ALL available components and choose earliest valid P and S
    according to phase priority.
    """
    p_candidates = []
    s_candidates = []

    for ch, tr, f, sac in all_items:
        phases = get_header_phases(sac)
        p_time, p_label = choose_best_p_from_phases(phases)
        s_time, s_label = choose_best_s_from_phases(phases)

        if np.isfinite(p_time):
            p_candidates.append((p_time, p_label, ch, f))
        if np.isfinite(s_time):
            s_candidates.append((s_time, s_label, ch, f))

    # choose earliest among all components
    if p_candidates:
        p_candidates.sort(key=lambda x: x[0])
        p_time, p_label, p_chan, p_file = p_candidates[0]
    else:
        p_time, p_label, p_chan, p_file = np.nan, "", "", None

    if s_candidates:
        s_candidates.sort(key=lambda x: x[0])
        s_time, s_label, s_chan, s_file = s_candidates[0]
    else:
        s_time, s_label, s_chan, s_file = np.nan, "", "", None

    return p_time, p_label, p_chan, s_time, s_label, s_chan


# ============================================================
# STA/LTA FALLBACK
# ============================================================
def preprocess_for_stalta(tr):
    trp = tr.copy()
    if STALTA_DEMEAN:
        trp.detrend("demean")
    if STALTA_DETREND:
        trp.detrend("linear")
    if STALTA_TAPER:
        trp.taper(max_percentage=0.05, type="hann")
    if STALTA_HIGHPASS_HZ is not None:
        trp.filter("highpass", freq=STALTA_HIGHPASS_HZ, corners=2, zerophase=True)
    return trp


def pick_first_trigger(tr, sta_sec, lta_sec, on, off, search_after=None):
    trp = preprocess_for_stalta(tr)
    y = trp.data.astype(float)
    t = get_relative_time_vector(trp)
    dt = trp.stats.delta

    nsta = max(1, int(round(sta_sec / dt)))
    nlta = max(nsta + 1, int(round(lta_sec / dt)))

    if len(y) < nlta + 2:
        return np.nan

    if search_after is not None:
        idx = np.where(t >= search_after)[0]
        if len(idx) == 0:
            return np.nan
        i0 = idx[0]
        y = y[i0:]
        t = t[i0:]
        if len(y) < nlta + 2:
            return np.nan

    cft = classic_sta_lta(y, nsta, nlta)
    onoff = trigger_onset(cft, on, off)
    if len(onoff) == 0:
        return np.nan

    ion = int(onoff[0][0])
    if ion < 0 or ion >= len(t):
        return np.nan

    return float(t[ion])


def classify_component(channel):
    ch = str(channel).upper()
    if ch.endswith("Z"):
        return "Z"
    if ch.endswith(("E", "N", "R", "T", "1", "2")):
        return "H"
    return "OTHER"


def choose_trace_for_p(comp_dict):
    for want in P_STALTA_TRACE_ORDER:
        if want in comp_dict:
            return comp_dict[want]["trace"], comp_dict[want]["path"], want

    for ch, item in comp_dict.items():
        if classify_component(ch) == "Z":
            return item["trace"], item["path"], ch

    for ch, item in comp_dict.items():
        return item["trace"], item["path"], ch

    return None, None, ""


def choose_trace_for_s(comp_dict):
    for want in S_STALTA_TRACE_ORDER:
        if want in comp_dict:
            return comp_dict[want]["trace"], comp_dict[want]["path"], want

    for ch, item in comp_dict.items():
        if classify_component(ch) == "H":
            return item["trace"], item["path"], ch

    return None, None, ""


# ============================================================
# ARRIVAL ESTIMATION
# ============================================================
def estimate_p_from_s(s_time, dist_km, vp_km_s, vs_km_s):
    if not (np.isfinite(s_time) and np.isfinite(dist_km) and dist_km > 0):
        return np.nan
    dt_sp = dist_km * (1.0 / vs_km_s - 1.0 / vp_km_s)
    return s_time - dt_sp


def estimate_s_from_p(p_time, dist_km, vp_km_s, vs_km_s):
    if not (np.isfinite(p_time) and np.isfinite(dist_km) and dist_km > 0):
        return np.nan
    dt_sp = dist_km * (1.0 / vs_km_s - 1.0 / vp_km_s)
    return p_time + dt_sp


def estimate_surface_from_dist(dist_km, vsurf_km_s):
    if not (np.isfinite(dist_km) and dist_km > 0 and vsurf_km_s > 0):
        return np.nan
    return dist_km / vsurf_km_s


# ============================================================
# CUTTING
# ============================================================
def cut_trace_by_relative_time(tr, t_start, t_end):
    t = get_relative_time_vector(tr)
    mask = (t >= t_start) & (t <= t_end)
    if np.count_nonzero(mask) < 2:
        return None

    idx = np.where(mask)[0]
    i0 = int(idx[0])
    i1 = int(idx[-1])

    tr2 = tr.copy()
    tr2.data = tr2.data[i0:i1 + 1].copy()

    old_b = 0.0
    if hasattr(tr2.stats, "sac") and tr2.stats.sac is not None:
        try:
            old_b = float(tr.stats.sac.b)
            if not np.isfinite(old_b):
                old_b = 0.0
        except Exception:
            old_b = 0.0

    new_b = old_b + i0 * tr.stats.delta
    new_e = new_b + (tr2.stats.npts - 1) * tr.stats.delta

    if not hasattr(tr2.stats, "sac") or tr2.stats.sac is None:
        tr2.stats.sac = {}

    tr2.stats.starttime = tr.stats.starttime + i0 * tr.stats.delta
    tr2.stats.npts = len(tr2.data)

    try:
        tr2.stats.sac.b = float(new_b)
        tr2.stats.sac.e = float(new_e)
    except Exception:
        pass

    return tr2


# ============================================================
# MAIN PROCESSING
# ============================================================
def process_event(event_dir: Path, out_event_dir: Path):
    files = sorted([p for p in event_dir.glob(INPUT_GLOB) if p.is_file()])
    if not files:
        return 0, 0

    ensure_dir(out_event_dir)

    comp_dict = {}
    all_items = []

    for f in files:
        try:
            st = read(str(f))
            if len(st) == 0:
                raise RuntimeError("Empty stream")

            tr = st[0]
            sac = get_sac_header(f)
            ch = infer_channel(tr, f)

            comp_dict[ch] = {"trace": tr, "path": f, "sac": sac}
            all_items.append((ch, tr, f, sac))

        except Exception as e:
            err = f"{f} | ERROR reading: {e}"
            print(f"[FAIL] {err}")
            write_log(FAIL_LOG, err)
            write_log(FAIL_LOG, traceback.format_exc())

    if not all_items:
        return 0, len(files)

    # distance: use first available valid sac distance
    dist_km = np.nan
    for ch, tr, f, sac in all_items:
        dist_km = get_dist_km_from_sac(sac)
        if np.isfinite(dist_km):
            break

    # ---------------- header picks from ALL components ----------------
    p_time = np.nan
    p_label = ""
    p_source = ""
    p_chan = ""

    s_time = np.nan
    s_label = ""
    s_source = ""
    s_chan = ""

    if USE_HEADER_PHASES:
        hp, hplabel, hpchan, hs, hslabel, hschan = collect_best_header_p_s(all_items)

        if np.isfinite(hp):
            p_time = hp
            p_label = hplabel
            p_source = "HEADER_ALLCOMP"
            p_chan = hpchan

        if np.isfinite(hs):
            s_time = hs
            s_label = hslabel
            s_source = "HEADER_ALLCOMP"
            s_chan = hschan

    # ---------------- STA/LTA fallback if needed ----------------
    if (not np.isfinite(p_time)) and USE_STALTA_FALLBACK:
        trp, pathp, pchan = choose_trace_for_p(comp_dict)
        if trp is not None:
            p_try = pick_first_trigger(
                trp,
                sta_sec=P_STA_SEC,
                lta_sec=P_LTA_SEC,
                on=P_ON,
                off=P_OFF,
                search_after=None
            )
            if np.isfinite(p_try):
                p_time = p_try
                p_label = "P_STALTA"
                p_source = "STA_LTA"
                p_chan = pchan

    if (not np.isfinite(s_time)) and USE_STALTA_FALLBACK:
        trs, paths, schan = choose_trace_for_s(comp_dict)
        if trs is not None:
            search_after = p_time + MIN_S_AFTER_P_SEC if np.isfinite(p_time) else None
            s_try = pick_first_trigger(
                trs,
                sta_sec=S_STA_SEC,
                lta_sec=S_LTA_SEC,
                on=S_ON,
                off=S_OFF,
                search_after=search_after
            )
            if np.isfinite(s_try):
                s_time = s_try
                s_label = "S_STALTA"
                s_source = "STA_LTA"
                s_chan = schan

    # ---------------- estimate missing body arrival ----------------
    if (not np.isfinite(p_time)) and np.isfinite(s_time):
        p_est = estimate_p_from_s(
            s_time=s_time,
            dist_km=dist_km,
            vp_km_s=ASSUME_P_VEL_KM_S,
            vs_km_s=ASSUME_S_VEL_KM_S
        )
        if np.isfinite(p_est):
            p_time = p_est
            p_label = "P_EST_FROM_S"
            p_source = "ESTIMATE_FROM_S"
            p_chan = s_chan or "NONE"

    if (not np.isfinite(s_time)) and np.isfinite(p_time):
        s_est = estimate_s_from_p(
            p_time=p_time,
            dist_km=dist_km,
            vp_km_s=ASSUME_P_VEL_KM_S,
            vs_km_s=ASSUME_S_VEL_KM_S
        )
        if np.isfinite(s_est):
            s_time = s_est
            s_label = "S_EST_FROM_P"
            s_source = "ESTIMATE_FROM_P"
            s_chan = p_chan or "NONE"

    # ---------------- estimate surface arrival ----------------
    surface_time = estimate_surface_from_dist(
        dist_km=dist_km,
        vsurf_km_s=ASSUME_SURFACE_VEL_KM_S
    )

    if not np.isfinite(p_time):
        msg = f"{event_dir.name} | cannot determine P arrival"
        print(f"[FAIL] {msg}")
        write_log(FAIL_LOG, msg)
        return 0, len(files)

    if not np.isfinite(surface_time):
        msg = f"{event_dir.name} | cannot determine surface-wave arrival | dist_km={dist_km}"
        print(f"[FAIL] {msg}")
        write_log(FAIL_LOG, msg)
        return 0, len(files)

    t_start = p_time - PRE_P_SEC
    t_end = surface_time + POST_SURFACE_SEC

    if np.isfinite(s_time) and (surface_time <= s_time):
        msg = (
            f"{event_dir.name} | suspicious arrival order | "
            f"P={p_time:.3f} | S={s_time:.3f} | Surf={surface_time:.3f}"
        )
        print(f"[FAIL] {msg}")
        write_log(FAIL_LOG, msg)
        return 0, len(files)

    if not np.isfinite(t_start) or not np.isfinite(t_end) or (t_end - t_start) < MIN_WINDOW_SEC:
        msg = (
            f"{event_dir.name} | invalid window | "
            f"dist_km={dist_km} | P={p_time} | S={s_time} | Surf={surface_time} | "
            f"start={t_start} | end={t_end}"
        )
        print(f"[FAIL] {msg}")
        write_log(FAIL_LOG, msg)
        return 0, len(files)

    # ---------------- cut all available traces ----------------
    n_ok = 0
    n_fail = 0

    p_str = f"{p_time:.3f}" if np.isfinite(p_time) else "nan"
    s_str = f"{s_time:.3f}" if np.isfinite(s_time) else "nan"
    surf_str = f"{surface_time:.3f}" if np.isfinite(surface_time) else "nan"
    dist_str = f"{dist_km:.3f}" if np.isfinite(dist_km) else "nan"

    for ch, tr, f, sac in all_items:
        try:
            b, e = get_trace_b_e(tr)

            cut_start = max(t_start, b)
            cut_end = min(t_end, e)

            if not np.isfinite(cut_start) or not np.isfinite(cut_end) or (cut_end - cut_start) < 1.0:
                raise RuntimeError(
                    f"window outside trace or too short | "
                    f"trace_b={b} trace_e={e} cut_start={cut_start} cut_end={cut_end}"
                )

            tr_cut = cut_trace_by_relative_time(tr, cut_start, cut_end)
            if tr_cut is None:
                raise RuntimeError("cut returned None")

            if not hasattr(tr_cut.stats, "sac") or tr_cut.stats.sac is None:
                tr_cut.stats.sac = {}

            try:
                tr_cut.stats.sac.kuser0 = "wawin"
                tr_cut.stats.sac.kuser1 = (p_label or "")[:8]
                tr_cut.stats.sac.kuser2 = (s_label or "")[:8]
            except Exception:
                pass

            outfile = build_output_path(out_event_dir, f, tr_cut)
            tr_cut.write(str(outfile), format="SAC")

            msg = (
                f"{f} -> {outfile} | "
                f"dist_km={dist_str} | "
                f"P={p_str} | P_label={p_label or 'NONE'} | P_source={p_source or 'NONE'} | P_chan={p_chan or 'NONE'} | "
                f"S={s_str} | S_label={s_label or 'NONE'} | S_source={s_source or 'NONE'} | S_chan={s_chan or 'NONE'} | "
                f"Surf={surf_str} | Surf_source=ASSUME_VSURF_{ASSUME_SURFACE_VEL_KM_S:.3f} | "
                f"start={cut_start:.3f} | end={cut_end:.3f} | "
                f"npts={tr_cut.stats.npts}"
            )

            print(f"[OK] {msg}")
            write_log(SUCCESS_LOG, msg)
            n_ok += 1

        except Exception as e:
            err = f"{f} | ERROR cutting: {e}"
            print(f"[FAIL] {err}")
            write_log(FAIL_LOG, err)
            write_log(FAIL_LOG, traceback.format_exc())
            n_fail += 1

    return n_ok, n_fail


# ============================================================
# MAIN
# ============================================================
def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    ensure_dir(OUTPUT_ROOT)
    reset_logs()

    event_dirs = [p for p in sorted(INPUT_ROOT.iterdir())
                  if p.is_dir() and is_event_dir_name(p.name)]

    print(f"INPUT_ROOT              : {INPUT_ROOT.resolve()}")
    print(f"OUTPUT_ROOT             : {OUTPUT_ROOT.resolve()}")
    print(f"N_EVENT_DIRS            : {len(event_dirs)}")
    print(f"INPUT_GLOB              : {INPUT_GLOB}")
    print(f"OUTPUT_SUFFIX           : {OUTPUT_SUFFIX}")
    print(f"PRE_P_SEC               : {PRE_P_SEC}")
    print(f"POST_SURFACE_SEC        : {POST_SURFACE_SEC}")
    print(f"ASSUME_P_VEL_KM_S       : {ASSUME_P_VEL_KM_S}")
    print(f"ASSUME_S_VEL_KM_S       : {ASSUME_S_VEL_KM_S}")
    print(f"ASSUME_SURFACE_VEL_KM_S : {ASSUME_SURFACE_VEL_KM_S}")
    print(f"MIN_WINDOW_SEC          : {MIN_WINDOW_SEC}")
    print(f"USE_HEADER_PHASES       : {USE_HEADER_PHASES}")
    print(f"USE_STALTA_FALLBACK     : {USE_STALTA_FALLBACK}")
    print(f"SUCCESS_LOG             : {SUCCESS_LOG}")
    print(f"FAIL_LOG                : {FAIL_LOG}")

    n_ok_total = 0
    n_fail_total = 0

    for event_dir in event_dirs:
        out_event_dir = OUTPUT_ROOT / event_dir.name
        print(f"\n[EVENT] {event_dir.name}")
        n_ok, n_fail = process_event(event_dir, out_event_dir)
        n_ok_total += n_ok
        n_fail_total += n_fail

    summary = f"SUMMARY | success={n_ok_total} | fail={n_fail_total} | time={now_str()}"
    print("\n" + "-" * 60)
    print(summary)
    write_log(SUCCESS_LOG, summary)
    write_log(FAIL_LOG, summary)


if __name__ == "__main__":
    main()