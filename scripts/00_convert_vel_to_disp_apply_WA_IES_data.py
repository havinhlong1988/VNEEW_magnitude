#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SAC data to true displacement and Wood-Anderson simulated displacement.

Input structure:
    input/2025_ML_VN/input_data/{event_id}/*.SAC

Optional response directory:
    input/resp/RESP.{net}.{sta}..{chan}

Output structure:
    output/IES_data/{event_id}/
        {sta}_{chan}.disp.SAC
        {sta}_{chan}.wa_disp.SAC
        RESP.{net}.{sta}..{chan}     (copied if used)

Logs:
    output/IES_data/success.log
    output/IES_data/fail.log

Main logic:
1) If input is NOT yet response removed:
   - find RESP file
   - copy RESP into output event folder
   - remove instrument response directly to displacement
2) If input IS already response removed:
   - interpret input as displacement / velocity / acceleration
   - convert to true displacement if needed
3) Simulate Wood-Anderson on true displacement
4) Save both true displacement and WA displacement
"""

import re
import sys
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from obspy import read, read_inventory
from scipy.fft import rfft, irfft, rfftfreq
from obspy.signal.invsim import simulate_seismometer


# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_ROOT = Path("input/2025_ML_VN/input_data")
RESP_ROOT = Path("input/resp")
OUTPUT_ROOT = Path("output/IES_data")

# ------------------------------------------------------------
# INPUT DATA STATUS
# ------------------------------------------------------------
# True  -> input data already has instrument response removed
# False -> remove instrument response using RESP file first
INPUT_ALREADY_RESPONSE_REMOVED = True

# Only used when INPUT_ALREADY_RESPONSE_REMOVED = True
# Choose one:
#   "disp", "vel", "acc", "auto"
INPUT_DATA_TYPE = "vel"

# Desired true displacement output unit
# Choose: "m", "cm", "mm", "um", "nm"
OUTPUT_DISP_UNIT = "nm"

# If detailed unit cannot be detected from SAC text headers,
# assume this unit for already-corrected input.
# Examples:
#   for INPUT_DATA_TYPE="vel"  -> "nm/s"
#   for INPUT_DATA_TYPE="disp" -> "nm"
#   for INPUT_DATA_TYPE="acc"  -> "nm/s2"
ASSUME_UNIT_IF_MISSING = "nm/s"

# If True, remove each existing output event folder before re-running
OVERWRITE_EXISTING = False

# ------------------------------------------------------------
# PREPROCESS BEFORE NUMERICAL CONVERSION
# (used only when input already response removed)
# ------------------------------------------------------------
DO_DEMEAN_BEFORE = True
DO_DETREND_BEFORE = True
DO_TAPER_BEFORE = True
TAPER_MAX_PERCENTAGE = 0.05   # 5%

APPLY_HIGHPASS_BEFORE = True
HP_FREQ = 0.02                # Hz
HP_CORNERS = 4
HP_ZEROPHASE = True

# Replace NaN/Inf with 0
REPLACE_BAD_VALUES = True

# Integration / differentiation methods
# "time_domain" or "frequency_domain"
NUMERICAL_METHOD = "frequency_domain"

# Only used when frequency domain is selected
FREQ_DOMAIN_FMIN = 0.02

# ------------------------------------------------------------
# RESPONSE REMOVAL PARAMETERS
# ------------------------------------------------------------
# Only used when INPUT_ALREADY_RESPONSE_REMOVED = False
# remove_response output is displacement directly
PRE_FILT = (0.005, 0.01, 40.0, 45.0)
WATER_LEVEL = 60

# ------------------------------------------------------------
# POSTPROCESS TRUE DISPLACEMENT
# ------------------------------------------------------------
DO_DEMEAN_AFTER = True
DO_DETREND_AFTER = True

# ------------------------------------------------------------
# WOOD-ANDERSON SIMULATION
# ------------------------------------------------------------
SAVE_TRUE_DISP = True
SAVE_WA_DISP = True

# WA input assumed to be displacement in meters
# WA output from simulate_seismometer is displacement in meters
# We convert it to OUTPUT_DISP_UNIT before saving

# Standard WA poles/zeros commonly used in local magnitude work
WOOD_ANDERSON_PAZ = {
    "poles": [-6.283185 + 4.712389j, -6.283185 - 4.712389j],
    "zeros": [0j],
    "gain": 1.0,
    "sensitivity": 2080.0,
}

SUCCESS_LOG = OUTPUT_ROOT / "success.log"
FAIL_LOG = OUTPUT_ROOT / "fail.log"


# ============================================================
# SAC ENUM VALUES
# ============================================================
IUNKN = 5
IDISP = 6
IVEL = 7
IACC = 8


# ============================================================
# UNIT HELPERS
# ============================================================
LENGTH_TO_M = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
}

VALID_DISP_UNITS = {"m", "cm", "mm", "um", "µm", "nm"}
VALID_VEL_UNITS = {"m/s", "cm/s", "mm/s", "um/s", "µm/s", "nm/s"}
VALID_ACC_UNITS = {"m/s2", "cm/s2", "mm/s2", "um/s2", "µm/s2", "nm/s2"}


# ============================================================
# BASIC HELPERS
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_log(path: Path, text: str):
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def normalize_unit_text(s):
    if s is None:
        return None

    s = str(s).strip().lower()
    if not s or s == "-12345":
        return None

    s = s.replace(" ", "")
    s = s.replace("μm", "um")
    s = s.replace("µm", "um")
    s = s.replace("sec", "s")
    s = s.replace("second", "s")
    s = s.replace("seconds", "s")

    s = s.replace("meter", "m")
    s = s.replace("meters", "m")
    s = s.replace("metre", "m")
    s = s.replace("metres", "m")

    s = s.replace("millimeter", "mm")
    s = s.replace("millimeters", "mm")
    s = s.replace("centimeter", "cm")
    s = s.replace("centimeters", "cm")
    s = s.replace("nanometer", "nm")
    s = s.replace("nanometers", "nm")
    s = s.replace("micrometer", "um")
    s = s.replace("micrometers", "um")

    s = s.replace("mps", "m/s")
    s = s.replace("cmps", "cm/s")
    s = s.replace("mmps", "mm/s")
    s = s.replace("umps", "um/s")
    s = s.replace("nmps", "nm/s")

    s = s.replace("m/s/s", "m/s2")
    s = s.replace("cm/s/s", "cm/s2")
    s = s.replace("mm/s/s", "mm/s2")
    s = s.replace("um/s/s", "um/s2")
    s = s.replace("nm/s/s", "nm/s2")

    return s


def extract_unit_from_text(text):
    s = normalize_unit_text(text)
    if s is None:
        return None

    candidates = [
        "nm/s2", "um/s2", "mm/s2", "cm/s2", "m/s2",
        "nm/s", "um/s", "mm/s", "cm/s", "m/s",
        "nm", "um", "mm", "cm", "m",
    ]
    for c in candidates:
        if c in s:
            return c
    return None


def detect_detailed_unit_from_sac(tr):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return None, "no_sac_header"

    checked_fields = []

    for key in ["kuser0", "kuser1", "kuser2", "kinst", "khole", "kevnm", "kstnm", "kcmpnm", "knetwk"]:
        val = getattr(sac, key, None)
        checked_fields.append(f"{key}={val}")
        unit = extract_unit_from_text(val)
        if unit is not None:
            return unit, f"from_{key}"

    for key in ["units", "unit"]:
        val = getattr(tr.stats, key, None)
        checked_fields.append(f"stats.{key}={val}")
        unit = extract_unit_from_text(val)
        if unit is not None:
            return unit, f"from_stats_{key}"

    return None, "; ".join(checked_fields)


# ============================================================
# NUMERIC HELPERS
# ============================================================
def integrate_time(data, delta):
    return np.cumsum(data) * float(delta)


def differentiate_time(data, delta):
    out = np.gradient(data, float(delta))
    return np.asarray(out, dtype=np.float64)


def integrate_freq(data, delta, fmin=0.02):
    n = len(data)
    freqs = rfftfreq(n, delta)
    spec = rfft(data)

    out = np.zeros_like(spec, dtype=np.complex128)
    mask = freqs >= float(fmin)
    out[mask] = spec[mask] / (1j * 2.0 * np.pi * freqs[mask])

    return np.asarray(irfft(out, n=n), dtype=np.float64)


def differentiate_freq(data, delta, fmin=0.02):
    n = len(data)
    freqs = rfftfreq(n, delta)
    spec = rfft(data)

    out = np.zeros_like(spec, dtype=np.complex128)
    mask = freqs >= float(fmin)
    out[mask] = spec[mask] * (1j * 2.0 * np.pi * freqs[mask])

    return np.asarray(irfft(out, n=n), dtype=np.float64)


def convert_length_unit(data, from_unit, to_unit):
    from_unit = normalize_unit_text(from_unit)
    to_unit = normalize_unit_text(to_unit)

    if from_unit not in VALID_DISP_UNITS:
        raise ValueError(f"Unsupported length unit: {from_unit}")
    if to_unit not in VALID_DISP_UNITS:
        raise ValueError(f"Unsupported length unit: {to_unit}")

    scale = LENGTH_TO_M[from_unit] / LENGTH_TO_M[to_unit]
    return np.asarray(data, dtype=np.float64) * scale


def preprocess_trace(tr):
    tr2 = tr.copy()

    if DO_DEMEAN_BEFORE:
        tr2.detrend("demean")
    if DO_DETREND_BEFORE:
        tr2.detrend("linear")
    if DO_TAPER_BEFORE:
        tr2.taper(max_percentage=TAPER_MAX_PERCENTAGE, type="hann")
    if APPLY_HIGHPASS_BEFORE:
        tr2.filter("highpass", freq=HP_FREQ, corners=HP_CORNERS, zerophase=HP_ZEROPHASE)

    return tr2


def postprocess_trace(tr):
    tr2 = tr.copy()

    if DO_DEMEAN_AFTER:
        tr2.detrend("demean")
    if DO_DETREND_AFTER:
        tr2.detrend("linear")

    return tr2


# ============================================================
# FILENAME / METADATA HELPERS
# ============================================================
def infer_sta_cha_net(tr, filepath: Path):
    net = getattr(tr.stats, "network", None)
    sta = getattr(tr.stats, "station", None)
    cha = getattr(tr.stats, "channel", None)

    sac = getattr(tr.stats, "sac", None)
    if sac is not None:
        if not net:
            net = getattr(sac, "knetwk", None)
        if not sta:
            sta = getattr(sac, "kstnm", None)
        if not cha:
            cha = getattr(sac, "kcmpnm", None)

    stem = filepath.stem
    parts_us = stem.split("_")

    if len(parts_us) >= 4:
        evt0 = parts_us[0].strip()
        net0 = parts_us[1].strip().upper()
        sta0 = parts_us[2].strip().upper()
        cha0 = parts_us[3].strip().upper().replace("-", "").replace(" ", "")

        if len(evt0) == 14 and evt0.isdigit():
            if not net:
                net = net0
            if not sta:
                sta = sta0
            if not cha:
                cha = cha0

    parts_dot = filepath.name.split(".")
    if len(parts_dot) >= 2:
        if not sta:
            sta = parts_dot[0].strip().upper()
        if not cha:
            cha = parts_dot[1].strip().upper().replace("-", "").replace(" ", "").replace("_", "")

    net = str(net).strip().upper() if net else "XX"
    sta = str(sta).strip().upper() if sta else "UNKNOWN"
    cha = str(cha).strip().upper().replace("-", "").replace(" ", "").replace("_", "") if cha else "UNK"

    return net, sta, cha


def find_resp_file(net, sta, cha):
    candidates = [
        RESP_ROOT / f"RESP.{net}.{sta}..{cha}",
        RESP_ROOT / f"RESP.{net}.{sta}..{cha[:2]}{cha[-1]}",
        RESP_ROOT / f"RESP.{net}.{sta}..{cha[:2]}*",
    ]

    for c in candidates[:2]:
        if c.exists():
            return c

    wildcard = list(RESP_ROOT.glob(f"RESP.{net}.{sta}..{cha[:2]}*"))
    if wildcard:
        return wildcard[0]

    return None


def set_sac_output_markers(tr, disp_unit, source_note, wa=False):
    if not hasattr(tr.stats, "sac") or tr.stats.sac is None:
        tr.stats.sac = {}

    tr.stats.sac.idep = IDISP
    tr.stats.sac.kuser0 = str(disp_unit)[:8]
    tr.stats.sac.kuser1 = str(source_note)[:8] if source_note else "disp"
    tr.stats.sac.kuser2 = "wa_disp" if wa else "disp"


# ============================================================
# EVENT DIRECTORY SCAN
# ============================================================
def is_event_dir_name(name: str) -> bool:
    return len(name) >= 14 and name[:14].isdigit()


def find_event_dirs(root: Path):
    event_dirs = []
    if not root.exists():
        return event_dirs

    for event_dir in sorted(root.iterdir()):
        if event_dir.is_dir() and is_event_dir_name(event_dir.name):
            event_dirs.append(event_dir)

    return event_dirs


# ============================================================
# TRUE DISPLACEMENT BUILDERS
# ============================================================
def true_displacement_from_already_corrected(tr):
    """
    Convert already response-removed input to true displacement in OUTPUT_DISP_UNIT.
    """
    sac = getattr(tr.stats, "sac", None)
    idep = getattr(sac, "idep", None) if sac is not None else None
    detected_unit, unit_source = detect_detailed_unit_from_sac(tr)

    dtype = INPUT_DATA_TYPE.lower().strip()
    if dtype == "auto":
        if idep == IDISP:
            dtype = "disp"
        elif idep == IVEL:
            dtype = "vel"
        elif idep == IACC:
            dtype = "acc"
        else:
            # fallback from detected unit
            if detected_unit in VALID_DISP_UNITS:
                dtype = "disp"
            elif detected_unit in VALID_VEL_UNITS:
                dtype = "vel"
            elif detected_unit in VALID_ACC_UNITS:
                dtype = "acc"
            else:
                raise ValueError(
                    f"Cannot auto-detect INPUT_DATA_TYPE for trace: idep={idep}, detected_unit={detected_unit}"
                )

    delta = float(tr.stats.delta)
    tr_work = preprocess_trace(tr)
    data = tr_work.data.astype(np.float64)

    if dtype == "disp":
        in_unit = detected_unit if detected_unit in VALID_DISP_UNITS else normalize_unit_text(ASSUME_UNIT_IF_MISSING)
        if in_unit not in VALID_DISP_UNITS:
            raise ValueError(f"Invalid displacement unit for INPUT_DATA_TYPE=disp: {in_unit}")
        disp = convert_length_unit(data, in_unit, OUTPUT_DISP_UNIT)
        source_note = f"disp->{OUTPUT_DISP_UNIT}"

    elif dtype == "vel":
        in_unit = detected_unit if detected_unit in VALID_VEL_UNITS else normalize_unit_text(ASSUME_UNIT_IF_MISSING)
        if in_unit not in VALID_VEL_UNITS:
            raise ValueError(f"Invalid velocity unit for INPUT_DATA_TYPE=vel: {in_unit}")

        if NUMERICAL_METHOD == "time_domain":
            disp_native = integrate_time(data, delta)
        elif NUMERICAL_METHOD == "frequency_domain":
            disp_native = integrate_freq(data, delta, FREQ_DOMAIN_FMIN)
        else:
            raise ValueError(f"Invalid NUMERICAL_METHOD={NUMERICAL_METHOD}")

        native_len_unit = in_unit.split("/")[0]
        disp = convert_length_unit(disp_native, native_len_unit, OUTPUT_DISP_UNIT)
        source_note = f"vel->{OUTPUT_DISP_UNIT}"

    elif dtype == "acc":
        in_unit = detected_unit if detected_unit in VALID_ACC_UNITS else normalize_unit_text(ASSUME_UNIT_IF_MISSING)
        if in_unit not in VALID_ACC_UNITS:
            raise ValueError(f"Invalid acceleration unit for INPUT_DATA_TYPE=acc: {in_unit}")

        if NUMERICAL_METHOD == "time_domain":
            vel_native = integrate_time(data, delta)
            disp_native = integrate_time(vel_native, delta)
        elif NUMERICAL_METHOD == "frequency_domain":
            vel_native = integrate_freq(data, delta, FREQ_DOMAIN_FMIN)
            disp_native = integrate_freq(vel_native, delta, FREQ_DOMAIN_FMIN)
        else:
            raise ValueError(f"Invalid NUMERICAL_METHOD={NUMERICAL_METHOD}")

        native_len_unit = in_unit.split("/")[0]
        disp = convert_length_unit(disp_native, native_len_unit, OUTPUT_DISP_UNIT)
        source_note = f"acc->{OUTPUT_DISP_UNIT}"

    else:
        raise ValueError(f"Unsupported INPUT_DATA_TYPE={INPUT_DATA_TYPE}")

    out_tr = tr.copy()
    out_tr.data = np.asarray(disp, dtype=np.float32)
    out_tr = postprocess_trace(out_tr)
    set_sac_output_markers(out_tr, OUTPUT_DISP_UNIT, source_note, wa=False)

    info = (
        f"already_corrected"
        f" | input_data_type={dtype}"
        f" | detected_unit={detected_unit}"
        f" | unit_source={unit_source}"
        f" | numerical_method={NUMERICAL_METHOD}"
    )
    return out_tr, info


def true_displacement_from_raw_with_resp(tr, net, sta, cha, out_event_dir):
    """
    Remove instrument response directly to displacement using RESP file.
    Output of ObsPy remove_response(output='DISP') is in meters.
    Then convert meters -> OUTPUT_DISP_UNIT.
    """
    resp_file = find_resp_file(net, sta, cha)
    if resp_file is None:
        raise FileNotFoundError(f"RESP file not found for {net}.{sta}.{cha} in {RESP_ROOT}")

    copied_resp = out_event_dir / resp_file.name
    if not copied_resp.exists():
        shutil.copy2(resp_file, copied_resp)

    inv = read_inventory(str(resp_file), format="RESP")

    tr_work = tr.copy()
    tr_work.detrend("demean")
    tr_work.detrend("linear")
    tr_work.taper(max_percentage=TAPER_MAX_PERCENTAGE, type="hann")

    tr_work.remove_response(
        inventory=inv,
        output="DISP",
        pre_filt=PRE_FILT,
        water_level=WATER_LEVEL,
        zero_mean=False,
        taper=False,
        plot=False,
    )

    # remove_response(output="DISP") -> meters
    disp_out = convert_length_unit(tr_work.data.astype(np.float64), "m", OUTPUT_DISP_UNIT)

    out_tr = tr.copy()
    out_tr.data = np.asarray(disp_out, dtype=np.float32)
    out_tr = postprocess_trace(out_tr)
    set_sac_output_markers(out_tr, OUTPUT_DISP_UNIT, "rmresp", wa=False)

    info = (
        f"response_removed"
        f" | resp_file={resp_file}"
        f" | copied_resp={copied_resp}"
        f" | pre_filt={PRE_FILT}"
        f" | water_level={WATER_LEVEL}"
        f" | output_from_remove_response=DISP(m)"
        f" | converted_to={OUTPUT_DISP_UNIT}"
    )
    return out_tr, info


# ============================================================
# WOOD-ANDERSON SIMULATION
# ============================================================
def simulate_wa_from_displacement(true_disp_tr):
    """
    Input:
        true_disp_tr.data in OUTPUT_DISP_UNIT
    Process:
        convert to meters
        simulate WA with paz_remove=None, paz_simulate=WOOD_ANDERSON_PAZ
        convert result back to OUTPUT_DISP_UNIT
    """
    delta = float(true_disp_tr.stats.delta)

    # convert current displacement to meters for WA simulation
    disp_m = convert_length_unit(true_disp_tr.data.astype(np.float64), OUTPUT_DISP_UNIT, "m")

    wa_m = simulate_seismometer(
        data=disp_m,
        samp_rate=1.0 / delta,
        paz_remove=None,
        paz_simulate=WOOD_ANDERSON_PAZ,
        water_level=0.0,
        zero_mean=False,
        taper=False,
        simulate_sensitivity=True,
        remove_sensitivity=False,
    )

    wa_out = convert_length_unit(wa_m, "m", OUTPUT_DISP_UNIT)

    wa_tr = true_disp_tr.copy()
    wa_tr.data = np.asarray(wa_out, dtype=np.float32)
    set_sac_output_markers(wa_tr, OUTPUT_DISP_UNIT, "woodand", wa=True)
    return wa_tr


# ============================================================
# CORE PROCESSING
# ============================================================
def process_one_file(infile: Path, out_event_dir: Path, event_id: str):
    try:
        st = read(str(infile))
        if len(st) == 0:
            return False, f"{infile} | empty stream"

        tr = st[0]
        multi_note = f"multi_trace={len(st)}; using_first_trace" if len(st) > 1 else "single_trace"

        net, sta, cha = infer_sta_cha_net(tr, infile)

        if REPLACE_BAD_VALUES:
            data = tr.data.astype(np.float64)
            badmask = ~np.isfinite(data)
            if np.any(badmask):
                data[badmask] = 0.0
                tr.data = data.astype(tr.data.dtype, copy=False)

        true_disp_file = out_event_dir / f"{sta}_{cha}.disp.SAC"
        wa_disp_file = out_event_dir / f"{sta}_{cha}.wa_disp.SAC"

        if INPUT_ALREADY_RESPONSE_REMOVED:
            true_disp_tr, info1 = true_displacement_from_already_corrected(tr)
        else:
            true_disp_tr, info1 = true_displacement_from_raw_with_resp(tr, net, sta, cha, out_event_dir)

        if SAVE_TRUE_DISP:
            true_disp_tr.write(str(true_disp_file), format="SAC")

        wa_info = "wa_not_saved"
        if SAVE_WA_DISP:
            wa_tr = simulate_wa_from_displacement(true_disp_tr)
            wa_tr.write(str(wa_disp_file), format="SAC")
            wa_info = f"wa_saved={wa_disp_file}"

        return True, (
            f"{infile}"
            f" | event={event_id}"
            f" | station={sta}"
            f" | channel={cha}"
            f" | true_disp_file={true_disp_file if SAVE_TRUE_DISP else 'not_saved'}"
            f" | {wa_info}"
            f" | {info1}"
            f" | {multi_note}"
        )

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{infile} | ERROR: {e}\n{tb}"


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    for logf in [SUCCESS_LOG, FAIL_LOG]:
        if logf.exists():
            logf.unlink()

    write_log(SUCCESS_LOG, f"# success log started: {now_str()}")
    write_log(FAIL_LOG, f"# fail log started: {now_str()}")

    if not INPUT_ROOT.exists():
        msg = f"INPUT_ROOT not found: {INPUT_ROOT}"
        write_log(FAIL_LOG, msg)
        print(msg)
        sys.exit(1)

    if not INPUT_ALREADY_RESPONSE_REMOVED and not RESP_ROOT.exists():
        msg = f"RESP_ROOT not found: {RESP_ROOT}"
        write_log(FAIL_LOG, msg)
        print(msg)
        sys.exit(1)

    event_dirs = find_event_dirs(INPUT_ROOT)
    if not event_dirs:
        msg = f"No valid event directories found in: {INPUT_ROOT}"
        write_log(FAIL_LOG, msg)
        print(msg)
        sys.exit(1)

    print(f"INPUT_ROOT : {INPUT_ROOT.resolve()}")
    print(f"RESP_ROOT  : {RESP_ROOT.resolve()}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT.resolve()}")
    print(f"N_EVENTS   : {len(event_dirs)}")
    print(f"INPUT_ALREADY_RESPONSE_REMOVED = {INPUT_ALREADY_RESPONSE_REMOVED}")
    print(f"INPUT_DATA_TYPE = {INPUT_DATA_TYPE}")
    print(f"OUTPUT_DISP_UNIT = {OUTPUT_DISP_UNIT}")

    total_files = 0
    n_success = 0
    n_fail = 0

    for event_dir in event_dirs:
        event_id = event_dir.name
        out_event_dir = OUTPUT_ROOT / event_id

        if OVERWRITE_EXISTING and out_event_dir.exists():
            shutil.rmtree(out_event_dir)

        ensure_dir(out_event_dir)

        sac_files = sorted([p for p in event_dir.glob("*.SAC") if p.is_file()])
        if not sac_files:
            msg = f"[{event_id}] no SAC files found in {event_dir}"
            write_log(FAIL_LOG, msg)
            print(msg)
            continue

        print(f"[EVENT] {event_id} | nfiles={len(sac_files)}")

        for sacfile in sac_files:
            total_files += 1
            ok, message = process_one_file(sacfile, out_event_dir, event_id)

            if ok:
                n_success += 1
                write_log(SUCCESS_LOG, message)
                print("[OK]", message)
            else:
                n_fail += 1
                write_log(FAIL_LOG, message)
                print("[FAIL]", message)

    summary = (
        f"SUMMARY | total_files={total_files} | success={n_success} | fail={n_fail}"
        f" | output_root={OUTPUT_ROOT} | time={now_str()}"
    )
    write_log(SUCCESS_LOG, summary)
    write_log(FAIL_LOG, summary)
    print(summary)


if __name__ == "__main__":
    main()