#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from obspy import read, UTCDateTime
import re

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/Data_Vung")
OUTPUT_ROOT = Path("input/Data_raw")
DEFAULT_NETWORK = "VN"

SUCCESS_LOG = OUTPUT_ROOT / "success.log"
FAIL_LOG = OUTPUT_ROOT / "fail.log"
# ================================================================= #

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# reset logs
open(SUCCESS_LOG, "w").close()
open(FAIL_LOG, "w").close()


def log_success(msg):
    with open(SUCCESS_LOG, "a") as f:
        f.write(msg + "\n")


def log_fail(msg):
    with open(FAIL_LOG, "a") as f:
        f.write(msg + "\n")


# only keep original raw SAC files starting with 20 and ending _SAC
sac_files = sorted([p for p in INPUT_ROOT.rglob("20*_SAC") if p.is_file()])
print(f"N_FILES: {len(sac_files)}")


# -------------------------
# find event dir and parse time from directory name
# input dir format:
#   YYYYMMDD_HHMMSS*
# examples:
#   20200108_221101
#   20200108_221101_M3.2
# -------------------------
def find_event_dir_and_time(sac_path):
    for p in [sac_path.parent] + list(sac_path.parents):
        m = re.match(r"^(\d{8})_(\d{6})(.*)$", p.name)
        if m:
            ymd = m.group(1)
            hms = m.group(2)
            rest = m.group(3)
            try:
                t = UTCDateTime.strptime(f"{ymd}{hms}", "%Y%m%d%H%M%S")
                return p.name, t
            except Exception:
                pass
    return None, None


# -------------------------
# build output dir name
# output dir format:
#   YYYYMMDDHHMMSS_M*
# -------------------------
def build_output_dir_name(src_dir_name, t):
    m = re.match(r"^(\d{8})_(\d{6})(.*)$", src_dir_name)
    if m:
        rest = m.group(3)
        if rest:
            return f"{t.strftime('%Y%m%d%H%M%S')}{rest}"
        else:
            return f"{t.strftime('%Y%m%d%H%M%S')}_M"
    return f"{t.strftime('%Y%m%d%H%M%S')}_M"


# -------------------------
# parse filename
# only station + component
# support examples:
# 2017-05-16-0804-54M.NNU2__003_NNU2__HH_E_SAC
# 2018009808.0230d58_DBVB__HH_Z_SAC
# 202000822210415e26_HGVB__HH_E_SAC
# -------------------------
def parse_filename(fname):
    try:
        parts = fname.split("__")

        sta = None
        comp_raw = None

        if len(parts) >= 3:
            p1 = parts[1].strip()
            p2 = parts[2].strip()

            if "_" in p1:
                sta_tmp = p1.split("_")[-1].strip()
            else:
                sta_tmp = p1.strip()

            if sta_tmp:
                sta = sta_tmp

            comp_raw = p2.replace("_SAC", "").strip()

        else:
            m = re.match(r"^.+?_([A-Za-z0-9]+)__([A-Za-z0-9_]+)_SAC$", fname)
            if m:
                sta = m.group(1).strip()
                comp_raw = m.group(2).strip()

        if not sta or not comp_raw:
            return None, None

        return sta, comp_raw

    except Exception:
        return None, None


# -------------------------
# normalize channel
# -------------------------
def fix_channel(comp_raw):
    comp = comp_raw.replace("_", "").replace("-", "").upper()

    if comp.endswith("E"):
        return "HHE"
    elif comp.endswith("N"):
        return "HHN"
    elif comp.endswith("Z"):
        return "HHZ"

    return None


# -------------------------
# get network from SAC header
# if missing -> DEFAULT_NETWORK
# -------------------------
def get_network_from_trace(tr):
    net = ""

    try:
        net = str(tr.stats.network).strip()
    except Exception:
        net = ""

    if not net or net in ["--", "None"]:
        try:
            sac = tr.stats.sac
            knetwk = str(getattr(sac, "knetwk", "")).strip()
            if knetwk and knetwk not in ["-12345", "None"]:
                net = knetwk
        except Exception:
            pass

    if not net:
        net = DEFAULT_NETWORK

    return net.upper()


# -------------------------
# main
# -------------------------
n_ok = 0
n_fail = 0
n_skip = 0

for sac_path in sac_files:
    fname = sac_path.name

    try:
        # ---- parse event time from parent directory tree ----
        src_folder_name, t = find_event_dir_and_time(sac_path)
        if t is None:
            msg = f"[SKIP][DIRTIME] no YYYYMMDD_HHMMSS* parent found | file={fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- parse station/channel from filename ----
        sta, comp_raw = parse_filename(fname)
        if sta is None:
            msg = f"[SKIP][NAME] {fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        comp = fix_channel(comp_raw)
        if comp is None:
            msg = f"[SKIP][CHAN] {fname} raw={comp_raw}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- read SAC ----
        st = read(str(sac_path))
        if len(st) == 0:
            msg = f"[SKIP][EMPTY] {fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        tr = st[0]

        # ---- get network from header, fallback VN ----
        net = get_network_from_trace(tr)

        # ---- output folder: YYYYMMDDHHMMSS_M* ----
        folder_name = build_output_dir_name(src_folder_name, t)
        out_dir = OUTPUT_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- update header ----
        tr.stats.network = net
        tr.stats.station = sta
        tr.stats.channel = comp
        tr.stats.location = ""

        try:
            tr.stats.sac.knetwk = net
        except Exception:
            pass
        try:
            tr.stats.sac.kstnm = sta
        except Exception:
            pass
        try:
            tr.stats.sac.kcmpnm = comp
        except Exception:
            pass

        # ---- output filename ----
        # YYYYMMDDHHMMSS_{net}_{sta}_{chan}.SAC
        out_name = f"{t.strftime('%Y%m%d%H%M%S')}_{net}_{sta}_{comp}.SAC"
        out_path = out_dir / out_name

        # ---- prevent overwrite ----
        if out_path.exists():
            msg = f"[SKIP][DUP] {out_path}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- write only SAC ----
        tr.write(str(out_path), format="SAC")

        msg = f"[OK] {fname} -> {folder_name}/{out_path.name}"
        print(msg)
        log_success(msg)
        n_ok += 1

    except Exception as e:
        msg = f"[FAIL] {fname} | {e}"
        print(msg)
        log_fail(msg)
        n_fail += 1


# -------------------------
# summary
# -------------------------
print("-" * 60)
print("DONE")
print(f"OK    : {n_ok}")
print(f"SKIP  : {n_skip}")
print(f"FAIL  : {n_fail}")
print(f"Logs  : {SUCCESS_LOG}, {FAIL_LOG}")