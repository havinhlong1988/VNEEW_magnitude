#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from shutil import copy2
from obspy import read, UTCDateTime
import re

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/Data_Vung")
OUTPUT_ROOT = Path("input/Data_raw_fixed_name")
NETWORK = "VN"

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


sac_files = sorted([p for p in INPUT_ROOT.rglob("*_SAC") if p.is_file()])
print(f"N_FILES: {len(sac_files)}")


# -------------------------
# parse filename
# -------------------------
def parse_filename(fname):
    try:
        # ---- time ----
        time_str = fname.split(".")[0]

        # remove trailing letter (S, M, etc.)
        time_str = re.sub(r"[A-Za-z]$", "", time_str)
        t = UTCDateTime.strptime(time_str, "%Y-%m-%d-%H%M-%S")

        parts = fname.split("__")

        # robust station extraction
        sta = parts[1].split("_")[-1].strip()
        if not sta:
            return None, None, None

        # channel
        comp_raw = parts[-1].replace("_SAC", "")

        return t, sta, comp_raw

    except Exception:
        return None, None, None


# -------------------------
# normalize channel
# -------------------------
def fix_channel(comp_raw):
    comp = comp_raw.replace("_", "").replace("-", "").upper()

    # force HH band
    if comp.endswith("E"):
        return "HHE"
    elif comp.endswith("N"):
        return "HHN"
    elif comp.endswith("Z"):
        return "HHZ"

    return None


# -------------------------
# main
# -------------------------
copied_dirs = set()

n_ok = 0
n_fail = 0
n_skip = 0
n_copy = 0

for sac_path in sac_files:
    fname = sac_path.name

    try:
        # ---- parse filename ----
        t, sta, comp_raw = parse_filename(fname)

        if t is None:
            msg = f"[SKIP][TIME] {fname}"
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

        # ---- folder (KEEP ORIGINAL NAME) ----
        folder_name = sac_path.parent.name
        out_dir = OUTPUT_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- copy aux files ----
        if sac_path.parent not in copied_dirs:
            for pattern in ["RESP*", "*.ts", "*.pz"]:
                for src in sac_path.parent.glob(pattern):
                    if src.is_file():
                        dst = out_dir / src.name
                        copy2(src, dst)
                        n_copy += 1
                        msg = f"[COPY] {src} -> {dst}"
                        print(msg)
                        log_success(msg)
            copied_dirs.add(sac_path.parent)

        # ---- read SAC ----
        st = read(str(sac_path))
        if len(st) == 0:
            msg = f"[SKIP][EMPTY] {fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        tr = st[0]

        # ---- update header ----
        tr.stats.network = NETWORK
        tr.stats.station = sta
        tr.stats.channel = comp
        tr.stats.location = ""

        # ---- output filename ----
        out_name = f"{sta}.{comp}..SAC"
        out_path = out_dir / out_name

        # ---- prevent overwrite ----
        if out_path.exists():
            msg = f"[SKIP][DUP] {out_path}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- write ----
        tr.write(str(out_path), format="SAC")

        msg = f"[OK] {fname} -> {out_path.name}"
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
print(f"COPY  : {n_copy}")
print(f"Logs  : {SUCCESS_LOG}, {FAIL_LOG}")