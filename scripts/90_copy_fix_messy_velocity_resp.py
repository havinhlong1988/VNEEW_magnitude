#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from shutil import copy2
from obspy import read

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/Data_Vung")
OUTPUT_ROOT = Path("input/Data_vpolz_fixed_name")
NETWORK = "VN"

# Only process files with this prefix and suffix *_SAC
FILE_PREFIX = "vpolz"

# Copy aux files in the same source folder once per event folder
COPY_AUX_FILES = False

SUCCESS_LOG = OUTPUT_ROOT / "success.log"
FAIL_LOG = OUTPUT_ROOT / "fail.log"
# ================================================================= #

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# reset logs
open(SUCCESS_LOG, "w", encoding="utf-8").close()
open(FAIL_LOG, "w", encoding="utf-8").close()


def log_success(msg):
    with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_fail(msg):
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def is_target_sac_file(path: Path, prefix: str) -> bool:
    """
    Only allow files:
      - regular file
      - basename starts with prefix
      - basename ends with '_SAC'
    """
    name = path.name
    return path.is_file() and name.startswith(prefix) and name.endswith("_SAC")


def clean_header_str(x):
    """
    Clean SAC/header string values.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s in {"", "-12345", "None"}:
        return ""
    return s


def normalize_channel(chan: str) -> str:
    """
    Normalize channel name from header.

    Accepted directly:
      HHE HHN HHZ
      BHE BHN BHZ
      EHE EHN EHZ

    Fallback:
      E -> HHE
      N -> HHN
      Z -> HHZ
      anything ending with E/N/Z -> HH + last char
    """
    chan = clean_header_str(chan).upper()
    if not chan:
        return ""

    if chan in {"HHE", "HHN", "HHZ", "BHE", "BHN", "BHZ", "EHE", "EHN", "EHZ"}:
        return chan

    if chan in {"E", "N", "Z"}:
        return "HH" + chan

    if chan.endswith("E"):
        return "HHE"
    if chan.endswith("N"):
        return "HHN"
    if chan.endswith("Z"):
        return "HHZ"

    return ""


def get_sta_chan_from_header(tr):
    """
    Get station and channel from ObsPy stats first,
    then fallback to SAC header.

    Priority:
      1) tr.stats.station / tr.stats.channel
      2) tr.stats.sac.kstnm / tr.stats.sac.kcmpnm
    """
    sta = ""
    chan = ""

    # priority 1: ObsPy stats
    if hasattr(tr.stats, "station"):
        sta = clean_header_str(tr.stats.station)
    if hasattr(tr.stats, "channel"):
        chan = clean_header_str(tr.stats.channel)

    # priority 2: SAC header
    sac = getattr(tr.stats, "sac", None)
    if sac is not None:
        if not sta and hasattr(sac, "kstnm"):
            sta = clean_header_str(sac.kstnm)
        if not chan and hasattr(sac, "kcmpnm"):
            chan = clean_header_str(sac.kcmpnm)

    chan = normalize_channel(chan)
    return sta, chan


def copy_aux_files(src_dir: Path, out_dir: Path):
    """
    Copy only auxiliary response files.
    Never copy waveform SAC files here.
    """
    n_copy = 0

    for pattern in ["RESP*", "*.ts", "*.pz"]:
        for src in src_dir.glob(pattern):
            if not src.is_file():
                continue

            # never copy SAC-like waveform files here
            if src.name.endswith("_SAC") or src.suffix.upper() == ".SAC":
                continue

            dst = out_dir / src.name
            if dst.exists():
                continue

            copy2(src, dst)
            n_copy += 1

            msg = f"[COPY] {src} -> {dst}"
            print(msg)
            log_success(msg)

    return n_copy


# only select files with wanted prefix
sac_files = sorted([p for p in INPUT_ROOT.rglob("*") if is_target_sac_file(p, FILE_PREFIX)])
print(f"N_FILES_MATCHED: {len(sac_files)}")
print(f"FILE_PREFIX    : {FILE_PREFIX}")

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
        # ---- double-safe prefix/suffix check ----
        if not is_target_sac_file(sac_path, FILE_PREFIX):
            msg = f"[SKIP][PREFIX] {fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- read waveform ----
        st = read(str(sac_path))
        if len(st) == 0:
            msg = f"[SKIP][EMPTY] {fname}"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        tr = st[0]

        # ---- station/channel from SAC header ----
        sta, comp = get_sta_chan_from_header(tr)

        if not sta:
            msg = f"[SKIP][STA] {fname} | station not found in SAC header"
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        if not comp:
            raw_stats_chan = clean_header_str(getattr(tr.stats, "channel", ""))
            raw_sac_chan = clean_header_str(getattr(getattr(tr.stats, "sac", None), "kcmpnm", ""))
            msg = (
                f"[SKIP][CHAN] {fname} | cannot determine channel from header "
                f"(stats.channel={raw_stats_chan}, sac.kcmpnm={raw_sac_chan})"
            )
            print(msg)
            log_fail(msg)
            n_skip += 1
            continue

        # ---- keep original parent folder name ----
        folder_name = sac_path.parent.name
        out_dir = OUTPUT_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- copy aux files once per source folder ----
        if COPY_AUX_FILES and sac_path.parent not in copied_dirs:
            n_copy += copy_aux_files(sac_path.parent, out_dir)
            copied_dirs.add(sac_path.parent)

        # ---- update headers ----
        tr.stats.network = NETWORK
        tr.stats.station = sta
        tr.stats.channel = comp
        tr.stats.location = ""

        # also update SAC string headers if available
        sac = getattr(tr.stats, "sac", None)
        if sac is not None:
            try:
                sac.kstnm = sta
            except Exception:
                pass
            try:
                sac.kcmpnm = comp
            except Exception:
                pass
            try:
                sac.knetwk = NETWORK
            except Exception:
                pass
            try:
                sac.khole = ""
            except Exception:
                pass

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
        msg = f"[FAIL] {fname} | {type(e).__name__}: {e}"
        print(msg)
        log_fail(msg)
        n_fail += 1

# -------------------------
# summary
# -------------------------
print("-" * 60)
print("DONE")
print(f"PREFIX: {FILE_PREFIX}")
print(f"OK    : {n_ok}")
print(f"SKIP  : {n_skip}")
print(f"FAIL  : {n_fail}")
print(f"COPY  : {n_copy}")
print(f"Logs  : {SUCCESS_LOG}, {FAIL_LOG}")