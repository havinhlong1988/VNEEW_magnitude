#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from shutil import copy2
from obspy import read

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/Data")
OUTPUT_ROOT = Path("input/Data_fixed_name")
NETWORK = "VN"
# ================================================================= #

if not INPUT_ROOT.exists():
    raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT.resolve()}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

sac_files = sorted([p for p in INPUT_ROOT.rglob("*_SAC") if p.is_file()])

print(f"INPUT_ROOT   : {INPUT_ROOT.resolve()}")
print(f"OUTPUT_ROOT  : {OUTPUT_ROOT.resolve()}")
print(f"N_CANDIDATES : {len(sac_files)}")

n_ok = 0
n_skip = 0
n_fail = 0
n_copy = 0

copied_files = set()
processed_dirs = set()

for sac_path in sac_files:
    try:
        rel_dir = sac_path.parent.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Copy aux files once per directory
        # ------------------------------------------------------------
        if sac_path.parent not in processed_dirs:
            for pattern in ["RESP*", "*.ts", "*.pz"]:
                for src in sac_path.parent.glob(pattern):
                    if src.is_file():
                        dst = out_dir / src.name
                        key = str(dst.resolve())
                        if key not in copied_files:
                            copy2(src, dst)
                            copied_files.add(key)
                            n_copy += 1
                            print(f"[COPY] {src} -> {dst}")
            processed_dirs.add(sac_path.parent)

        # ------------------------------------------------------------
        # Read SAC and rename by header info
        # ------------------------------------------------------------
        st = read(str(sac_path))
        if len(st) == 0:
            print(f"[SKIP] Empty stream: {sac_path}")
            n_skip += 1
            continue

        tr = st[0].copy()

        sta = None
        if hasattr(tr.stats, "sac") and hasattr(tr.stats.sac, "kstnm"):
            if tr.stats.sac.kstnm is not None:
                sta = str(tr.stats.sac.kstnm).strip()

        if not sta:
            sta = str(getattr(tr.stats, "station", "")).strip()

        if not sta:
            print(f"[SKIP] No station name in header: {sac_path}")
            n_skip += 1
            continue

        comp_raw = None
        if hasattr(tr.stats, "sac") and hasattr(tr.stats.sac, "kcmpnm"):
            if tr.stats.sac.kcmpnm is not None:
                comp_raw = str(tr.stats.sac.kcmpnm).strip()

        if not comp_raw:
            comp_raw = str(getattr(tr.stats, "channel", "")).strip()

        if not comp_raw:
            print(f"[SKIP] No component/channel in header: {sac_path}")
            n_skip += 1
            continue

        comp = comp_raw.replace("_", "").replace(" ", "").replace("-", "").upper()

        if comp in ["E", "N", "Z"]:
            comp = "HH" + comp

        if comp not in ["HHE", "HHN", "HHZ"]:
            print(f"[SKIP] Unsupported component '{comp_raw}' -> '{comp}': {sac_path}")
            n_skip += 1
            continue

        tr.stats.network = NETWORK
        tr.stats.station = sta
        tr.stats.location = ""
        tr.stats.channel = comp

        out_name = f"{NETWORK}.{sta}..{comp}.SAC"
        out_path = out_dir / out_name

        tr.write(str(out_path), format="SAC")
        print(f"[OK] {sac_path} -> {out_path}")
        n_ok += 1

    except Exception as e:
        print(f"[FAIL] {sac_path}: {e}")
        n_fail += 1

print("-" * 60)
print("Done.")
print(f"  Success SAC rename : {n_ok}")
print(f"  Skipped            : {n_skip}")
print(f"  Failed             : {n_fail}")
print(f"  Copied aux files   : {n_copy}")