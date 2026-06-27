#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from copy import deepcopy
import numpy as np
from obspy import read

# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/Data_fixed_name")   # new clean SAC files
OUTPUT_ROOT = None                       # None = save beside input SAC
PRE_FILT = (0.01, 0.02, 40.0, 45.0)
WATER_LEVEL = 60.0

NETWORK = "VN"
LOG_FILE = INPUT_ROOT / "remove_response_summary.txt"
# ================================================================= #
####  CAREFULLY CONSIDERATION THIS PART #### VIETNAM DATA HAVE DOUBLE SENSITIVITY REMOVAL ISSUE #######
REMOVE_SENSITIVITY=True # False if you found that double removal
LOC = "??" # LOC = ?? for all rather than ""
# ================================================================= #

if not INPUT_ROOT.exists():
    raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT.resolve()}")

if OUTPUT_ROOT is not None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# only process clean SAC files: VN.STA..CHAN.SAC
sac_files = sorted([p for p in INPUT_ROOT.rglob("VN.*..*.SAC") if p.is_file()])

print(f"INPUT_ROOT   : {INPUT_ROOT.resolve()}")
print(f"OUTPUT_ROOT  : {OUTPUT_ROOT if OUTPUT_ROOT is not None else '[same as input folder]'}")
print(f"PRE_FILT     : {PRE_FILT}")
print(f"WATER_LEVEL  : {WATER_LEVEL}")
print(f"N_CANDIDATES : {len(sac_files)}")

log_lines = []
log_lines.append(f"INPUT_ROOT   : {INPUT_ROOT.resolve()}")
log_lines.append(f"OUTPUT_ROOT  : {OUTPUT_ROOT if OUTPUT_ROOT is not None else '[same as input folder]'}")
log_lines.append(f"PRE_FILT     : {PRE_FILT}")
log_lines.append(f"WATER_LEVEL  : {WATER_LEVEL}")
log_lines.append(f"N_CANDIDATES : {len(sac_files)}")
log_lines.append("")

n_ok = 0
n_skip = 0
n_fail = 0

for sac_path in sac_files:
    parts = sac_path.name.split(".")

    # expected: VN STA '' CHAN SAC
    if len(parts) != 5:
        msg = f"[SKIP] Bad filename format: {sac_path}"
        print(msg)
        log_lines.append(msg)
        n_skip += 1
        continue

    net, sta, loc, comp, ext = parts

    if net != NETWORK or ext.upper() != "SAC":
        msg = f"[SKIP] Bad filename format: {sac_path}"
        print(msg)
        log_lines.append(msg)
        n_skip += 1
        continue

    resp_name = f"RESP.{NETWORK}.{sta}..{comp}"
    resp_path = sac_path.parent / resp_name

    if not resp_path.exists():
        msg = f"[MISS] RESP not found: {resp_path}"
        print(msg)
        log_lines.append(msg)
        n_fail += 1
        continue

    try:
        st = read(str(sac_path))
        if len(st) == 0:
            msg = f"[SKIP] Empty stream: {sac_path}"
            print(msg)
            log_lines.append(msg)
            n_skip += 1
            continue

        tr = st[0].copy()

        sac_header = deepcopy(getattr(tr.stats, "sac", None))

        tr.stats.network = NETWORK
        tr.stats.station = sta
        tr.stats.location = ""
        tr.stats.channel = comp

        # preprocess: rmean, rtr, taper
        tr.detrend("demean")
        tr.detrend("linear")
        tr.taper(max_percentage=0.05, type="hann")

        tr.simulate(
            paz_remove=None,
            pre_filt=PRE_FILT,
            remove_sensitivity=REMOVE_SENSITIVITY,
            seedresp={
                "filename": str(resp_path),
                "date": tr.stats.starttime,
                "units": "DIS",
                "network": NETWORK,
                "station": sta,
                "location": LOC,
                "channel": comp,
            }
        )

        if sac_header is not None:
            tr.stats.sac = sac_header
        else:
            from obspy.core import AttribDict
            tr.stats.sac = AttribDict()

        # keep header consistent with processed data
        tr.stats.sac.iftype = 1   # ITIME
        tr.stats.sac.idep = 6     # IDISP
        tr.stats.sac.kstnm = sta
        tr.stats.sac.kcmpnm = comp

        tr.stats.sac.depmin = float(np.min(tr.data))
        tr.stats.sac.depmax = float(np.max(tr.data))
        tr.stats.sac.depmen = float(np.mean(tr.data))
        

        out_name = f"{sta}.{comp}.disp.SAC"

        if OUTPUT_ROOT is None:
            out_path = sac_path.parent / out_name
        else:
            rel = sac_path.parent.relative_to(INPUT_ROOT)
            out_dir = OUTPUT_ROOT / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / out_name

        tr.write(str(out_path), format="SAC")
        msg = f"[OK] {sac_path} -> {out_path}"
        print(msg)
        log_lines.append(msg)
        n_ok += 1

    except Exception as e:
        msg = f"[FAIL] {sac_path}: {e}"
        print(msg)
        log_lines.append(msg)
        n_fail += 1

summary_lines = [
    "-" * 60,
    "Done.",
    f"  Success : {n_ok}",
    f"  Skipped : {n_skip}",
    f"  Failed  : {n_fail}",
]

for line in summary_lines:
    print(line)
    log_lines.append(line)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"[LOG] Summary saved to: {LOG_FILE}")