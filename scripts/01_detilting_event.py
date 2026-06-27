#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerically detilt displacement SAC files using ObsPy spline detrend.

Input structure:
    output/IES_data/{event_id}/*.disp.SAC

Output:
    output/IES_data/{event_id}/*.detilt.SAC
or overwrite original SAC if OVERWRITE_ORIGINAL = True

Method:
    1) read displacement SAC
    2) remove slow trend by ObsPy spline detrend
    3) optional final demean / linear detrend
    4) write new SAC

Logs:
    output/IES_data/detilt_success.log
    output/IES_data/detilt_fail.log

Notes:
- Apply this to displacement files, not raw velocity.
- dspline is in samples, so here we define SPLINE_SEC in seconds
  and convert it to dspline automatically.
"""

from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
from obspy import read


# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_ROOT = Path("output/IES_data")

# file pattern to process
INPUT_GLOB = "*.disp.SAC"

# output mode
OVERWRITE_ORIGINAL = False
OUTPUT_SUFFIX = ".detilt.SAC"   # only used if OVERWRITE_ORIGINAL=False

# -------- spline detrend parameters --------
# order of spline, 3 = cubic spline
SPLINE_ORDER = 3

# spacing between spline nodes in seconds
# try 10, 20, 30, 50 s depending on how strong the tilt is
SPLINE_SEC = 20.0

# optional cleanup before spline detrend
PRE_DEMEAN = True
PRE_LINEAR_DETREND = False

# optional final cleanup after spline detrend
FINAL_DEMEAN = True
FINAL_LINEAR_DETREND = True

# optional clipping of NaN/Inf
REPLACE_BAD_VALUES = True

# event folder: first 14 chars digits
ONLY_EVENT_DIR_14DIGIT = True

# log files
SUCCESS_LOG = INPUT_ROOT / "detilt_success.log"
FAIL_LOG = INPUT_ROOT / "detilt_fail.log"


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

    write_log(SUCCESS_LOG, f"# detilt success log started: {now_str()}")
    write_log(FAIL_LOG, f"# detilt fail log started: {now_str()}")


def is_event_dir_name(name: str) -> bool:
    if not ONLY_EVENT_DIR_14DIGIT:
        return True
    return len(name) >= 14 and name[:14].isdigit()


def linear_detrend(y):
    if len(y) < 2:
        return y.copy()
    x = np.arange(len(y), dtype=float)
    p = np.polyfit(x, y, 1)
    return y - (p[0] * x + p[1])


def build_output_path(infile: Path) -> Path:
    if OVERWRITE_ORIGINAL:
        return infile

    name = infile.name
    if name.endswith(".disp.SAC"):
        outname = name.replace(".disp.SAC", OUTPUT_SUFFIX)
    elif name.endswith(".SAC"):
        outname = name.replace(".SAC", OUTPUT_SUFFIX)
    else:
        outname = name + OUTPUT_SUFFIX

    return infile.with_name(outname)


def spline_sec_to_dspline(dt, spline_sec):
    """
    Convert spline spacing from seconds to samples.
    ObsPy detrend('spline') expects dspline in number of samples.
    """
    dspline = int(round(float(spline_sec) / float(dt)))
    if dspline < 2:
        dspline = 2
    return dspline


def process_one_file(infile: Path):
    st = read(str(infile))
    if len(st) == 0:
        raise RuntimeError("Empty stream")

    tr = st[0]
    y0 = tr.data.astype(np.float64)
    dt = float(tr.stats.delta)
    sr = float(tr.stats.sampling_rate)

    if REPLACE_BAD_VALUES:
        bad = ~np.isfinite(y0)
        if np.any(bad):
            y0[bad] = 0.0
            tr.data = y0.astype(tr.data.dtype, copy=False)

    # optional pre-cleaning
    if PRE_DEMEAN:
        tr.detrend("demean")
    if PRE_LINEAR_DETREND:
        tr.detrend("linear")

    # main de-tilt by spline
    dspline = spline_sec_to_dspline(dt, SPLINE_SEC)
    tr.detrend("spline", order=SPLINE_ORDER, dspline=dspline)

    # optional final cleanup
    if FINAL_DEMEAN:
        tr.detrend("demean")
    if FINAL_LINEAR_DETREND:
        tr.detrend("linear")

    out_tr = tr.copy()

    # save some note in SAC if possible
    if not hasattr(out_tr.stats, "sac") or out_tr.stats.sac is None:
        out_tr.stats.sac = {}

    try:
        out_tr.stats.sac.kuser1 = "detilt"
        out_tr.stats.sac.kuser2 = f"sp{SPLINE_ORDER}"
    except Exception:
        pass

    outfile = build_output_path(infile)
    out_tr.write(str(outfile), format="SAC")

    std0 = float(np.std(y0))
    std1 = float(np.std(out_tr.data.astype(np.float64)))

    return {
        "outfile": outfile,
        "dt": dt,
        "sr": sr,
        "dspline": dspline,
        "std_before": std0,
        "std_after": std1,
    }


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    reset_logs()

    event_dirs = [p for p in sorted(INPUT_ROOT.iterdir())
                  if p.is_dir() and is_event_dir_name(p.name)]

    print(f"INPUT_ROOT       : {INPUT_ROOT.resolve()}")
    print(f"N_EVENT_DIRS     : {len(event_dirs)}")
    print(f"INPUT_GLOB       : {INPUT_GLOB}")
    print(f"SPLINE_ORDER     : {SPLINE_ORDER}")
    print(f"SPLINE_SEC       : {SPLINE_SEC}")
    print(f"PRE_DEMEAN       : {PRE_DEMEAN}")
    print(f"PRE_LINEAR_DT    : {PRE_LINEAR_DETREND}")
    print(f"FINAL_DEMEAN     : {FINAL_DEMEAN}")
    print(f"FINAL_LINEAR_DT  : {FINAL_LINEAR_DETREND}")
    print(f"OVERWRITE        : {OVERWRITE_ORIGINAL}")
    print(f"SUCCESS_LOG      : {SUCCESS_LOG}")
    print(f"FAIL_LOG         : {FAIL_LOG}")

    n_ok = 0
    n_fail = 0

    for event_dir in event_dirs:
        files = sorted(event_dir.glob(INPUT_GLOB))
        if not files:
            continue

        print(f"\n[EVENT] {event_dir.name} | nfiles={len(files)}")

        for f in files:
            try:
                result = process_one_file(f)
                outfile = result["outfile"]
                dt = result["dt"]
                sr = result["sr"]
                dspline = result["dspline"]
                std0 = result["std_before"]
                std1 = result["std_after"]

                msg = (
                    f"{f} -> {outfile} | "
                    f"dt={dt:.6f} | sr={sr:.6f} | "
                    f"SPLINE_SEC={SPLINE_SEC:.3f} | "
                    f"dspline={dspline} | "
                    f"std_before={std0:.6g} | std_after={std1:.6g}"
                )

                print(f"[OK] {msg}")
                write_log(SUCCESS_LOG, msg)
                n_ok += 1

            except Exception as e:
                err = f"{f} | ERROR: {e}"
                print(f"[FAIL] {err}")
                write_log(FAIL_LOG, err)
                write_log(FAIL_LOG, traceback.format_exc())
                n_fail += 1

    summary = f"SUMMARY | success={n_ok} | fail={n_fail} | time={now_str()}"
    print("\n" + "-" * 60)
    print(summary)
    write_log(SUCCESS_LOG, summary)
    write_log(FAIL_LOG, summary)


if __name__ == "__main__":
    main()