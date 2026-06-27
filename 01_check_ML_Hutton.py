#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate local magnitude ML directly from the Hutton equation.

Important assumption:
    Input SAC data must already be Wood-Anderson displacement traces.

Hutton ML equation:
    ML = log10(A) + 1.11 log10(R/100) + 0.00189(R - 100) + 3.0

where:
    A = zero-to-peak Wood-Anderson amplitude in mm
    R = hypocentral / epicentral distance in km, here taken from SAC header dist

Flow:
    1. Read event SAC files.
    2. Group by station.
    3. Require horizontal components only: HHE and HHN.
    4. Determine origin time and distance from SAC header.
    5. Determine S-wave window from S pick or theoretical S arrival.
    6. Measure maximum absolute amplitude in the window.
    7. Convert amplitude to mm using explicit scale.
    8. Calculate component ML with Hutton equation.
    9. Average HHE and HHN to obtain station ML.
    10. Average station ML values to obtain event ML.

No station correction.
No catalog correction.
No automatic amplitude scale guessing.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pygmt

from obspy import read


# ========================= USER PARAMETERS ========================= #

INPUT_ROOT = Path("output/01_aligned_vel_data")
# Example:
# INPUT_ROOT = Path("output/01_aligned_WA_data")
# Use the folder that contains event subdirectories.

OUTPUT_ML_ROOT = Path("output/02_check_amp_ML_Hutton_direct")
OUTPUT_ML_FIG_ROOT = Path("figures/02_check_amp_ML_Hutton_direct")
STATION_AMP_FIG_ROOT = OUTPUT_ML_FIG_ROOT / "station_amp_check"

# Use horizontal components only for ML.
REQUIRED_COMPONENTS = ["HHE", "HHN"]

# Velocity model for theoretical arrival time fallback.
VP = 6.5
VS = 3.5
VSURFACE = 1.5

# Amplitude measurement window.
# Window = S arrival + AMP_AFTER_S_SEC to surface-wave arrival - AMP_BEFORE_SURF_SEC
AMP_AFTER_S_SEC = 0.0
AMP_BEFORE_SURF_SEC = 2.0
MIN_AMP_WINDOW_SEC = 5.0

# Filtering:
# For final Wood-Anderson displacement traces, usually keep this as None.
# Set a value only if you intentionally want to filter before amplitude measurement.
HIGHPASS_FREQ = None
# Example:
# HIGHPASS_FREQ = 0.05

# Explicit amplitude conversion to millimeter.
#
# If input SAC amplitude is already WA displacement in meter:
#     INPUT_AMPLITUDE_TO_MM = 1000.0
#
# If input SAC amplitude is already WA displacement in millimeter:
#     INPUT_AMPLITUDE_TO_MM = 1.0
#
# If input SAC amplitude is already WA displacement in nanometer:
#     INPUT_AMPLITUDE_TO_MM = 1e-6
#
INPUT_AMPLITUDE_TO_MM = 1000.0

# SAC undefined value.
SAC_UNDEF = -12345.0

# Output files.
SUCCESS_LOG = OUTPUT_ML_ROOT / "success.log"
FAIL_LOG = OUTPUT_ML_ROOT / "fail.log"

COMPONENT_CSV = OUTPUT_ML_ROOT / "component_hutton_ml.csv"
STATION_CSV = OUTPUT_ML_ROOT / "station_hutton_ml.csv"
EVENT_CSV = OUTPUT_ML_ROOT / "event_hutton_ml_summary.csv"

# ================================================================== #


for p in [OUTPUT_ML_ROOT, OUTPUT_ML_FIG_ROOT, STATION_AMP_FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

SUCCESS_LOG.write_text("")
FAIL_LOG.write_text("")


def log_success(msg):
    with open(SUCCESS_LOG, "a") as f:
        f.write(msg + "\n")


def log_fail(msg):
    with open(FAIL_LOG, "a") as f:
        f.write(msg + "\n")


def is_undef(x):
    if x is None:
        return True

    try:
        return float(x) == SAC_UNDEF
    except Exception:
        return True


def get_sac_header(tr, key, default=None):
    sac = getattr(tr.stats, "sac", None)
    return getattr(sac, key, default) if sac else default


def get_header_mag(tr):
    mag = get_sac_header(tr, "mag")

    if is_undef(mag):
        return np.nan

    try:
        return float(mag)
    except Exception:
        return np.nan


def get_origin_time(tr):
    """
    Get origin time from SAC header O.

    SAC O is relative to trace starttime.
    """
    o = get_sac_header(tr, "o")

    if is_undef(o):
        return None

    return tr.stats.starttime + float(o)


def get_distance_km(tr):
    """
    Get distance in km from SAC header.

    Priority:
        1. dist
        2. gcarc * 111.19
    """
    dist = get_sac_header(tr, "dist")

    if not is_undef(dist) and float(dist) > 0:
        return float(dist)

    gcarc = get_sac_header(tr, "gcarc")

    if not is_undef(gcarc) and float(gcarc) > 0:
        return float(gcarc) * 111.19

    return None


def classify_pick(label):
    """
    Classify SAC phase label into P or S group.

    Accepted examples:
        P, Pg, Pn
        S, Sg, Sn
    """
    if not label:
        return None

    s = str(label).strip().upper().replace(" ", "")

    if "PG" in s:
        return "Pg"
    if "PN" in s:
        return "Pn"
    if s == "P" or s.startswith("P"):
        return "P"

    if "SG" in s:
        return "Sg"
    if "SN" in s:
        return "Sn"
    if s == "S" or s.startswith("S"):
        return "S"

    return None


def collect_picks(traces):
    """
    Collect P and S picks from SAC headers.

    Uses:
        A / KA
        T0-T9 / KT0-KT9

    Returns:
        earliest P pick, earliest S pick
    """
    P = []
    S = []

    for comp, tr in traces.items():
        # A pick.
        a = get_sac_header(tr, "a")
        ka = get_sac_header(tr, "ka")

        if not is_undef(a):
            lab = classify_pick(ka)

            # If A exists but label is missing, assume P.
            if lab is None:
                lab = "P"

            if lab.startswith("P"):
                P.append(tr.stats.starttime + float(a))
            elif lab.startswith("S"):
                S.append(tr.stats.starttime + float(a))

        # T0-T9 picks.
        for i in range(10):
            t = get_sac_header(tr, f"t{i}")
            k = get_sac_header(tr, f"kt{i}")

            if is_undef(t):
                continue

            lab = classify_pick(k)

            if lab and lab.startswith("P"):
                P.append(tr.stats.starttime + float(t))
            elif lab and lab.startswith("S"):
                S.append(tr.stats.starttime + float(t))

    p_pick = min(P) if P else None
    s_pick = min(S) if S else None

    return p_pick, s_pick


def preprocess_trace(tr):
    """
    Optional preprocessing before amplitude measurement.

    For final Wood-Anderson displacement traces, HIGHPASS_FREQ should usually be None.
    """
    tr = tr.copy()

    if HIGHPASS_FREQ is not None and HIGHPASS_FREQ > 0:
        tr.detrend("demean")
        tr.detrend("linear")
        tr.taper(max_percentage=0.05, type="hann")
        tr.filter(
            "highpass",
            freq=HIGHPASS_FREQ,
            corners=4,
            zerophase=True,
        )

    return tr


def hutton_ml(amplitude_mm, distance_km):
    """
    Calculate ML using Hutton equation.

    amplitude_mm:
        Zero-to-peak Wood-Anderson displacement amplitude in mm.

    distance_km:
        Distance in km.
    """
    A = float(amplitude_mm)
    R = float(distance_km)

    if A <= 0 or R <= 0:
        return np.nan

    return (
        np.log10(A)
        + 1.11 * np.log10(R / 100.0)
        + 0.00189 * (R - 100.0)
        + 3.0
    )


def measure_zero_to_peak_amplitude(tr, t0, t1):
    """
    Measure maximum absolute amplitude in the selected window.

    Returns:
        amp_raw:
            Raw amplitude in the input SAC unit.

        amp_mm:
            Amplitude converted to mm.

        peak_time:
            Time of maximum absolute amplitude.

        peak_signed:
            Signed amplitude value at peak.

        npts:
            Number of samples in the measurement window.
    """
    tr_win = tr.copy()
    tr_win.trim(t0, t1, pad=False)

    if tr_win.stats.npts < 2:
        return np.nan, np.nan, None, np.nan, 0

    data = tr_win.data.astype(np.float64)

    if data.size < 2:
        return np.nan, np.nan, None, np.nan, 0

    if not np.any(np.isfinite(data)):
        return np.nan, np.nan, None, np.nan, int(data.size)

    i = int(np.nanargmax(np.abs(data)))

    peak_signed = float(data[i])
    amp_raw = abs(peak_signed)
    amp_mm = amp_raw * float(INPUT_AMPLITUDE_TO_MM)

    peak_time = tr_win.stats.starttime + i / tr_win.stats.sampling_rate

    return amp_raw, amp_mm, peak_time, peak_signed, int(tr_win.stats.npts)


def group_files(event_dir):
    """
    Group SAC files by net.station.

    Only keeps REQUIRED_COMPONENTS.
    """
    groups = defaultdict(dict)

    for f in sorted(event_dir.rglob("*.SAC")):
        try:
            tr = read(str(f), headonly=True)[0]

            net = str(tr.stats.network).strip()
            sta = str(tr.stats.station).strip()
            cha = str(tr.stats.channel).strip()

            if cha in REQUIRED_COMPONENTS:
                groups[(net, sta)][cha] = f

        except Exception as e:
            log_fail(f"[READ-HEAD-FAIL] {f} :: {e}")

    return groups


def process_station(event_dir, net, sta, files):
    """
    Process one station for one event.

    Returns component-level rows.
    """
    missing = [c for c in REQUIRED_COMPONENTS if c not in files]

    if missing:
        log_fail(f"[MISS-COMP] {event_dir.name} {net}.{sta} missing {missing}")
        return []

    try:
        traces = {c: read(str(files[c]))[0] for c in REQUIRED_COMPONENTS}
    except Exception as e:
        log_fail(f"[READ-FAIL] {event_dir.name} {net}.{sta} :: {e}")
        return []

    # Use first horizontal component as reference for origin, distance, header ML.
    ref_comp = REQUIRED_COMPONENTS[0]
    tr_ref = traces[ref_comp]

    origin = get_origin_time(tr_ref)
    distance_km = get_distance_km(tr_ref)
    header_ml = get_header_mag(tr_ref)

    if origin is None:
        log_fail(f"[NO-ORIGIN] {event_dir.name} {net}.{sta}")
        return []

    if distance_km is None or distance_km <= 0:
        log_fail(f"[NO-DIST] {event_dir.name} {net}.{sta}")
        return []

    p_pick, s_pick = collect_picks(traces)

    p_source = "header"
    s_source = "header"

    if p_pick is None:
        p_pick = origin + distance_km / VP
        p_source = "theory"

    if s_pick is None:
        s_pick = origin + distance_km / VS
        s_source = "theory"

    surface_pick = origin + distance_km / VSURFACE

    amp_window_start = s_pick + AMP_AFTER_S_SEC
    amp_window_end = surface_pick - AMP_BEFORE_SURF_SEC

    amp_window_sec = amp_window_end - amp_window_start

    if amp_window_sec < MIN_AMP_WINDOW_SEC:
        log_fail(
            f"[BAD-AMP-WINDOW] {event_dir.name} {net}.{sta} "
            f"S={s_pick.isoformat()} SURF={surface_pick.isoformat()} "
            f"WIN={amp_window_start.isoformat()}->{amp_window_end.isoformat()} "
            f"LEN={amp_window_sec:.2f}s"
        )
        return []

    rows = []

    for comp in REQUIRED_COMPONENTS:
        tr = traces[comp]
        tr_proc = preprocess_trace(tr)

        amp_raw, amp_mm, peak_time, peak_signed, npts = measure_zero_to_peak_amplitude(
            tr_proc,
            amp_window_start,
            amp_window_end,
        )

        ml = hutton_ml(amp_mm, distance_km)

        if not np.isfinite(ml):
            log_fail(
                f"[BAD-ML] {event_dir.name} {net}.{sta}.{comp} "
                f"A_raw={amp_raw} A_mm={amp_mm} R={distance_km}"
            )
            continue

        rows.append({
            "event": event_dir.name,
            "net": net,
            "sta": sta,
            "comp": comp,
            "file": files[comp].name,

            "distance_km": distance_km,
            "header_ML": header_ml,

            "ML_component": ml,

            "origin_time": origin.isoformat(),
            "p_pick": p_pick.isoformat(),
            "p_source": p_source,
            "s_pick": s_pick.isoformat(),
            "s_source": s_source,
            "surface_pick": surface_pick.isoformat(),

            "amp_window_start": amp_window_start.isoformat(),
            "amp_window_end": amp_window_end.isoformat(),
            "amp_window_sec": float(amp_window_sec),

            "peak_time": peak_time.isoformat() if peak_time else "",
            "peak_signed_input_unit": peak_signed,

            "amplitude_raw_input_unit": amp_raw,
            "amplitude_mm": amp_mm,
            "input_amplitude_to_mm": float(INPUT_AMPLITUDE_TO_MM),

            "npts_window": npts,
        })

    if rows:
        log_success(
            f"[OK] {event_dir.name} {net}.{sta} "
            f"dist={distance_km:.2f} km "
            f"header_ML={header_ml} "
            f"P={p_pick.isoformat()}({p_source}) "
            f"S={s_pick.isoformat()}({s_source}) "
            f"SURF={surface_pick.isoformat()} "
            f"WIN={amp_window_start.isoformat()}->{amp_window_end.isoformat()}"
        )

    return rows


def make_station_summary(component_df):
    """
    Average HHE and HHN ML values to obtain station ML.

    One row per event-station.
    """
    station_df = component_df.groupby(
        ["event", "net", "sta"],
        as_index=False,
    ).agg(
        ML_station=("ML_component", "mean"),
        ML_station_std=("ML_component", "std"),
        n_component=("ML_component", "count"),

        distance_km=("distance_km", "mean"),
        distance_min_km=("distance_km", "min"),
        distance_max_km=("distance_km", "max"),

        header_ML=("header_ML", "mean"),

        amplitude_mm_max=("amplitude_mm", "max"),
        amplitude_mm_mean=("amplitude_mm", "mean"),

        amp_window_start=("amp_window_start", "first"),
        amp_window_end=("amp_window_end", "first"),
        s_pick=("s_pick", "first"),
        s_source=("s_source", "first"),
    )

    # Keep only stations that have both horizontal components.
    station_df = station_df[station_df["n_component"] >= 2].copy()

    return station_df


def make_event_summary(station_df):
    """
    Average station ML values to obtain event ML.

    One row per event.
    """
    event_df = station_df.groupby(
        "event",
        as_index=False,
    ).agg(
        ML_event_mean=("ML_station", "mean"),
        ML_event_std=("ML_station", "std"),
        ML_event_median=("ML_station", "median"),

        n_station=("ML_station", "count"),

        distance_min_km=("distance_km", "min"),
        distance_max_km=("distance_km", "max"),
        distance_mean_km=("distance_km", "mean"),

        header_ML_mean=("header_ML", "mean"),
    )

    return event_df


def plot_distance_component_ml(component_df):
    """
    Plot distance versus component ML.
    """
    if component_df.empty:
        return

    tmp = component_df.dropna(subset=["distance_km", "ML_component"]).copy()

    if tmp.empty:
        return

    ml_min = float(tmp["ML_component"].min())
    ml_max = float(tmp["ML_component"].max())

    if ml_min == ml_max:
        ml_min -= 0.5
        ml_max += 0.5

    d_min = max(0.0, float(tmp["distance_km"].min()) - 10.0)
    d_max = float(tmp["distance_km"].max()) + 10.0

    if d_min == d_max:
        d_max = d_min + 10.0

    pygmt.makecpt(
        cmap="turbo",
        series=[ml_min, ml_max],
    )

    fig = pygmt.Figure()

    fig.basemap(
        region=[d_min, d_max, ml_min, ml_max],
        projection="X12c/8c",
        frame=[
            'xaf+l"Distance (km)"',
            'yaf+l"Component Hutton ML"',
            "WSen",
        ],
    )

    fig.plot(
        x=tmp["distance_km"],
        y=tmp["ML_component"],
        style="c0.2c",
        fill=tmp["ML_component"],
        cmap=True,
        pen="0.25p,black",
    )

    fig.colorbar(frame='af+l"Component ML"')

    fig.savefig(
        OUTPUT_ML_FIG_ROOT / "distance_vs_component_ml.png",
        dpi=300,
    )


def plot_distance_station_ml(station_df):
    """
    Plot distance versus station ML.
    """
    if station_df.empty:
        return

    tmp = station_df.dropna(subset=["distance_km", "ML_station"]).copy()

    if tmp.empty:
        return

    ml_min = float(tmp["ML_station"].min())
    ml_max = float(tmp["ML_station"].max())

    if ml_min == ml_max:
        ml_min -= 0.5
        ml_max += 0.5

    d_min = max(0.0, float(tmp["distance_km"].min()) - 10.0)
    d_max = float(tmp["distance_km"].max()) + 10.0

    if d_min == d_max:
        d_max = d_min + 10.0

    pygmt.makecpt(
        cmap="turbo",
        series=[ml_min, ml_max],
    )

    fig = pygmt.Figure()

    fig.basemap(
        region=[d_min, d_max, ml_min, ml_max],
        projection="X12c/8c",
        frame=[
            'xaf+l"Distance (km)"',
            'yaf+l"Station Hutton ML"',
            "WSen",
        ],
    )

    fig.plot(
        x=tmp["distance_km"],
        y=tmp["ML_station"],
        style="c0.25c",
        fill=tmp["ML_station"],
        cmap=True,
        pen="0.25p,black",
    )

    fig.colorbar(frame='af+l"Station ML"')

    fig.savefig(
        OUTPUT_ML_FIG_ROOT / "distance_vs_station_ml.png",
        dpi=300,
    )


def plot_event_ml_histogram(event_df):
    """
    Plot histogram of event ML values.
    """
    if event_df.empty:
        return

    tmp = event_df.dropna(subset=["ML_event_mean"]).copy()

    if tmp.empty:
        return

    ml_min = float(tmp["ML_event_mean"].min())
    ml_max = float(tmp["ML_event_mean"].max())

    if ml_min == ml_max:
        ml_min -= 0.5
        ml_max += 0.5

    hist, edges = np.histogram(
        tmp["ML_event_mean"].values,
        bins=20,
        range=(ml_min, ml_max),
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig = pygmt.Figure()

    fig.basemap(
        region=[
            ml_min,
            ml_max,
            0,
            max(hist) * 1.15 if len(hist) > 0 and max(hist) > 0 else 1,
        ],
        projection="X12c/8c",
        frame=[
            'xaf+l"Event Hutton ML"',
            'yaf+l"Number of events"',
            "WSen",
        ],
    )

    fig.plot(
        x=centers,
        y=hist,
        style=f"b{widths[0]}u",
        fill="gray70",
        pen="0.25p,black",
    )

    fig.savefig(
        OUTPUT_ML_FIG_ROOT / "event_ml_histogram.png",
        dpi=300,
    )


def main():
    all_component_rows = []

    event_dirs = sorted([d for d in INPUT_ROOT.glob("20*") if d.is_dir()])

    if not event_dirs:
        print(f"[ERROR] No event directories found: {INPUT_ROOT}/20*")
        return

    print(f"[INFO] Found {len(event_dirs)} event directories")

    for event_dir in event_dirs:
        groups = group_files(event_dir)

        if not groups:
            log_fail(f"[NO-SAC] {event_dir}")
            continue

        for (net, sta), files in sorted(groups.items()):
            rows = process_station(event_dir, net, sta, files)
            all_component_rows.extend(rows)

    if not all_component_rows:
        print("[ERROR] No valid component ML rows created.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    component_df = pd.DataFrame(all_component_rows)

    component_df = component_df.replace([np.inf, -np.inf], np.nan)
    component_df = component_df.dropna(
        subset=[
            "ML_component",
            "distance_km",
            "amplitude_mm",
        ]
    ).copy()

    if component_df.empty:
        print("[ERROR] Component DataFrame is empty after dropping NaN values.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    station_df = make_station_summary(component_df)

    if station_df.empty:
        print("[ERROR] Station DataFrame is empty.")
        print("This usually means no station has both HHE and HHN valid ML values.")
        print(f"Check fail log: {FAIL_LOG}")
        return

    event_df = make_event_summary(station_df)

    component_df.to_csv(COMPONENT_CSV, index=False)
    station_df.to_csv(STATION_CSV, index=False)
    event_df.to_csv(EVENT_CSV, index=False)

    plot_distance_component_ml(component_df)
    plot_distance_station_ml(station_df)
    plot_event_ml_histogram(event_df)

    print("========== DONE ==========")
    print(f"Input root        : {INPUT_ROOT}")
    print(f"Amplitude to mm   : {INPUT_AMPLITUDE_TO_MM}")
    print(f"Component CSV     : {COMPONENT_CSV}")
    print(f"Station CSV       : {STATION_CSV}")
    print(f"Event summary CSV : {EVENT_CSV}")
    print(f"Figure component  : {OUTPUT_ML_FIG_ROOT / 'distance_vs_component_ml.png'}")
    print(f"Figure station    : {OUTPUT_ML_FIG_ROOT / 'distance_vs_station_ml.png'}")
    print(f"Figure event hist : {OUTPUT_ML_FIG_ROOT / 'event_ml_histogram.png'}")
    print(f"Success log       : {SUCCESS_LOG}")
    print(f"Fail log          : {FAIL_LOG}")


if __name__ == "__main__":
    main()