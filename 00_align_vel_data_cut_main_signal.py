#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from obspy import read, UTCDateTime, Trace
from obspy.signal.trigger import classic_sta_lta, trigger_onset


# ========================= USER PARAMETERS ========================= #
INPUT_ROOT = Path("input/complete_data")

OUTPUT_SAC_ROOT = Path("output/01_aligned_vel_data")
OUTPUT_FIG_ROOT = Path("figures/01_aligned_vel_data")

REQUIRED_COMPONENTS = ["HHE", "HHN", "HHZ"]

VP = 6.5         # km/s
VS = 3.5         # km/s
VSURFACE = 1.5   # km/s

PRE_P = 30.0      # sec before P
POST_SURF = 30.0  # sec after surface wave

# high-pass filter for cut data
HIGHPASS_FREQ = 0.05   # Hz ; change as needed
FILTER_CORNERS = 4
FILTER_ZEROPHASE = True

# STA/LTA
STA_SEC = 1.0
LTA_SEC = 20.0
TRIG_ON = 3.0
TRIG_OFF = 1.0

# search around theoretical P when no header P  
P_SEARCH_BEFORE = 20.0
P_SEARCH_AFTER = 40.0

ALIGN_TOL_SEC = 0.01
SAC_UNDEF = -12345.0

SUCCESS_LOG = OUTPUT_SAC_ROOT / "success.log"
FAIL_LOG = OUTPUT_SAC_ROOT / "fail.log"
REPORT_LOG = OUTPUT_SAC_ROOT / "report.log"
# ================================================================== #


OUTPUT_SAC_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_ROOT.mkdir(parents=True, exist_ok=True)

SUCCESS_LOG.write_text("")
FAIL_LOG.write_text("")
REPORT_LOG.write_text("")


def log_success(msg: str):
    with open(SUCCESS_LOG, "a") as f:
        f.write(msg + "\n")


def log_fail(msg: str):
    with open(FAIL_LOG, "a") as f:
        f.write(msg + "\n")


def log_report(msg: str):
    with open(REPORT_LOG, "a") as f:
        f.write(msg + "\n")


def is_defined(val) -> bool:
    if val is None:
        return False
    try:
        return float(val) != SAC_UNDEF
    except Exception:
        return False


def get_sac_header(tr: Trace, key: str, default=None):
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return default
    return getattr(sac, key, default)


def get_reference_time_from_sac(tr: Trace) -> UTCDateTime:
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return tr.stats.starttime

    nzyear = get_sac_header(tr, "nzyear")
    nzjday = get_sac_header(tr, "nzjday")
    nzhour = get_sac_header(tr, "nzhour", 0)
    nzmin = get_sac_header(tr, "nzmin", 0)
    nzsec = get_sac_header(tr, "nzsec", 0)
    nzmsec = get_sac_header(tr, "nzmsec", 0)

    if not all(is_defined(x) for x in [nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec]):
        return tr.stats.starttime

    return UTCDateTime(
        year=int(nzyear),
        julday=int(nzjday),
        hour=int(nzhour),
        minute=int(nzmin),
        second=int(nzsec),
        microsecond=int(nzmsec) * 1000
    )


def get_origin_time(tr: Trace) -> Optional[UTCDateTime]:
    o = get_sac_header(tr, "o")
    if not is_defined(o):
        return None
    ref = get_reference_time_from_sac(tr)
    return ref + float(o)


def get_pick_abs(tr: Trace, rel_pick_sec: float) -> Optional[UTCDateTime]:
    ref = get_reference_time_from_sac(tr)
    return ref + float(rel_pick_sec)


def clean_phase_label(label: str) -> str:
    if label is None:
        return ""
    s = str(label).strip().upper()
    s = s.replace(" ", "")
    return s


def classify_pick_label(label: str) -> Optional[str]:
    """
    Return one of:
      Pg, P, Pn, Sg, S, Sn
    """
    s = clean_phase_label(label)
    if not s:
        return None

    # exact / strong match first
    if "PG" in s:
        return "Pg"
    if s == "P":
        return "P"
    if "PN" in s:
        return "Pn"

    if "SG" in s:
        return "Sg"
    if s == "S":
        return "S"
    if "SN" in s:
        return "Sn"

    # fallback:
    # something like "PICKP", "P1", "P?" -> P
    # something like "S1", "S?" -> S
    if s.startswith("P"):
        return "P"
    if s.startswith("S"):
        return "S"

    return None


def get_distance_km(tr: Trace) -> Optional[float]:
    dist = get_sac_header(tr, "dist")
    if is_defined(dist) and float(dist) > 0:
        return float(dist)

    gcarc = get_sac_header(tr, "gcarc")
    if is_defined(gcarc) and float(gcarc) > 0:
        return float(gcarc) * 111.19

    return None


def normalize_trace_reference_to_origin(tr: Trace) -> Trace:
    """
    Change SAC reference time to origin.
    Keep absolute waveform timing unchanged.
    Rewrite b/e/a/t0..t9 relative to origin.
    """
    tr = tr.copy()
    sac = getattr(tr.stats, "sac", None)
    if sac is None:
        return tr

    origin = get_origin_time(tr)
    if origin is None:
        return tr

    old_ref = get_reference_time_from_sac(tr)
    abs_start = tr.stats.starttime
    abs_end = tr.stats.endtime

    def shift_to_origin(old_rel):
        if not is_defined(old_rel):
            return old_rel
        abs_pick = old_ref + float(old_rel)
        return abs_pick - origin

    sac.nzyear = origin.year
    sac.nzjday = origin.julday
    sac.nzhour = origin.hour
    sac.nzmin = origin.minute
    sac.nzsec = origin.second
    sac.nzmsec = int(origin.microsecond / 1000)

    sac.b = float(abs_start - origin)
    sac.e = float(abs_end - origin)
    sac.o = 0.0

    a = get_sac_header(tr, "a")
    if is_defined(a):
        sac.a = float(shift_to_origin(a))

    for i in range(10):
        tkey = f"t{i}"
        tval = get_sac_header(tr, tkey)
        if is_defined(tval):
            setattr(sac, tkey, float(shift_to_origin(tval)))

    tr.stats.starttime = abs_start
    return tr


def detect_p_with_stalta(tr: Trace, p_theory: UTCDateTime) -> Optional[UTCDateTime]:
    tr2 = tr.copy()
    tr2.detrend("demean")
    tr2.detrend("linear")
    tr2.taper(max_percentage=0.05, type="hann")

    t0 = max(tr2.stats.starttime, p_theory - P_SEARCH_BEFORE)
    t1 = min(tr2.stats.endtime, p_theory + P_SEARCH_AFTER)
    if t1 <= t0:
        return None

    tr2.trim(t0, t1, pad=False)
    if tr2.stats.npts < 10:
        return None

    df = tr2.stats.sampling_rate
    nsta = max(1, int(STA_SEC * df))
    nlta = max(nsta + 1, int(LTA_SEC * df))

    data = tr2.data.astype(np.float64)
    if np.allclose(data, 0.0):
        return None

    cft = classic_sta_lta(data, nsta, nlta)
    on_off = trigger_onset(cft, TRIG_ON, TRIG_OFF)
    if len(on_off) == 0:
        return None

    i0 = int(on_off[0][0])
    return tr2.stats.starttime + i0 / df


def check_alignment(traces: Dict[str, Trace]) -> Tuple[bool, str]:
    starts = {k: v.stats.starttime for k, v in traces.items()}
    srs = {k: v.stats.sampling_rate for k, v in traces.items()}

    tmin = min(starts.values())
    offsets = {k: float(v - tmin) for k, v in starts.items()}
    max_off = max(offsets.values()) if offsets else 0.0
    same_sr = len(set(round(v, 6) for v in srs.values())) == 1

    aligned = (max_off <= ALIGN_TOL_SEC) and same_sr
    msg = (
        f"start_offsets_sec={{{', '.join(f'{k}:{offsets[k]:.4f}' for k in sorted(offsets))}}}, "
        f"same_sr={same_sr}"
    )
    return aligned, msg


def group_files_by_station(event_dir: Path):
    groups = defaultdict(dict)

    for f in sorted(event_dir.glob("*.SAC")):
        try:
            tr = read(str(f), headonly=True)[0]

            net = getattr(tr.stats, "network", "") or get_sac_header(tr, "knetwk", "")
            sta = getattr(tr.stats, "station", "") or get_sac_header(tr, "kstnm", "")
            cha = getattr(tr.stats, "channel", "") or get_sac_header(tr, "kcmpnm", "")

            net = str(net).strip()
            sta = str(sta).strip()
            cha = str(cha).strip()

            if sta and cha in REQUIRED_COMPONENTS:
                groups[(net, sta)][cha] = f

        except Exception as e:
            log_fail(f"[READ-HEAD-FAIL] {f} :: {e}")

    return groups


def write_sac(tr: Trace, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tr.write(str(outpath), format="SAC")


def collect_station_header_picks(traces: Dict[str, Trace]) -> Dict[str, List[Tuple[UTCDateTime, str, str, int]]]:
    """
    Return dict:
      {
        "P": [(abs_time, label, comp, priority), ...],
        "S": [(abs_time, label, comp, priority), ...]
      }

    Priority:
      Pg=0, P=1, Pn=2
      Sg=0, S=1, Sn=2

    Scan all channels and all candidate pick headers:
      A, T0..T9 with KT0..KT9 labels
    """
    out = {"P": [], "S": []}

    p_pri = {"Pg": 0, "P": 1, "Pn": 2}
    s_pri = {"Sg": 0, "S": 1, "Sn": 2}

    for comp, tr in traces.items():
        # A header: if exists, usually P
        a = get_sac_header(tr, "a")
        ka = get_sac_header(tr, "ka", "")
        if is_defined(a):
            label = classify_pick_label(ka)
            if label is None:
                label = "P"
            abs_time = get_pick_abs(tr, float(a))
            if label in p_pri:
                out["P"].append((abs_time, label, comp, p_pri[label]))
            elif label in s_pri:
                out["S"].append((abs_time, label, comp, s_pri[label]))

        # T0..T9
        for i in range(10):
            tkey = f"t{i}"
            ktkey = f"kt{i}"
            tval = get_sac_header(tr, tkey)
            kval = get_sac_header(tr, ktkey, "")

            if not is_defined(tval):
                continue

            label = classify_pick_label(kval)
            if label is None:
                continue

            abs_time = get_pick_abs(tr, float(tval))

            if label in p_pri:
                out["P"].append((abs_time, label, comp, p_pri[label]))
            elif label in s_pri:
                out["S"].append((abs_time, label, comp, s_pri[label]))

    return out


def choose_best_station_pick(picks: List[Tuple[UTCDateTime, str, str, int]]) -> Optional[Tuple[UTCDateTime, str, str]]:
    """
    picks item = (time, label, comp, priority)
    Choose:
      1) lowest priority number
      2) earliest time among same priority
    """
    if not picks:
        return None
    picks_sorted = sorted(picks, key=lambda x: (x[3], x[0]))
    best = picks_sorted[0]
    return best[0], best[1], best[2]


def highpass_trace(tr: Trace, freq: float) -> Trace:
    trf = tr.copy()
    trf.detrend("demean")
    trf.detrend("linear")
    trf.taper(max_percentage=0.05, type="hann")
    if freq is not None and freq > 0:
        trf.filter(
            "highpass",
            freq=float(freq),
            corners=FILTER_CORNERS,
            zerophase=FILTER_ZEROPHASE,
        )
    return trf


def make_station_plot(
    traces: Dict[str, Trace],
    figpath: Path,
    event_name: str,
    net: str,
    sta: str,
    dist_km: float,
    p_pick: UTCDateTime,
    s_pick: UTCDateTime,
    surf_pick: UTCDateTime,
    p_source: str,
    s_source: str,
    surf_source: str,
    p_label: str,
    s_label: str,
    p_comp: str,
    s_comp: str,
    cut_start: UTCDateTime,
):
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 7.8), sharex=True, constrained_layout=True
    )

    for ax, comp in zip(axes, REQUIRED_COMPONENTS):
        tr = traces[comp]
        t = tr.times("utcdatetime")
        x = np.array([tt - cut_start for tt in t], dtype=float)
        y = tr.data.astype(np.float64)

        ax.plot(x, y, "k-", linewidth=0.8)
        ax.axhline(0.0, color="0.7", linewidth=0.6)

        p_ls = "-" if p_source == "header" else "--"
        s_ls = "-" if s_source == "header" else "--"
        surf_ls = "-" if surf_source == "header" else "--"

        ax.axvline(float(p_pick - cut_start), color="blue", linewidth=1.2, linestyle=p_ls)
        ax.axvline(float(s_pick - cut_start), color="red", linewidth=1.2, linestyle=s_ls)
        ax.axvline(float(surf_pick - cut_start), color="green", linewidth=1.2, linestyle=surf_ls)

        ax.text(
            0.01, 0.92, comp,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7")
        )

        ymax = np.max(np.abs(y)) if len(y) else 1.0
        if ymax <= 0:
            ymax = 1.0
        ax.set_ylim(-ymax * 1.05, ymax * 1.05)

    axes[-1].set_xlabel("Time after cut start (s)")
    axes[1].set_ylabel("Amplitude")

    title = (
        f"{event_name}  {net}.{sta}  Dist={dist_km:.2f} km\n"
        f"P={p_label} ({p_source},{p_comp})   "
        f"S={s_label} ({s_source},{s_comp})   "
        f"Surface={surf_source}   HP={HIGHPASS_FREQ:.4f} Hz"
    )
    axes[0].set_title(title, fontsize=11)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="blue", lw=1.5, linestyle="-", label="P header"),
        Line2D([0], [0], color="blue", lw=1.5, linestyle="--", label="P estimated"),
        Line2D([0], [0], color="red", lw=1.5, linestyle="-", label="S header"),
        Line2D([0], [0], color="red", lw=1.5, linestyle="--", label="S estimated"),
        Line2D([0], [0], color="green", lw=1.5, linestyle="-", label="Surface header"),
        Line2D([0], [0], color="green", lw=1.5, linestyle="--", label="Surface estimated"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=3)

    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def process_station(event_dir: Path, net: str, sta: str, comp_files: Dict[str, Path]):
    missing = [c for c in REQUIRED_COMPONENTS if c not in comp_files]
    if missing:
        log_fail(f"[MISS-COMP] {event_dir.name} {net}.{sta} missing {missing}")
        return

    traces = {}
    for comp in REQUIRED_COMPONENTS:
        try:
            traces[comp] = read(str(comp_files[comp]))[0]
        except Exception as e:
            log_fail(f"[READ-FAIL] {comp_files[comp]} :: {e}")
            return

    # normalize SAC ref to origin
    for comp in REQUIRED_COMPONENTS:
        traces[comp] = normalize_trace_reference_to_origin(traces[comp])

    aligned_before, msg_before = check_alignment(traces)

    tr_ref = traces["HHZ"]

    origin = get_origin_time(tr_ref)
    if origin is None:
        log_fail(f"[NO-ORIGIN] {event_dir.name} {net}.{sta}")
        return

    dist_km = get_distance_km(tr_ref)
    if dist_km is None or dist_km <= 0:
        log_fail(f"[NO-DIST] {event_dir.name} {net}.{sta}")
        return

    p_theory = origin + dist_km / VP
    s_theory = origin + dist_km / VS
    surf_theory = origin + dist_km / VSURFACE

    # -------- station-level header picks from all components -------- #
    station_picks = collect_station_header_picks(traces)

    best_p = choose_best_station_pick(station_picks["P"])
    best_s = choose_best_station_pick(station_picks["S"])

    if best_p is not None:
        p_pick, p_label, p_comp = best_p
        p_source = "header"
    else:
        p_stalta = detect_p_with_stalta(traces["HHZ"], p_theory)
        if p_stalta is not None:
            p_pick = p_stalta
            p_label = "P"
            p_comp = "HHZ"
            p_source = "stalta"
        else:
            p_pick = p_theory
            p_label = "P"
            p_comp = "theory"
            p_source = "theory"

    if best_s is not None:
        s_pick, s_label, s_comp = best_s
        s_source = "header"
    else:
        s_pick = s_theory
        s_label = "S"
        s_comp = "theory"
        s_source = "theory"

    surf_pick = surf_theory
    surf_source = "theory"

    cut_start = p_pick - PRE_P
    cut_end = surf_pick + POST_SURF

    # trim and high-pass filter cut data
    cut_traces = {}
    for comp, tr in traces.items():
        trc = tr.copy()
        trc.trim(cut_start, cut_end, pad=False)

        if trc.stats.npts <= 1:
            log_fail(f"[EMPTY-CUT] {event_dir.name} {net}.{sta}.{comp}")
            return

        # high-pass the cut data
        trc = highpass_trace(trc, HIGHPASS_FREQ)

        sac = getattr(trc.stats, "sac", None)
        if sac is not None:
            sac.b = float(trc.stats.starttime - origin)
            sac.e = float(trc.stats.endtime - origin)
            sac.o = 0.0

        cut_traces[comp] = trc

    aligned_after, msg_after = check_alignment(cut_traces)

    # write SAC
    out_event_dir = OUTPUT_SAC_ROOT / event_dir.name
    out_event_dir.mkdir(parents=True, exist_ok=True)

    for comp, trc in cut_traces.items():
        outname = f"{event_dir.name}_{net}_{sta}_{comp}.SAC"
        write_sac(trc, out_event_dir / outname)

    # plot cut traces
    figpath = OUTPUT_FIG_ROOT / event_dir.name / f"{net}.{sta}.png"
    make_station_plot(
        traces=cut_traces,
        figpath=figpath,
        event_name=event_dir.name,
        net=net,
        sta=sta,
        dist_km=dist_km,
        p_pick=p_pick,
        s_pick=s_pick,
        surf_pick=surf_pick,
        p_source=p_source,
        s_source=s_source,
        surf_source=surf_source,
        p_label=p_label,
        s_label=s_label,
        p_comp=p_comp,
        s_comp=s_comp,
        cut_start=cut_start,
    )

    log_success(
        f"[OK] {event_dir.name} {net}.{sta} "
        f"dist_km={dist_km:.2f} "
        f"P={p_pick.isoformat()} ({p_label},{p_source},{p_comp}) "
        f"S={s_pick.isoformat()} ({s_label},{s_source},{s_comp}) "
        f"SURF={surf_pick.isoformat()} ({surf_source}) "
        f"HP={HIGHPASS_FREQ:.4f}Hz "
        f"cut=[{cut_start.isoformat()} -> {cut_end.isoformat()}]"
    )

    log_report(
        f"[ALIGN] {event_dir.name} {net}.{sta} "
        f"before_aligned={aligned_before} {msg_before} | "
        f"after_aligned={aligned_after} {msg_after}"
    )


def main():
    event_dirs = sorted([d for d in INPUT_ROOT.glob("*") if d.is_dir()])
    if not event_dirs:
        print(f"No event directories found in {INPUT_ROOT}")
        return

    for event_dir in event_dirs:
        groups = group_files_by_station(event_dir)
        if not groups:
            log_fail(f"[NO-SAC] {event_dir}")
            continue

        for (net, sta), comp_files in sorted(groups.items()):
            try:
                process_station(event_dir, net, sta, comp_files)
            except Exception as e:
                log_fail(f"[PROCESS-FAIL] {event_dir.name} {net}.{sta} :: {e}")

    print("Done.")
    print(f"SAC output : {OUTPUT_SAC_ROOT}")
    print(f"FIG output : {OUTPUT_FIG_ROOT}")
    print(f"Success log: {SUCCESS_LOG}")
    print(f"Fail log   : {FAIL_LOG}")
    print(f"Report log : {REPORT_LOG}")


if __name__ == "__main__":
    main()