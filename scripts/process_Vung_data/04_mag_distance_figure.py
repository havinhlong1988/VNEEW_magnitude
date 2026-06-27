#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import pygmt

# =========================================================
# USER PARAMETERS
# =========================================================
SUMMARY_CSV = Path("output/event_magnitude_summary.csv")
OUT_FIG     = Path("figures/event_dist_mag_joint_pygmt.png")

# Main panel axes
XCOL = "mean_header_hypdist_km"   # x-axis = distance
YCOL = "mean_header_mag"          # y-axis = magnitude

# ---- figure geometry (cm) ----
MAIN_W  = 12
MAIN_H  = 9
TOP_H   = 3.0
RIGHT_W = 3.2
GAP     = 0.25

# ---- scatter style ----
SYMBOL_SIZE = 0.15  # cm
POINT_FILL = "lightgray"
POINT_PEN  = "0.35p,navy"
POINT_TRANSPARENCY = 15

# ---- histogram style ----
HIST_FILL_TOP   = "goldenrod1"   # distance histogram (top)
HIST_FILL_RIGHT = "tomato2"      # magnitude histogram (right)
HIST_PEN        = "0.4p,black"

# ---- optional fit line ----
PLOT_FIT_LINE = True
FIT_PEN = "1.2p,red,-"

# ---- bin width ----
DIST_BIN_WIDTH = None
MAG_BIN_WIDTH  = 0.25


def nice_range(vmin, vmax, pad_frac=0.06, fallback=(0, 1)):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return list(fallback)
    if vmin == vmax:
        dv = max(abs(vmin) * 0.1, 1.0)
        return [vmin - dv, vmax + dv]
    dv = vmax - vmin
    pad = dv * pad_frac
    return [vmin - pad, vmax + pad]


def nice_step(vmin, vmax, target_n=5):
    span = abs(vmax - vmin)
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / target_n
    power = 10 ** np.floor(np.log10(raw))
    base = raw / power
    if base <= 1:
        step = 1 * power
    elif base <= 2:
        step = 2 * power
    elif base <= 5:
        step = 5 * power
    else:
        step = 10 * power
    return float(step)


def auto_bin_width(values, n_target=15):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    span = vmax - vmin
    if span <= 0:
        return 1.0
    bw = span / n_target
    if not np.isfinite(bw) or bw <= 0:
        bw = 1.0
    return bw


def safe_bin_width(bin_width, values):
    try:
        bw = float(bin_width)
    except (TypeError, ValueError):
        bw = np.nan
    if (not np.isfinite(bw)) or (bw <= 0):
        bw = auto_bin_width(values)
    return float(bw)


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    if XCOL not in df.columns:
        raise KeyError(f"Column not found in CSV: {XCOL}")
    if YCOL not in df.columns:
        raise KeyError(f"Column not found in CSV: {YCOL}")

    plot_df = df[[XCOL, YCOL]].copy()
    plot_df[XCOL] = pd.to_numeric(plot_df[XCOL], errors="coerce")
    plot_df[YCOL] = pd.to_numeric(plot_df[YCOL], errors="coerce")
    plot_df = plot_df.dropna(subset=[XCOL, YCOL])

    if len(plot_df) == 0:
        raise ValueError("No valid rows after dropping NaN values.")

    x = plot_df[XCOL].to_numpy(dtype=float)   # distance
    y = plot_df[YCOL].to_numpy(dtype=float)   # magnitude

    # Main panel ranges
    xr = nice_range(np.nanmin(x), np.nanmax(x), pad_frac=0.08)
    yr = nice_range(np.nanmin(y), np.nanmax(y), pad_frac=0.10)

    xstep = nice_step(xr[0], xr[1], target_n=6)
    ystep = nice_step(yr[0], yr[1], target_n=5)

    main_region = [xr[0], xr[1], yr[0], yr[1]]
    main_proj   = f"X{MAIN_W}c/{MAIN_H}c"

    # Bin widths
    dist_bw = safe_bin_width(DIST_BIN_WIDTH, x)
    mag_bw  = safe_bin_width(MAG_BIN_WIDTH, y)

    # Histogram maxima
    dist_edges = np.arange(
        np.floor(np.nanmin(x) / dist_bw) * dist_bw,
        np.ceil(np.nanmax(x) / dist_bw) * dist_bw + 1.5 * dist_bw,
        dist_bw,
    )
    mag_edges = np.arange(
        np.floor(np.nanmin(y) / mag_bw) * mag_bw,
        np.ceil(np.nanmax(y) / mag_bw) * mag_bw + 1.5 * mag_bw,
        mag_bw,
    )

    dist_counts, _ = np.histogram(x, bins=dist_edges)
    mag_counts, _ = np.histogram(y, bins=mag_edges)

    top_max   = max(1.0, float(np.nanmax(dist_counts)) * 1.15 if len(dist_counts) else 1.0)
    right_max = max(1.0, float(np.nanmax(mag_counts)) * 1.15 if len(mag_counts) else 1.0)

    # Top histogram follows main x-axis => distance
    top_region = [xr[0], xr[1], 0, top_max]
    top_proj   = f"X{MAIN_W}c/{TOP_H}c"

    # Right histogram follows main y-axis => magnitude
    right_region = [yr[0], yr[1], 0, right_max]
    right_proj   = f"X{RIGHT_W}c/{MAIN_H}c"

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="fancy",
        MAP_FRAME_PEN="1.0p,black",
        MAP_TICK_PEN_PRIMARY="0.8p,black",
        MAP_TICK_LENGTH_PRIMARY="0.10c",
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
    )

    # ---------------- Main scatter ----------------

    # fig.basemap(
    #     region=main_region,
    #     projection=main_proj,
    #     frame=[
    #         "WSen",
    #         f"xa{xstep:g}+l\"Hypocenter distance (km)\"",
    #         f"ya{ystep:g}+l\"Magnitude\"",
    #         f"g{xstep:g}",
    #         f"g{ystep:g}",
    #     ],
    # )

    fig.basemap(
        region=main_region,
        projection=main_proj,
        frame=[
            "WSen",
            f"xagf+l\"Hypocenter distance (km)\"",
            f"yagf+l\"Magnitude\"",
        ],
    )
    fig.plot(
        x=x,
        y=y,
        style=f"c{SYMBOL_SIZE}c",
        fill=POINT_FILL,
        pen=POINT_PEN,
        transparency=POINT_TRANSPARENCY,
    )

    # if PLOT_FIT_LINE and len(x) >= 2:
    #     coef = np.polyfit(x, y, 1)
    #     xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    #     yfit = coef[0] * xfit + coef[1]
    #     fig.plot(x=xfit, y=yfit, pen=FIT_PEN)

    fig.text(
        x=main_region[0] + 0.03 * (main_region[1] - main_region[0]),
        y=main_region[3] - 0.05 * (main_region[3] - main_region[2]),
        text=f"Total data: {len(plot_df)}",
        font="11p,Times-Bold,black",
        justify="TL",
        no_clip=True,
    )

    # ---------------- Top histogram: distance ----------------
    fig.shift_origin(yshift=f"{MAIN_H + GAP}c")

    fig.basemap(
        region=top_region,
        projection=top_proj,
        frame=[
            "WseN",
            f"xa{xstep:g}",
            "ya",
        ],
    )

    fig.histogram(
        data=x,
        region=top_region,
        projection=top_proj,
        series=dist_bw,
        histtype=0,
        fill=HIST_FILL_TOP,
        pen=HIST_PEN,
    )

    # ---------------- Right histogram: magnitude ----------------
    fig.shift_origin(xshift=f"{MAIN_W + GAP}c", yshift=f"{-(MAIN_H + GAP)}c")

    # fig.basemap(
    #     region=right_region,
    #     projection=right_proj,
    #     frame=[
    #         "wsen",
    #         "xa",
    #         f"ya{ystep:g}",
    #     ],
    # )

    fig.histogram(
        horizontal=True,
        projection=right_proj,
        data=y,
        region=right_region,
        histtype=0,
        frame=["wSrt", "xf", "yaf+lCounts"],
        fill=HIST_FILL_RIGHT,
        pen=HIST_PEN,
        series=mag_bw,   # force bins to follow main y-axis
    )

    fig.savefig(OUT_FIG, dpi=300)
    print(f"[OK] Saved figure: {OUT_FIG}")
    fig.show()


if __name__ == "__main__":
    main()