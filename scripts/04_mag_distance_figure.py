#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import pygmt

# =========================================================
# USER PARAMETERS
# =========================================================
SUMMARY_CSV = Path("output/03_ml_check_wa/event_station_ml.csv")

# component filter
# "HHZ"                  -> only HHZ
# ["HHE", "HHN", "HHZ"]  -> only these components
COMPONENT_SELECT = ["HHZ"]
COMPONENT_COL = "component"

# output figure name
if COMPONENT_SELECT is None:
    comp_tag = "all"
elif isinstance(COMPONENT_SELECT, str):
    comp_tag = COMPONENT_SELECT.strip().upper()
else:
    comp_tag = "_".join([str(c).strip().upper() for c in COMPONENT_SELECT])

OUT_FIG = Path(f"figures/03_event_magnitude_summary_WA_{comp_tag}.png")

# Main panel axes
XCOL = "distance"
YCOL = "magnitude"

# ---- filter range ----
X_RANGE = (0, 500)
Y_RANGE = (1, 6)

# ---- figure geometry (cm) ----
MAIN_W  = 12
MAIN_H  = 9
TOP_H   = 3.0
RIGHT_W = 3.2
GAP     = 0.4

# ---- scatter style ----
SYMBOL_SIZE = 0.15
POINT_FILL = "lightgray"
POINT_PEN  = "0.35p,navy"
POINT_TRANSPARENCY = 15

# ---- histogram style ----
HIST_FILL_TOP   = "goldenrod1"
HIST_FILL_RIGHT = "tomato2"
HIST_PEN        = "0.4p,black"

# ---- bin width ----
DIST_BIN_WIDTH = 25
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
    if COMPONENT_COL not in df.columns:
        raise KeyError(f"Column not found in CSV: {COMPONENT_COL}")

    plot_df = df[[XCOL, YCOL, COMPONENT_COL]].copy()

    plot_df[COMPONENT_COL] = (
        plot_df[COMPONENT_COL]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # correct filter for both string and list
    if COMPONENT_SELECT is not None:
        if isinstance(COMPONENT_SELECT, str):
            selected_components = [COMPONENT_SELECT.strip().upper()]
        else:
            selected_components = [str(c).strip().upper() for c in COMPONENT_SELECT]

        plot_df = plot_df[plot_df[COMPONENT_COL].isin(selected_components)]

    plot_df[XCOL] = pd.to_numeric(plot_df[XCOL], errors="coerce")
    plot_df[YCOL] = pd.to_numeric(plot_df[YCOL], errors="coerce")
    plot_df = plot_df.dropna(subset=[XCOL, YCOL])

    # ---------------- apply range filters ----------------
    if X_RANGE is not None:
        xmin, xmax = X_RANGE
        if xmin is not None:
            plot_df = plot_df[plot_df[XCOL] >= xmin]
        if xmax is not None:
            plot_df = plot_df[plot_df[XCOL] <= xmax]

    if Y_RANGE is not None:
        ymin, ymax = Y_RANGE
        if ymin is not None:
            plot_df = plot_df[plot_df[YCOL] >= ymin]
        if ymax is not None:
            plot_df = plot_df[plot_df[YCOL] <= ymax]

    if len(plot_df) == 0:
        raise ValueError(f"No valid rows for component={COMPONENT_SELECT} after filtering.")

    x = plot_df[XCOL].to_numpy(dtype=float)
    y = plot_df[YCOL].to_numpy(dtype=float)

    xr_auto = nice_range(np.nanmin(x), np.nanmax(x), pad_frac=0.08)
    yr_auto = nice_range(np.nanmin(y), np.nanmax(y), pad_frac=0.10)

    xr = list(X_RANGE) if X_RANGE is not None else xr_auto
    yr = list(Y_RANGE) if Y_RANGE is not None else yr_auto

    xstep = nice_step(xr[0], xr[1], target_n=6)
    ystep = nice_step(yr[0], yr[1], target_n=5)

    main_region = [xr[0], xr[1], yr[0], yr[1]]
    main_proj   = f"X{MAIN_W}c/{MAIN_H}c"

    dist_bw = safe_bin_width(DIST_BIN_WIDTH, x)
    mag_bw  = safe_bin_width(MAG_BIN_WIDTH, y)

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

    top_region = [xr[0], xr[1], 0, top_max]
    top_proj   = f"X{MAIN_W}c/{TOP_H}c"

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
    fig.basemap(
        region=main_region,
        projection=main_proj,
        frame=[
            "WSen",
            'xagf+l"Hypocenter distance (km)"',
            'yagf+l"Magnitude"',
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

    fig.text(
        x=main_region[0] + 0.03 * (main_region[1] - main_region[0]),
        y=main_region[3] - 0.05 * (main_region[3] - main_region[2]),
        text=f"Total data: {len(plot_df)} | component: {comp_tag}",
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

    fig.histogram(
        horizontal=True,
        projection=right_proj,
        data=y,
        region=right_region,
        histtype=0,
        frame=["wSrt", "xf", "yaf+lCounts"],
        fill=HIST_FILL_RIGHT,
        pen=HIST_PEN,
        series=mag_bw,
    )

    fig.savefig(OUT_FIG, dpi=300)
    print(f"[OK] Saved figure: {OUT_FIG}")
    fig.show()


if __name__ == "__main__":
    main()