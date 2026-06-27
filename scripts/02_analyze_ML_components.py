#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import pygmt


# ========================= USER PARAMETERS ========================= #

INPUT_CSV = Path("output/02_check_amp_ML_Hutton/event_station_ml.csv")

OUTPUT_DIR = Path("output/02_check_amp_ML_Hutton")
FIG_DIR = Path("figures/02_check_amp_ML_Hutton")

FIG_NAME = FIG_DIR / "03_ML_Hutton_Z_N_E_Havg.png"

COMP_MAP = {
    "Z": "HHZ",
    "N": "HHN",
    "E": "HHE",
}

# If True: average horizontal from N and E
MAKE_HAVG = True

# ================================================================== #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_df(df):
    rename_map = {}

    if "comp" in df.columns:
        rename_map["comp"] = "component"
    if "ML" in df.columns:
        rename_map["ML"] = "ML_cal"
    if "header_ML" in df.columns:
        rename_map["header_ML"] = "ML_header"
    if "ML_diff_calc_minus_ref" in df.columns:
        rename_map["ML_diff_calc_minus_ref"] = "ML_diff"

    df = df.rename(columns=rename_map)

    required = ["event", "sta", "component", "ML_cal", "ML_header"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    if "ML_diff" not in df.columns:
        df["ML_diff"] = df["ML_cal"] - df["ML_header"]

    df["ML_cal"] = pd.to_numeric(df["ML_cal"], errors="coerce")
    df["ML_header"] = pd.to_numeric(df["ML_header"], errors="coerce")
    df["ML_diff"] = pd.to_numeric(df["ML_diff"], errors="coerce")

    return df


def write_component_dat(df, label, component):
    sub = df[df["component"] == component].copy()
    sub = sub.dropna(subset=["ML_cal", "ML_header", "ML_diff"])

    out = OUTPUT_DIR / f"02_ML_Hutton_{label}.dat"

    sub_out = sub[["event", "sta", "ML_cal", "ML_header", "ML_diff"]]
    sub_out.to_csv(out, index=False, header=True)

    return sub_out, out


def write_havg_dat(df):
    h = df[df["component"].isin(["HHN", "HHE"])].copy()

    havg = (
        h.groupby(["event", "sta"], as_index=False)
        .agg(
            ML_cal=("ML_cal", "mean"),
            ML_header=("ML_header", "mean"),
            ML_diff=("ML_diff", "mean"),
        )
    )

    havg = havg.dropna(subset=["ML_cal", "ML_header", "ML_diff"])

    out = OUTPUT_DIR / "02_ML_Hutton_Havg.dat"
    havg[["event", "sta", "ML_cal", "ML_header", "ML_diff"]].to_csv(
        out, index=False, header=True
    )

    return havg, out


def plot_panel(fig, data, title, region, panel_idx):
    x = np.arange(len(data))
    y = data["ML_cal"].values
    yref = data["ML_header"].values
    diff = data["ML_diff"].values

    if panel_idx == 0:
        fig.basemap(
            region=region,
            projection="X7c/5c",
            frame=['xaf+l"Record index"', 'yaf+l"ML"', f'WSen+t"{title}"'],
        )
    else:
        fig.basemap(
            region=region,
            projection="X7c/5c",
            frame=['xaf+l"Record index"', 'yaf+l"ML"', f'WSen+t"{title}"'],
        )

    fig.plot(x=x, y=yref, style="c0.12c", fill="gray", pen="0.2p,black")
    fig.plot(x=x, y=y, style="c0.16c", fill=diff, cmap=True, pen="0.25p,black")

    for xi, yy, yr in zip(x, y, yref):
        if np.isfinite(yy) and np.isfinite(yr):
            fig.plot(x=[xi, xi], y=[yr, yy], pen="0.4p,black")


def main():
    df = pd.read_csv(INPUT_CSV)
    df = clean_df(df)

    outputs = {}

    for label, comp in COMP_MAP.items():
        sub, out = write_component_dat(df, label, comp)
        outputs[label] = sub
        print(f"[WRITE] {out}")

    if MAKE_HAVG:
        havg, out = write_havg_dat(df)
        outputs["Havg"] = havg
        print(f"[WRITE] {out}")

    all_plot = pd.concat(outputs.values(), ignore_index=True)

    y_min = float(np.nanmin([all_plot["ML_cal"].min(), all_plot["ML_header"].min()])) - 0.3
    y_max = float(np.nanmax([all_plot["ML_cal"].max(), all_plot["ML_header"].max()])) + 0.3

    diff_abs = float(np.nanmax(np.abs(all_plot["ML_diff"])))
    if diff_abs == 0 or not np.isfinite(diff_abs):
        diff_abs = 0.5

    pygmt.makecpt(cmap="polar", series=[-diff_abs, diff_abs])

    fig = pygmt.Figure()
    pygmt.config(
        MAP_FRAME_TYPE="fancy",
        MAP_FRAME_PEN="1.0p,black",
        MAP_TICK_PEN_PRIMARY="0.8p,black",
        MAP_TICK_LENGTH_PRIMARY="0.10c",
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
    )    

    panel_order = ["Z", "N", "E", "Havg"]
    panel_titles = {
        "Z": "HHZ",
        "N": "HHN",
        "E": "HHE",
        "Havg": "Average HHN-HHE",
    }

    for i, key in enumerate(panel_order):
        data = outputs[key].reset_index(drop=True)

        if data.empty:
            continue

        region = [-1, len(data), y_min, y_max]

        if i == 0:
            pass
        elif i == 1:
            fig.shift_origin(xshift="8.2c")
        elif i == 2:
            fig.shift_origin(xshift="-8.2c", yshift="-6.2c")
        elif i == 3:
            fig.shift_origin(xshift="8.2c")

        plot_panel(fig, data, panel_titles[key], region, i)

    fig.shift_origin(xshift="1.2c", yshift="-1.2c")
    fig.colorbar(
        position="JBC+w8c/0.25c+h",
        frame='xaf+l"ML calculated - ML header/reference"',
    )

    fig.savefig(FIG_NAME, dpi=300)

    print("DONE")
    print(f"Figure: {FIG_NAME}")


if __name__ == "__main__":
    main()