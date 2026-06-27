#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import pygmt

# =========================================================
# INPUT / OUTPUT
# =========================================================
INPUT_CSV = Path("output/03_ml_check_wa/event_station_ml.csv")
OUTFIG = Path("figures/03_ml_check_wa/00_ml_compare_use_wa.png")

# =========================================================
# USER SETTINGS
# =========================================================

# plot size for each panel
PANEL_W = "5i" 
PANEL_H = "5i"

title="Estimate from peak displacement (WA)"

# =========================================================
# READ DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)
df = df.replace([np.inf, -np.inf], np.nan)

if df.empty:
    raise ValueError("No valid rows after filtering positive distance/amplitude.")

# =========================================================
# PLOT
# =========================================================
OUTFIG.parent.mkdir(parents=True, exist_ok=True)

pygmt.config(
    FONT_LABEL="11p,Times-Bold,black",
    FONT_ANNOT_PRIMARY="10p,Times-Roman,black",
    FONT_TITLE="12p,Times-Bold,black",
    MAP_FRAME_PEN="0.8p,black",
    MAP_TICK_PEN_PRIMARY="0.8p,black",
)

fig = pygmt.Figure()

# -----------------------------------------------------
# LEFT PANEL: log distance vs log amplitude
# both x and y axes logarithmic
# -----------------------------------------------------
fig.basemap(
    region=[1e0,7e2,1e-0,1e1],
    # region=[xmin, xmax, yamp_min, yamp_max],
    projection=f"X{PANEL_W}l/{PANEL_H}l",
    frame=[
        f'WSen+t{title}',
        'pxa1g3f100p+l"log@-10@-(distance)"', 
        'pya0.25g3f0.25p+l"log@-10@-(amp@-WA@-)"',
    ],
)
fig.plot(
    x=df['distance'],
    y=df['LogA0Huton83']*-1,
    style="c0.25c",
    fill="red",
    pen="0.25p,black",
    transparency=20,
    label="Huton83 (nm)"
)
fig.plot(
    x=df['distance'],
    y=df['LogA0Le08']*-1,
    style="a0.25c",
    fill="blue",
    pen="0.25p,black",
    transparency=20,
    label="Le08 (nm))"
)

fig.plot(
    x=df['distance'],
    y=df['LogA0Nguyen11']*-1,
    style="s0.25c",
    fill="green",
    pen="0.25p,black",
    transparency=20,
    label="Nguyen11 (µm)"
)
fig.legend(
    position="JTL+jTL+w3c/1.5c+o0.2c/0.2c",
    # box="1p,black+gwhite"
    box=True
)
# ===========================================================
# Shift origin to plot the magnitude
# ===========================================================
fig.shift_origin(xshift="5.5i")
# -----------------------------------------------------
# RIGHT PANEL: log distance vs ML
# x logarithmic, y linear
# -----------------------------------------------------
fig.basemap(
    region=[1e0,7e2,-5,10], # [xmin, xmax, yml_min, yml_max],
    projection=f"X{PANEL_W}l/{PANEL_H}",
    frame=[
        f'wSEn+t{title}',
        'pxa1g3f100p+l"log@-10@-(distance)"', 
        'yagf1+l"Local magnitude (M@-L@-)"',
    ],
)
fig.plot(
    x=df['distance'],
    y=df['MLHuton83'],
    style="c0.25c",
    fill="red",
    pen="0.25p,black",
    transparency=20,
    label="Huton83 (nm)" 
)
fig.plot(
    x=df['distance'],
    y=df['MLLe08'],
    style="a0.25c",
    fill="blue",
    pen="0.25p,black",
    transparency=20,
    label="Le08 (nm)"
)

fig.plot(
    x=df['distance'],
    y=df['MLNguyen11'],
    style="s0.25c",
    fill="green",
    pen="0.25p,black",
    transparency=20,
    label="Nguyen11 (µm)"
)
fig.legend(
    position="JBL+jBL+o0.2c/0.2c",
    box="+gwhite"
    # box=True
)
fig.savefig(str(OUTFIG))
print(f"[OK] Saved figure: {OUTFIG}")
fig.show()