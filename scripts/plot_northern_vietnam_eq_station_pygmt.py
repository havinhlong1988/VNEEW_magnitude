#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import os
import numpy as np
import pandas as pd
import pygmt

# =========================
# User settings
# =========================
PWD = os.getcwd()

REPORT_FILES = [
    os.path.join(PWD, "report.out"),
]

OUTPUT_FIG_DIR = os.path.join(PWD, "figures")
OUTPUT_FIG = os.path.join(OUTPUT_FIG_DIR, "Fig01_report_earthquake_distribution.png")
OUTPUT_CSV = os.path.join(OUTPUT_FIG_DIR, "report_catalog_parsed.csv")

REGION = [100., 110., 17., 25.]
N
RELIEF_RES = "15s"

MAG_SCALE = 0.15
MAG_EXP_FACT = 1.2

FAULT_LV1 = os.path.join(PWD, "input", "faults_VN", "fault_lv1.txt")
FAULT_LV2 = os.path.join(PWD, "input", "faults_VN", "fault_lv2_use.txt")

# Optional station file
PLOT_STATION = False
STATION_FILE = os.path.join(PWD, "input", "station.txt")


# =========================
# Functions
# =========================
def parse_date_tokens(tokens):
    """
    Date format can be:
      410  -> Apr 10
      511  -> May 11
      5 1  -> May 01
    """
    if len(tokens) == 1:
        s = str(tokens[0]).strip()

        if len(s) == 4:
            mm = int(s[:2])
            dd = int(s[2:])
        elif len(s) == 3:
            mm = int(s[0])
            dd = int(s[1:])
        elif len(s) == 2:
            mm = int(s[0])
            dd = int(s[1])
        else:
            raise ValueError(f"Cannot parse Date = {s}")

    elif len(tokens) == 2:
        mm = int(tokens[0])
        dd = int(tokens[1])

    else:
        raise ValueError(f"Cannot parse Date tokens = {tokens}")

    return mm, dd


def parse_hrmm(hrmm):
    s = str(hrmm).strip().zfill(4)
    hh = int(s[:2])
    minute = int(s[2:])
    return hh, minute


def read_report_file(report_file):
    rows = []

    with open(report_file, "r") as f:
        header = f.readline().split()

        for iline, line in enumerate(f, start=2):
            if not line.strip():
                continue

            p = line.split()

            try:
                if len(p) == 11:
                    year = int(p[0])
                    mm, dd = parse_date_tokens([p[1]])
                    hrmm = p[2]
                    sec = float(p[3])
                    lat = float(p[4])
                    lon = float(p[5])
                    dep = float(p[6])
                    nst = int(p[7])
                    rms = float(p[8])
                    gap = float(p[9])
                    ml = float(p[10])

                elif len(p) == 12:
                    year = int(p[0])
                    mm, dd = parse_date_tokens([p[1], p[2]])
                    hrmm = p[3]
                    sec = float(p[4])
                    lat = float(p[5])
                    lon = float(p[6])
                    dep = float(p[7])
                    nst = int(p[8])
                    rms = float(p[9])
                    gap = float(p[10])
                    ml = float(p[11])

                else:
                    print(f"[SKIP] line {iline}: unexpected columns = {len(p)}")
                    continue

                hh, minute = parse_hrmm(hrmm)

                sec_int = int(sec)
                micro = int(round((sec - sec_int) * 1_000_000))

                evtime = datetime(year, mm, dd, hh, minute, sec_int, micro)

                rows.append({
                    "time": evtime,
                    "year": year,
                    "month": mm,
                    "day": dd,
                    "hrmm": str(hrmm).zfill(4),
                    "sec": sec,
                    "latitude": lat,
                    "longitude": lon,
                    "depth": dep,
                    "nst": nst,
                    "rms": rms,
                    "gap": gap,
                    "magnitude": ml,
                    "source_file": str(report_file),
                })

            except Exception as e:
                print(f"[SKIP] line {iline}: {e}")
                print("       ", line.strip())

    return pd.DataFrame(rows)


# =========================
# Read report catalog
# =========================
dfs = []

for rf in REPORT_FILES:
    if not os.path.exists(rf):
        print(f"[SKIP] report file not found: {rf}")
        continue

    df0 = read_report_file(rf)
    dfs.append(df0)

if len(dfs) == 0:
    raise RuntimeError("No report file was read.")

cat = pd.concat(dfs, ignore_index=True)

for col in ["latitude", "longitude", "depth", "magnitude"]:
    cat[col] = pd.to_numeric(cat[col], errors="coerce")

cat = cat.dropna(subset=["latitude", "longitude", "depth", "magnitude"]).copy()

cat = cat[
    cat["latitude"].between(-90, 90)
    & cat["longitude"].between(-180, 180)
    & cat["depth"].between(-5, 700)
    & cat["magnitude"].between(0, 10)
].copy()

cat["size_cm"] = MAG_SCALE * MAG_EXP_FACT ** cat["magnitude"]

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
cat.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] Parsed events: {len(cat)}")
print(f"[OK] Saved CSV: {OUTPUT_CSV}")


# =========================
# Optional station file
# =========================
sta = None

if PLOT_STATION and os.path.exists(STATION_FILE):
    sta = pd.read_csv(STATION_FILE, sep=r"\s+|\t+", engine="python")

    for col in ["latitude", "longitude", "elevation"]:
        if col in sta.columns:
            sta[col] = sta[col].astype(str).str.replace(",", ".", regex=False)
            sta[col] = pd.to_numeric(sta[col], errors="coerce")

    sta = sta.dropna(subset=["latitude", "longitude"]).copy()


# =========================
# Load topography
# =========================
relief = pygmt.datasets.load_earth_relief(
    resolution=RELIEF_RES,
    region=REGION,
)


# =========================
# Plot
# =========================
pygmt.config(
    FONT="15p,Times-Bold,black",
    FONT_TITLE="15p,Times-Bold,black",
    FONT_LABEL="15p,Times-Bold,black",
    FONT_ANNOT_PRIMARY="15p,Times-Bold,black",
    FONT_ANNOT_SECONDARY="15p,Times-Bold,black",
    MAP_FRAME_TYPE="fancy",
    FORMAT_GEO_MAP="dddF",
)

fig = pygmt.Figure()

# Topography
pygmt.makecpt(cmap="gebco", series=[-500, 1000, 10], continuous=True)

fig.grdimage(
    grid=relief,
    region=REGION,
    projection="M16c",
    shading=True,
    frame=["WSNe", "xaf1f0.5", "yaf1f0.5"],
)

fig.coast(
    region=REGION,
    projection="M16c",
    shorelines="0.8p,black",
    borders=["1/0.8p,black", "2/0.5p,gray40"],
    water="lightblue",
)

# Faults
if os.path.exists(FAULT_LV1):
    fig.plot(data=FAULT_LV1, pen="1.0p,red")

if os.path.exists(FAULT_LV2):
    fig.plot(data=FAULT_LV2, pen="1.0p,red")

# Topography colorbar
fig.colorbar(
    position="JBR+jBR+w4c/0.25c+h+o0.5c/0.9c",
    frame=['xaf500+l"Topography (m)"'],
    box="+gwhite+p0.5p",
)

# Earthquake depth CPT
depth_min = float(np.floor(cat["depth"].min()))
depth_max = float(np.ceil(cat["depth"].max()))

if depth_min == depth_max:
    depth_max = depth_min + 1.0

# same style: fixed 0–30 km depth scale
pygmt.makecpt(
    cmap="jet",
    series=[0, 30, 2],
    continuous=True,
    reverse=True,
)

fig.plot(
    x=cat["longitude"].values,
    y=cat["latitude"].values,
    style="cc",
    size=cat["size_cm"].values,
    fill=cat["depth"].values,
    cmap=True,
    pen="0.25p,black",
)

# Depth colorbar
fig.colorbar(
    cmap=True,
    position="JMR+w8c/0.35c+v+o1.0c/0c+r",
    frame=['xaf5+l"Depth (km)"'],
)

# Magnitude legend
for mag in [2.5, 3.0, 3.5, 4.0]:
    size = MAG_SCALE * MAG_EXP_FACT ** mag
    fig.plot(
        x=[-100],
        y=[-100],
        style=f"c{size}c",
        pen="0.25p,black",
        fill="white",
        label=f"M = {mag:.1f}",
    )

# Optional stations
if sta is not None:
    fig.plot(
        x=sta["longitude"],
        y=sta["latitude"],
        style="t0.35c",
        fill="blue",
        pen="0.3p,black",
        label="Station",
    )

fig.legend(
    position="JBL+jBL+o0.2c/0.2c",
    box="+gwhite+p0.5p",
)

# Scale bar
fig.basemap(
    map_scale="jBC+JBC+o0.0c/0.75c+c30+w200k+u"
)

# =========================
# Inset map
# =========================
with fig.inset(
    position="jTR+w3c+o0.2c",
    box="+p0.5p,black+gwhite",
):
    inset_region = [98, 118, 6, 26]

    fig.coast(
        region=inset_region,
        projection="M3c",
        land="lightgray",
        water="skyblue",
        shorelines="0.5p,black",
        borders="1/0.5p,black",
        frame="af",
        dcw="VN+ggold+p0.5p,black",
    )

    west, east, south, north = REGION
    fig.plot(
        x=[west, east, east, west, west],
        y=[south, south, north, north, south],
        pen="1p,red",
    )

    fig.text(
        x=[114],
        y=[14.25],
        text="EAST",
        font="6p,Times-Bold,black",
        justify="CM",
    )

    fig.text(
        x=[114],
        y=[13],
        text="VIETNAM",
        font="6p,Times-Bold,black",
        justify="CM",
    )

    fig.text(
        x=[114],
        y=[11.75],
        text="SEA",
        font="6p,Times-Bold,black",
        justify="CM",
    )

    fig.text(
        x=[110.5],
        y=[16.5],
        text="Hoang Sa",
        font="6p,Times-Roman,gold",
        justify="LM",
    )

    fig.text(
        x=[111.5],
        y=[9.5],
        text="Truong Sa",
        font="6p,Times-Roman,gold",
        justify="LM",
    )

    fig.plot(
        x=[105.8342],
        y=[21.0278],
        style="a0.15c",
        fill="red",
        pen="0.3p,black",
    )

fig.savefig(OUTPUT_FIG, dpi=300)
print(f"[OK] Saved figure: {OUTPUT_FIG}")