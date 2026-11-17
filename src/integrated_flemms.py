#!/usr/bin/env python3
"""
FLEMMS 2008 / 2013 / 2019 / 2024 Consolidated Longitudinal Integration
Merges FLEMMS_2008_2013_integrated.xlsx and FLEMMS_2019_2024_Integrated.xlsx
(built from the 4 harmonized microdata PUFs) into FLEMMS_2008-2024_integrated.xlsx.

Usage:
  python3 2008_2024_integrated_flemms.py --data-dir .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUT_NAME = "FLEMMS_2008-2024_integrated.xlsx"
KEY = ["indicator", "geography_code", "area", "age_group", "sex"]
YEARS = [2008, 2013, 2019, 2024]
BARMM = "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)"
IND_MAP = {
    "basic_5plus": "BASIC_RW",
    "basic_10_64": "BASIC_RW_10_64",
    "functional_10_64": "FUNC_RWCC",
}
AGE_BASE = {
    "basic_5plus": "5+",
    "basic_10_64": "10-64",
    "functional_10_64": "10-64",
}


def find_file(folder: Path, tag: str) -> Path:
    """Locate the single .xlsx workbook in `folder` whose name contains `tag` (case-insensitive)."""
    hits = [f for f in folder.glob("*.xlsx") if tag in f.name.lower() and f.name != OUT_NAME]
    if len(hits) != 1:
        sys.exit(f"Expected exactly one .xlsx containing '{tag}' in {folder}, found {[h.name for h in hits]}")
    return hits[0]


def consolidate_flemms(in_dir: Path, out_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the 4-wave consolidation pipeline and export the formatted Excel deliverable."""
    f_0813 = find_file(in_dir, "2008_2013")
    f_1924 = find_file(in_dir, "2019_2024")

    counts: list[dict] = []
    checks: list[dict] = []

    def load(file_path: Path, sheet: str) -> pd.DataFrame:
        df = pd.read_excel(file_path, sheet_name=sheet)
        counts.append(dict(stage="read", source_file=file_path.name, sheet=sheet, rows=len(df)))
        return df


    def check(name: str, expected: float, actual: float, note: str = "", tol: float = 1e-6) -> None:
        passed = abs(float(expected) - float(actual)) <= tol
        checks.append(
            dict(check=name, expected=expected, actual=actual, result="PASS" if passed else "FAIL", note=note)
        )

    a = load(f_0813, "merged")
    rc = load(f_0813, "row_counts").set_index("item").rows
    cw = load(f_0813, "region_crosswalk")
    b = load(f_1924, "Merged_2019_2024")
    st = load(f_1924, "Stacked_Long")
    ck = load(f_1924, "Checks").set_index("check").expected

    region_code = dict(zip(cw.label_2008, cw.code)) | {BARMM: "ARMM"}
    names = dict(zip(cw.code, cw.name)) | {"ARMM": "ARMM / BARMM"}
    cw["label_2019_2024"] = cw.code.map({v: k for k, v in region_code.items()})

    labels = a.drop_duplicates("indicator").set_index("indicator").indicator_label.to_dict()
    labels["BASIC_RW_10_64"] = "Basic literacy, ages 10-64"

    lvl = b.geo_level
    keys_b = pd.DataFrame(
        {
            "indicator": b.indicator.map(IND_MAP),
            "geography_code": np.select([lvl == "Region"], [b.area.map(region_code)], "PH"),
            "area": "All",
            "age_group": np.where(lvl == "Age group", b.area.replace({"60 and over": "60+"}), "All"),
            "sex": b.sex.replace({"Both Sexes": "Both"}),
        }
    )
    check("2019/2024 indicator codes unmapped", 0, int(keys_b.indicator.isna().sum()))
    check("2019/2024 region labels unmapped (incl. NIR)", 0, int(keys_b.geography_code.isna().sum()))
    check(
        "Indicator codes shared across files: before vs after harmonising",
        0,
        len(set(a.indicator) & set(b.indicator)),
        f"after harmonising: {len(set(a.indicator) & set(keys_b.indicator))} shared",
        tol=1e9,
    )

    tables = st.groupby(["year", "indicator"], observed=False).source_table.agg(lambda s: "; ".join(sorted(set(s)))).to_dict()

    long_parts = []
    for y in (2008, 2013):
        d = a[a.match_status.isin(["both", f"{y}_only"])]
        long_parts.append(
            d[KEY].assign(
                year=y,
                population=d[f"population_{y}"],
                count=d[f"count_{y}"],
                rate=d[f"rate_{y}"],
                age_base=d[f"age_base_{y}"],
                source_table=d[f"source_{y}"],
                source_file=f_0813.name,
                source_sheet="merged",
            )
        )
    for y in (2019, 2024):
        valid_mask = b[f"pop_{y}"].notna()
        long_parts.append(
            keys_b.loc[valid_mask].assign(
                year=y,
                population=b.loc[valid_mask, f"pop_{y}"] * 1000.0,
                count=b.loc[valid_mask, f"num_{y}"] * 1000.0,
                rate=b.loc[valid_mask, f"rate_{y}"],
                age_base=b.loc[valid_mask, "indicator"].map(AGE_BASE),
                source_table=b.loc[valid_mask, "indicator"].map(lambda i: tables[(y, i)]),
                source_file=f_1924.name,
                source_sheet="Merged_2019_2024",
            )
        )

    long = pd.concat(long_parts, ignore_index=True)
    long["geography_name"] = long.geography_code.map(names)
    long["indicator_label"] = long.indicator.map(labels)
    for y, g in long.groupby("year", observed=False):
        counts.append(
            dict(stage="tidy rows", source_file=g.source_file.iloc[0], sheet=g.source_sheet.iloc[0] + f" ({y})", rows=len(g))
        )

    vals = long.pivot(index=KEY, columns="year", values=["population", "count", "rate", "age_base", "source_table"])
    vals.columns = [f"{v}_{y}" for v, y in vals.columns]
