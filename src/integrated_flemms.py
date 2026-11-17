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

from matplotlib.pyplot import rc
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
 info = long.groupby(KEY, observed=False).agg(
        years_present=("year", lambda s: ",".join(map(str, sorted(s)))),
        n_years=("year", "size"),
        source_files=("source_file", lambda s: "; ".join(sorted(set(s)))),
    )
    w = vals.join(info).reset_index()
    w = w.merge(
        a[KEY + ["unmatched_reason"]].rename(columns={"unmatched_reason": "reason_in_2008_2013_file"}),
        on=KEY,
        how="left",
    )
    w["geography_name"] = w.geography_code.map(names)
    w["indicator_label"] = w.indicator.map(labels)

    early = w.years_present.str.contains("2008|2013")
    late = w.years_present.str.contains("2019|2024")
    w["match_status"] = np.select(
        [w.n_years == 4, early & late, early],
        ["all_4_years", "cross_period_partial", "2008_2013_only"],
        "2019_2024_only",
    )
    w["comparability"] = np.select(
        [(w.indicator == "BASIC_RW") & (w.age_group == "All"), w.indicator == "FUNC_RWCC"],
        [
            "CAUTION: age base differs by year (see age_base_*)",
            "INDICATIVE: 2008-2019 use Levels 3-4 comprehension threshold; 2024 uses revised PSA definition",
        ],
        "COMPARABLE",
    )

    def reason(r: pd.Series) -> str:
        if r.n_years == 4:
            return ""
        if r.area != "All":
            return "Urban/rural breakdown published in 2013 summary slice only"
        if r.indicator == "BASIC_RW_10_64":
            return "Basic literacy 10-64 slice generated in 2019/2024 table only"
        if r.age_group in ("6-9", "5-9"):
            return "Age band differs: 6-9 in 2008 vs 5-9 in 2019/2024"
        src = r.reason_in_2008_2013_file if isinstance(r.reason_in_2008_2013_file, str) else ""
        if r.indicator == "FUNC_RWC" or r.indicator.startswith("LEVEL_"):
            return "Indicator is in 2008/2013 file only" + (f"; {src}" if src else "")
        return src or f"Missing in year(s): {', '.join(str(y) for y in YEARS if str(y) not in r.years_present)}"

    w["unmatched_reason"] = w.apply(reason, axis=1)

    order = {
        "indicator": ["BASIC_RW", "BASIC_RW_10_64", "FUNC_RWC", "FUNC_RWCC", "LEVEL_ILLITERATE", "LEVEL_LOW", "LEVEL_UNDET"],
        "geography_code": list(cw.code),
        "area": ["All", "Urban", "Rural"],
        "sex": ["Both", "Male", "Female"],
        "age_group": ["All", "5-9", "6-9", "10-14", "15-19", "20-24", "25-29", "30-39", "40-49", "50-59", "60-64", "60+"],
    }
    for c, cats in order.items():
        w[c] = pd.Categorical(w[c], cats, ordered=True)
    check("Key values outside expected lists", 0, int(w[KEY].isna().sum().sum()))
    w = w.sort_values(KEY).reset_index(drop=True)
    w[KEY] = w[KEY].astype(str)

    for p in ["2008_2013", "2013_2019", "2019_2024"]:
        w[f"change_pp_{p}"] = None
    meta = ["years_present", "n_years", "match_status", "comparability", "unmatched_reason", "source_files"]
    wide_cols = (
        KEY
        + ["geography_name", "indicator_label"]
        + [f"{v}_{y}" for y in YEARS for v in ("population", "count", "rate")]
        + [c for c in w if c.startswith("change_pp")]
        + [f"{v}_{y}" for y in YEARS for v in ("age_base", "source_table")]
        + meta
    )
    w = w[wide_cols]

    unm = w[w.match_status != "all_4_years"]
    summary = unm.groupby(["match_status", "indicator", "unmatched_reason"], observed=False).size().reset_index(name="rows")

    exp_0813 = rc["merged total rows"]
    exp_1924 = ck["Merged rows = |2019 keys ∪ 2024 keys|"]
    check("Rows read: 2008/2013 'merged' vs its own row_counts sheet", exp_0813, len(a), f_0813.name)
    check("Rows read: 2019/2024 'Merged_2019_2024' vs its own Checks sheet", exp_1924, len(b), f_1924.name)
    check("Long rows 2008", rc["2008 rows after cleaning (duplicates removed)"], int((long.year == 2008).sum()))
    check("Long rows 2013", rc["2013 rows after age harmonisation"], int((long.year == 2013).sum()))
    check("Long rows 2019", int(b["pop_2019"].notna().sum()), int((long.year == 2019).sum()))
    check("Long rows 2024", int(b["pop_2024"].notna().sum()), int((long.year == 2024).sum()))
    check("Duplicate keys within a year", 0, int(long.duplicated(KEY + ["year"]).sum()))

    keys_a = set(a[KEY].itertuples(index=False, name=None))
    keys_b2 = set(keys_b[KEY].itertuples(index=False, name=None))
    check(
        "Consolidated rows = |2008/13 keys| + |2019/24 keys| - |overlap|",
        len(keys_a) + len(keys_b2) - len(keys_a & keys_b2),
        len(w),
        f"{len(keys_a)} + {len(keys_b2)} - {len(keys_a & keys_b2)} overlapping keys",
    )
    check("Sum of n_years = long rows", len(long), int(w.n_years.sum()))
    check("Rows with a reason = rows not matched in all 4 years", len(unm), int((w.unmatched_reason != "").sum()))
    check("Max |rate - count/population*100| (pp)", 0, float((long["rate"] - long["count"] / long.population * 100).abs().max()), tol=0.01)
    for y in YEARS:
        check(f"Non-blank rate_{y} in consolidated = long rows {y}", int((long.year == y).sum()), int(w[f"rate_{y}"].notna().sum()))

    counts += [
        dict(stage="stacked long", source_file=out_path.name, sheet="stacked_long", rows=len(long)),
        dict(stage="consolidated merged", source_file=out_path.name, sheet="consolidated_merged", rows=len(w)),
        dict(stage="unmatched records", source_file=out_path.name, sheet="unmatched_records", rows=len(unm)),
    ]

    readme = [
        "FLEMMS 2008 / 2013 / 2019 / 2024 - Consolidated Microdata Dataset (built by 2008_2024_integrated_flemms.py)",
        f"Sources: {f_0813.name} (years 2008 + 2013) and {f_1924.name} (years 2019 + 2024), derived from the 4 clean FLEMMS PUFs.",
        "Harmonised: indicator codes (basic_5plus -> BASIC_RW, functional_10_64 -> FUNC_RWCC, basic_10_64 -> BASIC_RW_10_64), "
        "all 18 regions including reconstructed Negros Island Region (NIR) across all 4 waves, 'Both Sexes' -> 'Both', '60 and over' -> '60+', persons units.",
        "consolidated_merged: one row per indicator x geography x area x age_group x sex across all 4 years, with live Excel change_pp_* formulas.",
        "stacked_long: tidy panel with one row per record per year.",
        "unmatched_records / unmatched_summary: rows not present in all 4 years with explicit classification reasons.",
        "Caveats: ARMM (2008/2013) and BARMM (2019/2024) share one code but boundaries differ; Region VIII was excluded by PSA in 2013; "
        "PSA revised functional literacy in 2024 (Board Resolution No. 13, s. 2024).",
    ]
    sheets = {
        "README": pd.DataFrame({"FLEMMS consolidation": readme}),
        "consolidated_merged": w,
        "stacked_long": long[
            ["year", "source_file", "source_sheet", "source_table"]
            + KEY
            + ["geography_name", "indicator_label", "population", "count", "rate", "age_base"]
        ],
        "unmatched_records": unm.drop(columns=[c for c in unm if c.startswith("change_pp")]),
        "unmatched_summary": summary,
        "row_counts": pd.DataFrame(counts),
        "checks": pd.DataFrame(checks),
        "region_crosswalk": cw[["code", "name", "label_2008", "label_2013", "label_2019_2024", "note"]],
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    wb = load_workbook(out_path)
    for ws in wb:
        head = {c.value: c.column_letter for c in ws[1]}
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=10, bold=c.row == 1, color="FFFFFF" if c.row == 1 else "000000")
                if c.row == 1:
                    c.fill = PatternFill("solid", fgColor="1F3864")
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 45)
        ws.freeze_panes = "A2"
        for name, letter in head.items():
            fmt = "#,##0" if name.startswith(("population", "count")) else "0.00" if name.startswith(("rate", "change_pp")) else None
            if fmt:
                for c in ws[letter][1:]:
                    c.number_format = fmt

    ws = wb["README"]
    ws.column_dimensions["A"].width = 140
    for c in ws["A"]:
        c.alignment = Alignment(wrap_text=True, vertical="top")

    ws = wb["consolidated_merged"]
    col = {c.value: c.column_letter for c in ws[1]}
    for name, (old, new) in {
        "change_pp_2008_2013": (2008, 2013),
        "change_pp_2013_2019": (2013, 2019),
        "change_pp_2019_2024": (2019, 2024),
    }.items():
        o, n = col[f"rate_{old}"], col[f"rate_{new}"]
        for r in range(2, ws.max_row + 1):
            ws[f"{col[name]}{r}"] = f'=IF(AND(ISNUMBER({o}{r}),ISNUMBER({n}{r})),{n}{r}-{o}{r},"")'
    wb.save(out_path)

    checks_df = pd.DataFrame(checks)
    failed = checks_df[checks_df["result"] == "FAIL"]["check"].tolist()
    print(f"Saved {out_path.name}: {len(w)} consolidated rows, {len(long)} long rows, {len(unm)} unmatched, {len(failed)} failed checks")
    for f in failed:
        print("FAILED:", f)
    return w, long, checks_df

