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
