from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure structured console logging for audit traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("flemms_validator")

# Canonical 22-column harmonized schema shared across all 4 FLEMMS survey waves
EXPECTED_COLUMNS: list[str] = [
    "survey_year", "reg_orig", "reg_harmonized", "reg_name", "is_nir",
    "prov_code", "nir_prov_name", "urbanity", "psu", "hh_id", "line_no",
    "sex", "age", "highest_grade", "attending_school", "employed",
    "basic_literate", "func_literacy_level", "func_literate",
    "hh_electricity", "pop_weight", "literacy_weight",
]

# PSGC province codes belonging to Negros Island Region (Region 18) across waves
NIR_PROVINCE_CODES: set[int] = {45, 302, 46, 61}

# Standardized 18 harmonized Philippine region codes (excluding unused code 15)
VALID_HARMONIZED_REGIONS: set[int] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19
}

# Mapping of FLEMMS survey wave years to expected cleaned CSV filenames
WAVE_FILES: dict[int, str] = {
    2008: "flemms_2008_clean.csv",
    2013: "flemms_2013_clean.csv",
    2019: "flemms_2019_clean.csv",
    2024: "flemms_2024_clean.csv",
}


class FLEMMSValidationError(Exception):
    """Base exception for FLEMMS dataset validation failures."""


class SchemaValidationError(FLEMMSValidationError):
    """Raised when required columns are missing, extra, or misordered."""


class IntegrityValidationError(FLEMMSValidationError):
    """Raised when datasets are empty, contain duplicate keys, or wrong survey years."""


class DomainValidationError(FLEMMSValidationError):
    """Raised when variables contain out-of-range or invalid category codes."""


class NIRHarmonizationError(FLEMMSValidationError):
    """Raised when Negros Island Region (Region 18) province mapping is inconsistent."""

def validate_schema(df: pd.DataFrame, expected_cols: list[str] = EXPECTED_COLUMNS) -> None:
    """Verify exact column names and order match the harmonized 22-column schema."""
    if df.empty:
        raise IntegrityValidationError("Dataset is empty (0 rows).")
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        missing = sorted(set(expected_cols) - set(actual_cols))
        extra = sorted(set(actual_cols) - set(expected_cols))
        raise SchemaValidationError(
            f"Schema mismatch. Missing={missing}, Extra={extra}, OrderMatch={actual_cols == expected_cols}"
        )


def validate_primary_keys(df: pd.DataFrame, expected_year: int) -> None:
    """Verify survey_year uniformity and uniqueness of composite member key (hh_id, line_no)."""
    years = df["survey_year"].dropna().unique().tolist()
    if years != [expected_year]:
        raise IntegrityValidationError(f"Expected survey_year=[{expected_year}], found {years}.")
    if df["hh_id"].isna().any() or df["line_no"].isna().any():
        raise IntegrityValidationError("Null values found in primary key columns (hh_id, line_no).")

    # Composite key (hh_id, line_no) must uniquely identify every household respondent
    dup_count = int(df.duplicated(subset=["hh_id", "line_no"]).sum())
    if dup_count > 0:
        raise IntegrityValidationError(f"Found {dup_count} duplicate records on (hh_id, line_no).")


def validate_nir_harmonization(df: pd.DataFrame) -> dict[str, int]:
    """Validate deterministic Negros Island Region (reg_harmonized == 18) reconstruction."""
    bad_regs = set(df["reg_harmonized"].dropna().astype(int).unique()) - VALID_HARMONIZED_REGIONS
    if bad_regs:
        raise NIRHarmonizationError(f"Unstandardized reg_harmonized codes found: {sorted(bad_regs)}")

    prov_in_nir = df["prov_code"].isin(NIR_PROVINCE_CODES)
    flag_is_nir = df["is_nir"] == 1
    reg_is_18 = df["reg_harmonized"] == 18

    # Ensure bidirectional equivalence between province code, NIR flag, and Region 18 assignment
    if not (prov_in_nir == flag_is_nir).all():
        mismatch = int((prov_in_nir != flag_is_nir).sum())
        raise NIRHarmonizationError(f"Mismatch ({mismatch} rows) between prov_code and is_nir flag.")
    if not (flag_is_nir == reg_is_18).all():
        mismatch = int((flag_is_nir != reg_is_18).sum())
        raise NIRHarmonizationError(f"Mismatch ({mismatch} rows) between is_nir == 1 and reg_harmonized == 18.")
    if df.loc[flag_is_nir, "nir_prov_name"].isna().any():
        raise NIRHarmonizationError("Missing nir_prov_name for one or more NIR rows.")

    prov_counts = df.loc[flag_is_nir, "nir_prov_name"].value_counts().to_dict()
    return {str(k): int(v) for k, v in prov_counts.items()}
