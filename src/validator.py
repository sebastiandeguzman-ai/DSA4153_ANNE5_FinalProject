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

def validate_value_domains(df: pd.DataFrame) -> None:
    """Validate allowed values for sex, age, binary indicators, and positive survey weights."""
    # PSA coding: 1 = Male, 2 = Female
    invalid_sex = set(df["sex"].dropna().unique()) - {1, 2}
    if invalid_sex:
        raise DomainValidationError(f"Invalid sex codes found: {invalid_sex}")

    if df["age"].isna().any() or (df["age"] < 0).any() or (df["age"] > 120).any():
        raise DomainValidationError(
            f"Invalid age values (min={df['age'].min()}, max={df['age'].max()}, nulls={df['age'].isna().sum()})."
        )

    # Socioeconomic and literacy flags must strictly use 0/1 binary encoding
    binary_cols = ["attending_school", "employed", "basic_literate", "func_literate", "hh_electricity"]
    for col in binary_cols:
        invalid_vals = set(df[col].dropna().unique()) - {0, 1, 0.0, 1.0}
        if invalid_vals:
            raise DomainValidationError(f"Column '{col}' has non-binary values: {invalid_vals}")

    # Every survey record requires a strictly positive population expansion weight
    if df["pop_weight"].isna().any() or (df["pop_weight"] <= 0).any():
        bad_w = int((df["pop_weight"].isna() | (df["pop_weight"] <= 0)).sum())
        raise DomainValidationError(f"Found {bad_w} rows with missing or non-positive pop_weight.")

    if (df["literacy_weight"].dropna() < 0).any():
        raise DomainValidationError("Negative literacy_weight values found.")


def validate_literacy_consistency(df: pd.DataFrame, strict: bool = False) -> list[str]:
    """Check age eligibility (10-64) and cross-wave literacy coding nuances."""
    warnings_found: list[str] = []

    # Functional literacy assessment applies strictly to respondents aged 10-64
    assessed = df[df["func_literate"].notna()]
    if not assessed.empty:
        min_age, max_age = int(assessed["age"].min()), int(assessed["age"].max())
        if min_age < 10 or max_age > 64:
            raise DomainValidationError(f"Assessed func_literate outside 10-64 age scope: [{min_age}, {max_age}].")

    year = int(df["survey_year"].iloc[0])

    # Check 1: 2008/2013 Level 2 ("Can read, write and compute") functional literacy threshold nuance
    if year in (2008, 2013):
        lvl2_not_lit = int(((df["func_literacy_level"] == 2) & (df["func_literate"] == 0)).sum())
        if lvl2_not_lit > 0:
            warnings_found.append(
                f"[{year}] {lvl2_not_lit:,} rows have func_literacy_level==2 coded as func_literate==0 "
                "(PSA 2008-2019 official functional literacy includes Levels 2, 3, and 4)."
            )

    # Check 2: Undetermined literacy level (NaN) paired with non-null func_literate
    undet_coded = int((df["func_literacy_level"].isna() & df["func_literate"].notna()).sum())
    if undet_coded > 0:
        warnings_found.append(
            f"[{year}] {undet_coded:,} rows have func_literacy_level==NaN (Undetermined code 9) with non-null func_literate."
        )

    if strict and warnings_found:
        raise DomainValidationError("Strict literacy validation failed: " + " | ".join(warnings_found))

    return warnings_found

def validate_wave_dataset(
    filepath: Path,
    expected_year: int,
    strict: bool = False,
    check_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run all 5 validation checks on a single wave CSV and return a flat summary row."""
    if not filepath.exists():
        raise FileNotFoundError(f"Clean CSV not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)

    def record(category: str, check_name: str, expected: object, actual: object, status: str, detail: str = "") -> None:
        if check_records is not None:
            check_records.append({
                "survey_year": expected_year, "file": filepath.name, "category": category,
                "check_name": check_name, "expected": str(expected), "actual": str(actual),
                "status": status, "detail": detail,
            })

    # Run and log all 5 modular checks
    for cat, chk, exp, fn in [
        ("Schema", "exact_22_column_parity", len(EXPECTED_COLUMNS), lambda: validate_schema(df)),
        ("Integrity", "unique_composite_key_hh_id_line_no", 0, lambda: validate_primary_keys(df, expected_year)),
        ("Domain", "valid_domains_and_positive_weights", "valid ranges & pop_weight > 0", lambda: validate_value_domains(df)),
    ]:
        try:
            fn()
            record(cat, chk, exp, "OK", "PASS", f"rows={len(df):,}")
        except FLEMMSValidationError as exc:
            record(cat, chk, exp, "violation", "FAIL", str(exc))
            raise

    try:
        nir_breakdown = validate_nir_harmonization(df)
        nir_detail = "; ".join(f"{k}: {v:,}" for k, v in sorted(nir_breakdown.items()))
        record("Geography", "nir_region_18_reconstruction", "18 regions + NIR", f"{df['reg_harmonized'].nunique()} regions", "PASS", nir_detail)
    except FLEMMSValidationError as exc:
        record("Geography", "nir_region_18_reconstruction", "18 regions + NIR", "mismatch", "FAIL", str(exc))
        raise

    try:
        warnings_found = validate_literacy_consistency(df, strict=strict)
        for w in warnings_found:
            logger.warning(w)
            record("Literacy", "func_literacy_coding_consistency", "0 nuances", "nuance flagged", "WARN", w)
        if not warnings_found:
            record("Literacy", "func_literacy_coding_consistency", "consistent 10-64 coding", "consistent", "PASS", "No discrepancies")
    except FLEMMSValidationError as exc:
        record("Literacy", "func_literacy_coding_consistency", "consistent 10-64 coding", "failed", "FAIL", str(exc))
        raise

    logger.info("PASS | %s | rows=%s | regions=%d", filepath.name, f"{len(df):,}", df["reg_harmonized"].nunique())
    return {
        "survey_year": expected_year, "file": filepath.name, "rows": int(len(df)), "cols": int(df.shape[1]),
        "unique_regions": int(df["reg_harmonized"].nunique()), "nir_rows": int(df["is_nir"].sum()),
        "nir_negros_occidental": int(nir_breakdown.get("Negros Occidental", 0)),
        "nir_negros_oriental": int(nir_breakdown.get("Negros Oriental", 0)),
        "nir_siquijor": int(nir_breakdown.get("Siquijor", 0)),
        "assessed_10_64": int(df["func_literate"].notna().sum()),
        "total_pop_weight": round(float(df["pop_weight"].sum()), 2),
        "total_literacy_weight": round(float(df["literacy_weight"].dropna().sum()), 2),
        "warnings_count": len(warnings_found),
        "warnings_detail": " | ".join(warnings_found) if warnings_found else "None",
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

def validate_all_waves(
    data_dir: str = ".",
    strict: bool = False,
    out_dir: str | None = "reports",
) -> pd.DataFrame:
    """Validate all 4 clean FLEMMS CSVs and export summary + granular check CSVs."""
    base = Path(data_dir)
    results: list[dict[str, object]] = []
    check_records: list[dict[str, object]] = []

    try:
        for year, filename in WAVE_FILES.items():
            summary = validate_wave_dataset(
                base / filename, expected_year=year, strict=strict, check_records=check_records
            )
            results.append(summary)
    finally:
        # Always persist granular check logs even if an exception halts mid-run
        if out_dir is not None and check_records:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            checks_csv = out_path / "flemms_validation_checks.csv"
            pd.DataFrame(check_records).to_csv(checks_csv, index=False)
            logger.info("Saved granular validation checks CSV -> %s", checks_csv)

    summary_df = pd.DataFrame(results)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_csv = out_path / "flemms_validation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info("Saved wave validation summary CSV -> %s", summary_csv)

    return summary_df
