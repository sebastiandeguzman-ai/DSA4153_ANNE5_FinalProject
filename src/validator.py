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
