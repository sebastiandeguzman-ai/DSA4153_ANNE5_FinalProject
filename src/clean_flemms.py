"""
FLEMMS Longitudinal Cleaning & Negros Island Region (NIR) Harmonization Pipeline
Generates 4 clean, schema-harmonized member-level datasets (2008, 2013, 2019, 2024):
  - flemms_2008_clean.csv
  - flemms_2013_clean.csv
  - flemms_2019_clean.csv
  - flemms_2024_clean.csv
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("flemms_pipeline")

# Negros Island Region (RA 12000): Negros Occidental (45), Bacolod City HUC (302),
# Negros Oriental (46), and Siquijor (61)
NIR_PROVINCE_CODES = np.array([45, 302, 46, 61], dtype=np.int64)

REGION_NAMES = {
    1: "Region I - Ilocos Region",
    2: "Region II - Cagayan Valley",
    3: "Region III - Central Luzon",
    4: "Region IV-A - CALABARZON",
    5: "Region V - Bicol Region",
    6: "Region VI - Western Visayas (Residual)",
    7: "Region VII - Central Visayas (Residual)",
    8: "Region VIII - Eastern Visayas",
    9: "Region IX - Zamboanga Peninsula",
    10: "Region X - Northern Mindanao",
    11: "Region XI - Davao Region",
    12: "Region XII - SOCCSKSARGEN",
    13: "National Capital Region (NCR)",
    14: "Cordillera Administrative Region (CAR)",
    16: "Region XIII - Caraga",
    17: "MIMAROPA Region",
    18: "Negros Island Region (NIR)",
    19: "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)",
}

NIR_PROVINCE_NAMES = {
    45: "Negros Occidental",
    302: "City of Bacolod (HUC)",
    46: "Negros Oriental",
    61: "Siquijor",
}

# ---------------------------------------------------------------------------
# 1. ACQUISITION HELPERS
# ---------------------------------------------------------------------------
def read_csv_checked(filepath: Path, required_cols: list[str], **kwargs) -> pd.DataFrame:
    """Load a CSV file and verify non-empty row count and required columns."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required raw dataset: {filepath}")
    df = pd.read_csv(filepath, low_memory=False, **kwargs)
    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {filepath}")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(f"{filepath.name} missing required columns: {sorted(missing)}")
    return df


def read_cspro_fixed_comma_csv(filepath: Path, required_cols: list[str]) -> pd.DataFrame:
    """
    Safely read PSA CSPro comma-delimited exports where unquoted commas appear
    inside fixed-width *_SPECIFY text fields (documented 2019 PUF defect).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required raw dataset: {filepath}")

    with filepath.open("r", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().rstrip("\r\n")
        cols = header_line.split(",")
        n_cols = len(cols)
        lines = [ln.rstrip("\r\n") for ln in fh]

    sample_line = next(ln for ln in lines if ln.count(",") == n_cols - 1)
    spans = []
    pos = 0
    for token in sample_line.split(","):
        width = len(token)
        spans.append((pos, pos + width))
        pos += width + 1

    records = []
    repaired_count = 0
    for ln in lines:
        if ln.count(",") == n_cols - 1:
            records.append(ln.split(","))
        else:
            records.append([ln[s:e] for s, e in spans])
            repaired_count += 1

    df = pd.DataFrame(records, columns=cols)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(f"{filepath.name} missing required columns: {sorted(missing)}")
    logger.info(
        "Loaded %s (%d rows; repaired %d unquoted-comma records)",
        filepath.name,
        len(df),
        repaired_count,
    )
    return df

