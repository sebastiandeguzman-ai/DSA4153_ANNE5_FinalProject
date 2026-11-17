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

# ---------------------------------------------------------------------------
# 2. VECTORIZED NIR & REGION HARMONIZATION (NUMPY)
# ---------------------------------------------------------------------------
def harmonize_region_and_nir(reg_orig: pd.Series, prov_code: pd.Series) -> pd.DataFrame:
    """
    Reconstruct Negros Island Region (REG_HARMONIZED = 18) and standardize
    legacy region codes (41->4, 42->17, 15->19) using vectorized NumPy masks.
    """
    reg_arr = pd.to_numeric(reg_orig, errors="coerce").to_numpy(dtype=np.float64)
    prov_arr = pd.to_numeric(prov_code, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

    is_nir = np.isin(prov_arr, NIR_PROVINCE_CODES)

    reg_harmonized = np.select(
        condlist=[
            is_nir,
            reg_arr == 41,  # Legacy CALABARZON -> 4
            reg_arr == 42,  # Legacy MIMAROPA -> 17
            reg_arr == 15,  # Legacy ARMM -> 19 (BARMM)
        ],
        choicelist=[18, 4, 17, 19],
        default=reg_arr,
    ).astype(np.int64)

    reg_series = pd.Series(reg_harmonized, index=reg_orig.index)
    prov_series = pd.Series(prov_arr, index=reg_orig.index).replace(-1, np.nan)

    return pd.DataFrame(
        {
            "reg_orig": pd.to_numeric(reg_orig, errors="coerce").astype("Int64"),
            "reg_harmonized": reg_series.astype("Int64"),
            "reg_name": reg_series.map(REGION_NAMES),
            "is_nir": is_nir.astype(np.int8),
            "prov_code": prov_series.astype("Int64"),
            "nir_prov_name": prov_series.map(NIR_PROVINCE_NAMES),
        },
        index=reg_orig.index,
    )


def to_clean_numeric(series: pd.Series, na_codes: set | None = None) -> pd.Series:
    """Strip whitespace strings and convert specified special/refusal codes to NaN."""
    cleaned = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    num = pd.to_numeric(cleaned, errors="coerce")
    if na_codes:
        num = num.where(~num.isin(na_codes), np.nan)
    return num

# ---------------------------------------------------------------------------
# 3. WAVE-SPECIFIC CLEANING & INTEGRATION FUNCTIONS
# ---------------------------------------------------------------------------
def build_2008_clean(data_dir: Path) -> pd.DataFrame:
    """Clean and integrate FLEMMS 2008 RT01 (All Members) and RT02 (10-64 Assessment)."""
    rt01 = read_csv_checked(
        data_dir / "FLEMMS PUF 2008 RT01 - All Members.CSV",
        required_cols=[
            "PREG", "PSTRATUM", "PPSU", "PHHSEQNO", "PC01_LNO",
            "PC04_SEX", "PC05_AGE", "PC11_ABLERW", "PC12_HGRADE",
            "PC15_CURSCH", "PXC22_JOB", "PD1_ELECT", "PFPOPWGT",
        ],
    )
    rt02 = read_csv_checked(
        data_dir / "FLEMMS PUF 2008 RT02 - 10-64 Years Old.CSV",
        required_cols=["PREG", "PSTRATUM", "PPSU", "PHHSEQNO", "LNO_10_64", "PLITERACY", "PF2FWGT"],
    )

    keys_01 = ["PREG", "PSTRATUM", "PPSU", "PHHSEQNO", "PC01_LNO"]
    keys_02 = ["PREG", "PSTRATUM", "PPSU", "PHHSEQNO", "LNO_10_64"]
    assert rt01.duplicated(keys_01).sum() == 0, "Duplicate member keys in 2008 RT01"
    assert rt02.duplicated(keys_02).sum() == 0, "Duplicate respondent keys in 2008 RT02"

    merged = rt01.merge(
        rt02[keys_02 + ["PLITERACY", "PF2FWGT"]],
        left_on=keys_01,
        right_on=keys_02,
        how="left",
        validate="one_to_one",
    )
    assert len(merged) == len(rt01), "Row count changed during 2008 left join"

    # Extract 2-digit province code from PPSU (e.g., 45001 -> 45, 46001 -> 46, 61001 -> 61)
    prov_code = pd.to_numeric(merged["PPSU"], errors="coerce") // 1000
    geo = harmonize_region_and_nir(merged["PREG"], prov_code)

    basic_lit = to_clean_numeric(merged["PC11_ABLERW"], na_codes={9})
    flevel = to_clean_numeric(merged["PLITERACY"], na_codes={9})
    cursch = to_clean_numeric(merged["PC15_CURSCH"], na_codes={9})
    job = to_clean_numeric(merged["PXC22_JOB"], na_codes={9})
    elect = to_clean_numeric(merged["PD1_ELECT"], na_codes={9})

    clean = pd.concat(
        [
            pd.DataFrame({"survey_year": np.int16(2008)}, index=merged.index),
            geo,
            pd.DataFrame(
                {
                    "urbanity": pd.Series(pd.NA, index=merged.index, dtype="Int64"),  # Not in 2008 PUF
                    "psu": to_clean_numeric(merged["PPSU"]).astype("Int64"),
                    "hh_id": (
                        merged["PSTRATUM"].astype(str).str.zfill(3)
                        + "_" + merged["PPSU"].astype(str).str.zfill(5)
                        + "_" + merged["PHHSEQNO"].astype(str).str.zfill(4)
                    ),
                    "line_no": to_clean_numeric(merged["PC01_LNO"]).astype("Int64"),
                    "sex": to_clean_numeric(merged["PC04_SEX"], na_codes={9}).astype("Int64"),
                    "age": to_clean_numeric(merged["PC05_AGE"], na_codes={999}).astype("Int64"),
                    "highest_grade": to_clean_numeric(merged["PC12_HGRADE"], na_codes={99, 999}).astype("Int64"),
                    "attending_school": np.where(cursch == 1, 1, np.where(cursch == 2, 0, np.nan)),
                    "employed": np.where(job == 1, 1, np.where(job == 2, 0, np.nan)),
                    "basic_literate": np.where(basic_lit == 1, 1, np.where(basic_lit == 2, 0, np.nan)),
                    "func_literacy_level": flevel.astype("Int64"),
                    "func_literate": np.where(flevel.isin([3, 4]), 1, np.where(flevel.isin([0, 1, 2]), 0, np.nan)),
                    "hh_electricity": np.where(elect == 1, 1, np.where(elect == 2, 0, np.nan)),
                    "pop_weight": to_clean_numeric(merged["PFPOPWGT"]),
                    "literacy_weight": to_clean_numeric(merged["PF2FWGT"]),
                },
                index=merged.index,
            ),
        ],
        axis=1,
    )
    logger.info("2008 clean rows: %d | NIR members: %d", len(clean), int(clean["is_nir"].sum()))
    return clean

def build_2013_clean(data_dir: Path) -> pd.DataFrame:
    """
    Clean and integrate FLEMMS 2013 UrbanRural RT01 & RT02.
    Uses UrbanRural PUF because PRV (province) and URB are preserved there.
    Note: Region VIII (Eastern Visayas) was excluded in 2013 due to Typhoon Yolanda.
    """
    rt01 = read_csv_checked(
        data_dir / "FLEMMS PUF 2013 UrbanRural RT01 - Household Members.CSV",
        required_cols=[
            "REG", "PRV", "STRATUM", "PSU", "HHSEQNO", "QID", "URB",
            "C01_LNO", "C04_SEX", "C05_AGE", "C11_ABLERW", "C12_HGRADE",
            "C14_CURSCH", "C21_JOB", "E1_ELECT", "PWGT",
        ],
    )
    rt02 = read_csv_checked(
        data_dir / "FLEMMS PUF 2013 UrbanRural RT02 - Individual.CSV",
        required_cols=["REG", "PRV", "STRATUM", "PSU", "HHSEQNO", "QID", "IRESP", "LLEVEL", "PWGT2"],
    )

    keys_01 = ["REG", "PRV", "STRATUM", "PSU", "HHSEQNO", "QID", "C01_LNO"]
    keys_02 = ["REG", "PRV", "STRATUM", "PSU", "HHSEQNO", "QID", "IRESP"]
    assert rt01.duplicated(keys_01).sum() == 0, "Duplicate member keys in 2013 RT01"
    assert rt02.duplicated(keys_02).sum() == 0, "Duplicate respondent keys in 2013 RT02"

    merged = rt01.merge(
        rt02[keys_02 + ["LLEVEL", "PWGT2"]],
        left_on=keys_01,
        right_on=keys_02,
        how="left",
        validate="one_to_one",
    )
    assert len(merged) == len(rt01), "Row count changed during 2013 left join"

    geo = harmonize_region_and_nir(merged["REG"], merged["PRV"])

    basic_lit = to_clean_numeric(merged["C11_ABLERW"], na_codes={9})
    flevel = to_clean_numeric(merged["LLEVEL"], na_codes={9})
    cursch = to_clean_numeric(merged["C14_CURSCH"], na_codes={9})
    job = to_clean_numeric(merged["C21_JOB"], na_codes={9})
    elect = to_clean_numeric(merged["E1_ELECT"], na_codes={9})

    clean = pd.concat(
        [
            pd.DataFrame({"survey_year": np.int16(2013)}, index=merged.index),
            geo,
            pd.DataFrame(
                {
                    "urbanity": to_clean_numeric(merged["URB"], na_codes={9}).astype("Int64"),
                    "psu": to_clean_numeric(merged["PSU"]).astype("Int64"),
                    "hh_id": merged["QID"].astype(str).str.strip(),
                    "line_no": to_clean_numeric(merged["C01_LNO"]).astype("Int64"),
                    "sex": to_clean_numeric(merged["C04_SEX"], na_codes={9}).astype("Int64"),
                    "age": to_clean_numeric(merged["C05_AGE"], na_codes={999}).astype("Int64"),
                    "highest_grade": to_clean_numeric(merged["C12_HGRADE"], na_codes={999}).astype("Int64"),
                    # In 2013 C14_CURSCH: 1 = Yes (Public), 2 = Yes (Private), 3 = No
                    "attending_school": np.where(cursch.isin([1, 2]), 1, np.where(cursch == 3, 0, np.nan)),
                    "employed": np.where(job == 1, 1, np.where(job == 2, 0, np.nan)),
                    "basic_literate": np.where(basic_lit == 1, 1, np.where(basic_lit == 2, 0, np.nan)),
                    "func_literacy_level": flevel.astype("Int64"),
                    "func_literate": np.where(flevel.isin([3, 4]), 1, np.where(flevel.isin([0, 1, 2]), 0, np.nan)),
                    "hh_electricity": np.where(elect == 1, 1, np.where(elect == 2, 0, np.nan)),
                    "pop_weight": to_clean_numeric(merged["PWGT"]),
                    "literacy_weight": to_clean_numeric(merged["PWGT2"]),
                },
                index=merged.index,
            ),
        ],
        axis=1,
    )
    logger.info("2013 clean rows: %d | NIR members: %d", len(clean), int(clean["is_nir"].sum()))
    return clean

def build_2019_clean(data_dir: Path) -> pd.DataFrame:
    """
    Clean and integrate FLEMMS 2019 across all 4 tables:
      1. HOUSEHOLD RECORD (filter FINAL_VISIT == 1; recover RPROV & RPSU)
      2. HOUSING CHARACTERISTICS (recover Q14_ELECTRICITY)
      3. DEMOGRAPHIC CHARACTERISTICS (all household members)
      4. HOUSEHOLD NAME (individual literacy assessment, ages 10-64)
    """
    hh = read_csv_checked(
        data_dir / "FLEMMS PUF 2019 HOUSEHOLD RECORD.CSV",
        required_cols=["REG", "URB", "HHID", "FINAL_VISIT", "RPROV", "RPSU", "RFACT"],
    )
    hsg = read_csv_checked(
        data_dir / "FLEMMS PUF 2019 HOUSING CHARACTERISTICS.CSV",
        required_cols=["REG", "URB", "HHID", "Q14_ELECTRICITY"],
    )
    dem = read_cspro_fixed_comma_csv(
        data_dir / "FLEMMS PUF 2019 DEMOGRAPHIC CHARACTERISTICS.CSV",
        required_cols=[
            "REG", "URB", "HHID", "C1_LNO", "C4_SEX", "C5_AGE",
            "C11_BASIC_LITERACY", "C12_HEA", "C14_ATTEND_SCHOOL",
            "C20_BUSINESS", "MEM_RFACT",
        ],
    )
    ind = read_cspro_fixed_comma_csv(
        data_dir / "FLEMMS PUF 2019 HOUSEHOLD NAME.CSV",
        required_cols=["REG", "URB", "HHID", "RESP_LNO", "LIT_LEVEL", "FL_RATE", "IND_MEM_RFACT"],
    )

    # Standardize join keys to integer across all 4 tables
    for df_part in (hh, hsg, dem, ind):
        for k in ("REG", "URB", "HHID"):
            df_part[k] = to_clean_numeric(df_part[k]).astype("Int64")

    dem["C1_LNO"] = to_clean_numeric(dem["C1_LNO"]).astype("Int64")
    ind["RESP_LNO"] = to_clean_numeric(ind["RESP_LNO"]).astype("Int64")

    # Keep completed household interviews (FINAL_VISIT == 1)
    hh_valid = hh[to_clean_numeric(hh["FINAL_VISIT"]) == 1].copy()
    assert hh_valid.duplicated(["REG", "URB", "HHID"]).sum() == 0, "Duplicate HH keys in 2019"

    hh_hsg = hh_valid[["REG", "URB", "HHID", "RPROV", "RPSU"]].merge(
        hsg[["REG", "URB", "HHID", "Q14_ELECTRICITY"]],
        on=["REG", "URB", "HHID"],
        how="left",
        validate="one_to_one",
    )

    mem = hh_hsg.merge(
        dem,
        on=["REG", "URB", "HHID"],
        how="inner",
        validate="one_to_many",
    )
    assert mem.duplicated(["REG", "URB", "HHID", "C1_LNO"]).sum() == 0, "Duplicate member keys in 2019"
    assert ind.duplicated(["REG", "URB", "HHID", "RESP_LNO"]).sum() == 0, "Duplicate ind keys in 2019"

    merged = mem.merge(
        ind[["REG", "URB", "HHID", "RESP_LNO", "LIT_LEVEL", "FL_RATE", "IND_MEM_RFACT"]],
        left_on=["REG", "URB", "HHID", "C1_LNO"],
        right_on=["REG", "URB", "HHID", "RESP_LNO"],
        how="left",
        validate="one_to_one",
    )
    assert len(merged) == len(mem), "Row count changed during 2019 member-individual join"

    # RPROV stores 4-digit province code (e.g., 4500 -> 45, 4600 -> 46, 6100 -> 61)
    prov_code = to_clean_numeric(merged["RPROV"]) // 100
    geo = harmonize_region_and_nir(merged["REG"], prov_code)

    basic_lit = to_clean_numeric(merged["C11_BASIC_LITERACY"], na_codes={9})
    flevel = to_clean_numeric(merged["LIT_LEVEL"], na_codes={9})
    fl_rate = to_clean_numeric(merged["FL_RATE"], na_codes={9})
    cursch = to_clean_numeric(merged["C14_ATTEND_SCHOOL"], na_codes={9})
    job = to_clean_numeric(merged["C20_BUSINESS"], na_codes={9})
    elect = to_clean_numeric(merged["Q14_ELECTRICITY"], na_codes={9})

    clean = pd.concat(
        [
            pd.DataFrame({"survey_year": np.int16(2019)}, index=merged.index),
            geo,
            pd.DataFrame(
                {
                    "urbanity": merged["URB"].astype("Int64"),
                    "psu": to_clean_numeric(merged["RPSU"]).astype("Int64"),
                    "hh_id": (
                        merged["REG"].astype(str).str.zfill(2)
                        + "_" + merged["URB"].astype(str)
                        + "_" + merged["HHID"].astype(str).str.zfill(6)
                    ),
                    "line_no": merged["C1_LNO"].astype("Int64"),
                    "sex": to_clean_numeric(merged["C4_SEX"], na_codes={9}).astype("Int64"),
                    "age": to_clean_numeric(merged["C5_AGE"], na_codes={999}).astype("Int64"),
                    "highest_grade": to_clean_numeric(merged["C12_HEA"], na_codes={99999}).astype("Int64"),
                    # In 2019 C14_ATTEND_SCHOOL: 1-3 = Yes (Public/Private/Home), 4 = No
                    "attending_school": np.where(cursch.isin([1, 2, 3]), 1, np.where(cursch == 4, 0, np.nan)),
                    "employed": np.where(job == 1, 1, np.where(job == 2, 0, np.nan)),
                    "basic_literate": np.where(basic_lit == 1, 1, np.where(basic_lit == 2, 0, np.nan)),
                    "func_literacy_level": flevel.astype("Int64"),
                    "func_literate": np.where(fl_rate == 1, 1, np.where(fl_rate == 2, 0, np.nan)),
                    "hh_electricity": np.where(elect == 1, 1, np.where(elect == 2, 0, np.nan)),
                    "pop_weight": to_clean_numeric(merged["MEM_RFACT"]),
                    "literacy_weight": to_clean_numeric(merged["IND_MEM_RFACT"]),
                },
                index=merged.index,
            ),
        ],
        axis=1,
    )
    logger.info("2019 clean rows: %d | NIR members: %d", len(clean), int(clean["is_nir"].sum()))
    return clean

