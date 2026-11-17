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