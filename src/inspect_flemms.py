"""
===============================================================================
Title: FLEMMS 2019 & 2024 Raw Data Inspection and Validation
Description: Exploratory inspection script to audit sheet structures, column 
             headers, and initial data quality for the FLEMMS survey datasets.
Project: DSA4153 Final Project
===============================================================================
"""

import numpy as np
import pandas as pd

# 1. File Paths
file_2019 = "datasets/2019 FLEMMS.xlsx"
file_2024 = "datasets/2024 FLEMMS.xlsx"


def inspect_sheet_numbers(file_path, workbook_label):
    excel_file = pd.ExcelFile(file_path)

    print(f"\n==================================================")
    print(f"      {workbook_label} WORKBOOK INSPECTION")
    print(f"==================================================")

    for sheet in excel_file.sheet_names:
        # Load raw sheet
        df_raw = pd.read_excel(file_path, sheet_name=sheet)

        # Header offset: "List of Tables" starts at row 1, statistical sheets skip top 4 title rows
        header_offset = 0 if sheet == "List of Tables" else 4
        df_data = df_raw.iloc[header_offset:]

        # Raw grid statistics
        raw_rows, raw_cols = df_raw.shape
        raw_nans = df_raw.isna().sum().sum()
        raw_blank_r = df_raw.isna().all(axis=1).sum()
        raw_blank_c = df_raw.isna().all(axis=0).sum()

        # Data section statistics (header rows excluded)
        data_rows, data_cols = df_data.shape
        data_nans = df_data.isna().sum().sum()
        data_blank_r = df_data.isna().all(axis=1).sum()
        data_blank_c = df_data.isna().all(axis=0).sum()

        # Print sheet results
        print(f"Sheet: {sheet}")
        print(
            f"  • Total Grid    : {raw_rows} rows × {raw_cols} cols | Total NaNs: {raw_nans} | Blank Rows: {raw_blank_r} | Blank Cols: {raw_blank_c}"
        )
        print(
            f"  • Data Section  : {data_rows} rows × {data_cols} cols | Data NaNs : {data_nans} | Blank Rows: {data_blank_r} | Blank Cols: {data_blank_c}\n"
        )


# Run inspection on both workbooks
inspect_sheet_numbers(file_2019, "2019 FLEMMS")
inspect_sheet_numbers(file_2024, "2024 FLEMMS")
