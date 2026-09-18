"""
app.py — Mental Health in Tech Survey: Data Cleaning & Loading Pipeline
==========================================================================
Source: OSMI (Open Sourcing Mental Illness) 2014 "Mental Health in Tech"
survey — 1,259 responses, 27 columns (data/survey.csv).

Reads the raw CSV, cleans it, engineers a few analysis-ready features, and
persists the result to:
  1. data/survey_cleaned.csv   (flat file, easy to inspect / share)
  2. db/mental_health.db       (SQLite database, table: survey)

This is the script the notebook and the Streamlit dashboard both build on.
Run it directly whenever the source data changes:

    python app.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

RAW_CSV_PATH = DATA_DIR / "survey.csv"
CLEANED_CSV_PATH = DATA_DIR / "survey_cleaned.csv"
SQLITE_DB_PATH = DB_DIR / "mental_health.db"
TABLE_NAME = "survey"

# --------------------------------------------------------------------------
# Gender standardization map
# --------------------------------------------------------------------------
MALE_PATTERNS = [
    "male", "m", "man", "cis male", "cis man", "malr", "mal", "maile",
    "make", "guy (-ish) ^_^", "male (cis)", "male-ish", "male leaning androgynous",
    "msle", "mail", "cis-male", "man ", "malr", "ostensibly male, unsure what that really means",
]
FEMALE_PATTERNS = [
    "female", "f", "woman", "cis female", "cis-female/femme", "femake", "femail",
    "female (cis)", "female (trans)", "trans-female", "trans woman", "woman ",
]


def standardize_gender(raw: str) -> str:
    if pd.isna(raw):
        return "Other/Unspecified"
    val = str(raw).strip().lower()
    if val in MALE_PATTERNS:
        return "Male"
    if val in FEMALE_PATTERNS:
        return "Female"
    # fallback fuzzy match for anything not in the explicit lists above
    if re.fullmatch(r"m+a*l*e?", val) and "female" not in val:
        return "Male"
    if "fem" in val or val.startswith("f") and val in ("f", "fm"):
        return "Female"
    return "Other/Unspecified"


AGE_MIN, AGE_MAX = 15, 90  # plausible working-age survey bounds

AGE_BINS = [14, 24, 34, 44, 54, 64, 91]
AGE_LABELS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]

SUPPORT_YES_COLS = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_CSV_PATH}. Place the OSMI survey.csv there and retry."
        )
    return pd.read_csv(RAW_CSV_PATH)


# --------------------------------------------------------------------------
# 2. Clean & standardize
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    df = df.drop_duplicates()
    dupes_removed = before - len(df)

    # --- Timestamp ---
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # --- Age: drop implausible values (negative, 0, or absurdly large like 99999999999) ---
    before = len(df)
    df = df[(df["Age"] >= AGE_MIN) & (df["Age"] <= AGE_MAX)]
    age_dropped = before - len(df)
    df["Age_Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    # --- Gender standardization (the classic messy free-text field) ---
    df["Gender_Clean"] = df["Gender"].apply(standardize_gender)

    # --- Country / state ---
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Is_US"] = df["Country"].eq("United States")
    df["state"] = df["state"].fillna("N/A (Outside US)")

    # --- self_employed: 18 missing -> fill with mode ("No") ---
    df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])

    # --- work_interfere: missing almost always means the respondent hasn't
    #     sought treatment, so the conditional question doesn't apply ---
    df["work_interfere"] = df["work_interfere"].fillna("Not applicable")

    # --- comments: mostly empty free text; keep a boolean flag, drop the raw text
    #     from the analysis-ready table (not useful for charts, adds noise) ---
    df["Has_Comment"] = df["comments"].notna()
    df = df.drop(columns=["comments"])

    # --- Standardize Yes/No/Don't know style columns (trim whitespace) ---
    yesno_cols = [
        "self_employed", "family_history", "treatment", "remote_work", "tech_company",
        "benefits", "care_options", "wellness_program", "seek_help", "anonymity",
        "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor",
        "mental_health_interview", "phys_health_interview", "mental_vs_physical", "obs_consequence",
    ]
    for c in yesno_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # --------------------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------------------
    df["Treatment_Flag"] = df["treatment"].eq("Yes")
    df["Family_History_Flag"] = df["family_history"].eq("Yes")

    # Composite "workplace support index" (0-5): count of supportive Yes answers
    def _support_score(row):
        score = 0
        for c in SUPPORT_YES_COLS:
            if row[c] == "Yes":
                score += 1
        return score

    df["Workplace_Support_Score"] = df.apply(_support_score, axis=1)

    df = df.reset_index(drop=True)

    print(
        f"Cleaning summary: removed {dupes_removed} duplicate rows, "
        f"{age_dropped} rows with implausible age. Final row count: {len(df):,}"
    )
    print(f"Gender standardized into: {df['Gender_Clean'].value_counts().to_dict()}")

    return df


# --------------------------------------------------------------------------
# 3. Persist
# --------------------------------------------------------------------------
def save_outputs(df: pd.DataFrame):
    out = df.copy()
    out["Timestamp"] = out["Timestamp"].astype(str)
    out.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Saved cleaned CSV -> {CLEANED_CSV_PATH} ({len(out):,} rows)")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    out.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_country ON {TABLE_NAME}(Country)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_treatment ON {TABLE_NAME}(treatment)")
    conn.commit()
    conn.close()
    print(f"Saved SQLite DB -> {SQLITE_DB_PATH} (table: {TABLE_NAME})")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    print("Loading raw survey data...")
    raw_df = load_raw_data()
    print(f"Raw rows loaded: {len(raw_df):,}, columns: {len(raw_df.columns)}")

    print("Cleaning & engineering features...")
    clean_df = clean_data(raw_df)

    print("Saving outputs...")
    save_outputs(clean_df)

    return clean_df


if __name__ == "__main__":
    run_pipeline()
    print("Pipeline complete.")
