# sso_pipeline.py
# Download EPA sewer overflow / collection system tables,
# merge collection_system_permit with sewer_overflow_bypass_event,
# and write raw + merged + summary outputs.

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

# -----------------------------
# CONFIG
# -----------------------------

# EPA ZIP containing:
#   - collection_system_permit
#   - sewer_overflow_bypass_event
SEWER_ZIP_URL = (
    "https://echo.epa.gov/files/echodownloads/"
    "all_sewer_overflow_and_collection_systems_tables.zip"
)

# Repo-relative paths (works nicely in GitHub Actions)
REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "data_cache"
OUTPUT_DIR = REPO_ROOT / "outputs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEWER_ZIP_PATH = CACHE_DIR / "all_sewer_overflow_and_collection_systems_tables.zip"


# =========================================================
# 1. Download & extract SSO tables
# =========================================================
def download_sso_zip(force: bool = False) -> Path:
    """
    Download the sewer overflow / collection system ZIP from EPA
    and save it to CACHE_DIR.
    """
    if SEWER_ZIP_PATH.exists() and not force:
        print(f"[sso] Using cached ZIP: {SEWER_ZIP_PATH}")
        return SEWER_ZIP_PATH

    print(f"[sso] Downloading: {SEWER_ZIP_URL}")
    resp = requests.get(SEWER_ZIP_URL, timeout=120)
    resp.raise_for_status()
    SEWER_ZIP_PATH.write_bytes(resp.content)
    print(f"[sso] Saved ZIP to: {SEWER_ZIP_PATH}")
    return SEWER_ZIP_PATH


def load_sso_tables(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (if needed) and extract:
      - collection_system_permit
      - sewer_overflow_bypass_event

    Returns (df_collection_system_permit, df_sewer_overflow_bypass_event).
    """
    zip_path = download_sso_zip(force=force_download)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        coll_name = next(
            n for n in names if "collection_system_permit" in n.lower()
        )
        sso_name = next(
            n for n in names if "sewer_overflow_bypass_event" in n.lower()
        )

        print("[sso] Found tables in ZIP:")
        print("   collection_system_permit ->", coll_name)
        print("   sewer_overflow_bypass_event ->", sso_name)

        with z.open(coll_name) as f:
            df_coll = pd.read_csv(f, low_memory=False)

        with z.open(sso_name) as f:
            df_sso = pd.read_csv(f, low_memory=False)

    print("[sso] collection_system_permit shape:", df_coll.shape)
    print("[sso] sewer_overflow_bypass_event shape:", df_sso.shape)

    return df_coll, df_sso


# =========================================================
# 2. Merge + summary
# =========================================================
def merge_sso_with_collection_system(
    df_coll: pd.DataFrame, df_sso: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Merge sewer_overflow_bypass_event with collection_system_permit
    on (permit_identifier, collection_system_identifier).

    Returns a dict of:
      - collection_system_permit_raw
      - sewer_overflow_bypass_event_raw
      - sso_events_with_collection_system
      - sso_summary_by_permit_year
    """

    # Keep raw copies
    df_coll_raw = df_coll.copy()
    df_sso_raw = df_sso.copy()

    # Normalized join keys (robust to whitespace/case)
    for frame in (df_coll, df_sso):
        frame["_permit_join"] = (
            frame["permit_identifier"].astype(str).str.strip().str.upper()
        )

    # collection_system_identifier exists in both tables
    df_coll["_cs_join"] = (
        df_coll["collection_system_identifier"].astype(str).str.strip().str.upper()
    )
    df_sso["_cs_join"] = (
        df_sso["collection_system_identifier"].astype(str).str.strip().str.upper()
    )

    # --------------------------------------------------
    # Coordinate cleanup (fix sign errors for US systems)
    # --------------------------------------------------
    # Convert to numeric, coerce bad text to NaN
    df_sso["latitude_measure"] = pd.to_numeric(
        df_sso.get("latitude_measure"), errors="coerce"
    )
    df_sso["longitude_measure"] = pd.to_numeric(
        df_sso.get("longitude_measure"), errors="coerce"
    )

    # 1) Rows with negative latitude (e.g. -41, 98) -> flip both
    mask_both = df_sso["latitude_measure"] < 0
    df_sso.loc[mask_both, ["latitude_measure", "longitude_measure"]] = (
        -df_sso.loc[mask_both, ["latitude_measure", "longitude_measure"]]
    )

    # 2) Rows with plausible US mainland lat (>= 0) and positive lon 60–110 -> flip lon only
    #    (e.g. 40, 88 -> 40, -88). This will NOT touch Guam (lon ~144).
    mask_lon_only = (
        df_sso["latitude_measure"].ge(0)
        & df_sso["longitude_measure"].between(60, 110)
    )
    df_sso.loc[mask_lon_only, "longitude_measure"] = -df_sso.loc[
        mask_lon_only, "longitude_measure"
    ]

    # Event-level merge: all SSO events, with collection system attributes appended
    df_merged = df_sso.merge(
        df_coll,
        how="left",
        on=["_permit_join", "_cs_join"],
        suffixes=("", "_coll"),
    )

    # Clean up join helper columns (but keep original identifiers)
    df_merged = df_merged.drop(columns=["_permit_join", "_cs_join"])
    df_coll_raw = df_coll_raw.drop(columns=["_permit_join", "_cs_join"], errors="ignore")
    df_sso_raw = df_sso_raw.drop(columns=["_permit_join", "_cs_join"], errors="ignore")

    print("[sso] Merged events shape:", df_merged.shape)

    # ---- Build a simple per-permit/year summary ----
    df_summary = df_merged.copy()

    # Convert datetimes / volume
    df_summary["sso_start_dt"] = pd.to_datetime(
        df_summary["sewer_overflow_bypass_start_datetime"], errors="coerce"
    )
    df_summary["sso_end_dt"] = pd.to_datetime(
        df_summary["sewer_overflow_bypass_end_datetime"], errors="coerce"
    )

    df_summary["sso_volume_gal"] = pd.to_numeric(
        df_summary["sewer_overflow_bypass_discharge_volume_gallons"],
        errors="coerce",
    )

    df_summary["year"] = df_summary["sso_start_dt"].dt.year

    # Aggregate by permit + collection system + year
    summary_group_cols = [
        "permit_identifier",
        "collection_system_identifier",
        "collection_system_name",
        "collection_system_owner_type_desc",
        "collection_system_population",
        "year",
    ]

    sso_summary = (
        df_summary.groupby(summary_group_cols, dropna=False)
        .agg(
            sso_event_count=("sewer_overflow_bypass_event_key", "nunique"),
            sso_total_volume_gal=("sso_volume_gal", "sum"),
            sso_first_event=("sso_start_dt", "min"),
            sso_last_event=("sso_end_dt", "max"),
        )
        .reset_index()
    )

    print("[sso] Summary by permit/year shape:", sso_summary.shape)

    return {
        "collection_system_permit_raw": df_coll_raw,
        "sewer_overflow_bypass_event_raw": df_sso_raw,
        "sso_events_with_collection_system": df_merged,
        "sso_summary_by_permit_year": sso_summary,
    }

# =========================================================
# 3. Orchestration
# =========================================================
def run_pipeline() -> None:
    print("=== EPA SSO / Collection System Pipeline: START ===")

    # 1) Download + load tables
    df_coll, df_sso = load_sso_tables(force_download=False)

    # 2) Merge + summarize
    results = merge_sso_with_collection_system(df_coll, df_sso)

    # 3) Write outputs
    results["collection_system_permit_raw"].to_csv(
        OUTPUT_DIR / "collection_system_permit_raw.csv",
        index=False,
    )
    results["sewer_overflow_bypass_event_raw"].to_csv(
        OUTPUT_DIR / "sewer_overflow_bypass_event_raw.csv",
        index=False,
    )
    results["sso_events_with_collection_system"].to_csv(
        OUTPUT_DIR / "sso_events_with_collection_system.csv",
        index=False,
    )
    results["sso_summary_by_permit_year"].to_csv(
        OUTPUT_DIR / "sso_summary_by_permit_year.csv",
        index=False,
    )

    print("=== EPA SSO / Collection System Pipeline: DONE ===")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()
