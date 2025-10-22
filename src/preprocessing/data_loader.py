import os
import pandas as pd
from typing import Tuple

def load_expressora_datasets(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both the main dataset and corrections file into Pandas DataFrames.
    Returns (main_dataset, corrections_dataset).
    """
    main_path = os.path.join(base_dir, "data/expressora_dataset_augmented.csv")
    corr_path = os.path.join(base_dir, "data/corrections.csv")

    if not os.path.exists(main_path):
        raise FileNotFoundError(f"❌ Main dataset not found: {main_path}")
    if not os.path.exists(corr_path):
        raise FileNotFoundError(f"❌ Corrections file not found: {corr_path}")

    print(f"INFO: Loading datasets from {base_dir} ...")
    df_main = pd.read_csv(main_path)
    df_corr = pd.read_csv(corr_path)

    print(f"✅ Main dataset: {len(df_main)} samples")
    print(f"✅ Corrections: {len(df_corr)} samples")

    return df_main, df_corr


def prepare_dataframe_for_training(df_main: pd.DataFrame, df_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Combines both DataFrames, regardless of structural differences,
    and produces a unified bilingual training dataset.
    """
    print("INFO: Preparing combined training dataset...")

    # --- Normalize corrections file if it's simpler ---
    if "gloss" in df_corr.columns and "translation" in df_corr.columns:
        df_corr = df_corr.rename(columns={
            "gloss": "gloss_input",
            "translation": "english_input"
        })
        df_corr["filipino_output"] = df_corr["english_input"]
        df_corr["origin"] = "MIXED"
        df_corr["context_tag"] = "NEUTRAL"

    # --- Concatenate ---
    df = pd.concat([df_main, df_corr], ignore_index=True)

    # --- Verify required columns ---
    required_cols = {"gloss_input", "english_input", "filipino_output"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Missing required columns. Found: {list(df.columns)}")

    # --- Clean text ---
    for col in ["gloss_input", "english_input", "filipino_output"]:
        df[col] = df[col].astype(str).str.strip()

    # --- Handle optional metadata ---
    if "origin" not in df.columns:
        df["origin"] = "UNKNOWN"
    if "context_tag" not in df.columns:
        df["context_tag"] = "NEUTRAL"

    df = df.dropna(subset=["gloss_input", "english_input", "filipino_output"]).drop_duplicates()
    print(f"✅ Cleaned dataset size: {len(df)} samples")

    # --- Expand bilingual training pairs ---
    records = []
    for _, row in df.iterrows():
        gloss = row["gloss_input"].upper().strip()
        en = row["english_input"].strip()
        fil = row["filipino_output"].strip()
        origin = row["origin"].upper()
        tag = row["context_tag"].upper()

        records.append({"gloss": f"[{tag}] [{origin}] {gloss}", "translation": en})
        records.append({"gloss": f"[{tag}] [{origin}] {gloss}", "translation": fil})

    df_final = pd.DataFrame(records)
    print(f"✅ Final bilingual dataset ready: {len(df_final)} samples")

    return df_final
