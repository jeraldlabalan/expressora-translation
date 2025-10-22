import argparse
import os
import torch
import pandas as pd

from src.training.train import train_model
from src.preprocessing.data_loader import load_expressora_datasets, prepare_dataframe_for_training
from src.inference.translator import Translator
from src.evaluation.metrics import compute_metrics

# =========================================================
# EXPRESSORA MAIN LAUNCHER
# =========================================================

BASE_DIR = "/content/drive/MyDrive/college-papers/Thesis/Expressora/expressora-translation"
MODEL_DIR = "/content/drive/MyDrive/college-papers/Thesis/Expressora/expressora-models"
FINAL_MODEL = os.path.join(MODEL_DIR, "expressora-mt5", "final-model")

def run_training():
    """Train a new Expressora translation model."""
    print("\n🧠 Starting Expressora training pipeline...\n")
    train_model(
        base_dir=BASE_DIR,
        output_name="expressora-mt5",
        train_frac=0.9,
        epochs=8,
        batch_size=4,
        lr=2e-4,
        max_len=128,
    )

def run_evaluation():
    """Run evaluation on a sample set using the trained model."""
    print("\n🔍 Evaluating Expressora model...\n")

    df_main, df_corr = load_expressora_datasets(BASE_DIR)
    df_all = prepare_dataframe_for_training(df_main, df_corr)

    model_path = FINAL_MODEL
    tr = Translator(model_path)
    tr.set_style("beam")

    sample_df = df_all.sample(n=min(50, len(df_all)), random_state=42)
    preds, refs = [], []

    for _, row in sample_df.iterrows():
        gloss = row["gloss"]
        ref = row["translation"]
        out = tr._generate(f"translate gloss to english and filipino: {gloss}")
        preds.append(out)
        refs.append(ref)

    print("\n--- Sample predictions ---")
    for i in range(3):
        print(f"GLOSS: {sample_df.iloc[i]['gloss']}")
        print(f"REF  : {refs[i]}")
        print(f"PRED : {preds[i]}\n")

    from evaluate import load
    bleu = load("sacrebleu").compute(predictions=preds, references=[[r] for r in refs])["score"]
    rouge = load("rouge").compute(predictions=preds, references=refs)
    print(f"✅ BLEU: {bleu:.2f}")
    print(f"✅ ROUGE-L: {rouge['rougeL']:.2f}")

def run_translation(gloss_input: str):
    """Translate a gloss string using the latest trained model."""
    print("\n🗣️  Translating gloss input...\n")

    if not os.path.exists(FINAL_MODEL):
        raise FileNotFoundError(f"❌ Model not found at {FINAL_MODEL}")

    tr = Translator(FINAL_MODEL)
    result = tr.translate(gloss_input)
    print("\n--- Translation Output ---")
    print(f"Gloss Input : {gloss_input}")
    print(f"English     : {result['english']}")
    print(f"Filipino    : {result['filipino']}")

def main():
    parser = argparse.ArgumentParser(description="Expressora Translation Module CLI")
    parser.add_argument("--train", action="store_true", help="Train the Expressora model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on dataset")
    parser.add_argument("--translate", type=str, help="Translate a gloss string")
    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.evaluate:
        run_evaluation()
    elif args.translate:
        run_translation(args.translate)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
