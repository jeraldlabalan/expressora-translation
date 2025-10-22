import os
import math
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from src.preprocessing.data_loader import load_expressora_datasets, prepare_dataframe_for_training
from src.evaluation.metrics import compute_metrics

def _tokenize_pairs(df: pd.DataFrame, tokenizer: T5Tokenizer, max_len: int = 128):
    inputs = [f"translate gloss to english and filipino: {g}" for g in df["gloss"]]
    targets = df["translation"].tolist()
    model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_len, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(
    base_dir: str = "/content/drive/MyDrive/college-papers/Thesis/Expressora/expressora-translation",
    output_name: str = "expressora-mt5",
    train_frac: float = 0.9,
    epochs: int = 8,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_len: int = 128,
):
    os.makedirs(os.path.join(base_dir, "models", output_name), exist_ok=True)

    # 1) Load + normalize data
    df_main, df_corr = load_expressora_datasets(base_dir)
    df_all = prepare_dataframe_for_training(df_main, df_corr)

    # 2) Split
    train_df = df_all.sample(frac=train_frac, random_state=42)
    val_df = df_all.drop(train_df.index)
    print(f"INFO: Split -> train: {len(train_df)} | val: {len(val_df)}")

    # 3) Model & tokenizer
    print("INFO: Loading tokenizer and model (google/mt5-small)…")
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    # 4) Datasets (HuggingFace)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenized_train = train_ds.map(lambda e: _tokenize_pairs(pd.DataFrame(e), tokenizer, max_len), batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(lambda e: _tokenize_pairs(pd.DataFrame(e), tokenizer, max_len), batched=True, remove_columns=val_ds.column_names)
    dsd = DatasetDict({"train": tokenized_train, "validation": tokenized_val})

    # 5) Trainer setup
    save_dir = os.path.join(base_dir, "models", output_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    fp16 = torch.cuda.is_available()
    bf16 = False
    if not fp16 and torch.cuda.is_available():
        bf16 = True

    args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=fp16,
        bf16=bf16,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",          # keep W&B off by default
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    # 6) Train
    print("INFO: Starting training…")
    trainer.train()

    # 7) Final eval
    print("INFO: Final evaluation…")
    metrics = trainer.evaluate()
    print(metrics)

    # 8) Save best/final
    final_path = os.path.join(save_dir, "final-model")
    print(f"INFO: Saving final model → {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("✅ Done.")

if __name__ == "__main__":
    train_model()
