import numpy as np
import evaluate

# =========================================================
# EXPRESSORA TRANSLATION METRICS
# =========================================================

def _safe_load_metric(name: str):
    """Try to load metric safely, falling back gracefully if unavailable."""
    try:
        return evaluate.load(name)
    except Exception as e:
        print(f"⚠️ Warning: Could not load metric '{name}': {e}")
        return None

metric_bleu = _safe_load_metric("sacrebleu")
metric_rouge = _safe_load_metric("rouge")

def postprocess_text(preds, labels):
    """Clean predictions and labels for proper scoring."""
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]  # sacrebleu expects nested list
    rouge_labels = [l[0] for l in labels]   # rouge expects flat list
    return preds, labels, rouge_labels

def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU, ROUGE, and average generation length."""
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Debug output for first few samples
    print("\n--- Sample Decodes ---")
    for i in range(min(5, len(decoded_preds))):
        print(f"Pred {i+1}: {decoded_preds[i]}")
        print(f"Gold {i+1}: {decoded_labels[i]}")
    print("----------------------\n")

    decoded_preds, bleu_labels, rouge_labels = postprocess_text(decoded_preds, decoded_labels)

    results = {}

    # BLEU
    if metric_bleu:
        bleu = metric_bleu.compute(predictions=decoded_preds, references=bleu_labels)
        results["bleu"] = round(bleu["score"], 4)
    else:
        results["bleu"] = 0.0

    # ROUGE
    if metric_rouge:
        rouge = metric_rouge.compute(predictions=decoded_preds, references=rouge_labels)
        for k, v in rouge.items():
            if isinstance(v, float):
                results[k] = round(v, 4)
    else:
        results["rouge1"] = results["rouge2"] = results["rougeL"] = 0.0

    # Generation length
    gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
    results["gen_len"] = round(np.mean(gen_lens), 4)

    return results
