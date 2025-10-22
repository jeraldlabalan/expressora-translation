import random
import pandas as pd
from typing import List, Tuple

# =========================================================
# EXPRESSORA DATA AUGMENTATION UTILS
# =========================================================

def shuffle_gloss_order(gloss: str) -> str:
    """Randomly shuffle tokens inside gloss to simulate sign ordering variance."""
    tokens = gloss.strip().split()
    if len(tokens) > 2:
        random.shuffle(tokens)
    return " ".join(tokens)

def random_synonym(phrase: str, lang: str = "en") -> str:
    """Replace some simple words with synonyms (for English or Filipino)."""
    syn_en = {
        "eat": ["consume", "have", "take"],
        "go": ["walk", "move", "travel"],
        "help": ["assist", "support"],
        "want": ["wish", "desire"],
        "good": ["nice", "great"],
        "bad": ["poor", "awful"],
        "happy": ["glad", "cheerful"],
        "sad": ["unhappy", "down"],
        "scared": ["afraid", "frightened"],
    }
    syn_fil = {
        "kumain": ["magkain", "kainin"],
        "gusto": ["nais", "ibig"],
        "maganda": ["maayos", "mabuti"],
        "tulong": ["saklolo", "assistensya"],
        "masaya": ["maligaya", "magalak"],
        "malungkot": ["nalulungkot", "lungkot"],
    }
    pool = syn_en if lang == "en" else syn_fil
    words = phrase.split()
    out = []
    for w in words:
        low = w.lower().strip(",.")
        if low in pool and random.random() < 0.3:
            out.append(random.choice(pool[low]))
        else:
            out.append(w)
    return " ".join(out)

def augment_pair(gloss: str, en: str, fil: str) -> List[Tuple[str, str, str]]:
    """Generate multiple augmented triples (gloss, en, fil)."""
    results = []
    for _ in range(2):  # two variants per sample
        g2 = shuffle_gloss_order(gloss)
        e2 = random_synonym(en, "en")
        f2 = random_synonym(fil, "fil")
        results.append((g2, e2, f2))
    return results

def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply augmentation to full DataFrame of gloss, english, filipino."""
    aug_records = []
    for _, row in df.iterrows():
        gloss, en, fil = row["gloss_input"], row["english_input"], row["filipino_output"]
        aug_records.append((gloss, en, fil))
        for g2, e2, f2 in augment_pair(gloss, en, fil):
            aug_records.append((g2, e2, f2))
    aug_df = pd.DataFrame(aug_records, columns=["gloss_input", "english_input", "filipino_output"])
    print(f"✅ Augmented dataset size: {len(aug_df)} (from {len(df)})")
    return aug_df
