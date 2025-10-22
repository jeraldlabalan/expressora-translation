import re
import torch
from typing import Dict, List, Literal, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ABBREVIATION_MAP = {"DOC": "DOCTOR", "PLS": "PLEASE"}

CONTEXT_LEXICON = {
    "QUESTION": ["WHO","WHAT","WHERE","WHEN","WHY","HOW","WHICH"],
    "IMPERATIVE": ["STOP","WAIT","COME","GO","HELP","CALL","PLEASE","NEED","GIVE","TELL","OPEN"],
    "EMOTION_NEG": ["SAD","SORRY","ANGRY","TIRED","HURT","PAIN","SICK"],
    "EMOTION_POS": ["HAPPY","THANK","LOVE","EXCITED","GOOD","NICE"],
}

def _expand_abbreviations(text: str) -> str:
    tokens = text.upper().split()
    return " ".join(ABBREVIATION_MAP.get(t, t) for t in tokens)

def _infer_context_tag(gloss_upper: str) -> str:
    toks = set(gloss_upper.split())
    for tag, keys in CONTEXT_LEXICON.items():
        if not toks.isdisjoint(keys):
            return tag
    return "NEUTRAL"

def _polish_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+,", ",", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text

def _polish_filipino(text: str) -> str:
    return text.replace("Pakiusap tumulong", "Pakiusap, tulungan")

class Translator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        print("INFO: Initializing Translator...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Using device: {self.device}")

        # ✅ Try to load tokenizer from checkpoint, else fallback to base mt5 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except Exception as e:
            print(f"⚠️ Tokenizer not found in checkpoint ({e}); loading base mT5 tokenizer instead.")
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("✅ Translator initialized successfully.")

        # Deterministic profile (beam)
        self._style: Literal["beam","sampling"] = "beam"
        self._gen_beam = dict(
            max_new_tokens=80,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
        )
        # Creative profile (sampling)
        self._gen_sample = dict(
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    def set_style(self, mode: Literal["beam","sampling"]="beam", **overrides) -> None:
        if mode not in ("beam","sampling"):
            raise ValueError("mode must be 'beam' or 'sampling'")
        self._style = mode
        if overrides:
            if mode == "beam":
                self._gen_beam.update(overrides)
            else:
                self._gen_sample.update(overrides)

    def translate(self, gloss_sequence: str, origin: str = "ASL") -> Dict[str, str]:
        if not gloss_sequence or not gloss_sequence.strip():
            return {"english": "", "filipino": ""}
        gloss_clean = _expand_abbreviations(gloss_sequence.strip())
        ctx = _infer_context_tag(gloss_clean)
        base = f"[{ctx}] [{origin.upper()}] {gloss_clean}"
        en  = self._generate(f"translate to English: {base}")
        fil = self._generate(f"translate to Filipino: {base}")
        return {
            "english": _polish_sentence(en),
            "filipino": _polish_sentence(_polish_filipino(fil))
        }

    def translate_batch(self, gloss_list: List[str], origin: str = "ASL") -> List[Dict[str, str]]:
        return [self.translate(g, origin=origin) for g in gloss_list]

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_cfg = self._gen_beam if self._style == "beam" else self._gen_sample
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **gen_cfg)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
