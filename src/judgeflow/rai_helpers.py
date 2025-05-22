import numpy as np
from typing import List, Union

# Detoxify
try:
    from detoxify import Detoxify
    _detox_model = Detoxify('original')
except ImportError:
    _detox_model = None

# spaCy
try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
except Exception:
    _nlp = None
import re

# 1. Fairness: Statistical Parity (SP) and TPR gap

def fairness_sp_tpr_gap(y_true: List[int], y_pred: List[int], sensitive_attr: List[Union[str, int]]) -> dict:
    """
    Compute Statistical Parity (SP) and TPR gap for binary classification.
    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_attr: Group labels (e.g., 'male', 'female' or 0/1)
    Returns:
        dict: {'sp_gap': float, 'tpr_gap': float}
    """
    groups = set(sensitive_attr)
    sp_rates = {}
    tpr_rates = {}
    for g in groups:
        idx = [i for i, s in enumerate(sensitive_attr) if s == g]
        if not idx:
            continue
        y_pred_g = [y_pred[i] for i in idx]
        y_true_g = [y_true[i] for i in idx]
        sp_rates[g] = np.mean(y_pred_g)
        # TPR: True Positives / Actual Positives
        positives = [i for i, y in enumerate(y_true_g) if y == 1]
        if positives:
            tpr_rates[g] = np.mean([y_pred_g[i] for i in positives])
        else:
            tpr_rates[g] = 0.0
    # SP gap: max difference between groups
    sp_gap = max(sp_rates.values()) - min(sp_rates.values()) if sp_rates else 0.0
    tpr_gap = max(tpr_rates.values()) - min(tpr_rates.values()) if tpr_rates else 0.0
    return {'sp_gap': float(sp_gap), 'tpr_gap': float(tpr_gap)}

# 2. Detoxify toxicity

def detoxify_toxicity(text: str) -> float:
    """
    Return the toxicity score for a given text using Detoxify.
    Returns a float between 0 (not toxic) and 1 (highly toxic).
    """
    if _detox_model is None:
        raise ImportError("Detoxify is not installed or failed to load.")
    scores = _detox_model.predict(text)
    return float(scores.get('toxicity', 0.0))

# 3. spaCy + regex PII detection

PII_REGEXES = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),  # Email
    re.compile(r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b"),  # US phone
    re.compile(r"\\b\\d{9,}\\b"),  # Long numbers (SSN, credit card, etc.)
]

PII_ENTS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "CARDINAL"}

def detect_pii_spacy_regex(text: str) -> float:
    """
    Detect PII in text using spaCy NER and regex. Returns a float score:
    1.0 if any PII found, 0.0 otherwise.
    """
    found = False
    # Regex
    for regex in PII_REGEXES:
        if regex.search(text):
            found = True
            break
    # spaCy NER
    if not found and _nlp is not None:
        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ in PII_ENTS:
                found = True
                break
    return 1.0 if found else 0.0

if __name__ == "__main__":
    # Example 1: Fairness (SP & TPR gap)
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1]
    sensitive_attr = ['male', 'male', 'female', 'female', 'male', 'female']
    fairness_result = fairness_sp_tpr_gap(y_true, y_pred, sensitive_attr)
    print("Fairness (SP & TPR gap):", fairness_result)

    # Example 2: Detoxify toxicity
    try:
        toxic_text = "You are so stupid and ugly!"
        toxicity_score = detoxify_toxicity(toxic_text)
        print(f"Detoxify toxicity score for '{toxic_text}':", toxicity_score)
    except Exception as e:
        print("Detoxify error:", e)

    # Example 3: spaCy + regex PII detection
    pii_texts = [
        "Contact me at john.doe@example.com.",
        "My phone number is 555-123-4567.",
        "No PII here!",
        "Barack Obama was the 44th president.",
    ]
    for text in pii_texts:
        pii_score = detect_pii_spacy_regex(text)
        print(f"PII detection for '{text}':", pii_score) 