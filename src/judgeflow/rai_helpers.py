"""Responsible-AI helper functions"""
import numpy as np
from typing import List, Union
import re
import warnings
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

# Detoxify: lazy load
_detox_model = None

def _get_detox():
    global _detox_model
    if _detox_model is None:
        from detoxify import Detoxify
        _detox_model = Detoxify('original')
    return _detox_model

# spaCy: try to load, fallback to regex only
try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
except Exception as e:
    _nlp = None
    warnings.warn(f"spaCy model could not be loaded: {e}. Falling back to regex-only PII detection.")

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
    model = _get_detox()
    scores = model.predict(text)
    return float(scores.get('toxicity', 0.0))

# 3. spaCy + regex PII detection

PII_REGEXES = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{9,}\b"),  # Long numbers (SSN, credit card, etc.)
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
        try:
            doc = _nlp(text)
            for ent in doc.ents:
                if ent.label_ in PII_ENTS:
                    found = True
                    break
        except Exception as e:
            warnings.warn(f"spaCy NER failed: {e}. Using regex-only.")
    return 1.0 if found else 0.0

# Calibration gap (probability calibration)

def demographic_parity(y_true, y_pred, sensitive):
    """Return (difference, ratio)."""
    diff = demographic_parity_difference(y_true, y_pred, sensitive)
    ratio = demographic_parity_ratio(y_true, y_pred, sensitive)
    return {"dp_diff": float(diff), "dp_ratio": float(ratio)}

def equal_opportunity(y_true, y_pred, sensitive):
    """Return (difference, ratio) of TPRs."""
    diff = equalized_odds_difference(y_true, y_pred, sensitive, method='threshold')
    ratio = equalized_odds_ratio(y_true, y_pred, sensitive, method='threshold')
    return {"eo_diff": float(diff), "eo_ratio": float(ratio)}

def calibration_gap(
    y_true: List[int],
    y_prob: List[float],
    sensitive_attr: List[Union[str, int]],
    n_bins: int = 10
) -> dict:
    """
    Per-group calibration gap: max over groups of | accuracy − confidence |.
    Args
    ----
    y_true : binary ground-truth labels (0 / 1)
    y_prob : predicted probability for the positive class (float 0-1)
    sensitive_attr : group label for each sample
    n_bins : number of reliability bins
    Returns
    -------
    dict with {'calib_gap': float, 'group_gaps': dict}
    """
    y_true  = np.asarray(y_true)
    y_prob  = np.asarray(y_prob)
    groups  = np.asarray(sensitive_attr)

    unique_groups = np.unique(groups)
    group_gaps = {}

    # bin boundaries
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for g in unique_groups:
        mask  = groups == g
        if mask.sum() == 0:
            continue
        prob_g = y_prob[mask]
        true_g = y_true[mask]

        # reliability diagram in n_bins
        digitized = np.digitize(prob_g, bins) - 1
        max_gap = 0.0
        for b in range(n_bins):
            bin_mask = digitized == b
            if bin_mask.sum() == 0:
                continue
            conf = prob_g[bin_mask].mean()
            acc  = true_g[bin_mask].mean()
            max_gap = max(max_gap, abs(acc - conf))
        group_gaps[g] = float(max_gap)

    overall_gap = float(max(group_gaps.values()) if group_gaps else 0.0)
    return {"calib_gap": overall_gap, "group_gaps": group_gaps}

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