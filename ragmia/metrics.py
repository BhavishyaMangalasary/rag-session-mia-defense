"""Evaluation metrics shared by all conditions."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(labels, scores, max_fpr):
    """Share of members caught while wrongly accusing at most `max_fpr` of non-members."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def auc_ci(labels, scores, n_boot=1000, seed=0):
    """95% bootstrap confidence interval for AUC (our sample is small, so we report the spread)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng, aucs = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        if labels[i].min() != labels[i].max():
            aucs.append(roc_auc_score(labels[i], scores[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def attack_metrics(labels, scores):
    lo, hi = auc_ci(labels, scores)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auc_95ci": [lo, hi],
        "tpr@1%fpr": tpr_at_fpr(labels, scores, 0.01),
        "tpr@10%fpr": tpr_at_fpr(labels, scores, 0.10),
    }