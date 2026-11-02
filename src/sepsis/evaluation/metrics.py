"""Değerlendirme metrikleri."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def discrimination(y, p) -> dict:
    return {"AUROC": round(float(roc_auc_score(y, p)), 4), "AUPRC": round(float(average_precision_score(y, p)), 4),
            "pozitif_orani": round(float(np.mean(y)), 4)}
