"""Lojistik regresyon baseline: C yalnızca DOĞRULAMA AUPRC'sine göre seçilir; test kümesi seçimde kullanılmaz."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

def _fit(X, y, C, max_iter):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter).fit(X, y)

def select_and_fit(Xtr, ytr, Xval, yval, c_grid, max_iter):
    scores = {C: average_precision_score(yval, _fit(Xtr, ytr, C, max_iter).predict_proba(Xval)[:, 1]) for C in c_grid}
    best_c = max(scores, key=scores.get)
    return _fit(Xtr, ytr, best_c, max_iter), best_c, scores

def learning_curve(Xtr, ytr, ptr, Xval, yval, C, max_iter, fractions, seed):
    rng = np.random.default_rng(seed)
    patients = np.unique(ptr); rng.shuffle(patients)
    out = []
    for f in fractions:
        m = np.isin(ptr, patients[: int(len(patients) * f)])
        mod = _fit(Xtr[m], ytr[m], C, max_iter)
        out.append((f, average_precision_score(ytr[m], mod.predict_proba(Xtr[m])[:, 1]),
                    average_precision_score(yval, mod.predict_proba(Xval)[:, 1])))
    return out

def alarm_threshold(model, Xval, alarm_rate: float) -> float:
    """Doğrulama saatlerinin alarm_rate oranı alarm üretecek eşik (test kullanılmaz)."""
    return float(np.quantile(model.predict_proba(Xval)[:, 1], 1 - alarm_rate))
