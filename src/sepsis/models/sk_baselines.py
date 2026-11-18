"""Ek baseline'lar (scikit-learn): lojistik regresyon, random forest, ExtraTrees, MLP. Hepsi LightGBM ile AYNI nedensel pencere
özniteliklerini (gbm.build_features, davranış sütunları DAHİL = standart karşılaştırma) kullanır.

Aşırı öğrenme protokolü (GRU/GBM ile aynı ölçüt): her model için küçük, önceden belirlenmiş bir düzenlileştirme ızgarası denenir;
her aday için eğitim ve doğrulama AUPRC'si ölçülür; eğitim−doğrulama farkı <= gap_limit olan adaylar arasında en yüksek
doğrulama AUPRC'li aday seçilir (hiçbiri sağlamazsa en küçük farklı aday). Test ve adaptasyon kümeleri kullanılmaz.
'Eğri' = ızgara boyunca eğitim/doğrulama ölçütleri (bu modellerde epoch yok; x ekseni = aday sırası, düzenlileştirme azalan sırada)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss
from sepsis.models.gru import _scores, _acc_at

IZGARA = {   # düzenlileştirme GÜÇLÜDEN ZAYIFA (eğri soldan sağa karmaşıklık artar)
    "lr":  [{"C": 1e-4}, {"C": 1e-3}, {"C": 1e-2}, {"C": 1e-1}],
    "rf":  [{"min_samples_leaf": 5000}, {"min_samples_leaf": 2000}, {"min_samples_leaf": 500}],
    "et":  [{"min_samples_leaf": 5000}, {"min_samples_leaf": 2000}, {"min_samples_leaf": 500}],
    "mlp": [{"alpha": 100.0}, {"alpha": 30.0}, {"alpha": 10.0}, {"alpha": 3.0}],   # ilk ızgara (1, 0,1, 0,01) 0,05 ölçütünü hiçbir koşuda sağlamadı (fark 0,14–0,28)
}


def _make(name: str, p: dict, seed: int, cfg: dict):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n_jobs = cfg.get("threads", 2)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=cfg.get("lr_max_iter", 300), solver="lbfgs"))
    if name in ("rf", "et"):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        K = RandomForestClassifier if name == "rf" else ExtraTreesClassifier
        return K(n_estimators=cfg.get("n_estimators", 100), min_samples_leaf=p["min_samples_leaf"], max_features="sqrt",
                 bootstrap=True, max_samples=cfg.get("max_samples", 0.2), n_jobs=n_jobs, random_state=seed)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=p["alpha"], batch_size=1024,
                             learning_rate_init=1e-3, max_iter=cfg.get("mlp_max_iter", 30), early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=3, random_state=seed))
    raise ValueError(name)


class _Model:
    def __init__(self, m): self.m = m
    def predict(self, F): return self.m.predict_proba(F)[:, 1]


def train_sk(name: str, Ftr, ytr, Fva, yva, cfg: dict, seed: int, verbose: bool = True):
    alarm = cfg.get("alarm_rate", 0.05); gap_limit = cfg.get("gap_limit", 0.05)
    hist, fitted = [], []
    for i, p in enumerate(IZGARA[name], 1):
        m = _make(name, p, seed, cfg).fit(Ftr, ytr)
        ptr, pv = m.predict_proba(Ftr)[:, 1], m.predict_proba(Fva)[:, 1]
        tr_auprc, tr_auroc = _scores(ytr, ptr); v_auprc, v_auroc = _scores(yva, pv)
        thr = float(np.quantile(pv, 1 - alarm)); tr_acc, tr_bacc = _acc_at(ytr, ptr, thr); v_acc, v_bacc = _acc_at(yva, pv, thr)
        hist.append({"epoch": i, "ayar": p, "train_loss": float(log_loss(ytr, ptr, labels=[0, 1])), "train_loss_eval": float(log_loss(ytr, ptr, labels=[0, 1])),
                     "val_loss": float(log_loss(yva, pv, labels=[0, 1])), "train_auprc": tr_auprc, "val_auprc": v_auprc, "train_auroc": tr_auroc,
                     "val_auroc": v_auroc, "train_acc": tr_acc, "val_acc": v_acc, "train_bacc": tr_bacc, "val_bacc": v_bacc, "esik": thr})
        fitted.append(m)
        if verbose: print(f"[{name}] aday {i} {p} | AUPRC {tr_auprc:.4f} / {v_auprc:.4f} | fark {tr_auprc - v_auprc:.3f}", flush=True)
    uygun = [h for h in hist if h["train_auprc"] - h["val_auprc"] <= gap_limit]
    sec = max(uygun, key=lambda h: h["val_auprc"]) if uygun else min(hist, key=lambda h: h["train_auprc"] - h["val_auprc"])
    if not uygun and verbose: print(f"UYARI: gap_limit={gap_limit} hiçbir adayda sağlanmadı; en küçük farklı aday seçildi.")
    model = _Model(fitted[sec["epoch"] - 1]); model.best_epoch = sec["epoch"]; model.ayar = sec["ayar"]
    return model, hist
