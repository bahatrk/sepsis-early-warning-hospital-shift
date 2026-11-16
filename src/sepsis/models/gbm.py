"""Gradyan artırmalı ağaç (LightGBM) taban modeli: zengin, NEDENSEL pencere öznitelikleri + doğrulamada erken durdurma.

Öznitelik grupları (her saat t için yalnızca ≤ t bilgisi; hasta sınırları aşılmaz):
  değer     : son gözlenen (ileri doldurulmuş) z-skorlar (34) + Age, HospAdmTime, ICULOS (3) + ikili (3)           -> 40
  pencere   : 6 saatlik ortalama / min / maks ve 6 saatlik değişim (x_t - x_{t-6})  (34 değişken x 4)               -> 136
  davranış  : "gözlendi" (34), "son ölçüm süresi" (34), son 6 saatteki ölçüm sayısı (34): hastanenin ÖLÇÜM ALIŞKANLIĞINI kodlar -> 102
`davranis=False` sürümü (gbm_davranissiz) üçüncü grubu çıkarır: model yalnızca fizyolojik değerleri görür (Reyna ve ark. 2020'nin
'hastaneye özgü veri toplama davranışı' uyarısının sınaması). Not: ön işlemede eksik değerler z=0 (kaynak ortalaması) ile
doldurulduğundan değer sütunları eksikliği dolaylı olarak hâlâ taşıyabilir; bu bir sınırlılıktır."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sepsis.models.gru import _scores, _acc_at

WIN = 6


def _lagged(a: np.ndarray, j: int) -> np.ndarray:
    """a'nın j satır geriye kaydırılmışı (ilk j satır kendi değeri; geçerlilik maskesi ayrıca uygulanır)."""
    if j == 0: return a
    out = np.empty_like(a); out[j:] = a[:-j]; out[:j] = a[:j]; return out


def build_features(X: np.ndarray, saat: np.ndarray, names: list[str], davranis: bool = True):
    """X: (N, 108) ön işlenmiş satırlar (hasta başına ardışık, saat sırasıyla); saat: hasta içi saat indeksi. -> (F, adlar)"""
    from sepsis.models.grud import grud_indices
    idx = grud_indices(names)
    val, mask, delta = idx["deger"], idx["maske"], idx["delta"]
    Z = X[:, val].astype(np.float32); N, d = Z.shape
    saat = saat.astype(np.int32)
    s = np.zeros_like(Z); mn = np.full_like(Z, np.inf); mx = np.full_like(Z, -np.inf); cnt = np.zeros(N, dtype=np.float32)
    parts = [("deger", Z), ("diger", X[:, idx["diger"]].astype(np.float32))]
    # pencere (yalnızca aynı hastanın önceki saatleri): lag j geçerli <=> saat >= j
    valid_cnt = np.zeros((N, 1), dtype=np.float32)
    for j in range(WIN):
        v = (saat >= j)[:, None]; L = _lagged(Z, j)
        s += np.where(v, L, 0.0); valid_cnt += v
        mn = np.minimum(mn, np.where(v, L, np.inf)); mx = np.maximum(mx, np.where(v, L, -np.inf))
    lag = np.minimum(saat, WIN)                                        # x_t - x_{t-lag}
    dlt = Z - Z[np.arange(N) - lag]
    parts += [("ort6", s / valid_cnt), ("min6", mn), ("maks6", mx), ("delta6", dlt)]
    names_out = ([f"{names[i]}" for i in val] + [names[i] for i in idx["diger"]]
                 + [f"{names[i]}__ort6" for i in val] + [f"{names[i]}__min6" for i in val]
                 + [f"{names[i]}__maks6" for i in val] + [f"{names[i]}__delta6" for i in val])
    if davranis:
        M = X[:, mask].astype(np.float32); c = np.zeros_like(M)
        for j in range(WIN): c += np.where((saat >= j)[:, None], _lagged(M, j), 0.0)
        parts += [("maske", M), ("delta_ol", X[:, delta].astype(np.float32)), ("sayi6", c)]
        names_out += [names[i] for i in mask] + [names[i] for i in delta] + [f"{names[i]}__sayi6" for i in mask]
    F = np.hstack([p for _, p in parts]).astype(np.float32)
    assert F.shape[1] == len(names_out)
    return F, names_out


def train_gbm(Ftr, ytr, Fva, yva, cfg: dict, seed: int, verbose: bool = True):
    """LightGBM'i max_rounds tur eğitir; her `kontrol` turda (eğitim alt örneği, doğrulama) AUPRC/AUROC/kayıp/accuracy kaydeder.
    Seçim: eğitim−doğrulama AUPRC farkı <= gap_limit olan kontrol noktaları arasında en yüksek doğrulama AUPRC
    (hiçbiri sağlanmazsa en küçük farklı nokta); patience kontrol noktası iyileşme olmazsa durur. Yalnızca eğitim ve doğrulama kullanılır."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ytr), size=min(cfg.get("train_eval_rows", 200_000), len(ytr)), replace=False)   # eğitim AUPRC'si için alt örnek
    params = dict(objective="binary", learning_rate=cfg["lr"], num_leaves=cfg["num_leaves"], max_depth=cfg["max_depth"],
                  min_child_samples=cfg["min_child_samples"], feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                  bagging_freq=1, lambda_l2=cfg["lambda_l2"], max_bin=cfg.get("max_bin", 63), seed=seed, n_jobs=cfg.get("threads", 2), verbose=-1,
                  deterministic=True, force_row_wise=True)
    booster = lgb.train(params, lgb.Dataset(Ftr, label=ytr), num_boost_round=cfg["max_rounds"])
    alarm = cfg.get("alarm_rate", 0.05); gap_limit = cfg.get("gap_limit"); step = cfg["kontrol"]
    hist, best, best_it, min_gap, gap_it, wait = [], -1.0, None, 9.0, None, 0
    for k, it in enumerate(range(step, cfg["max_rounds"] + 1, step), 1):
        ptr = booster.predict(Ftr[sub], num_iteration=it); pv = booster.predict(Fva, num_iteration=it); ys = ytr[sub]
        tr_auprc, tr_auroc = _scores(ys, ptr); v_auprc, v_auroc = _scores(yva, pv)
        thr = float(np.quantile(pv, 1 - alarm)); tr_acc, tr_bacc = _acc_at(ys, ptr, thr); v_acc, v_bacc = _acc_at(yva, pv, thr)
        hist.append({"epoch": it, "train_loss": float(log_loss(ys, ptr, labels=[0, 1])), "train_loss_eval": float(log_loss(ys, ptr, labels=[0, 1])),
                     "val_loss": float(log_loss(yva, pv, labels=[0, 1])), "train_auprc": tr_auprc, "val_auprc": v_auprc, "train_auroc": tr_auroc,
                     "val_auroc": v_auroc, "train_acc": tr_acc, "val_acc": v_acc, "train_bacc": tr_bacc, "val_bacc": v_bacc, "esik": thr})
        if verbose: print(f"tur {it:4d} | AUPRC {tr_auprc:.4f} / {v_auprc:.4f} | AUROC {tr_auroc:.3f} / {v_auroc:.3f}", flush=True)
        gap = tr_auprc - v_auprc
        if gap < min_gap: min_gap, gap_it = gap, it
        if (gap_limit is None or gap <= gap_limit) and v_auprc > best: best, best_it, wait = v_auprc, it, 0
        else:
            wait += 1
            if wait >= cfg["patience"]: break
    if best_it is None:
        best_it = gap_it
        if verbose: print(f"UYARI: gap_limit={gap_limit} sağlanmadı; en küçük farklı nokta ({gap_it}) seçildi.")
    booster.best_it = best_it
    return booster, hist
