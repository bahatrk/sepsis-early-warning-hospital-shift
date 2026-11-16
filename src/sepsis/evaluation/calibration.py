"""Olasılık kalibrasyonu: kaynak hastanede Platt ölçekleme + hedef hastanede ETİKETSİZ öncül (prevalans) kayması düzeltmesi.

Neden gerekli: model ağırlıklı kayıpla (pos_weight ≈ negatif/pozitif) eğitildiği için ham çıktı olasılık gibi okunamaz.
Üstüne hedef hastanenin sepsis oranı farklıdır (öncül kayması). İki adım:
  1) Platt: kaynak DOĞRULAMA kümesinde (etiketli) lojistik ölçekleme  p_cal = sigmoid(a * logit(p) + b).
  2) Öncül düzeltmesi (Saerens, Latinne, Decaestecker 2002, EM): hedef hastanenin etiketsiz `adaptasyon` kümesindeki
     skorlardan hedef sepsis oranı EM ile kestirilir, olasılıklar buna göre yeniden ağırlıklandırılır.
Sıralama (AUROC/AUPRC) iki adımda da değişmez; yalnızca olasılıkların doğruluğu (ECE, Brier) değişir.
Hedef test etiketleri yöntem seçmek ya da parametre kestirmek için KULLANILMAZ; yalnızca raporlama içindir."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def platt_fit(y, p) -> tuple[float, float]:
    """(a, b): p_cal = sigmoid(a*logit(p)+b). Düzenlileştirme yok (C büyük); tek özellikli lojistik regresyon."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(p, a: float, b: float) -> np.ndarray:
    return sigmoid(a * logit(p) + b)


def prior_shift_adjust(p, pi_kaynak: float, pi_hedef: float) -> np.ndarray:
    """Kaynak öncülüne göre kalibre olasılıkları hedef öncülüne taşır (Bayes yeniden ağırlıklandırma)."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    w1, w0 = pi_hedef / pi_kaynak, (1 - pi_hedef) / (1 - pi_kaynak)
    return w1 * p / (w1 * p + w0 * (1 - p))


def em_prior(p_cal, pi_kaynak: float, n_iter: int = 200, tol: float = 1e-8) -> float:
    """Etiketsiz hedef verisinden hedef sepsis oranını EM ile kestirir (Saerens ve ark. 2002). p_cal: KAYNAKTA kalibre olasılıklar."""
    pi = float(pi_kaynak)
    for _ in range(n_iter):
        yeni = float(prior_shift_adjust(p_cal, pi_kaynak, pi).mean())
        yeni = min(max(yeni, EPS), 1 - EPS)
        if abs(yeni - pi) < tol:
            pi = yeni; break
        pi = yeni
    return pi


def ece(y, p, n_bins: int = 15) -> float:
    """Beklenen kalibrasyon hatası, eşit-frekanslı kutular (sepsis nadir olduğundan eşit genişlik kutular çoğu boş kalır)."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p); bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)) / len(y))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def metrics(y, p) -> dict:
    y = np.asarray(y)
    pi = float(y.mean()); b = brier(y, p); b_ref = brier(y, np.full(len(y), pi))
    return {"ece": ece(y, p), "brier": b, "brier_beceri": 1 - b / b_ref if b_ref > 0 else float("nan"),
            "ortalama_tahmin": float(np.mean(p)), "gercek_oran": pi}


def calibration_report(y_val, p_val, y_hedef, p_hedef, p_adapt, y_kaynak_test=None, p_kaynak_test=None) -> dict:
    """Dört biçim: ham | platt (yalnızca kaynak kalibrasyonu) | platt_em (+ etiketsiz öncül düzeltmesi) | platt_oracle_oncul
    (tanı amaçlı: gerçek hedef oranını bilseydik; etiket kullandığından yöntem DEĞİL, EM kestiriminin kalitesini ölçer)."""
    a, b = platt_fit(y_val, p_val)
    pi_s = float(np.mean(y_val))
    pa_cal = platt_apply(p_adapt, a, b)
    pi_em = em_prior(pa_cal, pi_s)
    pt_cal = platt_apply(p_hedef, a, b)
    pi_gercek = float(np.mean(y_hedef))
    out = {"platt": {"a": a, "b": b}, "oncul": {"kaynak": pi_s, "hedef_em": pi_em, "hedef_gercek_tani": pi_gercek},
           "hedef_test": {
               "ham": metrics(y_hedef, p_hedef),
               "platt": metrics(y_hedef, pt_cal),
               "platt_em": metrics(y_hedef, prior_shift_adjust(pt_cal, pi_s, pi_em)),
               "platt_oracle_oncul": metrics(y_hedef, prior_shift_adjust(pt_cal, pi_s, pi_gercek))}}
    if y_kaynak_test is not None:
        out["kaynak_ici_test"] = {"ham": metrics(y_kaynak_test, p_kaynak_test),
                                  "platt": metrics(y_kaynak_test, platt_apply(p_kaynak_test, a, b))}
    return out


def calibration_markdown(cal: dict) -> str:
    o = cal["oncul"]
    s = ["### Olasılık kalibrasyonu", "",
         "Platt ölçekleme kaynak doğrulamasında, öncül kestirimi (EM) hedefin etiketsiz `adaptasyon` kümesinde yapılır. "
         "Hedef test etiketleri yalnızca raporlama içindir. `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır), yöntem değildir.", "",
         f"Sepsis oranı: kaynak {o['kaynak']:.4f} | hedef, EM kestirimi {o['hedef_em']:.4f} | hedef, gerçek (tanı) {o['hedef_gercek_tani']:.4f}", "",
         "| Küme | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Gerçek oran |", "|---|---|---|---|---|---|---|"]
    for kume in ("kaynak_ici_test", "hedef_test"):
        for ad, m in cal.get(kume, {}).items():
            s.append(f"| {kume} | {ad} | {m['ece']:.4f} | {m['brier']:.4f} | {m['brier_beceri']:.3f} | {m['ortalama_tahmin']:.4f} | {m['gercek_oran']:.4f} |")
    return "\n".join(s) + "\n"
