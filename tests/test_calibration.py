import numpy as np
from sepsis.evaluation import calibration as C


def _sentetik(n, pi, rng, ayrim=1.5, sismis=4.0):
    """y ~ Bernoulli(pi); gerçek log-oran skoru ayrim ile ayrışır; model çıktısı 'sismis' kadar yukarı kaydırılmış (pos_weight etkisi)."""
    y = (rng.random(n) < pi).astype(int)
    z = rng.normal(0, 1, n) + ayrim * y
    p_dogru = C.sigmoid(ayrim * z - ayrim ** 2 / 2 + np.log(pi / (1 - pi)))     # N(0,1)/N(ayrim,1) için gerçek posterior
    p_model = C.sigmoid(C.logit(p_dogru) + sismis)
    return y, p_model


def test_platt_ece_duser():
    rng = np.random.default_rng(0)
    y, p = _sentetik(200_000, 0.02, rng)
    a, b = C.platt_fit(y, p)
    assert C.ece(y, C.platt_apply(p, a, b)) < 0.2 * C.ece(y, p)
    assert abs(C.platt_apply(p, a, b).mean() - y.mean()) < 0.002


def test_siralama_degismez():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y, p = _sentetik(20_000, 0.05, rng)
    a, b = C.platt_fit(y, p)
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, C.platt_apply(p, a, b))) < 1e-9
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, C.prior_shift_adjust(C.platt_apply(p, a, b), 0.05, 0.1))) < 1e-9


def test_em_hedef_oranini_kestirir_ve_brier_duser():
    rng = np.random.default_rng(2)
    yk, pk = _sentetik(200_000, 0.014, rng)                 # kaynak
    # hedef: oncul 0.03'e kaymış; sınıflayıcı (kaynak öncülüne göre kurulmuş posterior) aynı
    yh = (rng.random(300_000) < 0.03).astype(int); z = rng.normal(0, 1, 300_000) + 1.5 * yh
    ph = C.sigmoid(C.logit(C.sigmoid(1.5 * z - 1.5 ** 2 / 2 + np.log(0.014 / 0.986))) + 4.0)
    pa = ph[:100_000]; yt, pt = yh[100_000:], ph[100_000:]
    rep = C.calibration_report(yk, pk, yt, pt, pa)
    assert abs(rep["oncul"]["hedef_em"] - 0.03) < 0.004
    assert rep["hedef_test"]["platt_em"]["brier"] <= rep["hedef_test"]["platt"]["brier"] + 1e-6
    assert rep["hedef_test"]["platt_em"]["ece"] < rep["hedef_test"]["ham"]["ece"]
    assert "ECE" in C.calibration_markdown(rep)


def test_ece_mukemmel_kalibre():
    rng = np.random.default_rng(3)
    p = rng.random(100_000) ** 3; y = (rng.random(100_000) < p).astype(int)
    assert C.ece(y, p) < 0.01
