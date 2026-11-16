"""Uçtan uca: sahte veriyle bölme -> ön işleme -> 06_train_model (iki model) -> beklenen çıktı dosyaları."""
import json, os, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def test_train_script_produces_all_outputs(raw_dir, tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    cfg["paths"].update(raw_dir=raw_dir, split_csv=str(tmp_path / "bolme.csv"), processed_dir=str(tmp_path / "proc"),
                        artifacts_dir=str(tmp_path / "art"), results_dir=str(tmp_path / "res"))
    for k in ("gru", "grud"): cfg[k].update(hidden=4, max_epochs=2, patience=2, batch_size=32, threads=1)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {**os.environ, "SEPSIS_CONFIG": str(tmp_path / "cfg.yaml"), "PYTHONPATH": str(ROOT / "src")}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    run("scripts/01_make_split.py"); run("scripts/02_preprocess.py", "A->B")
    for model in ("grud", "gru_ozelliksiz", "gru_maskdrop"):
        run("scripts/06_train_model.py", "A->B", "--model", model, "--seed", "0")
        d = tmp_path / "res" / "modeller" / f"{model}_A_B_seed0"
        for f in ["sonuc.json", "tablo.md", "kesisim.md", "egriler.png", "roc_pr.png", "karisiklik_kaynak_ici_test.png",
                  "karisiklik_hedef_test.png", "rapor_dogrulama.md", "rapor_hedef_test.md",
                  "rapor_hedef_test_uyarlanmis_esik.md", "karisiklik_hedef_test_uyarlanmis_esik.png", "rapor_kalibrasyon.md", "skorlar.npz"]:
            assert (d / f).exists(), f"{model}: {f} yok"
        r = json.load(open(d / "sonuc.json", encoding="utf-8"))
        assert r["n_ozellik"] == (40 if model == "gru_ozelliksiz" else 108)
        assert {"dogrulama_esigi", "uyarlanmis_esik"} <= set(r["hedef_uyarlama"])
        assert {"ham", "platt", "platt_em", "platt_oracle_oncul"} <= set(r["kalibrasyon"]["hedef_test"])
        assert "Toplam kesişen hasta: 0" in (d / "kesisim.md").read_text(encoding="utf-8")
        assert (tmp_path / "art" / model / "A_B" / "seed0.pt").exists()


def test_sweep_uses_only_train_and_val(raw_dir, tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    cfg["paths"].update(raw_dir=raw_dir, split_csv=str(tmp_path / "bolme.csv"), processed_dir=str(tmp_path / "proc"),
                        artifacts_dir=str(tmp_path / "art"), results_dir=str(tmp_path / "res"))
    for k in ("gru", "grud"): cfg[k].update(hidden=4, max_epochs=2, patience=2, batch_size=32, threads=1)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {**os.environ, "SEPSIS_CONFIG": str(tmp_path / "cfg.yaml"), "PYTHONPATH": str(ROOT / "src")}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    run("scripts/01_make_split.py"); run("scripts/02_preprocess.py", "A->B")
    for f in ("kaynak_ici_test", "hedef_test", "adaptasyon"):               # test dosyaları yoksa betik yine çalışmalı
        p = tmp_path / "proc" / "A_B" / f"{f}.npz"
        if p.exists(): p.unlink()
    for model in ("gru", "grud"):
        run("scripts/08_regularization_sweep.py", "--model", model, "--yonler", "A->B", "--seeds", "0",
            "--dropout", "0.2", "--weight-decay", "1e-3", "--gap-limit", "none", "0.05")
        r = json.load(open(tmp_path / "res" / "tarama" / f"{model}_A_B.json", encoding="utf-8"))
        assert len(r["tum"]) == 2 and "secilen" in r
        assert (tmp_path / "res" / "tarama" / f"{model}_A_B.md").exists()


def test_hedef_norm_degistirir_yalniz_hedefi(raw_dir, tmp_path):
    import numpy as np
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    cfg["paths"].update(raw_dir=raw_dir, split_csv=str(tmp_path / "bolme.csv"), processed_dir=str(tmp_path / "proc"),
                        artifacts_dir=str(tmp_path / "art"), results_dir=str(tmp_path / "res"))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg2 = dict(cfg); cfg2["paths"] = dict(cfg["paths"], processed_dir=str(tmp_path / "proc_hn"))
    (tmp_path / "cfg2.yaml").write_text(yaml.safe_dump(cfg2), encoding="utf-8")
    env = lambda c: {**os.environ, "SEPSIS_CONFIG": str(tmp_path / c), "PYTHONPATH": str(ROOT / "src")}
    run = lambda c, *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env(c), check=True, capture_output=True, text=True)
    run("cfg.yaml", "scripts/01_make_split.py"); run("cfg.yaml", "scripts/02_preprocess.py", "A->B")
    run("cfg2.yaml", "scripts/02_preprocess.py", "A->B", "--hedef-norm", "adaptasyon")
    a, b = tmp_path / "proc" / "A_B", tmp_path / "proc_hn" / "A_B"
    for k in ("egitim", "dogrulama", "kaynak_ici_test"):                       # kaynak kümeleri birebir aynı
        assert np.array_equal(np.load(a / f"{k}.npz")["X"], np.load(b / f"{k}.npz")["X"])
    for k in ("adaptasyon", "hedef_test"):                                     # hedef kümeleri değişir
        assert not np.array_equal(np.load(a / f"{k}.npz")["X"], np.load(b / f"{k}.npz")["X"])
    assert (b / "kayma_tanisi.md").exists() and (b / "parametreler_hedef.json").exists()
    # ölçülmeyen sütunlar (Age, ikili, gösterge) aynı kalır
    names = json.load(open(a / "ozellikler.json")); xa, xb = np.load(a / "hedef_test.npz")["X"], np.load(b / "hedef_test.npz")["X"]
    for c in ("Age", "Gender", "HR_gozlendi", "HR_son_olcum_suresi"):
        j = names.index(c); assert np.array_equal(xa[:, j], xb[:, j])
    # adaptasyon kümesinde hedef istatistiğiyle standartlaştırılan ölçülen değişkenin ortalaması ~0 (kırpma/ffill payı)
    j = names.index("HR"); assert abs(np.load(b / "adaptasyon.npz")["X"][:, j].mean()) < 0.15


def test_gbm_modelleri_ve_pencere_ozellikleri(raw_dir, tmp_path):
    import numpy as np
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    cfg["paths"].update(raw_dir=raw_dir, split_csv=str(tmp_path / "bolme.csv"), processed_dir=str(tmp_path / "proc"),
                        artifacts_dir=str(tmp_path / "art"), results_dir=str(tmp_path / "res"))
    cfg["gbm"].update(max_rounds=20, kontrol=10, patience=2, min_child_samples=5, threads=1)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {**os.environ, "SEPSIS_CONFIG": str(tmp_path / "cfg.yaml"), "PYTHONPATH": str(ROOT / "src")}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    run("scripts/01_make_split.py"); run("scripts/02_preprocess.py", "A->B")
    boy = {}
    for m in ("gbm", "gbm_davranissiz"):
        run("scripts/06_train_model.py", "A->B", "--model", m, "--seed", "0")
        o = tmp_path / "res" / "modeller" / f"{m}_A_B_seed0"
        for f in ["sonuc.json", "egriler.png", "roc_pr.png", "karisiklik_hedef_test.png", "rapor_hedef_test.md", "tablo.md", "kesisim.md",
                  "rapor_hedef_test_uyarlanmis_esik.md", "rapor_kalibrasyon.md", "skorlar.npz"]:
            assert (o / f).exists(), (m, f)
        boy[m] = json.load(open(o / "sonuc.json"))["n_ozellik"]
    assert boy["gbm"] == boy["gbm_davranissiz"] + 102 and boy["gbm_davranissiz"] == 176


def test_pencere_ozellikleri_nedensel_ve_hasta_sinirina_saygili():
    import numpy as np
    from sepsis.data.preprocess import feature_names
    from sepsis.models.gbm import build_features
    names = feature_names(True); rng = np.random.default_rng(0)
    n1, n2 = 9, 7; X = rng.normal(size=(n1 + n2, len(names))).astype(np.float32); saat = np.r_[np.arange(n1), np.arange(n2)]
    F, adlar = build_features(X, saat, names, True)
    i = adlar.index("HR__ort6"); j = adlar.index("HR"); col = X[:, j]
    assert np.isclose(F[n1 + 2, i], col[n1:n1 + 3].mean())            # 2. hasta: yalnızca kendi 3 saati (hasta sınırı aşılmaz)
    assert np.isclose(F[8, i], col[3:9].mean())                        # tam 6 saatlik pencere
    assert np.isclose(F[8, adlar.index("HR__delta6")], col[8] - col[2])
    X2 = X.copy(); X2[8:, :] += 100                                    # gelecek değişirse (saat 8 ve sonrası) önceki saatlerin öznitelikleri değişmemeli
    F2, _ = build_features(X2, saat, names, True); assert np.allclose(F[:8], F2[:8])


def test_gbm_sweep_yalniz_egitim_ve_dogrulama(raw_dir, tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    cfg["paths"].update(raw_dir=raw_dir, split_csv=str(tmp_path / "bolme.csv"), processed_dir=str(tmp_path / "proc"),
                        artifacts_dir=str(tmp_path / "art"), results_dir=str(tmp_path / "res"))
    cfg["gbm"].update(threads=1)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {**os.environ, "SEPSIS_CONFIG": str(tmp_path / "cfg.yaml"), "PYTHONPATH": str(ROOT / "src")}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    run("scripts/01_make_split.py"); run("scripts/02_preprocess.py", "A->B")
    for f in ("kaynak_ici_test", "hedef_test", "adaptasyon"):
        p = tmp_path / "proc" / "A_B" / f"{f}.npz"
        if p.exists(): p.unlink()
    run("scripts/09_gbm_sweep.py", "--yonler", "A->B", "--num-leaves", "4", "--min-child", "5", "50", "--lambda-l2", "10",
        "--max-rounds", "20", "--kontrol", "10")
    r = json.load(open(tmp_path / "res" / "tarama" / "gbm_A_B.json", encoding="utf-8"))
    assert len(r["satirlar"]) == 2 and "secilen" in r and (tmp_path / "res" / "tarama" / "gbm_A_B.md").exists()
