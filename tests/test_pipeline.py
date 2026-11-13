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
    for model in ("grud", "gru_ozelliksiz"):
        run("scripts/06_train_model.py", "A->B", "--model", model, "--seed", "0")
        d = tmp_path / "res" / "modeller" / f"{model}_A_B_seed0"
        for f in ["sonuc.json", "tablo.md", "kesisim.md", "egriler.png", "roc_pr.png", "karisiklik_kaynak_ici_test.png",
                  "karisiklik_hedef_test.png", "rapor_dogrulama.md", "rapor_hedef_test.md"]:
            assert (d / f).exists(), f"{model}: {f} yok"
        r = json.load(open(d / "sonuc.json", encoding="utf-8"))
        assert r["n_ozellik"] == (108 if model == "grud" else 40)
        assert "Toplam kesişen hasta: 0" in (d / "kesisim.md").read_text(encoding="utf-8")
        assert (tmp_path / "art" / model / "A_B" / "seed0.pt").exists()
