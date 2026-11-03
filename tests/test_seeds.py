import json
import pytest
from sepsis.evaluation.seeds import summarize, to_markdown


def _yaz(d, seed, auroc, auprc):
    r = {"yon": "A->B", "seed": seed}
    for k in ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]:
        r[k] = {"AUROC": auroc, "AUPRC": auprc, "pozitif_orani": 0.02}
    (d / f"gru_A_B_seed{seed}.json").write_text(json.dumps(r))


def test_ortalama_std(tmp_path):
    for s, a in enumerate([0.70, 0.72, 0.74]):
        _yaz(tmp_path, s, a, 0.10)
    o = summarize(str(tmp_path), "A_B")
    assert o["n"] == 3 and o["seedler"] == [0, 1, 2]
    assert o["hedef_test"]["AUROC"]["ort"] == pytest.approx(0.72)
    assert o["hedef_test"]["AUROC"]["std"] == pytest.approx(0.02)       # ddof=1
    assert o["hedef_test"]["AUPRC"]["std"] == pytest.approx(0.0, abs=1e-12)
    assert "A->B (n=3)" in to_markdown([o])


def test_seed_sirasi_sayisal(tmp_path):
    for s in [0, 2, 10]:
        _yaz(tmp_path, s, 0.7, 0.1)
    assert summarize(str(tmp_path), "A_B")["seedler"] == [0, 2, 10]


def test_dosya_yoksa_hata(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize(str(tmp_path), "A_B")
