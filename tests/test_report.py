import numpy as np, pandas as pd, pytest
from sepsis.evaluation import report as R

def _yp(n=2000, seed=0):
    rng = np.random.default_rng(seed); y = (rng.random(n) < 0.05).astype(int)
    return y, np.clip(0.2 * y + rng.random(n) * 0.8, 0, 1)

def test_threshold_gives_requested_alarm_rate():
    y, p = _yp(); thr = R.threshold_at_alarm_rate(p, 0.05)
    assert abs((p >= thr).mean() - 0.05) < 0.005

def test_hour_report_counts_and_keys():
    y, p = _yp(); thr = R.threshold_at_alarm_rate(p, 0.05); r = R.hour_report(y, p, thr)
    assert r["tn"] + r["fp"] + r["fn"] + r["tp"] == len(y) and r["tp"] + r["fn"] == y.sum()
    for k in R.SINIFLAR + ["macro avg", "weighted avg"]:
        assert {"precision", "recall", "f1-score", "support"} <= set(r["rapor"][k])
    assert "accuracy" in r and 0 <= r["dengeli_accuracy"] <= 1
    assert "Gerçek: sepsis var" in R.report_markdown(r, "deneme")

def test_summary_table_one_row_per_set():
    y, p = _yp(); md, rows = R.summary_table({"egitim": (y, p), "dogrulama": _yp(seed=1), "hedef_test": _yp(seed=2)}, 0.9)
    assert [r["kume"] for r in rows] == ["egitim", "dogrulama", "hedef_test"] and md.count("\n") == 4
    assert all(0 <= r["AUROC"] <= 1 for r in rows)

def _split(leak=False):
    rows = [(f"A{i}", "A", 0, "A->B", k) for k, rng in [("egitim", range(0, 6)), ("dogrulama", range(6, 8)), ("kaynak_ici_test", range(8, 10))] for i in rng]
    rows += [(f"B{i}", "B", 0, "A->B", k) for k, rng in [("adaptasyon", range(0, 3)), ("hedef_test", range(3, 8))] for i in rng]
    if leak: rows.append(("A0", "A", 0, "A->B", "hedef_test"))
    return pd.DataFrame(rows, columns=["hasta_id", "hastane", "sepsis", "yon", "kume"])

def test_intersection_report_zero_and_detects_leak():
    md, n = R.intersection_report(_split(), "A->B"); assert n == 0 and "sızıntı yok" in md
    md, n = R.intersection_report(_split(leak=True), "A->B"); assert n == 1 and "SIZINTI VAR" in md

def test_plots_write_files(tmp_path):
    y, p = _yp(); hist = [{"epoch": e, "train_loss_eval": 1 / e, "val_loss": 1 / e + .1, "train_acc": .9, "val_acc": .9,
                           "train_auroc": .8, "val_auroc": .7, "train_auprc": .2, "val_auprc": .1} for e in (1, 2, 3)]
    R.plot_training_curves(hist, str(tmp_path / "a.png"), "t"); R.plot_roc_pr({"dogrulama": (y, p)}, str(tmp_path / "b.png"), "t")
    R.plot_confusion(R.hour_report(y, p, 0.5), str(tmp_path / "c.png"), "t")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ["a.png", "b.png", "c.png"])
