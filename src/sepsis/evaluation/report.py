"""Hocanın istediği değerlendirme çıktıları: eğriler, sınıflandırma raporu, ROC/PR, karışıklık matrisi,
eğitim/doğrulama/test tablosu ve kümeler arası kesişim kontrolü.

Sınıflandırma raporu ve karışıklık matrisi bir EŞİK ister. Eşik yalnızca DOĞRULAMA saatlerinin %alarm_rate'i alarm
üretecek şekilde seçilir (test kümesine bakılmaz). Birim saattir (satır düzeyi)."""
import itertools
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

SINIFLAR = ["sepsis_yok", "sepsis_var"]
KUME_ADI = {"egitim": "Eğitim", "dogrulama": "Doğrulama", "kaynak_ici_test": "Kaynak içi test", "hedef_test": "Hedef test"}

def threshold_at_alarm_rate(p_val, alarm_rate=0.05) -> float:
    """Doğrulama saatlerinin alarm_rate oranı alarm üretecek eşik."""
    return float(np.quantile(p_val, 1 - alarm_rate))

def hour_report(y, p, thr) -> dict:
    """Eşikte saat düzeyi rapor: karışıklık matrisi + sınıf bazında precision/recall/F1 + macro/weighted + accuracy."""
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cr = classification_report(y, pred, labels=[0, 1], target_names=SINIFLAR, output_dict=True, zero_division=0)
    bacc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return {"esik": float(thr), "alarm_orani": float(pred.mean()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(cr["accuracy"]), "dengeli_accuracy": float(bacc), "rapor": cr}

def report_markdown(rep: dict, baslik: str, alarm_rate: float = 0.05) -> str:
    cr = rep["rapor"]
    satir = [f"### {baslik}", "",
             f"Eşik = {rep['esik']:.4f}: doğrulama saatlerinin %{100 * alarm_rate:.0f}'i alarm verecek şekilde seçildi "
             f"(test kümesine bakılmadı). Bu kümede gerçekleşen alarm oranı %{100 * rep['alarm_orani']:.2f}.", "",
             "| Sınıf | Precision | Recall | F1 | Destek (saat) |", "|---|---|---|---|---|"]
    for s in SINIFLAR + ["macro avg", "weighted avg"]:
        r = cr[s]; satir.append(f"| {s} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    satir += ["", f"Accuracy = {rep['accuracy']:.4f}; dengeli accuracy = {rep['dengeli_accuracy']:.4f} "
              f"(sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).", "",
              "| | Tahmin: sepsis yok | Tahmin: sepsis var |", "|---|---|---|",
              f"| **Gerçek: sepsis yok** | {rep['tn']} | {rep['fp']} |", f"| **Gerçek: sepsis var** | {rep['fn']} | {rep['tp']} |", ""]
    return "\n".join(satir)

def summary_table(sets: dict, thr: float) -> tuple[str, list]:
    """sets: {küme: (y, p)}. Eğitim / doğrulama / test sonuçları yan yana tek tablo (markdown) + satır sözlükleri."""
    rows = []
    for k, (y, p) in sets.items():
        y = np.asarray(y).astype(int); rep = hour_report(y, p, thr); s = rep["rapor"]["sepsis_var"]
        pos = float(y.mean())
        rows.append({"kume": k, "saat": int(len(y)), "pozitif_orani": pos, "AUROC": float(roc_auc_score(y, p)),
                     "AUPRC": float(average_precision_score(y, p)), "AUPRC_kat": float(average_precision_score(y, p)) / pos,
                     "accuracy": rep["accuracy"], "dengeli_accuracy": rep["dengeli_accuracy"],
                     "precision": s["precision"], "recall": s["recall"], "F1": s["f1-score"]})
    md = ["| Küme | Saat | Pozitif oranı | AUROC | AUPRC | AUPRC/oran | Accuracy | Dengeli acc. | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {KUME_ADI.get(r['kume'], r['kume'])} | {r['saat']} | {r['pozitif_orani']:.4f} | {r['AUROC']:.3f} | {r['AUPRC']:.3f} | "
                  f"{r['AUPRC_kat']:.1f} | {r['accuracy']:.3f} | {r['dengeli_accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['F1']:.3f} |")
    return "\n".join(md), rows

def intersection_report(split: pd.DataFrame, direction: str) -> tuple[str, int]:
    """Yönün kümeleri arasındaki ortak hasta sayılarını yazar. Döndürür: (markdown, toplam kesişen hasta)."""
    g = split[split.yon == direction]; sets = {k: set(x.hasta_id) for k, x in g.groupby("kume")}
    satir = [f"### Kümeler arası kesişim kontrolü ({direction})", "", "| Küme | Hasta |", "|---|---|"]
    satir += [f"| {k} | {len(v)} |" for k, v in sets.items()]
    satir += ["", "| Küme çifti | Ortak hasta |", "|---|---|"]; toplam = 0
    for (n1, s1), (n2, s2) in itertools.combinations(sets.items(), 2):
        c = len(s1 & s2); toplam += c; satir.append(f"| {n1} ∩ {n2} | {c} |")
    satir += ["", f"**Toplam kesişen hasta: {toplam}** " + ("(sızıntı yok)" if toplam == 0 else "(SIZINTI VAR)"), ""]
    return "\n".join(satir), toplam

def plot_training_curves(history: list, path: str, baslik: str) -> None:
    """Eğitim ve doğrulama eğrileri AYNI grafikte: kayıp, accuracy (doğrulama eşiğinde), AUROC, AUPRC."""
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    pan = [("Kayıp (ağırlıklı BCE)", "train_loss_eval", "val_loss"), ("Accuracy (doğrulama eşiğinde)", "train_acc", "val_acc"),
           ("AUROC", "train_auroc", "val_auroc"), ("AUPRC", "train_auprc", "val_auprc")]
    for a, (ad, tr, va) in zip(ax.ravel(), pan):
        a.plot(ep, [h[tr] for h in history], "o-", label="eğitim"); a.plot(ep, [h[va] for h in history], "s-", label="doğrulama")
        a.set_xlabel("Epoch"); a.set_ylabel(ad); a.grid(alpha=0.3); a.legend(); a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(baslik); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

def plot_roc_pr(sets: dict, path: str, baslik: str) -> None:
    """Kümeler için ROC (AUC) ve PR (AUPRC; yatay çizgi = pozitif oranı) eğrileri."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (y, p) in sets.items():
        y = np.asarray(y).astype(int); ad = KUME_ADI.get(k, k)
        fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=f"{ad} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr, label=f"{ad} (AUPRC {average_precision_score(y, p):.3f}; oran {y.mean():.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("Yanlış pozitif oranı"); ax[0].set_ylabel("Doğru pozitif oranı"); ax[0].set_title("ROC")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall")
    for a in ax: a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(baslik); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

def plot_confusion(rep: dict, path: str, baslik: str) -> None:
    cm = np.array([[rep["tn"], rep["fp"]], [rep["fn"], rep["tp"]]]); norm = cm / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, f"{cm[i, j]}\n(%{100 * norm[i, j]:.1f})", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["sepsis yok", "sepsis var"]); ax.set_yticks([0, 1], ["sepsis yok", "sepsis var"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(baslik, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
