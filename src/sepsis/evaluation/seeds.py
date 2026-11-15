"""Çoklu seed sonuçlarının özeti: ortalama ± standart sapma (seed'ler arası)."""
import glob, json, re
import numpy as np

KUMELER = ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]
METRIKLER = ["AUROC", "AUPRC"]


def summarize(results_dir: str, slug: str) -> dict:
    """results/gru_{slug}_seed*.json dosyalarını okur; seed'ler arası ort ve std (ddof=1) döndürür."""
    files = sorted(glob.glob(f"{results_dir}/gru_{slug}_seed*.json"),
                   key=lambda f: int(re.search(r"seed(\d+)\.json$", f).group(1)))
    if not files:
        raise FileNotFoundError(f"{results_dir}/gru_{slug}_seed*.json bulunamadı")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    out = {"yon": runs[0]["yon"], "seedler": [r["seed"] for r in runs], "n": len(runs)}
    for k in KUMELER:
        out[k] = {"pozitif_orani": runs[0][k]["pozitif_orani"]}
        for m in METRIKLER:
            v = np.array([r[k][m] for r in runs], dtype=float)
            out[k][m] = {"ort": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def to_markdown(summaries: list) -> str:
    satir = ["| Yön | Küme | Pozitif oranı | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / pozitif oranı |",
             "|---|---|---|---|---|---|"]
    for s in summaries:
        for k in KUMELER:
            a, p, po = s[k]["AUROC"], s[k]["AUPRC"], s[k]["pozitif_orani"]
            satir.append(f"| {s['yon']} (n={s['n']}) | {k} | {po:.4f} | {a['ort']:.3f} ± {a['std']:.3f} | "
                         f"{p['ort']:.3f} ± {p['std']:.3f} | {p['ort'] / po:.1f} |")
    return "\n".join(satir)


# ---- scripts/06_train_model.py çıktıları için (birden çok model) ----
def summarize_model(results_dir: str, model: str, slug: str) -> dict:
    """results/modeller/{model}_{slug}_seed*/sonuc.json dosyalarını okur; seed'ler arası ort ve std (ddof=1).
    Ayrıca eğitim-doğrulama AUPRC farkı (aşırı öğrenme ölçütü, ≤0,05) ve ölçütü aşan çalıştırma sayısı."""
    files = sorted(glob.glob(f"{results_dir}/modeller/{model}_{slug}_seed*/sonuc.json"),
                   key=lambda f: int(re.search(r"seed(\d+)/sonuc\.json$", f).group(1)))
    if not files:
        raise FileNotFoundError(f"{results_dir}/modeller/{model}_{slug}_seed*/sonuc.json bulunamadı")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    out = {"model": model, "yon": runs[0]["yon"], "seedler": [r["seed"] for r in runs], "n": len(runs)}
    for k in KUMELER:
        out[k] = {"pozitif_orani": runs[0][k]["pozitif_orani"]}
        for m in METRIKLER:
            v = np.array([r[k][m] for r in runs], dtype=float)
            out[k][m] = {"ort": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    gaps = np.array([r["egitim"]["AUPRC"] - r["dogrulama"]["AUPRC"] for r in runs])
    out["auprc_farki"] = {"ort": float(gaps.mean()), "ihlal": int((gaps > 0.05).sum())}
    if all("hedef_uyarlama" in r for r in runs):                       # etiketsiz eşik uyarlaması (hedef test)
        out["hedef_uyarlama"] = {}
        for ad in ("dogrulama_esigi", "uyarlanmis_esik"):
            out["hedef_uyarlama"][ad] = {}
            for m in ("alarm_orani", "precision", "recall", "f1", "dengeli_accuracy"):
                v = np.array([r["hedef_uyarlama"][ad][m] for r in runs], dtype=float)
                out["hedef_uyarlama"][ad][m] = {"ort": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def models_to_markdown(summaries: list) -> str:
    satir = ["| Model | Yön | Küme | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / oran |", "|---|---|---|---|---|---|"]
    for s in summaries:
        for k in KUMELER:
            a, p, po = s[k]["AUROC"], s[k]["AUPRC"], s[k]["pozitif_orani"]
            satir.append(f"| {s['model']} (n={s['n']}) | {s['yon']} | {k} | {a['ort']:.3f} ± {a['std']:.3f} | "
                         f"{p['ort']:.3f} ± {p['std']:.3f} | {p['ort'] / po:.1f} |")
    satir += ["", "Aşırı öğrenme ölçütü (eğitim − doğrulama AUPRC ≤ 0,05):", "",
              "| Model | Yön | Ortalama fark | Ölçütü aşan çalıştırma |", "|---|---|---|---|"]
    for s in summaries:
        g = s["auprc_farki"]; satir.append(f"| {s['model']} | {s['yon']} | {g['ort']:.3f} | {g['ihlal']} / {s['n']} |")
    if any("hedef_uyarlama" in s for s in summaries):
        satir += ["", "Hedef test, etiketsiz eşik uyarlaması (ort ± std; iki eşik de doğrulamadaki %5 alarm oranını hedefler):", "",
                  "| Model | Yön | Eşik | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |", "|---|---|---|---|---|---|---|---|"]
        for s in summaries:
            for ad, h in s.get("hedef_uyarlama", {}).items():
                f = lambda m: f"{h[m]['ort']:.3f} ± {h[m]['std']:.3f}"
                satir.append(f"| {s['model']} | {s['yon']} | {ad} | {f('alarm_orani')} | {f('precision')} | {f('recall')} | {f('f1')} | {f('dengeli_accuracy')} |")
    return "\n".join(satir)
