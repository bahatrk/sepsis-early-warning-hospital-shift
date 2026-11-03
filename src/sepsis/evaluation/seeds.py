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
