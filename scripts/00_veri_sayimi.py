"""Ham PhysioNet/CinC 2019 verisini KENDİ KODUMUZLA sayar -> results/veri_sayimi.json ve .md.
Hastane, klasör adından (setA/setB) bulunur; klasör adı yoksa dosya numarasından (p1xxxxx = B).
Kullanım:  python scripts/00_veri_sayimi.py [--raw DIZIN]"""
import argparse, json, os, re
from pathlib import Path
import pandas as pd

LABEL = "SepsisLabel"

def hastane_bul(path: Path) -> str:
    yol = "/".join(path.parts).lower()
    if re.search(r"set_?a|training_?a|(^|/)a(/|$)", yol): return "A"
    if re.search(r"set_?b|training_?b|(^|/)b(/|$)", yol): return "B"
    sayi = int(re.sub(r"\D", "", path.stem) or 0)
    return "B" if sayi >= 100000 else "A"

def say(raw_dir: str) -> dict:
    dosyalar = sorted(Path(raw_dir).expanduser().rglob("*.psv"))
    if not dosyalar: raise SystemExit(f"{raw_dir} altında .psv bulunamadı")
    kayit, eksik_say, eksik_top = {}, {}, {}
    for f in dosyalar:
        df = pd.read_csv(f, sep="|")
        h = hastane_bul(f)
        r = kayit.setdefault(h, dict(hasta=0, satir=0, sepsisli_hasta=0, pozitif_satir=0, saat_toplam=0))
        r["hasta"] += 1; r["satir"] += len(df)
        r["pozitif_satir"] += int(df[LABEL].sum()); r["sepsisli_hasta"] += int(df[LABEL].max() > 0)
        e = eksik_say.setdefault(h, df.isna().sum()*0); eksik_say[h] = e + df.isna().sum()
        eksik_top[h] = eksik_top.get(h, 0) + len(df)
    sonuc = {}
    for h, r in sorted(kayit.items()):
        sonuc[h] = dict(r,
            sepsisli_hasta_orani=round(r["sepsisli_hasta"]/r["hasta"], 4),
            pozitif_satir_orani=round(r["pozitif_satir"]/r["satir"], 4),
            ort_saat_hasta=round(r["satir"]/r["hasta"], 1),
            eksik_orani={c: round(float(v)/eksik_top[h], 3) for c, v in eksik_say[h].items() if c != LABEL})
    t = {k: sum(r[k] for r in kayit.values()) for k in ("hasta", "satir", "sepsisli_hasta", "pozitif_satir")}
    t["sepsisli_hasta_orani"] = round(t["sepsisli_hasta"]/t["hasta"], 4)
    t["pozitif_satir_orani"] = round(t["pozitif_satir"]/t["satir"], 4)
    sonuc["toplam"] = t
    return sonuc

def markdown(s: dict) -> str:
    sat = ["| Küme | Hasta | Satır (saat) | Sepsisli hasta | Hasta oranı | Pozitif satır | Satır oranı |", "|---|---|---|---|---|---|---|"]
    for h in [k for k in s if k != "toplam"] + ["toplam"]:
        r = s[h]
        sat.append(f"| {h} | {r['hasta']:,} | {r['satir']:,} | {r['sepsisli_hasta']:,} | {r['sepsisli_hasta_orani']:.2%} | {r['pozitif_satir']:,} | {r['pozitif_satir_orani']:.2%} |")
    return "\n".join(sat) + "\n"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.raw is None:
        from sepsis.config import load_config
        a.raw = load_config()["paths"]["raw_dir"]
    s = say(a.raw); os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(f"{a.out}/veri_sayimi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(f"{a.out}/veri_sayimi.md", "w", encoding="utf-8").write(markdown(s))
    print(markdown(s))
