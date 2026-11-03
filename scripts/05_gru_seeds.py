"""GRU'yu birden çok seed ile eğitir (varsa atlar) ve ortalama ± std özetini yazar.

Kullanım:  python scripts/05_gru_seeds.py                  # seed 0-4, iki yön
           python scripts/05_gru_seeds.py --seeds 0 1 2    # seçilen seed'ler
           python scripts/05_gru_seeds.py --sadece-ozet    # eğitmeden mevcut sonuçları özetle
Bölme (splits/bolme.csv) sabittir; seed yalnızca model başlangıcını ve mini-batch sırasını değiştirir.
"""
import argparse, json, os, subprocess, sys
from sepsis.config import load_config, direction_slug
from sepsis.evaluation.seeds import summarize, to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--sadece-ozet", action="store_true")
args = ap.parse_args()
cfg = load_config(); rd = cfg["paths"]["results_dir"]
yonler = ["A->B", "B->A"]

if not args.sadece_ozet:
    for yon in yonler:
        slug = direction_slug(yon)
        for s in args.seeds:
            if os.path.exists(f"{rd}/gru_{slug}_seed{s}.json"):
                print(f"[atla] {yon} seed {s} zaten var"); continue
            print(f"[eğit] {yon} seed {s}", flush=True)
            subprocess.run([sys.executable, "scripts/04_train_gru.py", yon, "--seed", str(s)], check=True)

ozetler = [summarize(rd, direction_slug(y)) for y in yonler]
json.dump(ozetler, open(f"{rd}/gru_ozet.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
md = to_markdown(ozetler); open(f"{rd}/gru_ozet.md", "w", encoding="utf-8").write(md + "\n")
print(md)
