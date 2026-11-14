"""Seçilen modelleri birden çok tohumla eğitir (varsa atlar) ve ortalama ± std özetini yazar.

Kullanım:  python scripts/07_model_seeds.py --models gru grud                 # seed 0-4, iki yön
           python scripts/07_model_seeds.py --models grud --seeds 0 --yonler "A->B"   # hızlı deneme
           python scripts/07_model_seeds.py --models gru grud --sadece-ozet            # eğitmeden özetle
Çıktı: results/modeller/ozet.md ve ozet.json"""
import argparse, json, os, subprocess, sys
from sepsis.config import load_config, direction_slug
from sepsis.evaluation.seeds import summarize_model, models_to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["gru", "grud"], choices=["gru", "gru_ozelliksiz", "grud", "grud_decaysiz"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--yonler", nargs="+", default=["A->B", "B->A"], choices=["A->B", "B->A"])
ap.add_argument("--device", default=None); ap.add_argument("--sadece-ozet", action="store_true")
args = ap.parse_args()
cfg = load_config(); rd = cfg["paths"]["results_dir"]

if not args.sadece_ozet:
    for model in args.models:
        for yon in args.yonler:
            for s in args.seeds:
                if os.path.exists(f"{rd}/modeller/{model}_{direction_slug(yon)}_seed{s}/sonuc.json"):
                    print(f"[atla] {model} {yon} seed {s} zaten var"); continue
                print(f"[eğit] {model} {yon} seed {s}", flush=True)
                cmd = [sys.executable, "scripts/06_train_model.py", yon, "--model", model, "--seed", str(s)]
                if args.device: cmd += ["--device", args.device]
                subprocess.run(cmd, check=True)

ozetler = [summarize_model(rd, m, direction_slug(y)) for m in args.models for y in args.yonler]
os.makedirs(f"{rd}/modeller", exist_ok=True)
json.dump(ozetler, open(f"{rd}/modeller/ozet.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
md = models_to_markdown(ozetler); open(f"{rd}/modeller/ozet.md", "w", encoding="utf-8").write(md + "\n")
print(md)
