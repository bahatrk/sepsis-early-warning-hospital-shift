"""Hasta düzeyinde bölme -> splits/bolme.csv.
Bölme BİR KEZ üretilir ve dondurulur: dosya varsa üzerine yazılmaz (sonuçların tekrar üretilebilirliği için);
yalnızca kesişim kontrolü yapılır. Yeniden üretmek için --force."""
import argparse, os
import pandas as pd
from sepsis.config import load_config
from sepsis.data.split import patient_table, make_split, assert_disjoint

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
cfg = load_config(); csv = cfg["paths"]["split_csv"]

if os.path.exists(csv) and not args.force:
    split = pd.read_csv(csv)
    print(f"{csv} zaten var (dondurulmuş bölme); yeniden üretilmedi.")
else:
    patients = patient_table(cfg["paths"]["raw_dir"])
    print(f"A: {(patients.hastane=='A').sum()} hasta, {patients[patients.hastane=='A'].sepsis.sum()} sepsisli | "
          f"B: {(patients.hastane=='B').sum()} hasta, {patients[patients.hastane=='B'].sepsis.sum()} sepsisli")
    split = make_split(patients, cfg["seed"], **cfg["split"])
    os.makedirs(os.path.dirname(csv), exist_ok=True); split.to_csv(csv, index=False)

assert_disjoint(split)
print("Kesişim yok ✔\n", split.groupby(["yon", "kume"]).sepsis.agg(["count", "sum", "mean"]).round(3))
