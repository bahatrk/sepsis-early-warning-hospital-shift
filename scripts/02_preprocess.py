"""Ön işleme: parametreler yalnızca eğitim hastalarından; çıktı data/processed/<yon>/*.npz"""
import argparse, os
import numpy as np, pandas as pd
from sepsis.config import load_config, direction_slug
from sepsis.data.loader import find_patient_files, read_patient
from sepsis.data.preprocess import fit_params, transform, save_params, feature_names
import json

ap = argparse.ArgumentParser(); ap.add_argument("direction", choices=["A->B", "B->A"]); args = ap.parse_args()
cfg = load_config(); pp = cfg["preprocess"]
files = find_patient_files(cfg["paths"]["raw_dir"])
split = pd.read_csv(cfg["paths"]["split_csv"]); split = split[split.yon == args.direction]
out = f"{cfg['paths']['processed_dir']}/{direction_slug(args.direction)}"; os.makedirs(out, exist_ok=True)

train_ids = split[split.kume == "egitim"].hasta_id.tolist()
params = fit_params((read_patient(files[i]) for i in train_ids), pp["use_iculos"])
save_params(params, f"{out}/parametreler.json")
json.dump(feature_names(pp["use_iculos"]), open(f"{out}/ozellikler.json", "w"))
print(f"Parametreler {len(train_ids)} eğitim hastasından öğrenildi.")

for kume, g in split.groupby("kume"):
    Xs, ys, pid, hour = [], [], [], []
    ids = g.hasta_id.tolist()
    for k, hid in enumerate(ids):
        df = read_patient(files[hid])
        Xs.append(transform(df, params, pp["clip"])); y = df["SepsisLabel"].to_numpy().astype(np.int8)
        ys.append(y); pid.append(np.full(len(y), k, np.int32)); hour.append(np.arange(len(y), dtype=np.int16))
    np.savez_compressed(f"{out}/{kume}.npz", X=np.vstack(Xs), y=np.concatenate(ys), pid=np.concatenate(pid),
                        saat=np.concatenate(hour), hasta_idleri=np.array(ids))
    print(f"{kume}: {len(ids)} hasta, {sum(map(len, ys))} satır")
