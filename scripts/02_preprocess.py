"""Ön işleme: parametreler yalnızca eğitim hastalarından; çıktı data/processed/<yon>/*.npz

--hedef-norm adaptasyon : DENEY KOLU (hastane bazlı normalizasyon). Hedef hastanenin kümeleri (adaptasyon, hedef_test) için
  34 ölçülen değişkenin ortalama/std'si hedef hastanenin ETİKETSİZ `adaptasyon` hastalarından hesaplanır (etiket ve test hastası kullanılmaz);
  Age, HospAdmTime, ICULOS ve ikili sütunlar kaynak parametreleriyle kalır; kaynak kümeleri değişmez.
  Ayrı bir processed_dir ile çalıştırılmalıdır; kayma_tanisi.md kaynak–hedef standartlaştırılmış ortalama farklarını yazar."""
import argparse, os
import numpy as np, pandas as pd
from sepsis.config import load_config, direction_slug
from sepsis.data.loader import find_patient_files, read_patient
from sepsis.data.preprocess import fit_params, transform, save_params, feature_names, MEASURED
import json

ap = argparse.ArgumentParser(); ap.add_argument("direction", choices=["A->B", "B->A"])
ap.add_argument("--hedef-norm", choices=["kaynak", "adaptasyon"], default="kaynak"); args = ap.parse_args()
cfg = load_config(); pp = cfg["preprocess"]
files = find_patient_files(cfg["paths"]["raw_dir"])
split = pd.read_csv(cfg["paths"]["split_csv"]); split = split[split.yon == args.direction]
out = f"{cfg['paths']['processed_dir']}/{direction_slug(args.direction)}"; os.makedirs(out, exist_ok=True)

train_ids = split[split.kume == "egitim"].hasta_id.tolist()
params = fit_params((read_patient(files[i]) for i in train_ids), pp["use_iculos"])
save_params(params, f"{out}/parametreler.json")
json.dump(feature_names(pp["use_iculos"]), open(f"{out}/ozellikler.json", "w"))
print(f"Parametreler {len(train_ids)} eğitim hastasından öğrenildi.")

params_hedef = None
if args.hedef_norm == "adaptasyon":                                    # hedef hastane: yalnızca etiketsiz adaptasyon hastaları
    ad_ids = split[split.kume == "adaptasyon"].hasta_id.tolist()
    pt = fit_params((read_patient(files[i]) for i in ad_ids), pp["use_iculos"])
    params_hedef = {"ortalama": dict(params["ortalama"]), "std": dict(params["std"]), "use_iculos": params["use_iculos"]}
    for c in MEASURED:                                                 # yalnızca ölçülen 34 değişken hedef istatistiğini alır
        params_hedef["ortalama"][c] = pt["ortalama"][c]; params_hedef["std"][c] = pt["std"][c]
    save_params(params_hedef, f"{out}/parametreler_hedef.json")
    satir = ["| Değişken | Kaynak ort. | Hedef ort. | (Hedef−Kaynak)/Kaynak std | Std oranı (hedef/kaynak) |", "|---|---|---|---|---|"]
    fark = {c: (pt["ortalama"][c] - params["ortalama"][c]) / params["std"][c] for c in MEASURED}
    for c in sorted(MEASURED, key=lambda c: -abs(fark[c]) if np.isfinite(fark[c]) else 0):
        satir.append(f"| {c} | {params['ortalama'][c]:.3g} | {pt['ortalama'][c]:.3g} | {fark[c]:+.3f} | {pt['std'][c] / params['std'][c]:.2f} |")
    open(f"{out}/kayma_tanisi.md", "w", encoding="utf-8").write(
        f"## Kaynak ve hedef hastane değişken dağılımları ({args.direction}; hedef: adaptasyon kümesi, etiketsiz)\n\n" + "\n".join(satir) + "\n")
    print(f"Hedef normalizasyonu: {len(ad_ids)} adaptasyon hastasından.")

for kume, g in split.groupby("kume"):
    Xs, ys, pid, hour = [], [], [], []
    ids = g.hasta_id.tolist()
    for k, hid in enumerate(ids):
        df = read_patient(files[hid])
        Xs.append(transform(df, params_hedef if (params_hedef is not None and kume in ("adaptasyon", "hedef_test")) else params, pp["clip"])); y = df["SepsisLabel"].to_numpy().astype(np.int8)
        ys.append(y); pid.append(np.full(len(y), k, np.int32)); hour.append(np.arange(len(y), dtype=np.int16))
    np.savez_compressed(f"{out}/{kume}.npz", X=np.vstack(Xs), y=np.concatenate(ys), pid=np.concatenate(pid),
                        saat=np.concatenate(hour), hasta_idleri=np.array(ids))
    print(f"{kume}: {len(ids)} hasta, {sum(map(len, ys))} satır")
