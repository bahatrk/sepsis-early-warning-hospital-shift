"""LR baseline: C doğrulamada seçilir; öğrenme eğrisi; test BİR KEZ; web için model kaydı."""
import argparse, json, os, shutil
import numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sepsis.config import load_config, direction_slug
from sepsis.evaluation.metrics import discrimination
from sepsis.models.lr import select_and_fit, learning_curve, alarm_threshold

ap = argparse.ArgumentParser(); ap.add_argument("direction", choices=["A->B", "B->A"]); args = ap.parse_args()
cfg = load_config(); L = cfg["lr"]; slug = direction_slug(args.direction)
D = f"{cfg['paths']['processed_dir']}/{slug}"
load = lambda k: (lambda d: (d["X"], d["y"], d["pid"]))(np.load(f"{D}/{k}.npz"))
(Xtr, ytr, ptr), (Xval, yval, _) = load("egitim"), load("dogrulama")

model, best_c, scores = select_and_fit(Xtr, ytr, Xval, yval, L["c_grid"], L["max_iter"])
for c, s in scores.items(): print(f"C={c}: doğrulama AUPRC={s:.4f}")
print("Seçilen C:", best_c)
curve = learning_curve(Xtr, ytr, ptr, Xval, yval, best_c, L["max_iter"], [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cfg["seed"])

res = {"yon": args.direction, "C": best_c, "ogrenme_egrisi": curve}
for k in ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]:        # test yalnızca model seçildikten sonra
    X, y, _ = load(k); res[k] = discrimination(y, model.predict_proba(X)[:, 1]); print(k, res[k])
os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
json.dump(res, open(f"{cfg['paths']['results_dir']}/lr_{slug}.json", "w"), indent=1, ensure_ascii=False)

plt.figure(figsize=(5, 3.5))
plt.plot([100 * c[0] for c in curve], [c[1] for c in curve], "o-", label="eğitim")
plt.plot([100 * c[0] for c in curve], [c[2] for c in curve], "s-", label="doğrulama")
plt.xlabel("Kullanılan eğitim hastası (%)"); plt.ylabel("AUPRC"); plt.title(f"LR öğrenme eğrisi ({args.direction})")
plt.legend(); plt.tight_layout(); plt.savefig(f"{cfg['paths']['results_dir']}/lr_ogrenme_egrisi_{slug}.png", dpi=150)

md = f"{cfg['paths']['artifacts_dir']}/lr/{slug}"; os.makedirs(md, exist_ok=True)
joblib.dump(model, f"{md}/lr.joblib"); shutil.copy(f"{D}/parametreler.json", md); shutil.copy(f"{D}/ozellikler.json", md)
esik = alarm_threshold(model, Xval, L["alarm_rate"])
json.dump({"yon": args.direction, "esik": esik, "alarm_orani_dogrulama": L["alarm_rate"]}, open(f"{md}/meta.json", "w"))
print("Model kaydedildi:", md, "| alarm eşiği:", round(esik, 4))
