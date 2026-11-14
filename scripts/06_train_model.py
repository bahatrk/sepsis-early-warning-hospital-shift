"""Tek bir modeli (tek yön, tek tohum) eğitir ve hocanın istediği tüm çıktıları üretir.

Modeller:
  gru            nedensel GRU, 108 özniteliğin hepsi (baz; scripts/04 ile aynı model)
  gru_ozelliksiz nedensel GRU, "gözlendi" ve "son ölçüm süresi" sütunları ÇIKARILMIŞ (40 öznitelik) -- ablasyon
  grud           GRU-D tarzı eksiklik farkındalıklı kodlayıcı (girdi + gizli durum çürümesi) -- ÖNERİLEN MODEL
  grud_decaysiz  grud'un çürüme mekanizmaları kapalı hali -- ablasyon

Kullanım:  python scripts/06_train_model.py "A->B" --model grud --seed 0 [--device cuda]
Çıktılar:  results/modeller/<model>_<yon>_seed<s>/  (sonuc.json, egriler.png, roc_pr.png, karisiklik_*.png,
           rapor_*.md, tablo.md, kesisim.md)  ve  artifacts/<model>/<yon>/seed<s>.pt
Eşik yalnızca DOĞRULAMA kümesinden seçilir; test kümeleri model seçiminde kullanılmaz (yalnızca en sonda bir kez)."""
import argparse, json, os
import numpy as np, pandas as pd, torch
from sepsis.config import load_config, direction_slug
from sepsis.data.sequences import load_sequences
from sepsis.evaluation.metrics import discrimination
from sepsis.evaluation import report as R
from sepsis.models.gru import CausalGRU, train_model, predict
from sepsis.models.grud import GRUD, grud_indices

MODELLER = ["gru", "gru_ozelliksiz", "grud", "grud_decaysiz"]
KUMELER = ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]

ap = argparse.ArgumentParser(); ap.add_argument("direction", choices=["A->B", "B->A"])
ap.add_argument("--model", choices=MODELLER, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default=None, help="cpu | cuda (varsayılan: config'teki train.device, yoksa cpu)")
args = ap.parse_args()
cfg = load_config(); slug = direction_slug(args.direction)
device = args.device or cfg.get("train", {}).get("device", "cpu")
if device == "cuda" and not torch.cuda.is_available():
    print("UYARI: cuda yok, cpu kullanılıyor."); device = "cpu"
alarm = float(cfg.get("report", {}).get("alarm_rate", 0.05))
H = dict(cfg["grud"] if args.model.startswith("grud") else cfg["gru"]); H["alarm_rate"] = alarm
if device == "cpu": torch.set_num_threads(H.get("threads", os.cpu_count() or 1))

# --- sızıntı kontrolü eğitimden ÖNCE: kümeler arasında ortak hasta olmamalı ---
out = f"{cfg['paths']['results_dir']}/modeller/{args.model}_{slug}_seed{args.seed}"; os.makedirs(out, exist_ok=True)
split = pd.read_csv(cfg["paths"]["split_csv"]); kes_md, kes = R.intersection_report(split, args.direction)
open(f"{out}/kesisim.md", "w", encoding="utf-8").write(kes_md)
assert kes == 0, "Kümeler arasında ortak hasta var!"

D = f"{cfg['paths']['processed_dir']}/{slug}"
names = json.load(open(f"{D}/ozellikler.json", encoding="utf-8")); idx = grud_indices(names)
seqs = {k: load_sequences(f"{D}/{k}.npz") for k in KUMELER}

# --- girdi sütunları ve model fabrikası ---
if args.model == "gru_ozelliksiz":
    keep = sorted(idx["deger"] + idx["diger"])                       # gözlendi + son_olcum_suresi sütunları çıkarılır
    seqs = {k: ([x[:, keep] for x in X], y) for k, (X, y) in seqs.items()}
    noise_idx = torch.tensor([keep.index(i) for i in idx["deger"]])
else:
    keep = list(range(len(names))); noise_idx = torch.tensor(idx["deger"])
n_features = seqs["egitim"][0][0].shape[1]
if args.model in ("gru", "gru_ozelliksiz"):
    make = lambda: CausalGRU(n_features, H["hidden"], H["layers"], H["dropout"])
else:
    dec = args.model == "grud"
    make = lambda: GRUD(idx, H["hidden"], H["dropout"], decay_x=dec, decay_h=dec)

print(f"[{args.model}] {args.direction} seed {args.seed} | cihaz {device} | öznitelik {n_features}")
model, hist = train_model(make, seqs["egitim"], seqs["dogrulama"], H, args.seed, device, noise_idx)

# --- tahminler: test kümeleri model seçildikten SONRA, bir kez ---
preds = {}
for k in KUMELER:
    y, p = predict(model, *seqs[k], device=device); preds[k] = (y, p)
thr = R.threshold_at_alarm_rate(preds["dogrulama"][1], alarm)          # eşik yalnızca doğrulamadan

res = {"yon": args.direction, "model": args.model, "seed": args.seed, "cihaz": device, "en_iyi_epoch": int(model.best_epoch),
       "hiperparametre": {k: v for k, v in H.items()}, "n_ozellik": int(n_features), "esik": thr, "alarm_orani_hedef": alarm,
       "gecmis": hist}
tablo_md, rows = R.summary_table(preds, thr)
for k in KUMELER:
    res[k] = discrimination(*preds[k]); rep = R.hour_report(*preds[k], thr); res[k]["saat_raporu"] = rep
    open(f"{out}/rapor_{k}.md", "w", encoding="utf-8").write(R.report_markdown(rep, f"{R.KUME_ADI[k]} ({args.model}, {args.direction})", alarm))
    if k in ("kaynak_ici_test", "hedef_test"):
        R.plot_confusion(rep, f"{out}/karisiklik_{k}.png", f"{R.KUME_ADI[k]} | {args.model} {args.direction} | eşik %{100*alarm:.0f} alarm")
res["tablo"] = rows
json.dump(res, open(f"{out}/sonuc.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(f"{out}/tablo.md", "w", encoding="utf-8").write(
    f"## {args.model} | {args.direction} | seed {args.seed}\n\nEn iyi epoch: {model.best_epoch} (seçim: doğrulama AUPRC). "
    f"Eşik: doğrulama saatlerinin %{100*alarm:.0f}'i alarm.\n\n{tablo_md}\n")

R.plot_training_curves(hist, f"{out}/egriler.png", f"{args.model} | {args.direction} | seed {args.seed}")
R.plot_roc_pr({k: preds[k] for k in ["dogrulama", "kaynak_ici_test", "hedef_test"]}, f"{out}/roc_pr.png", f"{args.model} | {args.direction} | seed {args.seed}")

md = f"{cfg['paths']['artifacts_dir']}/{args.model}/{slug}"; os.makedirs(md, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "n_features": n_features, "keep": keep, "idx": idx, "cfg": H, "esik": thr,
            "model": args.model}, f"{md}/seed{args.seed}.pt")
print(tablo_md); print("Çıktılar:", out)
