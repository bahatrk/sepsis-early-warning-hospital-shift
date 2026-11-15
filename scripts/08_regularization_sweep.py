"""Aşırı öğrenmeyi azaltmak için düzenlileştirme taraması -- YALNIZCA doğrulama kümesi kullanılır.

Test kümeleri bu betikte HİÇ yüklenmez. Her ayar için eğitim-doğrulama AUPRC farkı (en iyi epoch'ta) ve
doğrulama AUPRC'si tohumlar üzerinden ortalanır. Seçim kuralı: ortalama fark <= 0.05 olan ayarlar arasında en yüksek
doğrulama AUPRC; hiçbiri sağlamıyorsa en küçük ortalama fark. Kazanan ayar results/tarama/<model>_<yon>.json'a yazılır;
sonra configs/default.yaml'a elle işlenip 07_model_seeds.py ile 5 tohum koşulur.

Aşamalı kullanım (tek seferde tam ızgara yerine, süreyi kısaltmak için):
  Aşama 1 (weight decay; AdamW'de adım başına küçülme lr*wd olduğundan 1e-4 etkisizdir):
    python scripts/08_regularization_sweep.py --model gru --dropout 0.2 --weight-decay 1e-4 1e-2 5e-2 1e-1 3e-1 --gap-limit 0.05 --etiket _wd
  Aşama 2 (veri artırma, seçilen weight decay ile):
    python scripts/08_regularization_sweep.py --model gru --weight-decay <seçilen> --gap-limit 0.05 --noise-std 0 0.05 0.1 0.2 --etiket _jitter
    ... --scale-std 0 0.05 0.1 0.2 --etiket _scale ;  ... --shift-std 0 0.1 0.2 0.4 --etiket _offset
Kullanım:  python scripts/08_regularization_sweep.py --model grud --yonler "A->B" "B->A" --seeds 0 1
           python scripts/08_regularization_sweep.py --model gru --dropout 0.2 0.4 --weight-decay 1e-4 1e-3 --gap-limit none 0.05"""
import argparse, itertools, json, os
import numpy as np, torch
from sepsis.config import load_config, direction_slug
from sepsis.data.sequences import load_sequences
from sepsis.models.gru import CausalGRU, train_model
from sepsis.models.grud import GRUD, grud_indices

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["gru", "grud"], required=True)
ap.add_argument("--yonler", nargs="+", default=["A->B", "B->A"], choices=["A->B", "B->A"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--dropout", type=float, nargs="+", default=[0.2, 0.4])
ap.add_argument("--weight-decay", type=float, nargs="+", default=[1e-4, 1e-3])
ap.add_argument("--gap-limit", nargs="+", default=["none", "0.05"])
ap.add_argument("--noise-std", type=float, nargs="+", default=[0.0], help="jittering (yalnızca eğitimde)")
ap.add_argument("--scale-std", type=float, nargs="+", default=[0.0], help="magnitude scaling (yalnızca eğitimde)")
ap.add_argument("--shift-std", type=float, nargs="+", default=[0.0], help="hasta başına offset (yalnızca eğitimde)")
ap.add_argument("--etiket", default="", help="çıktı dosya adına eklenir (aşamaları ayırmak için)")
ap.add_argument("--device", default=None)
args = ap.parse_args()
cfg = load_config(); device = args.device or cfg.get("train", {}).get("device", "cpu")
if device == "cuda" and not torch.cuda.is_available(): device = "cpu"
gaps = [None if g.lower() == "none" else float(g) for g in args.gap_limit]
base = dict(cfg[args.model]); base["alarm_rate"] = float(cfg.get("report", {}).get("alarm_rate", 0.05))
if device == "cpu": torch.set_num_threads(base.get("threads", os.cpu_count() or 1))
out_dir = f"{cfg['paths']['results_dir']}/tarama"; os.makedirs(out_dir, exist_ok=True)

for yon in args.yonler:
    slug = direction_slug(yon); D = f"{cfg['paths']['processed_dir']}/{slug}"
    names = json.load(open(f"{D}/ozellikler.json", encoding="utf-8")); idx = grud_indices(names)
    tr, va = load_sequences(f"{D}/egitim.npz"), load_sequences(f"{D}/dogrulama.npz")   # test yüklenmez
    nf = tr[0][0].shape[1]; noise_idx = torch.tensor(idx["deger"])
    rows = []
    for do, wd, gl, ns, sc, sh in itertools.product(args.dropout, args.weight_decay, gaps, args.noise_std, args.scale_std, args.shift_std):
        H = dict(base, dropout=do, weight_decay=wd, gap_limit=gl, noise_std=ns, scale_std=sc, shift_std=sh)
        make = (lambda: CausalGRU(nf, H["hidden"], H["layers"], H["dropout"])) if args.model == "gru" \
            else (lambda: GRUD(idx, H["hidden"], H["dropout"]))
        v, g, ep = [], [], []
        for s in args.seeds:
            print(f"[{args.model}] {yon} dropout={do} wd={wd} gap={gl} noise={ns} scale={sc} shift={sh} seed={s}", flush=True)
            m, hist = train_model(make, tr, va, H, s, device, noise_idx, verbose=False)
            h = next(r for r in hist if r["epoch"] == m.best_epoch)
            v.append(h["val_auprc"]); g.append(h["train_auprc"] - h["val_auprc"]); ep.append(int(m.best_epoch))
        rows.append({"dropout": do, "weight_decay": wd, "gap_limit": gl, "noise_std": ns, "scale_std": sc, "shift_std": sh, "val_auprc": float(np.mean(v)),
                     "val_auprc_std": float(np.std(v)), "fark": float(np.mean(g)), "fark_max": float(np.max(g)),
                     "en_iyi_epoch": ep})
        print("   ->", rows[-1], flush=True)
    ok = [r for r in rows if r["fark"] <= 0.05]
    best = max(ok, key=lambda r: r["val_auprc"]) if ok else min(rows, key=lambda r: r["fark"])
    res = {"model": args.model, "yon": yon, "seeds": args.seeds, "olcut_saglandi": bool(ok), "secilen": best, "tum": rows}
    json.dump(res, open(f"{out_dir}/{args.model}_{slug}{args.etiket}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = [f"## {args.model} | {yon} | tohumlar {args.seeds}  (yalnızca doğrulama kümesi)\n",
          "| dropout | weight decay | gap_limit | jitter | scaling | offset | doğr. AUPRC | eğitim−doğr. fark (ort) | fark (maks) | en iyi epoch |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["val_auprc"]):
        md.append(f"| {r['dropout']} | {r['weight_decay']} | {r['gap_limit']} | {r['noise_std']} | {r['scale_std']} | {r['shift_std']} | {r['val_auprc']:.4f} ± {r['val_auprc_std']:.4f} | {r['fark']:.3f} | {r['fark_max']:.3f} | {r['en_iyi_epoch']} |")
    md.append(f"\nSeçilen: dropout={best['dropout']}, weight_decay={best['weight_decay']}, gap_limit={best['gap_limit']}, noise_std={best['noise_std']}, scale_std={best['scale_std']}, shift_std={best['shift_std']} "
              f"({'ölçüt sağlandı' if ok else 'HİÇBİR ayar fark<=0.05 sağlamadı; en küçük fark seçildi'}).")
    open(f"{out_dir}/{args.model}_{slug}{args.etiket}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
