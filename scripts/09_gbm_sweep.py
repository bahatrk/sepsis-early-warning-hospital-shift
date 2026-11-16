"""LightGBM düzenlileştirme taraması -- YALNIZCA eğitim ve doğrulama kümeleri yüklenir (test/adaptasyon HİÇ okunmaz).

Amaç: eğitim−doğrulama AUPRC farkı <= gap_limit (0.05) sağlayan en iyi ayarı bulmak. Seçim kuralı (08 ile aynı):
farkı sağlayan ayarlar arasında en yüksek doğrulama AUPRC; hiçbiri sağlamıyorsa en küçük ortalama fark.
Kullanım:  python scripts/09_gbm_sweep.py --yonler "A->B" "B->A" --num-leaves 4 8 --min-child 2000 5000 --lambda-l2 10 100 --etiket _a
Davranış sütunlu sürüm (gbm) taranır; kazanan ayar gbm_davranissiz için de kullanılır (aynı model ailesi, karşılaştırma adil kalsın)."""
import argparse, itertools, json, os
import numpy as np
from sepsis.config import load_config, direction_slug
from sepsis.models import gbm as G

ap = argparse.ArgumentParser()
ap.add_argument("--yonler", nargs="+", default=["A->B", "B->A"], choices=["A->B", "B->A"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--num-leaves", type=int, nargs="+", default=[4, 8])
ap.add_argument("--min-child", type=int, nargs="+", default=[2000, 5000])
ap.add_argument("--lambda-l2", type=float, nargs="+", default=[10.0, 100.0])
ap.add_argument("--lr", type=float, nargs="+", default=[0.05])
ap.add_argument("--max-rounds", type=int, default=300); ap.add_argument("--kontrol", type=int, default=10)
ap.add_argument("--gap-limit", type=float, default=0.05)
ap.add_argument("--etiket", default="")
args = ap.parse_args()
cfg = load_config(); base = dict(cfg["gbm"]); base.update(max_rounds=args.max_rounds, kontrol=args.kontrol, gap_limit=args.gap_limit,
                                                           alarm_rate=float(cfg.get("report", {}).get("alarm_rate", 0.05)))
os.makedirs(f"{cfg['paths']['results_dir']}/tarama", exist_ok=True)

for yon in args.yonler:
    slug = direction_slug(yon); D = f"{cfg['paths']['processed_dir']}/{slug}"
    names = json.load(open(f"{D}/ozellikler.json", encoding="utf-8"))
    tr, va = np.load(f"{D}/egitim.npz"), np.load(f"{D}/dogrulama.npz")        # YALNIZCA bu ikisi
    Ftr, _ = G.build_features(tr["X"], tr["saat"], names, True); Fva, _ = G.build_features(va["X"], va["saat"], names, True)
    ytr, yva = tr["y"].astype(np.int8), va["y"].astype(np.int8)
    rows = []
    for nl, mc, l2, lr in itertools.product(args.num_leaves, args.min_child, args.lambda_l2, args.lr):
        H = dict(base, num_leaves=nl, min_child_samples=mc, lambda_l2=l2, lr=lr, max_depth=max(3, int(np.ceil(np.log2(nl))) + 1))
        vs, gs, its = [], [], []
        for s in args.seeds:
            print(f"[gbm] {yon} leaves={nl} min_child={mc} l2={l2} lr={lr} seed={s}", flush=True)
            b, hist = G.train_gbm(Ftr, ytr, Fva, yva, H, s, verbose=False)
            h = next(x for x in hist if x["epoch"] == b.best_it)
            vs.append(h["val_auprc"]); gs.append(h["train_auprc"] - h["val_auprc"]); its.append(int(b.best_it))
        r = {"num_leaves": nl, "min_child_samples": mc, "lambda_l2": l2, "lr": lr, "val_auprc": float(np.mean(vs)),
             "val_auprc_std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0, "fark": float(np.mean(gs)), "fark_max": float(np.max(gs)), "en_iyi_tur": its}
        print("   ->", r, flush=True); rows.append(r)
    uygun = [r for r in rows if r["fark"] <= args.gap_limit]
    sec = max(uygun, key=lambda r: r["val_auprc"]) if uygun else min(rows, key=lambda r: r["fark"])
    md = [f"## gbm | {yon} | tohumlar {args.seeds}  (yalnızca doğrulama kümesi)", "",
          "| yaprak | min_child | lambda_l2 | lr | doğr. AUPRC | eğitim−doğr. fark (ort) | fark (maks) | en iyi tur |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["val_auprc"]):
        md.append(f"| {r['num_leaves']} | {r['min_child_samples']} | {r['lambda_l2']} | {r['lr']} | {r['val_auprc']:.4f} ± {r['val_auprc_std']:.4f} | {r['fark']:.3f} | {r['fark_max']:.3f} | {r['en_iyi_tur']} |")
    md.append(f"\nSeçilen: yaprak={sec['num_leaves']}, min_child={sec['min_child_samples']}, lambda_l2={sec['lambda_l2']}, lr={sec['lr']} "
              + ("(ölçüt sağlandı)." if uygun else "(ÖLÇÜT HİÇBİR AYARDA SAĞLANMADI; en küçük fark seçildi)."))
    print("\n".join(md))
    open(f"{cfg['paths']['results_dir']}/tarama/gbm_{slug}{args.etiket}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"yon": yon, "satirlar": rows, "secilen": sec, "olcut_saglandi": bool(uygun)},
              open(f"{cfg['paths']['results_dir']}/tarama/gbm_{slug}{args.etiket}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
