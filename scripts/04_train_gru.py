"""GRU baseline: doğrulama AUPRC'sine göre erken durdurma; test BİR KEZ; öğrenme eğrisi (epoch)."""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sepsis.config import load_config, direction_slug
from sepsis.data.sequences import load_sequences
from sepsis.evaluation.metrics import discrimination
from sepsis.models.gru import train_gru, predict

ap = argparse.ArgumentParser(); ap.add_argument("direction", choices=["A->B", "B->A"]); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cfg = load_config(); G = cfg["gru"]; slug = direction_slug(args.direction)
D = f"{cfg['paths']['processed_dir']}/{slug}"
seqs = {k: load_sequences(f"{D}/{k}.npz") for k in ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]}
n_features = seqs["egitim"][0][0].shape[1]
torch.set_num_threads(G.get("threads", os.cpu_count()))

model, history = train_gru(seqs["egitim"], seqs["dogrulama"], n_features, G, args.seed)
res = {"yon": args.direction, "seed": args.seed, "gru": {k: G[k] for k in ["hidden", "layers", "dropout", "lr", "weight_decay"]}, "gecmis": history}
for k in ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]:          # test yalnızca model seçildikten sonra
    y, p = predict(model, *seqs[k]); res[k] = discrimination(y, p); print(k, res[k])
rd = cfg["paths"]["results_dir"]; os.makedirs(rd, exist_ok=True)
json.dump(res, open(f"{rd}/gru_{slug}_seed{args.seed}.json", "w"), indent=1, ensure_ascii=False)

ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(ep, [h["train_loss"] for h in history], "o-"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Eğitim kaybı")
ax[1].plot(ep, [h["train_auprc"] for h in history], "o-", label="eğitim"); ax[1].plot(ep, [h["val_auprc"] for h in history], "s-", label="doğrulama")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("AUPRC"); ax[1].legend(); fig.suptitle(f"GRU öğrenme eğrisi ({args.direction}, seed {args.seed})")
plt.tight_layout(); plt.savefig(f"{rd}/gru_ogrenme_egrisi_{slug}_seed{args.seed}.png", dpi=150)

md = f"{cfg['paths']['artifacts_dir']}/gru/{slug}"; os.makedirs(md, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "n_features": n_features, "cfg": G}, f"{md}/gru_seed{args.seed}.pt")
print("Model kaydedildi:", md)
