"""3. Adım: Lojistik regresyon baseline. Kullanım: python3 lr_baseline.py "A->B" """
import sys, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

YON = sys.argv[1]
D = f"veri/{YON.replace('->', '_')}"
os.makedirs("sonuclar", exist_ok=True)
SEED = 42

def yukle(kume):
    d = np.load(f"{D}/{kume}.npz")
    return d["X"], d["y"], d["pid"]

def olc(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"AUROC": round(roc_auc_score(y, p), 4), "AUPRC": round(average_precision_score(y, p), 4)}

Xe, ye, pe = yukle("egitim"); Xv, yv, _ = yukle("dogrulama")

# --- 1) Hiperparametre (C) yalnızca DOĞRULAMA AUPRC'sine göre seçilir ---
en_iyi, en_iyi_c = -1, None
for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xe, ye)
    a = average_precision_score(yv, m.predict_proba(Xv)[:, 1])
    print(f"C={C}: doğrulama AUPRC={a:.4f}")
    if a > en_iyi: en_iyi, en_iyi_c = a, C
print("Seçilen C:", en_iyi_c)

# --- 2) Öğrenme eğrisi: eğitim hastalarının %10..100'üyle eğit; eğitim ve doğrulama AUPRC'si ---
rng = np.random.default_rng(SEED)
hastalar = np.unique(pe); rng.shuffle(hastalar)
oranlar = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]; egri = []
for o in oranlar:
    secili = np.isin(pe, hastalar[: int(len(hastalar) * o)])
    m = LogisticRegression(C=en_iyi_c, class_weight="balanced", max_iter=3000).fit(Xe[secili], ye[secili])
    egri.append((o, average_precision_score(ye[secili], m.predict_proba(Xe[secili])[:, 1]),
                 average_precision_score(yv, m.predict_proba(Xv)[:, 1])))
plt.figure(figsize=(5, 3.5))
plt.plot([100 * e[0] for e in egri], [e[1] for e in egri], "o-", label="eğitim")
plt.plot([100 * e[0] for e in egri], [e[2] for e in egri], "s-", label="doğrulama")
plt.xlabel("Kullanılan eğitim hastası (%)"); plt.ylabel("AUPRC"); plt.title(f"LR öğrenme eğrisi ({YON})")
plt.legend(); plt.tight_layout(); plt.savefig(f"sonuclar/lr_ogrenme_egrisi_{YON.replace('->','_')}.png", dpi=150)

# --- 3) Son model (tüm eğitim hastaları) ve BİR KEZ test ---
model = LogisticRegression(C=en_iyi_c, class_weight="balanced", max_iter=3000).fit(Xe, ye)
sonuc = {"yon": YON, "C": en_iyi_c, "ogrenme_egrisi": egri}
for kume in ["egitim", "dogrulama", "kaynak_ici_test", "hedef_test"]:
    X, y, _ = yukle(kume)
    sonuc[kume] = olc(model, X, y); print(kume, sonuc[kume])
json.dump(sonuc, open(f"sonuclar/lr_{YON.replace('->','_')}.json", "w"), indent=1, ensure_ascii=False)
