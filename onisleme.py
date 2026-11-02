"""2. Adım: Ön işleme. Parametreler (ortalama/std) YALNIZCA eğitim hastalarından öğrenilir.
Kullanım: python3 onisleme.py "A->B" [veri_klasoru]"""
import glob, os, sys, json
import numpy as np, pandas as pd

YON = sys.argv[1]
ROOT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/sepsis_s3")
CIKTI = f"veri/{YON.replace('->', '_')}"
os.makedirs(CIKTI, exist_ok=True)

VITALS = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
LABS = ["BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine",
        "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI",
        "Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
OLCUM = VITALS + LABS                         # 34 zamanla değişen değişken (eksik olabilen)
ZCOLS = OLCUM + ["Age", "HospAdmTime", "ICULOS"]   # z-skorlanacak 37 değişken
BIN = ["Gender", "Unit1", "Unit2"]            # ikili değişkenler (z-skor yok; Unit boşsa 0)
KULLAN_ICULOS = True                          # ablasyon için False yapılabilir
if not KULLAN_ICULOS:
    ZCOLS = [c for c in ZCOLS if c != "ICULOS"]

# --- dosya yolları ---
yol = {}
for h in "AB":
    for f in glob.glob(f"{ROOT}/**/training_set{h}/*.psv", recursive=True):
        yol[h + "_" + os.path.basename(f)[:-4]] = f
bolme = pd.read_csv("bolme.csv")
bolme = bolme[bolme.yon == YON]

def oku(hid):
    return pd.read_csv(yol[hid], sep="|")

# --- 1) Parametreleri YALNIZCA eğitim hastalarından öğren ---
egitim_idler = bolme[bolme.kume == "egitim"].hasta_id.tolist()
parcalar = pd.concat([oku(h)[ZCOLS] for h in egitim_idler])
ort, std = parcalar.mean(), parcalar.std().replace(0, 1.0)     # NaN'ları pandas otomatik atlar
json.dump({"ortalama": ort.to_dict(), "std": std.to_dict(), "egitim_hasta_sayisi": len(egitim_idler)},
          open(f"{CIKTI}/parametreler.json", "w"), indent=1)
print("Parametreler", len(egitim_idler), "eğitim hastasından öğrenildi.")

# --- 2) Her hastayı dönüştür (yalnızca geçmişe bakar) ---
def donustur(df):
    n = len(df)
    t = np.arange(n)
    gozlendi = df[OLCUM].notna().to_numpy()                      # (n, 34) bu saatte ölçüldü mü
    # son ölçümden beri geçen saat: ileri yönlü indeks taşıma
    son = np.where(gozlendi, t[:, None], np.nan)
    son = pd.DataFrame(son).ffill().to_numpy()
    gecen = np.where(np.isnan(son), t[:, None] + 1, t[:, None] - son)   # hiç ölçülmediyse: yatıştan beri süre
    gecen = np.log1p(gecen)
    # değerler: ileri taşı (geleceğe bakmaz), sonra eğitim ortalamasıyla doldur = z-skorda 0
    z = df[ZCOLS].ffill()
    z = ((z - ort) / std).clip(-5, 5).fillna(0.0)
    ikili = df[BIN].copy()
    ikili[["Unit1", "Unit2"]] = ikili[["Unit1", "Unit2"]].fillna(0)
    X = np.hstack([z.to_numpy(), ikili.to_numpy(), gozlendi.astype(float), gecen]).astype(np.float32)
    return X, df["SepsisLabel"].to_numpy().astype(np.int8)

OZELLIK = ZCOLS + BIN + [c + "_gozlendi" for c in OLCUM] + [c + "_son_olcum_suresi" for c in OLCUM]
json.dump(OZELLIK, open(f"{CIKTI}/ozellikler.json", "w"))

for kume, g in bolme.groupby("kume"):
    Xs, ys, pid, saat = [], [], [], []
    ids = g.hasta_id.tolist()
    for k, hid in enumerate(ids):
        X, y = donustur(oku(hid))
        Xs.append(X); ys.append(y); pid.append(np.full(len(y), k, np.int32)); saat.append(np.arange(len(y), dtype=np.int16))
    np.savez_compressed(f"{CIKTI}/{kume}.npz", X=np.vstack(Xs), y=np.concatenate(ys),
                        pid=np.concatenate(pid), saat=np.concatenate(saat), hasta_idleri=np.array(ids))
    print(f"{kume}: {len(ids)} hasta, {sum(len(y) for y in ys)} satır, {len(OZELLIK)} öznitelik")
