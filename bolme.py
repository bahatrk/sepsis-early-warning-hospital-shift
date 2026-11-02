"""1. Adım: Hasta düzeyinde bölme. Çıktı: bolme.csv + kesişim kontrolü."""
import glob, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sepsis_s3")

def hasta_tablosu(hastane):
    """Her hasta için tek satır: hasta_id, hastane, sepsisli_mi."""
    dosyalar = glob.glob(f"{ROOT}/**/training_set{hastane}/*.psv", recursive=True)
    kayit = []
    for f in dosyalar:
        df = pd.read_csv(f, sep="|", usecols=["SepsisLabel"])
        kayit.append((hastane + "_" + os.path.basename(f)[:-4], hastane, int(df["SepsisLabel"].max())))
    return pd.DataFrame(kayit, columns=["hasta_id", "hastane", "sepsis"])

A, B = hasta_tablosu("A"), hasta_tablosu("B")
print("A:", len(A), "hasta,", A.sepsis.sum(), "sepsisli | B:", len(B), "hasta,", B.sepsis.sum(), "sepsisli")

# A→B: kaynak A %70 eğitim / %15 doğrulama / %15 kaynak-içi test; hedef B %30 adaptasyon / %70 test
a_egitim, a_gecici = train_test_split(A, test_size=0.30, stratify=A.sepsis, random_state=SEED)
a_dogr, a_test = train_test_split(a_gecici, test_size=0.50, stratify=a_gecici.sepsis, random_state=SEED)
b_adapt, b_test = train_test_split(B, test_size=0.70, stratify=B.sepsis, random_state=SEED)

# B→A: roller tersine
b_egitim, b_gecici = train_test_split(B, test_size=0.30, stratify=B.sepsis, random_state=SEED)
b_dogr, b_test_ic = train_test_split(b_gecici, test_size=0.50, stratify=b_gecici.sepsis, random_state=SEED)
a_adapt, a_test_dis = train_test_split(A, test_size=0.70, stratify=A.sepsis, random_state=SEED)

def etiketle(df, yon, kume):
    d = df.copy(); d["yon"] = yon; d["kume"] = kume; return d

bolme = pd.concat([
    etiketle(a_egitim, "A->B", "egitim"), etiketle(a_dogr, "A->B", "dogrulama"), etiketle(a_test, "A->B", "kaynak_ici_test"),
    etiketle(b_adapt, "A->B", "adaptasyon"), etiketle(b_test, "A->B", "hedef_test"),
    etiketle(b_egitim, "B->A", "egitim"), etiketle(b_dogr, "B->A", "dogrulama"), etiketle(b_test_ic, "B->A", "kaynak_ici_test"),
    etiketle(a_adapt, "B->A", "adaptasyon"), etiketle(a_test_dis, "B->A", "hedef_test"),
])
bolme.to_csv("bolme.csv", index=False)

# --- Kesişim yok testi (hocanın istediği) ---
for yon, g in bolme.groupby("yon"):
    kumeler = {k: set(x.hasta_id) for k, x in g.groupby("kume")}
    isimler = list(kumeler)
    for i in range(len(isimler)):
        for j in range(i + 1, len(isimler)):
            ortak = kumeler[isimler[i]] & kumeler[isimler[j]]
            assert not ortak, f"SIZINTI! {yon}: {isimler[i]} ∩ {isimler[j]} = {len(ortak)} hasta"
    print(yon, "-> kesişim yok ✔ |", {k: len(v) for k, v in kumeler.items()})
print(bolme.groupby(["yon", "kume"]).sepsis.agg(["count", "sum", "mean"]).round(3))
