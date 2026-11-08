"""Ön işleme. Parametreler YALNIZCA eğitim hastalarından öğrenilir; dönüşüm yalnızca geçmiş saatlere bakar.
Hem eğitim hattı hem web API bu modülü kullanır (tek kaynak)."""
import json
import numpy as np, pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
        "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
MEASURED = VITALS + LABS                       # 34 zamanla değişen, eksik olabilen değişken
BINARY = ["Gender", "Unit1", "Unit2"]          # z-skor yok; Unit boşsa 0

def zscore_columns(use_iculos: bool = True) -> list[str]:
    cols = MEASURED + ["Age", "HospAdmTime", "ICULOS"]
    return cols if use_iculos else [c for c in cols if c != "ICULOS"]

def feature_names(use_iculos: bool = True) -> list[str]:
    return (zscore_columns(use_iculos) + BINARY + [c + "_gozlendi" for c in MEASURED]
            + [c + "_son_olcum_suresi" for c in MEASURED])

def fit_params(frames, use_iculos: bool = True) -> dict:
    """Ortalama ve std'yi yalnızca verilen (eğitim) hastalardan öğrenir."""
    cols = zscore_columns(use_iculos)
    allrows = pd.concat([f[cols] for f in frames])
    mean, std = allrows.mean(), allrows.std().replace(0, 1.0)
    return {"ortalama": mean.to_dict(), "std": std.to_dict(), "use_iculos": use_iculos}

def transform(df: pd.DataFrame, params: dict, clip: float = 5.0) -> np.ndarray:
    """Tek hastanın kaydını (saat x öznitelik) matrisine çevirir. Saat t'nin çıktısı yalnızca ≤ t bilgisine bağlıdır."""
    mean, std = pd.Series(params["ortalama"]), pd.Series(params["std"])
    cols = list(mean.index)
    n = len(df); t = np.arange(n)
    observed = df[MEASURED].notna().to_numpy()
    last = pd.DataFrame(np.where(observed, t[:, None], np.nan)).ffill().to_numpy()
    since = np.log1p(np.where(np.isnan(last), t[:, None] + 1, t[:, None] - last))
    z = ((df[cols].ffill() - mean) / std).clip(-clip, clip).fillna(0.0)
    binary = df[BINARY].copy(); binary[["Unit1", "Unit2"]] = binary[["Unit1", "Unit2"]].fillna(0)
    return np.hstack([z.to_numpy(), binary.to_numpy(), observed.astype(float), since]).astype(np.float32)

def save_params(params: dict, path: str) -> None:
    json.dump(params, open(path, "w", encoding="utf-8"), indent=1)

def load_params(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))
