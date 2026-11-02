"""Web API: .psv yükle -> saatlik sepsis riski + alarm noktaları. Araştırma prototipidir; klinik karar aracı DEĞİLDİR.
Çalıştırma:  make api   (veya: uvicorn sepsis.api.main:app)"""
import io, json, os
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sepsis.data.preprocess import transform, load_params, zscore_columns, BINARY

STATIC = Path(__file__).parent / "static"
MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/lr/A_B")
model = joblib.load(f"{MODEL_DIR}/lr.joblib")
params = load_params(f"{MODEL_DIR}/parametreler.json")
meta = json.load(open(f"{MODEL_DIR}/meta.json"))

app = FastAPI(title="Sepsis Erken Uyarı Prototipi")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/saglik")
def health():
    return {"durum": "ok", "model": meta["yon"], "esik": meta["esik"]}

@app.post("/tahmin")
async def predict(dosya: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await dosya.read()), sep="|")
    except Exception:
        raise HTTPException(400, "Dosya okunamadı: '|' ile ayrılmış .psv bekleniyor.")
    missing = (set(params["ortalama"]) | set(BINARY)) - set(df.columns)
    if missing:
        raise HTTPException(400, f"Eksik sütunlar: {sorted(missing)[:5]}")
    if len(df) < 1:
        raise HTTPException(400, "Boş kayıt.")
    risk = model.predict_proba(transform(df, params))[:, 1]
    alarm = risk >= meta["esik"]
    first = int(alarm.argmax()) + 1 if alarm.any() else None
    return {"saat": list(range(1, len(df) + 1)), "risk": [round(float(r), 4) for r in risk],
            "alarm": [bool(a) for a in alarm], "esik": meta["esik"], "ilk_alarm_saati": first,
            "uyari": "Araştırma prototipi; klinik karar için kullanılamaz."}
