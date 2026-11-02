"""API uçtan uca: küçük bir model eğitip /tahmin ucunu dener."""
import json, os
import joblib, numpy as np, pandas as pd, pytest
from sklearn.linear_model import LogisticRegression
from sepsis.data.loader import find_patient_files, read_patient
from sepsis.data.preprocess import fit_params, transform, save_params, feature_names

@pytest.fixture()
def client(raw_dir, tmp_path, monkeypatch):
    files = sorted(find_patient_files(raw_dir).values())[:80]
    frames = [read_patient(f) for f in files]
    params = fit_params(frames)
    X = np.vstack([transform(f, params) for f in frames]); y = np.concatenate([f.SepsisLabel for f in frames])
    model = LogisticRegression(max_iter=200).fit(X, y)
    joblib.dump(model, tmp_path / "lr.joblib"); save_params(params, str(tmp_path / "parametreler.json"))
    json.dump({"yon": "A->B", "esik": 0.5}, open(tmp_path / "meta.json", "w"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from importlib import reload
    import sepsis.api.main as m; reload(m)
    from fastapi.testclient import TestClient
    return TestClient(m.app), files[0]

def test_health_and_home(client):
    c, _ = client
    assert c.get("/saglik").json()["durum"] == "ok" and c.get("/").status_code == 200

def test_predict_returns_one_risk_per_hour(client):
    c, f = client
    r = c.post("/tahmin", files={"dosya": ("p.psv", open(f, "rb").read())}).json()
    n = len(read_patient(f))
    assert len(r["risk"]) == n == len(r["saat"]) == len(r["alarm"]) and all(0 <= x <= 1 for x in r["risk"])

def test_bad_file_gives_400(client):
    c, _ = client
    assert c.post("/tahmin", files={"dosya": ("k.psv", b"a|b\n1|2\n")}).status_code == 400
