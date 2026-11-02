"""Sahte PhysioNet benzeri veri (testler gerçek veriye ihtiyaç duymaz)."""
import numpy as np, pandas as pd, pytest
from sepsis.data.preprocess import MEASURED

COLS = MEASURED + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]

def fake_patient(rng, sepsis_p=0.1):
    n = int(rng.integers(8, 40))
    df = pd.DataFrame(rng.normal(50, 10, (n, len(COLS))), columns=COLS)
    df[MEASURED[8:]] = df[MEASURED[8:]].mask(rng.random((n, len(MEASURED) - 8)) < 0.9)
    df["Gender"] = rng.integers(0, 2, n); df["Unit1"] = np.nan; df["Unit2"] = 1.0; df["ICULOS"] = np.arange(1, n + 1)
    y = np.zeros(n, int)
    if rng.random() < sepsis_p: y[n // 2:] = 1
    df["SepsisLabel"] = y
    return df

@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw"); rng = np.random.default_rng(0)
    for h, n in (("A", 200), ("B", 160)):
        d = root / f"training_set{h}"; d.mkdir()
        for i in range(n):
            fake_patient(rng, 0.09 if h == "A" else 0.06).to_csv(d / f"p{i:06d}.psv", sep="|", index=False)
    return str(root)
