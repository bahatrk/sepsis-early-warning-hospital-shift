"""Ham PhysioNet 2019 .psv dosyalarını bulma ve okuma."""
import glob, os
import pandas as pd

HOSPITALS = ("A", "B")

def find_patient_files(raw_dir: str) -> dict[str, str]:
    """hasta_id ('A_p000001') -> dosya yolu. hasta_id hastane harfini içerir; iki hastane karışmaz."""
    files = {}
    for h in HOSPITALS:
        for f in glob.glob(f"{raw_dir}/**/training_set{h}/*.psv", recursive=True):
            files[f"{h}_{os.path.basename(f)[:-4]}"] = f
    return files

def read_patient(path: str, columns=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", usecols=columns)
