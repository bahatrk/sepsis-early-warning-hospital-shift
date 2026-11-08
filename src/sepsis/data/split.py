"""Hasta düzeyinde bölme ve kesişim kontrolü. Aynı hastanın tüm saatleri aynı kümede kalır."""
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sepsis.data.loader import find_patient_files, read_patient

def patient_table(raw_dir: str) -> pd.DataFrame:
    """Her hasta için tek satır: hasta_id, hastane, sepsis (hastanın herhangi bir saatinde SepsisLabel=1)."""
    rows = []
    for pid, path in sorted(find_patient_files(raw_dir).items()):
        rows.append((pid, pid[0], int(read_patient(path, ["SepsisLabel"])["SepsisLabel"].max())))
    return pd.DataFrame(rows, columns=["hasta_id", "hastane", "sepsis"])

def _tag(df, direction, name):
    d = df.copy(); d["yon"] = direction; d["kume"] = name; return d

def _split_direction(source, target, direction, seed, src_frac, tgt_frac):
    tr, rest = train_test_split(source, test_size=src_frac, stratify=source.sepsis, random_state=seed)
    val, src_test = train_test_split(rest, test_size=0.50, stratify=rest.sepsis, random_state=seed)
    adapt, tgt_test = train_test_split(target, test_size=tgt_frac, stratify=target.sepsis, random_state=seed)
    return [_tag(tr, direction, "egitim"), _tag(val, direction, "dogrulama"), _tag(src_test, direction, "kaynak_ici_test"),
            _tag(adapt, direction, "adaptasyon"), _tag(tgt_test, direction, "hedef_test")]

def make_split(patients: pd.DataFrame, seed=42, source_val_test_fraction=0.30, target_test_fraction=0.70) -> pd.DataFrame:
    a = patients[patients.hastane == "A"]; b = patients[patients.hastane == "B"]
    parts = _split_direction(a, b, "A->B", seed, source_val_test_fraction, target_test_fraction)
    parts += _split_direction(b, a, "B->A", seed, source_val_test_fraction, target_test_fraction)
    return pd.concat(parts, ignore_index=True)

def assert_disjoint(split: pd.DataFrame) -> None:
    """Her yönde hiçbir hasta iki kümede bulunmamalı; ihlalde AssertionError."""
    for direction, g in split.groupby("yon"):
        sets = {k: set(x.hasta_id) for k, x in g.groupby("kume")}
        for (n1, s1), (n2, s2) in itertools.combinations(sets.items(), 2):
            common = s1 & s2
            assert not common, f"SIZINTI: {direction}: {n1} ∩ {n2} = {len(common)} hasta"
