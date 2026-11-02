import pytest
from sepsis.data.split import patient_table, make_split, assert_disjoint

def test_split_has_no_overlap(raw_dir):
    split = make_split(patient_table(raw_dir))
    assert_disjoint(split)                       # ihlalde AssertionError

def test_every_patient_appears_once_per_direction(raw_dir):
    p = patient_table(raw_dir); s = make_split(p)
    for _, g in s.groupby("yon"):
        assert g.hasta_id.is_unique and set(g.hasta_id) == set(p.hasta_id)

def test_split_is_reproducible(raw_dir):
    p = patient_table(raw_dir)
    a, b = make_split(p, seed=42), make_split(p, seed=42)
    assert a.equals(b)

def test_overlap_is_detected(raw_dir):
    s = make_split(patient_table(raw_dir))
    bad = s.copy(); row = bad[(bad.yon == "A->B") & (bad.kume == "egitim")].iloc[[0]].copy(); row["kume"] = "hedef_test"
    import pandas as pd
    with pytest.raises(AssertionError):
        assert_disjoint(pd.concat([bad, row]))
