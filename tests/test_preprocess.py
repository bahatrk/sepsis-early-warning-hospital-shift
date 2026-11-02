import numpy as np
from sepsis.data.loader import find_patient_files, read_patient
from sepsis.data.preprocess import fit_params, transform, feature_names

def _frames(raw_dir, n=40):
    files = sorted(find_patient_files(raw_dir).values())[:n]
    return [read_patient(f) for f in files]

def test_transform_is_causal(raw_dir):
    """Son saatleri silmek, kalan saatlerin çıktısını DEĞİŞTİRMEMELİ (gelecek sızıntısı yok)."""
    frames = _frames(raw_dir); params = fit_params(frames)
    for df in frames:
        k = len(df) // 2
        assert np.allclose(transform(df, params)[:k], transform(df.iloc[:k].copy(), params))

def test_params_depend_only_on_given_training_frames(raw_dir):
    """Parametreler yalnızca verilen hastalardan öğrenilir: test hastalarını eklemek eğitim parametresini değiştirmemeli."""
    frames = _frames(raw_dir, 60)
    p_train = fit_params(frames[:30])
    p_again = fit_params(frames[:30])
    assert p_train == p_again
    assert fit_params(frames[:60]) != p_train           # farklı veriyle farklı çıkar (testin kendisi anlamlı)

def test_feature_count_and_no_nan(raw_dir):
    frames = _frames(raw_dir); params = fit_params(frames)
    X = transform(frames[0], params)
    assert X.shape == (len(frames[0]), len(feature_names())) and not np.isnan(X).any()

def test_iculos_ablation_removes_one_feature(raw_dir):
    assert len(feature_names(False)) == len(feature_names(True)) - 1
