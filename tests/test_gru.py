import numpy as np, torch
from sepsis.models.gru import CausalGRU, train_gru, predict
from sepsis.data.sequences import collate

def test_gru_is_causal():
    """Son saatleri değiştirmek, önceki saatlerin riskini DEĞİŞTİRMEMELİ."""
    torch.manual_seed(0); m = CausalGRU(10).eval()
    x = torch.randn(3, 20, 10); x2 = x.clone(); x2[:, 12:] = torch.randn(3, 8, 10)
    with torch.no_grad():
        assert torch.allclose(m(x)[:, :12], m(x2)[:, :12], atol=1e-6)

def test_padding_does_not_change_predictions():
    torch.manual_seed(0); m = CausalGRU(10).eval()
    a = np.random.randn(7, 10).astype(np.float32); b = np.random.randn(15, 10).astype(np.float32)
    X, _, _ = collate([a, b], [np.zeros(7), np.zeros(15)])
    with torch.no_grad(): padded = m(X)[0, :7]; alone = m(torch.from_numpy(a)[None])[0]
    assert torch.allclose(padded, alone, atol=1e-5)

def test_training_runs_and_predicts_one_value_per_hour():
    rng = np.random.default_rng(0)
    def mk(n):
        Xs, ys = [], []
        for _ in range(n):
            T = int(rng.integers(8, 30)); x = rng.normal(size=(T, 6)).astype(np.float32)
            y = (x[:, 0] > 1.0).astype(np.int8); Xs.append(x); ys.append(y)
        return Xs, ys
    tr, va = mk(60), mk(30)
    cfg = {"hidden": 8, "layers": 1, "dropout": 0.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 16, "max_epochs": 3, "patience": 3}
    model, hist = train_gru(tr, va, 6, cfg, seed=0)
    y, p = predict(model, *va)
    assert len(p) == sum(len(x) for x in va[0]) and 0 <= p.min() and p.max() <= 1 and len(hist) >= 1
