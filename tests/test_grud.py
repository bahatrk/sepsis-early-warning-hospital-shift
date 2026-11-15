import numpy as np, pytest, torch
from sepsis.data.preprocess import feature_names, MEASURED
from sepsis.data.sequences import collate
from sepsis.models.grud import GRUD, grud_indices
from sepsis.models.gru import train_model, predict

NAMES = feature_names(True); IDX = grud_indices(NAMES); D = len(NAMES)

def _x(B=3, T=20, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randn(B, T, D, generator=g)
    n = len(MEASURED)
    x[..., IDX["maske"]] = (torch.rand(B, T, n, generator=g) < 0.3).float()          # 0/1 maske
    x[..., IDX["delta"]] = torch.rand(B, T, n, generator=g) * 3                        # δ >= 0
    return x

def test_indices_partition_all_columns():
    allidx = IDX["deger"] + IDX["maske"] + IDX["delta"] + IDX["diger"]
    assert sorted(allidx) == list(range(D)) and len(IDX["deger"]) == len(IDX["maske"]) == len(IDX["delta"]) == 34
    assert len(IDX["diger"]) == 6                                                      # Age, HospAdmTime, ICULOS, Gender, Unit1, Unit2

@pytest.mark.parametrize("dec", [True, False])
def test_grud_is_causal(dec):
    """Son saatleri değiştirmek, önceki saatlerin riskini DEĞİŞTİRMEMELİ (çürüme açık/kapalı)."""
    torch.manual_seed(0); m = GRUD(IDX, 8, 0.0, decay_x=dec, decay_h=dec).eval()
    x = _x(); x2 = x.clone(); x2[:, 12:] = _x(seed=1)[:, 12:]
    with torch.no_grad(): assert torch.allclose(m(x)[:, :12], m(x2)[:, :12], atol=1e-6)

def test_grud_padding_does_not_change_predictions():
    torch.manual_seed(0); m = GRUD(IDX, 8, 0.0).eval()
    a = _x(1, 7)[0].numpy(); b = _x(1, 15, seed=2)[0].numpy()
    X, _, _ = collate([a, b], [np.zeros(7), np.zeros(15)])
    with torch.no_grad(): padded = m(X)[0, :7]; alone = m(torch.from_numpy(a)[None])[0]
    assert torch.allclose(padded, alone, atol=1e-5)

def test_decay_switches_change_output_and_ablation_ignores_delta():
    torch.manual_seed(0); full = GRUD(IDX, 8, 0.0, True, True).eval()
    abl = GRUD(IDX, 8, 0.0, False, False).eval(); abl.load_state_dict(full.state_dict())
    x = _x(); x_d = x.clone(); x_d[..., IDX["delta"]] = x_d[..., IDX["delta"]] * 5       # δ değişir
    with torch.no_grad():
        assert not torch.allclose(full(x), abl(x), atol=1e-5)                            # çürüme etkili
        assert torch.allclose(abl(x), abl(x_d), atol=1e-6)                               # ablasyon δ'yı kullanmaz
        assert not torch.allclose(full(x), full(x_d), atol=1e-6)                         # tam model δ'ya bağlı

def _data(n, rng):
    Xs, ys = [], []
    for _ in range(n):
        T = int(rng.integers(8, 25)); x = _x(1, T, int(rng.integers(1_000_000)))[0].numpy().astype(np.float32)
        ys.append((x[:, IDX["deger"][0]] > 0.8).astype(np.int8)); Xs.append(x)
    return Xs, ys

@pytest.mark.parametrize("extra", [{}, {"noise_std": 0.1}, {"gap_limit": 0.05}])
def test_train_model_runs_with_options(extra):
    rng = np.random.default_rng(0); tr, va = _data(50, rng), _data(25, rng)
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 16, "max_epochs": 3, "patience": 3, **extra}
    model, hist = train_model(lambda: GRUD(IDX, 8, 0.1), tr, va, cfg, 0, "cpu", torch.tensor(IDX["deger"]), verbose=False)
    y, p = predict(model, *va)
    assert len(p) == sum(len(x) for x in va[0]) and 0 <= p.min() and p.max() <= 1
    for k in ["train_loss_eval", "val_loss", "train_acc", "val_acc", "train_auroc", "val_auroc", "train_auprc", "val_auprc", "esik"]:
        assert k in hist[0]
    assert 1 <= model.best_epoch <= len(hist)

def test_training_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0); tr, va = _data(40, rng), _data(20, rng)
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 16, "max_epochs": 2, "patience": 2}
    torch.set_num_threads(1)
    ps = []
    for _ in range(2):
        m, _ = train_model(lambda: GRUD(IDX, 8, 0.1), tr, va, cfg, 3, "cpu", verbose=False); ps.append(predict(m, *va)[1])
    assert np.allclose(ps[0], ps[1], atol=1e-7)


class _Probe(torch.nn.Module):
    """Eğitim/değerlendirme sırasında modele giren girdileri kaydeder."""
    def __init__(self):
        super().__init__(); self.lin = torch.nn.Linear(D, 1); self.train_x, self.eval_x = [], []
    def forward(self, x):
        (self.train_x if self.training else self.eval_x).append(x.detach().clone()); return self.lin(x).squeeze(-1)

def test_mask_drop_zeroes_indicators_only_in_training():
    rng = np.random.default_rng(0); tr, va = _data(30, rng), _data(15, rng)
    cols = IDX["maske"] + IDX["delta"]
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 10, "max_epochs": 1, "patience": 1, "mask_drop": 1.0}
    holder = {}
    def make(): holder["m"] = _Probe(); return holder["m"]
    train_model(make, tr, va, cfg, 0, "cpu", verbose=False, mask_idx=torch.tensor(cols))
    m = holder["m"]
    assert all(float(x[..., cols].abs().sum()) == 0.0 for x in m.train_x)               # p=1: eğitimde hepsi sıfır
    assert all(float(x[..., IDX["deger"]].abs().sum()) > 0 for x in m.train_x)           # değer sütunlarına dokunulmaz
    assert any(float(x[..., cols].abs().sum()) > 0 for x in m.eval_x)                    # doğrulamada gösterge sütunları korunur

def test_mask_drop_zero_keeps_results_identical():
    rng = np.random.default_rng(0); tr, va = _data(30, rng), _data(15, rng)
    cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 10, "max_epochs": 2, "patience": 2}
    torch.set_num_threads(1); ps = []
    for extra in ({}, {"mask_drop": 0.0}):
        m, _ = train_model(lambda: GRUD(IDX, 8, 0.1), tr, va, {**cfg, **extra}, 3, "cpu", verbose=False,
                           mask_idx=torch.tensor(IDX["maske"])); ps.append(predict(m, *va)[1])
    assert np.allclose(ps[0], ps[1], atol=1e-7)


def test_augmentations_apply_only_in_training_and_zero_keeps_results_identical():
    rng = np.random.default_rng(0); tr, va = _data(30, rng), _data(15, rng)
    base = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 10, "max_epochs": 1, "patience": 1}
    deger = torch.tensor(IDX["deger"]); diger = IDX["maske"] + IDX["delta"] + IDX["diger"]
    seen = {}
    def run(extra, key):
        def make(): seen[key] = _Probe(); return seen[key]
        train_model(make, tr, va, {**base, **extra}, 0, "cpu", noise_idx=deger, verbose=False)
    run({}, "kapali"); run({"scale_std": 0.5, "shift_std": 0.5}, "acik")
    k_tr = torch.cat([x.reshape(-1, D) for x in seen["kapali"].train_x]); a_tr = torch.cat([x.reshape(-1, D) for x in seen["acik"].train_x])
    assert not torch.allclose(k_tr[:, IDX["deger"]], a_tr[:, IDX["deger"]])                    # değer sütunları eğitimde değişir
    assert torch.allclose(k_tr[:, diger], a_tr[:, diger])                                      # diğer sütunlara dokunulmaz
    k_ev = torch.cat([x.reshape(-1, D) for x in seen["kapali"].eval_x]); a_ev = torch.cat([x.reshape(-1, D) for x in seen["acik"].eval_x])
    assert torch.allclose(k_ev, a_ev)                                                           # doğrulamada hiçbir şey değişmez
    torch.set_num_threads(1); ps = []
    for extra in ({}, {"noise_std": 0.0, "scale_std": 0.0, "shift_std": 0.0}):
        m, _ = train_model(lambda: GRUD(IDX, 8, 0.1), tr, va, {**base, "max_epochs": 2, "patience": 2, **extra}, 3, "cpu", deger, verbose=False)
        ps.append(predict(m, *va)[1])
    assert np.allclose(ps[0], ps[1], atol=1e-7)
