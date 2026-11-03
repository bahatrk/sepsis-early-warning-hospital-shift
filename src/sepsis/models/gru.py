"""GRU baseline. Tek yönlü (nedensel) GRU: saat t'nin riski yalnızca ≤ t saatlerine bağlıdır.
Doğrulama AUPRC'sine göre erken durdurma; test kümesi model seçiminde kullanılmaz."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sepsis.data.sequences import collate

class CausalGRU(nn.Module):
    def __init__(self, n_features, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)        # bidirectional=False: gelecek yok
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(self.drop(h)).squeeze(-1)                         # (B, T) logit

def _batches(X_list, y_list, batch_size, shuffle, rng):
    order = np.arange(len(X_list))
    if shuffle: rng.shuffle(order)
    else: order = np.argsort([len(x) for x in X_list])                    # değerlendirmede benzer uzunluklar bir arada (hız)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield idx, *collate([X_list[j] for j in idx], [y_list[j] for j in idx])

@torch.no_grad()
def predict(model, X_list, y_list, batch_size=256):
    """Satır düzeyinde (y, p) döndürür; hasta sırası korunur."""
    model.eval(); probs = [None] * len(X_list)
    for idx, X, _, mask in _batches(X_list, y_list, batch_size, False, None):
        p = torch.sigmoid(model(X)).numpy()
        for k, j in enumerate(idx): probs[j] = p[k, : len(X_list[j])]
    return np.concatenate(y_list), np.concatenate(probs)

def _scores(y, p):
    return average_precision_score(y, p), roc_auc_score(y, p)

def train_gru(train, val, n_features, cfg, seed):
    """train/val: (X_list, y_list). Döndürür: en iyi doğrulama AUPRC'li model, epoch geçmişi."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = CausalGRU(n_features, cfg["hidden"], cfg["layers"], cfg["dropout"])
    ytr = np.concatenate(train[1]); pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train(); total, count = 0.0, 0
        for _, X, y, mask in _batches(train[0], train[1], cfg["batch_size"], True, rng):
            opt.zero_grad()
            loss = (lossf(model(X), y) * mask).sum() / mask.sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * mask.sum().item(); count += mask.sum().item()
        (ytr_, ptr_), (yv, pv) = predict(model, *train), predict(model, *val)
        tr_auprc, _ = _scores(ytr_, ptr_); v_auprc, v_auroc = _scores(yv, pv)
        history.append({"epoch": epoch, "train_loss": total / count, "train_auprc": tr_auprc, "val_auprc": v_auprc, "val_auroc": v_auroc})
        print(f"epoch {epoch:2d} | kayıp {total/count:.4f} | eğitim AUPRC {tr_auprc:.4f} | doğrulama AUPRC {v_auprc:.4f}")
        if v_auprc > best: best, best_state, wait = v_auprc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]: print("Erken durdurma."); break
    model.load_state_dict(best_state)
    return model, history
