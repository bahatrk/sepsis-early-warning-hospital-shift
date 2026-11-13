"""GRU baseline ve ortak eğitim döngüsü. Tek yönlü (nedensel) GRU: saat t'nin riski yalnızca ≤ t saatlerine bağlıdır.
Doğrulama AUPRC'sine göre erken durdurma; test kümesi model seçiminde kullanılmaz.

Ortak döngü (train_model) hem CausalGRU hem de GRU-D tarzı modeli (models/grud.py) eğitir.
Varsayılan cihaz CPU'dur: eski sonuçlar (results/gru_*.json) bire bir aynı kalır."""
import copy, time
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
def predict_with_loss(model, X_list, y_list, batch_size=256, device="cpu", pos_weight=None):
    """Satır düzeyinde (y, p, kayıp) döndürür; hasta sırası korunur. Kayıp: eğitimdekiyle aynı ağırlıklı BCE
    (pos_weight verilmezse None). Dropout kapalı (eval) olduğundan eğitim ve doğrulama kayıpları kıyaslanabilir."""
    model.eval(); probs = [None] * len(X_list); tot, cnt = 0.0, 0
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none") if pos_weight is not None else None
    for idx, X, y, mask in _batches(X_list, y_list, batch_size, False, None):
        X = X.to(device); logit = model(X)
        if lossf is not None:
            m = mask.to(device); tot += (lossf(logit, y.to(device)) * m).sum().item(); cnt += m.sum().item()
        p = torch.sigmoid(logit).cpu().numpy()
        for k, j in enumerate(idx): probs[j] = p[k, : len(X_list[j])]
    return np.concatenate(y_list), np.concatenate(probs), (tot / cnt if cnt else float("nan"))

def predict(model, X_list, y_list, batch_size=256, device="cpu"):
    """Satır düzeyinde (y, p) döndürür; hasta sırası korunur."""
    y, p, _ = predict_with_loss(model, X_list, y_list, batch_size, device)
    return y, p

def _scores(y, p):
    return average_precision_score(y, p), roc_auc_score(y, p)

def _acc_at(y, p, thr):
    """Eşikte accuracy ve dengeli accuracy (sepsis çok nadir olduğu için ikisi birlikte raporlanır)."""
    pred = p >= thr; y = y.astype(bool)
    tp, tn = (pred & y).sum(), (~pred & ~y).sum(); pos, neg = y.sum(), (~y).sum()
    return float((tp + tn) / len(y)), float(0.5 * (tp / max(pos, 1) + tn / max(neg, 1)))

def train_model(make_model, train, val, cfg, seed, device="cpu", noise_idx=None, verbose=True):
    """Ortak eğitim döngüsü. make_model(): torch modülü üretir (tohum burada sabitlenir, model oluşturulmadan önce).
    train/val: (X_list, y_list). Döndürür: en iyi doğrulama AUPRC'li model, epoch geçmişi.

    cfg anahtarları: lr, weight_decay, batch_size, max_epochs, patience, alarm_rate (varsayılan 0.05);
    isteğe bağlı: noise_std (yalnızca eğitimde, noise_idx sütunlarına Gauss gürültüsü; varsayılan 0),
    gap_limit (varsa: eğitim-doğrulama AUPRC farkı bu sınırı aşan epoch'lar en iyi seçilmez; seçim yalnızca
    doğrulama bilgisine dayanır)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model().to(device)
    ytr = np.concatenate(train[1]); pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    noise = float(cfg.get("noise_std", 0.0) or 0.0); gap_limit = cfg.get("gap_limit"); alarm = float(cfg.get("alarm_rate", 0.05))
    best, best_state, wait, history, best_epoch = -1.0, None, 0, [], 0
    min_gap, gap_state, gap_epoch = float("inf"), None, 0                  # gap_limit hiçbir epoch'ta sağlanmazsa yedek
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time(); model.train(); total, count = 0.0, 0
        for _, X, y, mask in _batches(train[0], train[1], cfg["batch_size"], True, rng):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            if noise > 0 and noise_idx is not None:                       # veri artırma: yalnızca eğitimde
                X = X.clone(); X[..., noise_idx] += noise * torch.randn_like(X[..., noise_idx])
            opt.zero_grad()
            loss = (lossf(model(X), y) * mask).sum() / mask.sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * mask.sum().item(); count += mask.sum().item()
        (ytr_, ptr_, ltr), (yv, pv, lv) = (predict_with_loss(model, *train, device=device, pos_weight=pos_weight),
                                           predict_with_loss(model, *val, device=device, pos_weight=pos_weight))
        tr_auprc, tr_auroc = _scores(ytr_, ptr_); v_auprc, v_auroc = _scores(yv, pv)
        thr = float(np.quantile(pv, 1 - alarm))                           # eşik yalnızca doğrulamadan
        tr_acc, tr_bacc = _acc_at(ytr_, ptr_, thr); v_acc, v_bacc = _acc_at(yv, pv, thr)
        history.append({"epoch": epoch, "train_loss": total / count, "train_loss_eval": ltr, "val_loss": lv,
                        "train_auprc": tr_auprc, "val_auprc": v_auprc, "train_auroc": tr_auroc, "val_auroc": v_auroc,
                        "train_acc": tr_acc, "val_acc": v_acc, "train_bacc": tr_bacc, "val_bacc": v_bacc, "esik": thr,
                        "sure_sn": time.time() - t0})
        if verbose:
            print(f"epoch {epoch:2d} | kayıp {total/count:.4f} (eğitim) {lv:.4f} (doğr.) | AUPRC {tr_auprc:.4f} / {v_auprc:.4f} | AUROC {tr_auroc:.3f} / {v_auroc:.3f} | {time.time() - t0:.0f} sn")
        gap = tr_auprc - v_auprc
        if gap_limit is not None and gap < min_gap: min_gap, gap_state, gap_epoch = gap, copy.deepcopy(model.state_dict()), epoch
        eligible = gap_limit is None or gap <= gap_limit
        if eligible and v_auprc > best: best, best_state, wait, best_epoch = v_auprc, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= cfg["patience"]:
                if verbose: print("Erken durdurma.")
                break
    if best_state is None:                                                # gap_limit hiçbir epoch'ta sağlanmadı
        best_state, best_epoch = gap_state, gap_epoch
        if verbose: print(f"UYARI: gap_limit={gap_limit} hiçbir epoch'ta sağlanmadı; en küçük farklı epoch ({gap_epoch}) seçildi.")
    model.load_state_dict(best_state); model.best_epoch = best_epoch
    return model, history

def train_gru(train, val, n_features, cfg, seed, device="cpu"):
    """Eski arayüz (scripts/04_train_gru.py ve testler): nedensel GRU eğitir."""
    return train_model(lambda: CausalGRU(n_features, cfg["hidden"], cfg["layers"], cfg["dropout"]),
                       train, val, cfg, seed, device)
