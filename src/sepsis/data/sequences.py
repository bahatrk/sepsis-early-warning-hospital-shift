"""Ön işlenmiş satır matrisini hasta dizilerine çevirir (hasta başına bir dizi, saat sırasıyla)."""
import numpy as np
import torch

def load_sequences(npz_path: str):
    """npz -> (X_list, y_list): her eleman bir hastanın (T, d) / (T,) dizisi. pid hasta sırasını verir."""
    d = np.load(npz_path)
    X, y, pid = d["X"], d["y"], d["pid"]
    cuts = np.flatnonzero(np.diff(pid)) + 1
    return np.split(X, cuts), np.split(y, cuts)

def collate(batch_X, batch_y):
    """Hastaları en uzununa kadar sıfırla doldurur; maske gerçek saatleri işaretler."""
    B, T, d = len(batch_X), max(len(x) for x in batch_X), batch_X[0].shape[1]
    X = torch.zeros(B, T, d); y = torch.zeros(B, T); mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (xi, yi) in enumerate(zip(batch_X, batch_y)):
        n = len(xi); X[i, :n] = torch.from_numpy(xi); y[i, :n] = torch.from_numpy(yi.astype(np.float32)); mask[i, :n] = True
    return X, y, mask
