"""GRU-D tarzı eksiklik farkındalıklı nedensel kodlayıcı (Che vd., 2018'den esinlenilmiştir).

Girdi matrisi sepsis.data.preprocess.feature_names() sırasındaki 108 öznitelik sütunudur. Bunlar:
  * 34 ölçülen değişken için z-değer (son gözlem taşınmış, LOCF; hiç gözlenmemişse 0 = eğitim ortalaması),
  * 34 "gözlendi" maskesi m,
  * 34 "son ölçümden beri geçen süre" δ (log1p saat),
  * 6 diğer öznitelik (yaş, kabul zamanı, ICULOS, cinsiyet, birim): doğrudan hücre girdisine eklenir.

Baz GRU bu sütunların hepsini düz öznitelik olarak görür. Burada fark, eksikliğin MODELİN İÇİNDE kullanılmasıdır:
  * girdi çürümesi:  x̂_t = m·x_t + (1−m)·γx·x_son   (γx = exp(−ReLU(w⊙δ + b)); eğitim ortalaması 0 olduğundan
    uzun süre ölçülmeyen değişken ortalamaya çekilir),
  * gizli durum çürümesi: h_{t−1} ← γh ⊙ h_{t−1}  (γh = exp(−ReLU(W δ + b))).
Model yalnızca geçmişe bakar (saat t'nin çıktısı ≤ t saatlerine bağlıdır).

Ablasyon anahtarları: decay_x, decay_h. İkisi de kapalıysa x̂ = x (LOCF) olur ve δ kullanılmaz."""
import torch
from torch import nn

def grud_indices(names: list[str]) -> dict:
    """ozellikler.json'daki sütun adlarından değer / maske / δ / diğer sütun indekslerini çıkarır."""
    from sepsis.data.preprocess import MEASURED
    val = [names.index(c) for c in MEASURED]
    mask = [names.index(c + "_gozlendi") for c in MEASURED]
    delta = [names.index(c + "_son_olcum_suresi") for c in MEASURED]
    used = set(val) | set(mask) | set(delta)
    other = [i for i in range(len(names)) if i not in used]
    return {"deger": val, "maske": mask, "delta": delta, "diger": other}

class GRUD(nn.Module):
    def __init__(self, idx: dict, hidden=64, dropout=0.2, decay_x=True, decay_h=True):
        super().__init__()
        self.hidden, self.decay_x, self.decay_h = hidden, decay_x, decay_h
        for k, v in idx.items(): self.register_buffer(f"i_{k}", torch.tensor(v, dtype=torch.long), persistent=False)
        n = len(idx["deger"]); n_in = n + n + len(idx["diger"])           # x̂, m, diğer
        self.wx = nn.Parameter(torch.full((n,), 0.1)); self.bx = nn.Parameter(torch.full((n,), 0.1))   # ReLU'nun ölü başlamaması için > 0
        self.dec_h = nn.Linear(n, hidden); nn.init.normal_(self.dec_h.weight, 0.0, 0.05); nn.init.constant_(self.dec_h.bias, 0.1)
        self.cell = nn.GRUCell(n_in, hidden)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)

    def forward(self, x):                                                  # x: (B, T, 108) -> (B, T) logit
        v, m, d, o = x[..., self.i_deger], x[..., self.i_maske], x[..., self.i_delta], x[..., self.i_diger]
        if self.decay_x:
            gx = torch.exp(-torch.relu(d * self.wx + self.bx))
            v = m * v + (1 - m) * gx * v
        inp = torch.cat([v, m, o], dim=-1)
        gh = torch.exp(-torch.relu(self.dec_h(d))) if self.decay_h else None   # (B, T, H)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden); outs = []
        for t in range(T):
            if gh is not None: h = h * gh[:, t]
            h = self.cell(inp[:, t], h); outs.append(h)
        return self.out(self.drop(torch.stack(outs, dim=1))).squeeze(-1)
