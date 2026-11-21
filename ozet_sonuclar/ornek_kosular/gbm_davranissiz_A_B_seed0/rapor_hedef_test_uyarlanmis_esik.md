### Hedef test, uyarlanmış eşik (gbm_davranissiz, A->B)

Eşik = 0.0727: hedef hastanenin ETİKETSİZ uyarlama kümesindeki saatlerin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %4.93.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9900 | 0.9547 | 0.9720 | 525321 |
| sepsis_var | 0.0932 | 0.3246 | 0.1448 | 7533 |
| macro avg | 0.5416 | 0.6396 | 0.5584 | 532854 |
| weighted avg | 0.9773 | 0.9458 | 0.9603 | 532854 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8078, sepsis_var = 0.8078, macro = 0.8078 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9458; dengeli accuracy = 0.6396 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 501521 | 23800 |
| **Gerçek: sepsis var** | 5088 | 2445 |

### Hedef test: etiketsiz eşik uyarlaması

İki eşik de %5 alarm oranını hedefler; fark eşiğin hangi kümeden seçildiğidir. Uyarlanmış eşik hedef hastanenin etiketsiz `adaptasyon` kümesinden hesaplanır; etiket ve test kümesi kullanılmaz.

| Eşik | Eşik değeri | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|
| dogrulama_esigi | 0.1100 | %2.85 | 0.115 | 0.233 | 0.154 | 0.604 |
| uyarlanmis_esik | 0.0727 | %4.93 | 0.093 | 0.325 | 0.145 | 0.640 |
