### Hedef test, uyarlanmış eşik (grud, A->B)

Eşik = 0.7153: hedef hastanenin ETİKETSİZ uyarlama kümesindeki saatlerin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %4.96.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9898 | 0.9542 | 0.9717 | 525321 |
| sepsis_var | 0.0897 | 0.3146 | 0.1396 | 7533 |
| macro avg | 0.5398 | 0.6344 | 0.5556 | 532854 |
| weighted avg | 0.9771 | 0.9452 | 0.9599 | 532854 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.7775, sepsis_var = 0.7775, macro = 0.7775 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9452; dengeli accuracy = 0.6344 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 501269 | 24052 |
| **Gerçek: sepsis var** | 5163 | 2370 |

### Hedef test: etiketsiz eşik uyarlaması

İki eşik de %5 alarm oranını hedefler; fark eşiğin hangi kümeden seçildiğidir. Uyarlanmış eşik hedef hastanenin etiketsiz `adaptasyon` kümesinden hesaplanır; etiket ve test kümesi kullanılmaz.

| Eşik | Eşik değeri | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|
| dogrulama_esigi | 0.8447 | %2.33 | 0.110 | 0.181 | 0.137 | 0.580 |
| uyarlanmis_esik | 0.7153 | %4.96 | 0.090 | 0.315 | 0.140 | 0.634 |
