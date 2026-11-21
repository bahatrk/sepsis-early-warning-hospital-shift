### Hedef test (gbm_davranissiz, A->B)

Eşik = 0.1100: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %2.85.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9888 | 0.9744 | 0.9816 | 525321 |
| sepsis_var | 0.1154 | 0.2327 | 0.1543 | 7533 |
| macro avg | 0.5521 | 0.6036 | 0.5679 | 532854 |
| weighted avg | 0.9765 | 0.9639 | 0.9699 | 532854 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8078, sepsis_var = 0.8078, macro = 0.8078 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9639; dengeli accuracy = 0.6036 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 511878 | 13443 |
| **Gerçek: sepsis var** | 5780 | 1753 |
