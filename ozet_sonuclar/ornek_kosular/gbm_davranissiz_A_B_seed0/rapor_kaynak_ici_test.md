### Kaynak içi test (gbm_davranissiz, A->B)

Eşik = 0.1100: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %4.01.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9838 | 0.9652 | 0.9744 | 116844 |
| sepsis_var | 0.1488 | 0.2768 | 0.1936 | 2572 |
| macro avg | 0.5663 | 0.6210 | 0.5840 | 119416 |
| weighted avg | 0.9658 | 0.9503 | 0.9576 | 119416 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8101, sepsis_var = 0.8101, macro = 0.8101 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9503; dengeli accuracy = 0.6210 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 112772 | 4072 |
| **Gerçek: sepsis var** | 1860 | 712 |
