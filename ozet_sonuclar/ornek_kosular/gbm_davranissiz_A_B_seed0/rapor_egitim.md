### Eğitim (gbm_davranissiz, A->B)

Eşik = 0.1100: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %3.48.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9839 | 0.9708 | 0.9773 | 539014 |
| sepsis_var | 0.1782 | 0.2844 | 0.2191 | 12003 |
| macro avg | 0.5810 | 0.6276 | 0.5982 | 551017 |
| weighted avg | 0.9663 | 0.9558 | 0.9608 | 551017 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8335, sepsis_var = 0.8335, macro = 0.8335 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9558; dengeli accuracy = 0.6276 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 523269 | 15745 |
| **Gerçek: sepsis var** | 8589 | 3414 |
