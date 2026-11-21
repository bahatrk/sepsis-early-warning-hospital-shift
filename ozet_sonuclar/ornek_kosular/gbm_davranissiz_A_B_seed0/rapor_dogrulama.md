### Doğrulama (gbm_davranissiz, A->B)

Eşik = 0.1100: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %5.00.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9837 | 0.9549 | 0.9691 | 117221 |
| sepsis_var | 0.1175 | 0.2749 | 0.1647 | 2561 |
| macro avg | 0.5506 | 0.6149 | 0.5669 | 119782 |
| weighted avg | 0.9652 | 0.9404 | 0.9519 | 119782 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.7933, sepsis_var = 0.7933, macro = 0.7933 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9404; dengeli accuracy = 0.6149 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 111935 | 5286 |
| **Gerçek: sepsis var** | 1857 | 704 |
