### Doğrulama (grud, A->B)

Eşik = 0.8447: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %5.00.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9838 | 0.9550 | 0.9692 | 117221 |
| sepsis_var | 0.1192 | 0.2788 | 0.1670 | 2561 |
| macro avg | 0.5515 | 0.6169 | 0.5681 | 119782 |
| weighted avg | 0.9653 | 0.9405 | 0.9520 | 119782 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.7910, sepsis_var = 0.7910, macro = 0.7910 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9405; dengeli accuracy = 0.6169 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 111945 | 5276 |
| **Gerçek: sepsis var** | 1847 | 714 |
