### Kaynak içi test (grud, A->B)

Eşik = 0.8447: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %3.51.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9828 | 0.9692 | 0.9759 | 116844 |
| sepsis_var | 0.1401 | 0.2282 | 0.1736 | 2572 |
| macro avg | 0.5614 | 0.5987 | 0.5748 | 119416 |
| weighted avg | 0.9646 | 0.9532 | 0.9586 | 119416 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8008, sepsis_var = 0.8008, macro = 0.8008 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9532; dengeli accuracy = 0.5987 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 113240 | 3604 |
| **Gerçek: sepsis var** | 1985 | 587 |
