### Hedef test (grud, A->B)

Eşik = 0.8447: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %2.33.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9881 | 0.9790 | 0.9836 | 525321 |
| sepsis_var | 0.1098 | 0.1807 | 0.1366 | 7533 |
| macro avg | 0.5490 | 0.5798 | 0.5601 | 532854 |
| weighted avg | 0.9757 | 0.9677 | 0.9716 | 532854 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.7775, sepsis_var = 0.7775, macro = 0.7775 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9677; dengeli accuracy = 0.5798 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 514291 | 11030 |
| **Gerçek: sepsis var** | 6172 | 1361 |
