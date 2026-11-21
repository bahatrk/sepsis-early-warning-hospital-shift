### Eğitim (grud, A->B)

Eşik = 0.8447: doğrulama saatlerinin %5'i alarm verecek şekilde seçildi (test kümesine ve etiketlere bakılmadı). Bu kümede gerçekleşen alarm oranı %3.22.

| Sınıf | Precision | Recall | F1 | Destek (saat) |
|---|---|---|---|---|
| sepsis_yok | 0.9835 | 0.9730 | 0.9782 | 539014 |
| sepsis_var | 0.1799 | 0.2661 | 0.2146 | 12003 |
| macro avg | 0.5817 | 0.6195 | 0.5964 | 551017 |
| weighted avg | 0.9660 | 0.9576 | 0.9616 | 551017 |

Sınıf bazında ROC AUC (bire karşı hepsi): sepsis_yok = 0.8379, sepsis_var = 0.8379, macro = 0.8379 (iki sınıflı problemde iki sınıfın AUC değeri aynıdır).

Accuracy = 0.9576; dengeli accuracy = 0.6195 (sepsis saatleri nadir olduğundan accuracy tek başına yanıltıcıdır).

| | Tahmin: sepsis yok | Tahmin: sepsis var |
|---|---|---|
| **Gerçek: sepsis yok** | 524450 | 14564 |
| **Gerçek: sepsis var** | 8809 | 3194 |
