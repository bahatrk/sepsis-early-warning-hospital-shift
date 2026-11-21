| Model | Yön | Küme | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / oran |
|---|---|---|---|---|---|
| gru (n=5) | A->B | egitim | 0.862 ± 0.011 | 0.140 ± 0.010 | 6.4 |
| gru (n=5) | A->B | dogrulama | 0.790 ± 0.002 | 0.098 ± 0.004 | 4.6 |
| gru (n=5) | A->B | kaynak_ici_test | 0.796 ± 0.005 | 0.101 ± 0.002 | 4.7 |
| gru (n=5) | A->B | hedef_test | 0.763 ± 0.013 | 0.073 ± 0.002 | 5.2 |
| gru (n=5) | B->A | egitim | 0.876 ± 0.013 | 0.153 ± 0.008 | 10.8 |
| gru (n=5) | B->A | dogrulama | 0.848 ± 0.005 | 0.125 ± 0.006 | 8.9 |
| gru (n=5) | B->A | kaynak_ici_test | 0.827 ± 0.003 | 0.157 ± 0.008 | 11.2 |
| gru (n=5) | B->A | hedef_test | 0.739 ± 0.008 | 0.076 ± 0.003 | 3.5 |

Aşırı öğrenme ölçütü (eğitim − doğrulama AUPRC ≤ 0,05):

| Model | Yön | Ortalama fark | Ölçütü aşan çalıştırma |
|---|---|---|---|
| gru | A->B | 0.042 | 0 / 5 |
| gru | B->A | 0.028 | 0 / 5 |

Hedef test, etiketsiz eşik uyarlaması (ort ± std; iki eşik de doğrulamadaki %5 alarm oranını hedefler):

| Model | Yön | Eşik | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|---|
| gru | A->B | dogrulama_esigi | 0.025 ± 0.003 | 0.105 ± 0.006 | 0.181 ± 0.013 | 0.132 ± 0.002 | 0.579 ± 0.005 |
| gru | A->B | uyarlanmis_esik | 0.051 ± 0.001 | 0.088 ± 0.003 | 0.316 ± 0.008 | 0.138 ± 0.005 | 0.634 ± 0.004 |
| gru | B->A | dogrulama_esigi | 0.052 ± 0.006 | 0.101 ± 0.005 | 0.243 ± 0.017 | 0.143 ± 0.003 | 0.597 ± 0.006 |
| gru | B->A | uyarlanmis_esik | 0.050 ± 0.001 | 0.103 ± 0.002 | 0.236 ± 0.010 | 0.143 ± 0.003 | 0.595 ± 0.004 |

Hedef test, olasılık kalibrasyonu (ort ± std). `platt_em`: kaynakta Platt + hedefin etiketsiz verisinden EM öncül düzeltmesi; `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır):

| Model | Yön | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Hedef oranı: EM kestirimi / gerçek |
|---|---|---|---|---|---|---|---|
| gru | A->B | ham | 0.2034 ± 0.0244 | 0.0898 ± 0.0104 | -5.4416 ± 0.7483 | 0.2176 ± 0.0244 |  |
| gru | A->B | platt | 0.0016 ± 0.0005 | 0.0135 ± 0.0000 | 0.0283 ± 0.0013 | 0.0144 ± 0.0007 |  |
| gru | A->B | platt_em | 0.0141 ± 0.0000 | 0.0141 ± 0.0000 | -0.0143 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 / 0.0141 |
| gru | A->B | platt_oracle_oncul | 0.0046 ± 0.0005 | 0.0136 ± 0.0000 | 0.0249 ± 0.0007 | 0.0096 ± 0.0005 |  |
| gru | B->A | ham | 0.2970 ± 0.0751 | 0.1646 ± 0.0443 | -6.7700 ± 2.0896 | 0.3186 ± 0.0751 |  |
| gru | B->A | platt | 0.0056 ± 0.0009 | 0.0207 ± 0.0000 | 0.0245 ± 0.0023 | 0.0165 ± 0.0008 |  |
| gru | B->A | platt_em | 0.0734 ± 0.0333 | 0.0348 ± 0.0092 | -0.6441 ± 0.4357 | 0.0951 ± 0.0333 | 0.0958 / 0.0217 |
| gru | B->A | platt_oracle_oncul | 0.0076 ± 0.0007 | 0.0208 ± 0.0001 | 0.0167 ± 0.0051 | 0.0248 ± 0.0012 |  |
