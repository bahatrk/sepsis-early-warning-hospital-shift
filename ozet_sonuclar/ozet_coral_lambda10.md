| Model | Yön | Küme | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / oran |
|---|---|---|---|---|---|
| gru_coral (n=5) | A->B | egitim | 0.855 ± 0.011 | 0.135 ± 0.007 | 6.2 |
| gru_coral (n=5) | A->B | dogrulama | 0.790 ± 0.001 | 0.096 ± 0.002 | 4.5 |
| gru_coral (n=5) | A->B | kaynak_ici_test | 0.794 ± 0.004 | 0.100 ± 0.003 | 4.7 |
| gru_coral (n=5) | A->B | hedef_test | 0.790 ± 0.017 | 0.082 ± 0.004 | 5.8 |
| gru_coral (n=5) | B->A | egitim | 0.877 ± 0.013 | 0.155 ± 0.008 | 10.9 |
| gru_coral (n=5) | B->A | dogrulama | 0.848 ± 0.005 | 0.125 ± 0.006 | 8.9 |
| gru_coral (n=5) | B->A | kaynak_ici_test | 0.827 ± 0.004 | 0.156 ± 0.008 | 11.2 |
| gru_coral (n=5) | B->A | hedef_test | 0.737 ± 0.007 | 0.075 ± 0.003 | 3.5 |

Aşırı öğrenme ölçütü (eğitim − doğrulama AUPRC ≤ 0,05):

| Model | Yön | Ortalama fark | Ölçütü aşan çalıştırma |
|---|---|---|---|
| gru_coral | A->B | 0.039 | 0 / 5 |
| gru_coral | B->A | 0.029 | 0 / 5 |

Hedef test, etiketsiz eşik uyarlaması (ort ± std; iki eşik de doğrulamadaki %5 alarm oranını hedefler):

| Model | Yön | Eşik | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|---|
| gru_coral | A->B | dogrulama_esigi | 0.032 ± 0.004 | 0.100 ± 0.007 | 0.226 ± 0.016 | 0.138 ± 0.006 | 0.599 ± 0.006 |
| gru_coral | A->B | uyarlanmis_esik | 0.051 ± 0.000 | 0.084 ± 0.004 | 0.303 ± 0.014 | 0.132 ± 0.006 | 0.628 ± 0.007 |
| gru_coral | B->A | dogrulama_esigi | 0.052 ± 0.006 | 0.101 ± 0.005 | 0.239 ± 0.018 | 0.141 ± 0.004 | 0.596 ± 0.006 |
| gru_coral | B->A | uyarlanmis_esik | 0.050 ± 0.001 | 0.101 ± 0.003 | 0.232 ± 0.010 | 0.141 ± 0.005 | 0.593 ± 0.005 |

Hedef test, olasılık kalibrasyonu (ort ± std). `platt_em`: kaynakta Platt + hedefin etiketsiz verisinden EM öncül düzeltmesi; `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır):

| Model | Yön | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Hedef oranı: EM kestirimi / gerçek |
|---|---|---|---|---|---|---|---|
| gru_coral | A->B | ham | 0.2131 ± 0.0241 | 0.1029 ± 0.0084 | -6.3864 ± 0.6017 | 0.2272 ± 0.0241 |  |
| gru_coral | A->B | platt | 0.0027 ± 0.0011 | 0.0135 ± 0.0001 | 0.0307 ± 0.0041 | 0.0165 ± 0.0013 |  |
| gru_coral | A->B | platt_em | 0.0141 ± 0.0000 | 0.0141 ± 0.0000 | -0.0143 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 / 0.0141 |
| gru_coral | A->B | platt_oracle_oncul | 0.0032 ± 0.0008 | 0.0135 ± 0.0000 | 0.0298 ± 0.0024 | 0.0110 ± 0.0008 |  |
| gru_coral | B->A | ham | 0.2929 ± 0.0721 | 0.1622 ± 0.0429 | -6.6574 ± 2.0230 | 0.3145 ± 0.0721 |  |
| gru_coral | B->A | platt | 0.0057 ± 0.0008 | 0.0207 ± 0.0000 | 0.0240 ± 0.0023 | 0.0164 ± 0.0007 |  |
| gru_coral | B->A | platt_em | 0.0696 ± 0.0282 | 0.0335 ± 0.0082 | -0.5807 ± 0.3853 | 0.0913 ± 0.0282 | 0.0920 / 0.0217 |
| gru_coral | B->A | platt_oracle_oncul | 0.0074 ± 0.0007 | 0.0208 ± 0.0001 | 0.0165 ± 0.0046 | 0.0247 ± 0.0010 |  |
