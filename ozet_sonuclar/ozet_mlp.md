| Model | Yön | Küme | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / oran |
|---|---|---|---|---|---|
| mlp (n=5) | A->B | egitim | 0.775 ± 0.009 | 0.094 ± 0.007 | 4.3 |
| mlp (n=5) | A->B | dogrulama | 0.742 ± 0.010 | 0.085 ± 0.004 | 4.0 |
| mlp (n=5) | A->B | kaynak_ici_test | 0.752 ± 0.014 | 0.083 ± 0.004 | 3.9 |
| mlp (n=5) | A->B | hedef_test | 0.731 ± 0.015 | 0.060 ± 0.008 | 4.3 |
| mlp (n=5) | B->A | egitim | 0.816 ± 0.010 | 0.129 ± 0.009 | 9.1 |
| mlp (n=5) | B->A | dogrulama | 0.809 ± 0.014 | 0.095 ± 0.006 | 6.7 |
| mlp (n=5) | B->A | kaynak_ici_test | 0.795 ± 0.014 | 0.123 ± 0.013 | 8.8 |
| mlp (n=5) | B->A | hedef_test | 0.677 ± 0.016 | 0.049 ± 0.006 | 2.3 |

Aşırı öğrenme ölçütü (eğitim − doğrulama AUPRC ≤ 0,05):

| Model | Yön | Ortalama fark | Ölçütü aşan çalıştırma |
|---|---|---|---|
| mlp | A->B | 0.008 | 0 / 5 |
| mlp | B->A | 0.034 | 0 / 5 |

Hedef test, etiketsiz eşik uyarlaması (ort ± std; iki eşik de doğrulamadaki %5 alarm oranını hedefler):

| Model | Yön | Eşik | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|---|
| mlp | A->B | dogrulama_esigi | 0.012 ± 0.002 | 0.122 ± 0.014 | 0.107 ± 0.016 | 0.113 ± 0.012 | 0.548 ± 0.007 |
| mlp | A->B | uyarlanmis_esik | 0.051 ± 0.003 | 0.076 ± 0.006 | 0.275 ± 0.014 | 0.119 ± 0.008 | 0.614 ± 0.008 |
| mlp | B->A | dogrulama_esigi | 0.183 ± 0.036 | 0.048 ± 0.005 | 0.398 ± 0.049 | 0.085 ± 0.008 | 0.610 ± 0.013 |
| mlp | B->A | uyarlanmis_esik | 0.049 ± 0.001 | 0.071 ± 0.009 | 0.159 ± 0.017 | 0.098 ± 0.012 | 0.556 ± 0.009 |

Hedef test, olasılık kalibrasyonu (ort ± std). `platt_em`: kaynakta Platt + hedefin etiketsiz verisinden EM öncül düzeltmesi; `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır):

| Model | Yön | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Hedef oranı: EM kestirimi / gerçek |
|---|---|---|---|---|---|---|---|
| mlp | A->B | ham | 0.0160 ± 0.0111 | 0.0140 ± 0.0004 | -0.0046 ± 0.0310 | 0.0294 ± 0.0113 |  |
| mlp | A->B | platt | 0.0030 ± 0.0006 | 0.0136 ± 0.0001 | 0.0209 ± 0.0037 | 0.0135 ± 0.0011 |  |
| mlp | A->B | platt_em | 0.0141 ± 0.0000 | 0.0141 ± 0.0000 | -0.0143 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 / 0.0141 |
| mlp | A->B | platt_oracle_oncul | 0.0052 ± 0.0007 | 0.0137 ± 0.0000 | 0.0155 ± 0.0030 | 0.0089 ± 0.0007 |  |
| mlp | B->A | ham | 0.0203 ± 0.0042 | 0.0222 ± 0.0008 | -0.0468 ± 0.0376 | 0.0410 ± 0.0040 |  |
| mlp | B->A | platt | 0.0139 ± 0.0035 | 0.0217 ± 0.0004 | -0.0234 ± 0.0206 | 0.0315 ± 0.0047 |  |
| mlp | B->A | platt_em | 0.6235 ± 0.1436 | 0.4660 ± 0.1573 | -21.0013 ± 7.4236 | 0.6452 ± 0.1436 | 0.6469 / 0.0217 |
| mlp | B->A | platt_oracle_oncul | 0.0265 ± 0.0062 | 0.0233 ± 0.0010 | -0.1023 ± 0.0493 | 0.0468 ± 0.0067 |  |
