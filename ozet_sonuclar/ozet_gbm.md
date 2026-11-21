| Model | Yön | Küme | AUROC (ort ± std) | AUPRC (ort ± std) | AUPRC / oran |
|---|---|---|---|---|---|
| gbm (n=5) | A->B | egitim | 0.837 ± 0.001 | 0.140 ± 0.001 | 6.4 |
| gbm (n=5) | A->B | dogrulama | 0.800 ± 0.002 | 0.091 ± 0.001 | 4.2 |
| gbm (n=5) | A->B | kaynak_ici_test | 0.815 ± 0.001 | 0.107 ± 0.002 | 5.0 |
| gbm (n=5) | A->B | hedef_test | 0.814 ± 0.001 | 0.071 ± 0.001 | 5.0 |
| gbm (n=5) | B->A | egitim | 0.877 ± 0.002 | 0.173 ± 0.006 | 12.2 |
| gbm (n=5) | B->A | dogrulama | 0.859 ± 0.003 | 0.131 ± 0.004 | 9.3 |
| gbm (n=5) | B->A | kaynak_ici_test | 0.845 ± 0.002 | 0.160 ± 0.005 | 11.4 |
| gbm (n=5) | B->A | hedef_test | 0.759 ± 0.004 | 0.077 ± 0.004 | 3.6 |
| gbm_davranissiz (n=5) | A->B | egitim | 0.835 ± 0.002 | 0.143 ± 0.002 | 6.6 |
| gbm_davranissiz (n=5) | A->B | dogrulama | 0.795 ± 0.002 | 0.094 ± 0.002 | 4.4 |
| gbm_davranissiz (n=5) | A->B | kaynak_ici_test | 0.811 ± 0.001 | 0.116 ± 0.001 | 5.4 |
| gbm_davranissiz (n=5) | A->B | hedef_test | 0.809 ± 0.002 | 0.092 ± 0.002 | 6.5 |
| gbm_davranissiz (n=5) | B->A | egitim | 0.862 ± 0.008 | 0.156 ± 0.010 | 11.0 |
| gbm_davranissiz (n=5) | B->A | dogrulama | 0.842 ± 0.009 | 0.122 ± 0.004 | 8.7 |
| gbm_davranissiz (n=5) | B->A | kaynak_ici_test | 0.831 ± 0.006 | 0.140 ± 0.009 | 10.0 |
| gbm_davranissiz (n=5) | B->A | hedef_test | 0.759 ± 0.004 | 0.087 ± 0.002 | 4.0 |

Aşırı öğrenme ölçütü (eğitim − doğrulama AUPRC ≤ 0,05):

| Model | Yön | Ortalama fark | Ölçütü aşan çalıştırma |
|---|---|---|---|
| gbm | A->B | 0.049 | 0 / 5 |
| gbm | B->A | 0.042 | 0 / 5 |
| gbm_davranissiz | A->B | 0.049 | 0 / 5 |
| gbm_davranissiz | B->A | 0.034 | 0 / 5 |

Hedef test, etiketsiz eşik uyarlaması (ort ± std; iki eşik de doğrulamadaki %5 alarm oranını hedefler):

| Model | Yön | Eşik | Gerçekleşen alarm oranı | Precision | Recall | F1 | Dengeli acc. |
|---|---|---|---|---|---|---|---|
| gbm | A->B | dogrulama_esigi | 0.025 ± 0.002 | 0.092 ± 0.002 | 0.160 ± 0.017 | 0.117 ± 0.006 | 0.569 ± 0.007 |
| gbm | A->B | uyarlanmis_esik | 0.049 ± 0.001 | 0.088 ± 0.001 | 0.305 ± 0.004 | 0.136 ± 0.001 | 0.630 ± 0.002 |
| gbm | B->A | dogrulama_esigi | 0.108 ± 0.011 | 0.077 ± 0.003 | 0.385 ± 0.029 | 0.128 ± 0.003 | 0.641 ± 0.010 |
| gbm | B->A | uyarlanmis_esik | 0.048 ± 0.001 | 0.102 ± 0.004 | 0.227 ± 0.012 | 0.141 ± 0.007 | 0.591 ± 0.006 |
| gbm_davranissiz | A->B | dogrulama_esigi | 0.028 ± 0.000 | 0.115 ± 0.002 | 0.231 ± 0.001 | 0.154 ± 0.001 | 0.603 ± 0.001 |
| gbm_davranissiz | A->B | uyarlanmis_esik | 0.049 ± 0.000 | 0.094 ± 0.000 | 0.328 ± 0.002 | 0.146 ± 0.001 | 0.641 ± 0.001 |
| gbm_davranissiz | B->A | dogrulama_esigi | 0.073 ± 0.005 | 0.100 ± 0.005 | 0.339 ± 0.007 | 0.155 ± 0.005 | 0.636 ± 0.002 |
| gbm_davranissiz | B->A | uyarlanmis_esik | 0.051 ± 0.001 | 0.112 ± 0.002 | 0.265 ± 0.009 | 0.157 ± 0.004 | 0.609 ± 0.004 |

Hedef test, olasılık kalibrasyonu (ort ± std). `platt_em`: kaynakta Platt + hedefin etiketsiz verisinden EM öncül düzeltmesi; `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır):

| Model | Yön | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Hedef oranı: EM kestirimi / gerçek |
|---|---|---|---|---|---|---|---|
| gbm | A->B | ham | 0.0047 ± 0.0004 | 0.0136 ± 0.0000 | 0.0246 ± 0.0006 | 0.0169 ± 0.0004 |  |
| gbm | A->B | platt | 0.0037 ± 0.0001 | 0.0136 ± 0.0000 | 0.0277 ± 0.0007 | 0.0150 ± 0.0004 |  |
| gbm | A->B | platt_em | 0.0141 ± 0.0000 | 0.0141 ± 0.0000 | -0.0143 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 / 0.0141 |
| gbm | A->B | platt_oracle_oncul | 0.0053 ± 0.0002 | 0.0136 ± 0.0000 | 0.0254 ± 0.0011 | 0.0100 ± 0.0003 |  |
| gbm | B->A | ham | 0.0065 ± 0.0017 | 0.0210 ± 0.0001 | 0.0109 ± 0.0070 | 0.0277 ± 0.0018 |  |
| gbm | B->A | platt | 0.0056 ± 0.0010 | 0.0209 ± 0.0001 | 0.0133 ± 0.0052 | 0.0253 ± 0.0015 |  |
| gbm | B->A | platt_em | 0.3758 ± 0.0302 | 0.2050 ± 0.0247 | -8.6798 ± 1.1674 | 0.3974 ± 0.0302 | 0.3995 / 0.0217 |
| gbm | B->A | platt_oracle_oncul | 0.0159 ± 0.0021 | 0.0219 ± 0.0003 | -0.0328 ± 0.0143 | 0.0375 ± 0.0021 |  |
| gbm_davranissiz | A->B | ham | 0.0052 ± 0.0002 | 0.0134 ± 0.0000 | 0.0357 ± 0.0007 | 0.0183 ± 0.0001 |  |
| gbm_davranissiz | A->B | platt | 0.0039 ± 0.0001 | 0.0134 ± 0.0000 | 0.0381 ± 0.0005 | 0.0163 ± 0.0001 |  |
| gbm_davranissiz | A->B | platt_em | 0.0141 ± 0.0000 | 0.0141 ± 0.0000 | -0.0143 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 / 0.0141 |
| gbm_davranissiz | A->B | platt_oracle_oncul | 0.0047 ± 0.0001 | 0.0135 ± 0.0000 | 0.0333 ± 0.0006 | 0.0109 ± 0.0001 |  |
| gbm_davranissiz | B->A | ham | 0.0024 ± 0.0008 | 0.0205 ± 0.0000 | 0.0321 ± 0.0011 | 0.0217 ± 0.0012 |  |
| gbm_davranissiz | B->A | platt | 0.0031 ± 0.0004 | 0.0205 ± 0.0000 | 0.0309 ± 0.0018 | 0.0204 ± 0.0007 |  |
| gbm_davranissiz | B->A | platt_em | 0.2272 ± 0.0240 | 0.0970 ± 0.0140 | -3.5806 ± 0.6602 | 0.2489 ± 0.0240 | 0.2488 / 0.0217 |
| gbm_davranissiz | B->A | platt_oracle_oncul | 0.0089 ± 0.0010 | 0.0209 ± 0.0001 | 0.0128 ± 0.0056 | 0.0304 ± 0.0010 |  |
