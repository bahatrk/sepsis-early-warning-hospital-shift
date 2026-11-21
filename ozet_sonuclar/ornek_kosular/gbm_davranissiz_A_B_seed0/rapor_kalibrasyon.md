### Olasılık kalibrasyonu

Platt ölçekleme kaynak doğrulamasında, öncül kestirimi (EM) hedefin etiketsiz `adaptasyon` kümesinde yapılır. Hedef test etiketleri yalnızca raporlama içindir. `platt_oracle_oncul` tanı amaçlıdır (gerçek hedef oranını kullanır), yöntem değildir.

Sepsis oranı: kaynak 0.0214 | hedef, EM kestirimi 0.0000 | hedef, gerçek (tanı) 0.0141

| Küme | Biçim | ECE | Brier | Brier becerisi | Ort. tahmin | Gerçek oran |
|---|---|---|---|---|---|---|
| kaynak_ici_test | ham | 0.0042 | 0.0200 | 0.049 | 0.0226 | 0.0215 |
| kaynak_ici_test | platt | 0.0050 | 0.0201 | 0.048 | 0.0199 | 0.0215 |
| hedef_test | ham | 0.0054 | 0.0134 | 0.035 | 0.0184 | 0.0141 |
| hedef_test | platt | 0.0041 | 0.0134 | 0.038 | 0.0165 | 0.0141 |
| hedef_test | platt_em | 0.0141 | 0.0141 | -0.014 | 0.0000 | 0.0141 |
| hedef_test | platt_oracle_oncul | 0.0046 | 0.0135 | 0.033 | 0.0110 | 0.0141 |
