# Sonuçlar (özet)

Bu klasör, Google Drive'daki tam sonuç klasörlerinden seçilmiş çıktıları içerir (ham skorlar ve model dosyaları boyut nedeniyle repoda yok).

| Dosya | İçerik |
|---|---|
| `ozet_gru.md` | gru, 5 tohum × 2 yön (wd 0,3) |
| `ozet_gru_digerleri_grud.md` | gru_ozelliksiz, gru_maskdrop, grud (kendi model v1), grud_decaysiz |
| `ozet_gbm.md` | gbm, gbm_davranissiz (hedefte en iyi model) |
| `ozet_ek_baselineler_lr_rf_et_iw.md` | lr, rf, et, gbm_davranissiz_iw |
| `ozet_mlp.md` | MLP, güçlü düzenlileştirmeyle yeniden eğitim |
| `ozet_coral_lambda10.md` | gru_coral, λ = 10 |
| `ornek_kosular/grud_A_B_seed0/` | Kendi model v1 (GRU-D), A→B, tohum 0: eğriler, ROC/PR, karışıklık matrisleri, sınıflandırma raporları, tablo, kesişim kontrolü |
| `ornek_kosular/gbm_davranissiz_A_B_seed0/` | En iyi model, aynı çıktılar |

`egriler.png` dosyaları kayıtlı eğitim geçmişinden güncel kodla (seçilen epoch işaretli) yeniden çizilmiştir.
Ayrıntılı yorum: `Baha Türk bitirme v3.docx`.
