# Mimari kararlar

1. **Tek ön işleme kaynağı.** `sepsis.data.preprocess` hem eğitim hattında hem API'de kullanılır; kopya kod yoktur.
2. **Dondurulmuş bölme.** `splits/bolme.csv` bir kez üretilir ve sürüm kontrolündedir; script varsa üzerine yazmaz.
3. **Yapılandırma dosyada.** Yollar, seed, oranlar ve model ızgarası `configs/default.yaml` içindedir.
4. **Kod İngilizce, veri şeması Türkçe.** Bölme CSV sütunları (`hasta_id, hastane, sepsis, yon, kume`) danışmana gönderilen
   sürümle uyumlu kalsın diye korunmuştur.
5. **Büyük çıktılar depoda yok.** `data/`, `artifacts/` `.gitignore`'dadır; `make` ile yeniden üretilir.
