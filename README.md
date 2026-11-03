# Sepsis Early Warning under Hospital Distribution Shift

Hastaneler arası dağılım kayması altında yoğun bakımda sepsis erken uyarısı (PhysioNet/CinC 2019, eğitim setleri A ve B).
Atatürk Üniversitesi, İngilizce Bilgisayar Mühendisliği — bitirme projesi. **Araştırma prototipidir; klinik karar aracı değildir.**

## Yapı

```
├── configs/default.yaml     # tüm yollar ve parametreler (tek yapılandırma kaynağı)
├── src/sepsis/              # kurulabilir paket
│   ├── data/                #   loader, hasta düzeyinde bölme, ön işleme (tek kaynak)
│   ├── models/              #   lr.py, gru.py (baseline'lar); özgün model eklenecek
│   ├── evaluation/          #   metrikler
│   └── api/                 #   FastAPI uygulaması + basit arayüz
├── scripts/                 # ince komutlar: 01_make_split, 02_preprocess, 03_train_lr, 04_train_gru
├── tests/                   # pytest: kesişim yok, nedensellik, API
├── splits/bolme.csv         # dondurulmuş hasta bölmesi (depoda)
├── results/                 # küçük çıktılar: JSON, grafikler (depoda)
├── data/, artifacts/        # ön işlenmiş veri, eğitilmiş modeller (depoda DEĞİL)
└── docs/
```

## Çalıştırma

```bash
python3 -m venv venv && source venv/bin/activate
make install                  # paketi kur
# ham veri: PhysioNet/CinC 2019 training_setA/B  ->  configs/default.yaml içinde paths.raw_dir
make split                    # 1) bölme (dondurulmuşsa yalnızca kesişim kontrolü)
make preprocess               # 2) ön işleme (parametreler yalnızca eğitim hastalarından)
make train                    # 3) LR baseline: eğitim/doğrulama/test + öğrenme eğrisi
make train-gru                # 4) GRU baseline (nedensel, tek yönlü): eğitim/doğrulama/test + epoch öğrenme eğrisi
make api                      # web uygulaması: http://127.0.0.1:8000
make test                     # birim testleri (gerçek veri gerekmez)
```

## Sızıntı ve aşırı öğrenme kontrolleri

| Kontrol | Nerede |
|---|---|
| Hasta düzeyinde bölme; kümeler arasında kesişim yok | `data/split.py::assert_disjoint`, `tests/test_split.py` |
| Bölme dondurulur, yeniden üretilebilir | `scripts/01_make_split.py`, `splits/bolme.csv` |
| Ölçekleme/doldurma parametreleri yalnızca eğitimden | `data/preprocess.py::fit_params`, `tests/test_preprocess.py` |
| Yalnızca geçmiş saatler (nedensel) | `tests/test_preprocess.py::test_transform_is_causal` |
| Hiperparametre yalnızca doğrulamada; test bir kez | `models/lr.py`, `scripts/03_train_lr.py` |
| Eğitim, doğrulama ve test birlikte raporlanır | `results/lr_*.json` |

## Sonuçlar (LR baseline, saat düzeyi)

`results/lr_A_B.json` ve `results/lr_B_A.json`. AUPRC'yi pozitif oranıyla birlikte okuyun (rastgele modelin AUPRC'si pozitif orana eşittir).

## Veri

PhysioNet/CinC Challenge 2019, v1.0.0, DOI 10.13026/v64v-d857, lisans CC BY 4.0. Ham veri bu depoda **yoktur**.
