.PHONY: install split preprocess train train-gru train-gru-seeds train-model train-model-seeds api test
install:        ## paketi geliştirme kipinde kur
	pip install -e ".[dev]"
split:          ## 1) hasta düzeyinde bölme + kesişim kontrolü
	python scripts/01_make_split.py
preprocess:     ## 2) ön işleme (iki yön)
	python scripts/02_preprocess.py "A->B" && python scripts/02_preprocess.py "B->A"
train:          ## 3) LR baseline (iki yön)
	python scripts/03_train_lr.py "A->B" && python scripts/03_train_lr.py "B->A"
train-gru:      ## 4) GRU baseline (iki yön; seed 0)
	python scripts/04_train_gru.py "A->B" && python scripts/04_train_gru.py "B->A"
train-gru-seeds: ## 5) GRU 5 seed + ortalama ± std özeti (results/gru_ozet.md)
	python scripts/05_gru_seeds.py
MODEL ?= grud
YON ?= A->B
SEED ?= 0
train-model:    ## 6) tek model + tüm raporlar (MODEL=gru|gru_ozelliksiz|grud|grud_decaysiz YON="A->B" SEED=0)
	python scripts/06_train_model.py "$(YON)" --model $(MODEL) --seed $(SEED)
train-model-seeds: ## 7) çoklu tohum + özet (results/modeller/ozet.md)
	python scripts/07_model_seeds.py --models gru grud
api:            ## web uygulaması: http://127.0.0.1:8000
	uvicorn sepsis.api.main:app --reload
test:           ## birim testleri (gerçek veri gerekmez)
	pytest -q
