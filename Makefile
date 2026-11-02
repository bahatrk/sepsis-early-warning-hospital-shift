.PHONY: install split preprocess train api test
install:        ## paketi geliştirme kipinde kur
	pip install -e ".[dev]"
split:          ## 1) hasta düzeyinde bölme + kesişim kontrolü
	python scripts/01_make_split.py
preprocess:     ## 2) ön işleme (iki yön)
	python scripts/02_preprocess.py "A->B" && python scripts/02_preprocess.py "B->A"
train:          ## 3) LR baseline (iki yön)
	python scripts/03_train_lr.py "A->B" && python scripts/03_train_lr.py "B->A"
api:            ## web uygulaması: http://127.0.0.1:8000
	uvicorn sepsis.api.main:app --reload
test:           ## birim testleri (gerçek veri gerekmez)
	pytest -q
