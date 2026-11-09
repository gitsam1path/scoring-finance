# Raccourcis (macOS / Linux, ou Windows avec make installé)
install:
	pip install -r requirements.txt

train:
	cd src && python -m scoring.train

api:
	uvicorn api.main:app --reload

test:
	pytest -v

docker:
	docker build -t scoring-finance . && docker run -p 8000:8000 scoring-finance
