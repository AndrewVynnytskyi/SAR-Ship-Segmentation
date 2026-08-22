.PHONY: setup format lint build jupyter

setup:
	pip install -r requirements.txt
	pip install ruff pre-commit
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .

build:
	docker build -t sar-ship-segmentation .

jupyter:
	docker run --rm -p 8888:8888 -v $(PWD):/workspace sar-ship-segmentation
