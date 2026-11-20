# Если рядом лежит файл .env (OPENAI_API_KEY=...), его переменные попадут в команды ниже.
-include .env
export

PYTHON = .venv/bin/python
LLM ?= replay

.PHONY: install eval record test

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

# make eval            — ответы из data/responses, ключ не нужен
# make eval LLM=openai — настоящая модель
eval:
	$(PYTHON) eval.py --llm $(LLM)

# Прогнать набор на настоящей модели и заменить заглушки в data/responses её ответами
record:
	$(PYTHON) eval.py --llm openai --record

test:
	$(PYTHON) -m pytest -q
