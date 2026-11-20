-include .env
export

PYTHON = .venv/bin/python
LLM ?= replay

.PHONY: install eval record test

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

eval:
	$(PYTHON) eval.py --llm $(LLM)

record:
	$(PYTHON) eval.py --llm openai --record

test:
	$(PYTHON) -m pytest -q
