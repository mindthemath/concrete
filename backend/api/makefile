lint:
	uvx black .
	uvx isort --profile black .

# .PHONY: requirements.txt
requirements.txt: ../pyproject.toml
	uv pip compile ../pyproject.toml --extra api -o requirements.txt

dev:
	ENVIRONMENT=dev uv run --extra api --extra cpu python server.py

build:
	docker build -t mindthemath/concrete-api .

run:
	docker run --rm -p 9600:9600 mindthemath/concrete-api

