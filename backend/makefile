upgrade:
	uv lock --upgrade

requirements-dev.txt: pyproject.toml uv.lock
	uv pip compile --extra cpu --extra api --extra train pyproject.toml --group dev -o requirements-dev.txt
