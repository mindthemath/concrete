dev:
	uv run python -i run_gp.py

debug:
	uv run python -i debug_ckpt.py

lint:
	uvx black .
	uvx isort --profile black .

clean:
	rm -rf checkpoints __pycache__ lightning_logs

mock:
	uv run mock_data.py