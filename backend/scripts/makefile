dev:
	uv run  --extra train --extra cpu python run_gp.py

debug:
	uv run --extra train --extra cpu python -i debug_ckpt.py

lint:
	uvx black .
	uvx isort --profile black .

clean:
	rm -rf checkpoints __pycache__ lightning_logs

mock:
	uv run mock_data.py

prep:
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_nn_ckpt.py
