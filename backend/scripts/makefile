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

physics_nn.onnx: prep_nn_ckpt.py run_gp.py
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_nn_ckpt.py

prep: physics_nn.onnx

.PHONY: physics_nn.onnx
