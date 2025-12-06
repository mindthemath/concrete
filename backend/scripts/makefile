train_nn:
	uv run  --extra train --extra cpu python train.py --model nn

train_gp:
	uv run  --extra train --extra cpu python train.py --model gp

debug:
	uv run --extra train --extra cpu python -i debug_ckpt.py

lint:
	uvx black .
	uvx isort --profile black .

clean:
	rm -rf checkpoints __pycache__ lightning_logs best_ckpt_path.txt

mock:
	uv run mock_data.py

physics_nn.onnx: prep_nn_ckpt.py train_nn
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_nn_ckpt.py

physics_gp.onnx: prep_gp_ckpt.py
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_gp_ckpt.py

physics_gp_cov.onnx: prep_gp_cov_ckpt.py
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_gp_cov_ckpt.py

prep_nn: physics_nn.onnx

prep_gp: physics_gp.onnx

prep_gp_cov: physics_gp_cov.onnx

.PHONY: physics_nn.onnx physics_gp.onnx
