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

physics_nn.onnx: prep_nn_ckpt.py
	uv run --extra cpu --with onnxruntime --with onnxscript python prep_nn_ckpt.py

prep: physics_nn.onnx

.PHONY: physics_nn.onnx
