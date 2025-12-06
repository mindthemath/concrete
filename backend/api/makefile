lint:
	uvx black .
	uvx isort --profile black .

# .PHONY: requirements.txt
requirements.txt: ../pyproject.toml
	uv pip compile ../pyproject.toml --extra api -o requirements.txt

dev:
	ENVIRONMENT=dev uv run --extra api --extra cpu python server.py

build:
	docker build -f Dockerfile -t mindthemath/concrete-api ../..

run: build
	docker run --rm -p 9600:9600 --name concrete-api -e ENVIRONMENT=dev mindthemath/concrete-api

# scalers are the same between both model architectures (share the same training script)
nn.tar.gz:
	cd ../scripts && tar -czvf ../api/nn.tar.gz \
	physics_nn.onnx physics_nn.onnx.data \
	scalers/scaler_x.joblib scalers/scaler_y.joblib

gp.tar.gz:
	cd ../scripts && tar -czvf ../api/gp.tar.gz \
	physics_gp.onnx \
	scalers/scaler_x.joblib scalers/scaler_y.joblib

gp_cov.tar.gz:
	cd ../scripts && tar -czvf ../api/gp_cov.tar.gz \
	physics_gp_cov.onnx \
	scalers/scaler_x.joblib scalers/scaler_y.joblib

data: nn.tar.gz gp.tar.gz gp_cov.tar.gz

clean:
	rm -f nn.tar.gz gp.tar.gz gp_cov.tar.gz
