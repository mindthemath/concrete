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

run: build
	docker run --rm -p 9600:9600 --name concrete-api mindthemath/concrete-api

artifacts.tar.gz:
	cd ../scripts && tar -czvf ../api/artifacts.tar.gz \
	physics_nn.onnx physics_nn.onnx.data \
	scalers/scaler_x.joblib scalers/scaler_y.joblib

data: artifacts.tar.gz

clean:
	rm -f artifacts.tar.gz
