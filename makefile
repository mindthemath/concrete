dev:
	./bin/bun run dev

install:
	./bin/bun install --frozen-lockfile

update:
	./bin/bun update

build:
	./bin/bun run build

lint:
	./bin/bun run lint

docker-build:
	docker build -t mindthemath/concrete-app .

docker-run:
	docker run --rm -p 3030:3000 mindthemath/concrete-app
