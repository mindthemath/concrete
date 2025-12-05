build:
	docker build -t mindthemath/concrete-app .

run:
	docker run --rm -p 3030:3000 mindthemath/concrete-app

