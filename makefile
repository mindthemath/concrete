build:
	docker build -t mindthemath/concrete-app .

run:
	docker run --rm -p 3030:3030 --name concrete mindthemath/concrete-app
