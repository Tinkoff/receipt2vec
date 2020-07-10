PEP8=pycodestyle
IMAGE_NAME=test

.PHONY: flake8
flake8:
	docker build --no-cache -t flake8:${IMAGE_NAME} -f ./test/flake8.Dockerfile --build-arg workDir=${pwd} . > /dev/null
	docker run --rm flake8:${IMAGE_NAME}

.PHONY: pep8
pep8:
	docker build --no-cache -t ${PEP8}:${IMAGE_NAME} -f ./test/pycodestyle.Dockerfile --build-arg workDir=${pwd} . > /dev/null
	docker run --rm ${PEP8}:${IMAGE_NAME}

.PHONY: check
check: pep8 flake8