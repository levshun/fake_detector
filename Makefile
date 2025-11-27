IMAGE_NAME = deepfake-swapping
CONTAINER_NAME = deepfake_runner

ifdef OS
	MODELS_PATH = $(shell echo %CD%\swapping\models)
else
	MODELS_PATH = $(PWD)/swapping/models
endif

.PHONY: build run run-cpu stop clean

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --name $(CONTAINER_NAME) --rm --gpus all -v "$(MODELS_PATH):/app/swapping/models" $(IMAGE_NAME)

run-cpu:
	docker run --name $(CONTAINER_NAME) --rm -v "$(MODELS_PATH):/app/swapping/models" $(IMAGE_NAME)

stop:
	-docker stop $(CONTAINER_NAME)

restart: stop build run

clean: stop
	-docker rmi $(IMAGE_NAME)