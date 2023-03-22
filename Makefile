BASE_DIR := $(CURDIR)
BIN_DIR := ${BASE_DIR}/bin
DATA_DIR := ${BASE_DIR}/data/${DATASET}
RUNS_DIR := ${BASE_DIR}/runs
CHESS_MOVES_CLASSIFICATION_MODEL_DIR := ${BASE_DIR}/chess_moves_classification_model

DOCKER_IMAGE := chess-vision-lab/chess_moves_classification_model
DOCKER_NAME := chess_moves_classification_model

DOCKER_CONSOLE_BASE := docker run -it --rm --gpus all \
	--shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
	--network="host" \
	-v ${BIN_DIR}:/workspace/bin \
	-v ${DATA_DIR}:/workspace/data \
	-v ${RUNS_DIR}:/workspace/runs \
	-v ${CHESS_MOVES_CLASSIFICATION_MODEL_DIR}:/workspace/chess_moves_classification_model \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-w /workspace/chess_moves_classification_model

DOCKER_CONSOLE := ${DOCKER_CONSOLE_BASE} --name ${DOCKER_NAME} ${DOCKER_IMAGE}

install:
	mkdir -p ${RUNS_DIR} && \
	ln -s ../runs chess_moves_classification_model/runs && \
	curl https://chessvisionlab.com/moves_dataset -L -o data/moves.zip && \
	unzip data/moves.zip -d data/

docker-build:
	docker images | grep -q "${DOCKER_IMAGE}" || \
	docker build -t ${DOCKER_IMAGE} .

docker-rm:
	docker rmi ${DOCKER_IMAGE}

console: docker-build
	${DOCKER_CONSOLE} /bin/bash

clean: docker-rm
	@true
