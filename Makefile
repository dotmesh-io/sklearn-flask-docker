.PHONY: build-base
build-base:
	docker build -t quay.io/dotmesh/sklearn-flask:dev -f Dockerfile .

.PHONY: push-base
push-base: build-base
	docker build -t quay.io/dotmesh/sklearn-flask:dev

.PHONY: image
image:
	docker build --build-arg model_name=model.joblib \
	 -t quay.io/dotmesh/sklearn-flask-model:dev \
	 -f Dockerfile.build example_model/

.PHONY: run
run:
	docker run -it -p 8501:8501 quay.io/dotmesh/sklearn-flask-model:dev
