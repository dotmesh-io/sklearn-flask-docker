
build-base:
	docker build -t quay.io/dotmesh/sklearn-flask:dev -f Dockerfile .

push-base: build-base
	docker build -t quay.io/dotmesh/sklearn-flask:dev

image: push-base
	docker build --build-arg model_name=model.joblib \
	 -t quay.io/dotmesh/sklearn-flask-model:dev \
	 -f Dockerfile.build test_data/

run:
	docker run -it -p 8501:8501 quay.io/dotmesh/sklearn-flask-model:dev