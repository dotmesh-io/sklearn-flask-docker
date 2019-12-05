
image:
	docker build --build-arg model_name=model_precision.joblib \
	 -t quay.io/dotmesh/sklearn-flask:dev \
	 -f Dockerfile.build test_data/

run:
	docker run -it -p 8501:8501 quay.io/dotmesh/sklearn-flask:dev