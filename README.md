# sklearn-flask-docker

[![Build Status](https://drone.app.cloud.dotscience.net/api/badges/dotmesh-io/sklearn-flask-docker/status.svg)](https://drone.app.cloud.dotscience.net/dotmesh-io/sklearn-flask-docker)

This is a minimal docker image which just contains flask and joblib. Its purpose is to provide a base image from which you can deploy sklearn models which can then be exposed as a http API.

## Using this docker image
In [dotscience](https://dotscience.com) we use this image as part of the model build phase. If you are running this standalone (or learning how to deploy it) you will need a joblib file containing the sklearn model object. At minimum this object will have the `predict` method which is called from the api. This can be mounted or copied in as part of a docker build phase.

The following environment variables are used to configure flask:

| Name  | Example  | Description  |
|---|---|---|
| `FLASK_HOST`  |  `0.0.0.0` | The hostname on which flask will run  |
| `FLASK_PORT`  | `80`  | The port on which flask will run  |
| `MODEL_JOBLIB_FILE`  | `/app/model.joblib`  | The full absolute path to your model file, saved using `joblib`  |

For a tutorial on how to label up model files for use with this image, see [dotscience documentation](https://docs.dotscience.com/tutorials/hyperparam/)

## Updating the docker image (dotscience maintainers)
This docker image is deployed to `quay.io/dotmesh/sklearn-flask` on every push. To update this docker image just submit a PR. To release it, use github releases and create a new tag, then update the base image tag [here](https://github.com/dotmesh-io/dotscience-agent/blob/master/dockerfiles/sklearn/Dockerfile#L1)


## Development and Testing

There's a test model in `/test_data/model.joblib`.

1. Create virtual env: 
  
  ```
  python3 -m venv venv
  ```


2. Activate it:

  ```
  source venv/bin/activate
  ```

3. Start server:

  ```
  python main.py
  ```

4. Call model "predict" endpoint:
  ```
  curl --request POST \
    --url http://localhost:8501/v1/models/model:predict \
    --header 'content-type: application/json' \
    --data '{
    "instances": [
      [6.8,  2.8,  4.8,  1.4],
      [6.0,  3.4,  4.5,  1.6]
    ]
  }'
```