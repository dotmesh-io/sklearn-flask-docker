from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import os
import http
import sys
import tarfile
import tempfile
from subprocess import check_call

app = Flask(__name__)

model = None


@app.route("/v1/healthcheck")
def healthcheck():
    return "OK"


# TF Serving here returns a bigger response with available models
# {
#   "model_version_status": [
#     {
#       "version": "1",
#       "state": "AVAILABLE",
#       "status": {
#         "error_code": "OK",
#         "error_message": ""
#       }
#     }
#   ]
# }
# We might want to return something similar
@app.route("/v1/models/model", methods=["GET"])
def models():
    return "AVAILABLE"


def predict(model, query):
    """Default implementation of model prediction.

    May be overriden by custom_predict.py if it exists.

    Accepts the unpickled model and a decoded-from-JSON query, must return a
    response that can be encoded as JSON.
    """
    instances = query["instances"]
    try:
        inputs = np.array(instances)
    except Exception as e:
        raise Exception(
            "Failed to initialize NumPy array from inputs: %s, %s" % (e, instances)
        )
    result = model.predict_proba(inputs).tolist()
    return {"predictions": result}


@app.route("/v1/models/model:predict", methods=["POST"])
def http_predict():
    data = request.get_json(force=True)

    print(data)

    try:
        return jsonify(predict(model, data))
    except Exception as e:
        raise Exception("Failed to predict %s" % e)


def maybe_untar(tarFileBaseDir, joblibFilePath):
    """Sometimes the model is actually a tarball with the model inside.

    Yes, this is terrible.
    """
    try:
        t = tarfile.open(tarFileBaseDir)
    except Exception as e:
        print("Not a tarfile: {} {}".format(e.__class__, e))
        return

    # Extract runtime-requirements.txt if it exists:
    try:
        infile = t.extractfile(os.path.join(tarFileBaseDir, "runtime-requirements.txt"))
    except KeyError:
        pass
    else:
        with tempfile.NamedTemporaryFile("wb") as f:
            f.write(infile.read())
            f.flush()
            check_call(["pip", "install", "-r", f.name])

    # Extract custom_predict.py if it exists:
    try:
        infile = t.extractfile(os.path.join(tarFileBaseDir, "custom_predict.py"))
    except KeyError:
        pass
    else:
        # Override the prediction function:
        global predict
        custom_predict = {}
        exec(infile.read(), custom_predict, custom_predict)
        predict = custom_predict["predict"]

    # Extract the model object:
    print(t.list())
    infile = t.extractfile(os.path.join(tarFileBaseDir, joblibFilePath))
    with open(joblibFilePath + ".tmp", "wb") as outfile:
        outfile.write(infile.read())
    os.rename(joblibFilePath + ".tmp", joblibFilePath)


def setup():
    """Setup the model object."""
    # if we haven't overriden the model filename using env var at runtime, use the args passed in
    if len(sys.argv) > 1:
        os.environ.setdefault("MODEL_JOBLIB_FILE", sys.argv[1])

    tarFileBaseDir = "model"
    joblibFilePath = os.environ.get("MODEL_JOBLIB_FILE", "example_model/model.joblib")
    maybe_untar(tarFileBaseDir, joblibFilePath)

    global model
    try:
        # Backwards compatibility, new code shouldn't use joblib:
        model = joblib.load(joblibFilePath)
    except Exception as e:
        print(e)
        print(
            "Failed to open with joblib %s, using regular pickle instead."
            % joblibFilePath
        )
        with open(joblibFilePath, "rb") as f:
            model = pickle.load(f)
    print(' * Model loaded from "%s"' % (joblibFilePath,))


if __name__ == "__main__":
    setup()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = os.environ.get("FLASK_PORT", "8501")
    app.run(host=host, port=port)
