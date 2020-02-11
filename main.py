from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import os
import http
import sys

app = Flask(__name__)

model = None

@app.route('/v1/healthcheck')
def healthcheck():
    return 'OK'

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
@app.route('/v1/models/model', methods=['GET'])
def models():
    return 'AVAILABLE'

@app.route('/v1/models/model:predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(data)

    instances = data["instances"]
    try:
        inputs = np.array(instances)
    except Exception as e:
        raise Exception(
            "Failed to initialize NumPy array from inputs: %s, %s" % (e, instances))
    try:
        result = model.predict_proba(inputs).tolist()
        return { "predictions" : result }
    except Exception as e:
            raise Exception("Failed to predict %s" % e)

if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = os.environ.get("FLASK_PORT", "8501")
    # if we haven't overriden the model filename using env var at runtime, use the args passed in
    if len(sys.argv) > 1:
        os.environ.setdefault("MODEL_JOBLIB_FILE", sys.argv[1])

    path = os.environ.get("MODEL_JOBLIB_FILE", "example_model/model.joblib")
    try:
        # Backwards compatibility, new code shouldn't use joblib:
        model = joblib.load(path)
    except:
        print("Failed to open with joblib, using regular pickle instead.")
        with open(path, "rb") as f:
            model = pickle.load(f)
    print(' * Model loaded from "%s"' % (path,))

    app.run(host=host, port=port)
