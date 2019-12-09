from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import os
import http
import sys

app = Flask(__name__)

model = None

@app.route('/healthcheck')
def healthcheck():
    return 'true'

@app.route('/', methods=['GET'])
def base():
    return 'OK'

@app.route('/predict', methods=['POST'])
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
        result = model.predict(inputs).tolist()
        return { "predictions" : result }
    except Exception as e:
            raise Exception("Failed to predict %s" % e)

if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = os.environ.get("FLASK_PORT", "8501")
    # if we haven't overriden the model filename using env var at runtime, use the args passed in
    if len(sys.argv) > 1:
        os.environ.setdefault("MODEL_JOBLIB_FILE", sys.argv[1])

    try:
        model = load(os.environ.get("MODEL_JOBLIB_FILE", "example_model/model.joblib"))
        print(' * Model loaded from "%s"' % os.environ.get("MODEL_JOBLIB_FILE", "example_model/model.joblib"))

    except Exception as e:
        print('No model here: %s' % os.environ.get("MODEL_JOBLIB_FILE"))
        print('Train first')
        print(str(e))
        sys.exit()

    app.run(host=host, port=port)