from flask import Flask, request
from joblib import load
import os
import http
import sys
app = Flask(__name__)


@app.route('/healthcheck')
def healthcheck():
    return 'true'

@app.route('/predict')
def predict():
    if request.args.get("parameters", None) == None:
        return {"error": "did not supply any parameters"}, http.HTTPStatus.BAD_REQUEST
    model = load(os.environ.get("MODEL_JOBLIB_FILE", "/app/model.joblib"))
    return model.predict(request.args["parameters"])

if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = os.environ.get("FLASK_PORT", "80")
    # if we haven't overriden the model filename using env var at runtime, use the args passed in
    if len(sys.argv) > 0:
        os.environ.setdefault("MODEL_JOBLIB_FILE", sys.argv[1])
    app.run(host=host, port=port)