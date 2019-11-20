from flask import Flask, request
from joblib import load
import os
import http
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
    app.run(host=host, port=port)