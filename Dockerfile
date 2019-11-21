FROM python:3.7

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV MODEL_ABS_PATH "/app/model.joblib"
CMD [ "python", "main.py", "${MODEL_ABS_PATH}" ]