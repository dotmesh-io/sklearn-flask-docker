FROM python:3.10-slim-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
ENV MODEL_ABS_PATH "/app/model.joblib"
ENV FLASK_PORT "8501"
CMD [ "python", "main.py" ]
