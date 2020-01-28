FROM python:3.8-slim

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV MODEL_ABS_PATH "/app/model.joblib"
ENV FLASK_PORT "8501"
CMD [ "python", "main.py" ]
