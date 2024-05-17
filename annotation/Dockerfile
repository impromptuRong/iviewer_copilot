FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY utils ./utils
COPY annotation/. .

RUN export no_proxy="localhost"

# gunicorn app_annotation:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:9020
CMD ["gunicorn", "app_annotation:app", "--workers", "8", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9020"]
