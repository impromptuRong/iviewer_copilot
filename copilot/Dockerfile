FROM python:3.11-slim

WORKDIR /app

# COPY .env .
COPY copilot/. .
COPY utils ./agents/utils
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app_copilot:app", "--host", "0.0.0.0", "--port", "9040"]
