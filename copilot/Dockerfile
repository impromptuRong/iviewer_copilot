FROM python:3.11-slim

WORKDIR /app

# COPY .env .
COPY utils ./utils
COPY copilot/. .
RUN pip install --no-cache-dir -r requirements.txt

# RUN export no_proxy="localhost"

CMD ["uvicorn", "app_copilot:app", "--host", "0.0.0.0", "--port", "9040"]
