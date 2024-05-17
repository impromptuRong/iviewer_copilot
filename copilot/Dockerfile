FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY .env .
COPY utils ./utils
COPY copilot/. .

# RUN export no_proxy="localhost"

CMD ["python", "app_copilot.py"]
