# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script
COPY scripts/download_model.py scripts/download_model.py

# Download the latest trained model from DagsHub
RUN --mount=type=secret,id=dagshub_token,required=true \
    MLFLOW_TRACKING_USERNAME=Candicemamou \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python scripts/download_model.py

# Copy FastAPI application
COPY app/ app/

EXPOSE 8000

# Start the API
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]