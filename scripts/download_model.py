import os
import shutil
import tempfile
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = (
    "https://dagshub.com/Candicemamou/"
    "mlflow-with-daghub.mlflow"
)

MODEL_NAME = "DigitsRandomForest"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

ready_versions = [
    v for v in versions if v.status == "READY"
]

if not ready_versions:
    raise RuntimeError("No trained model found")

latest = max(ready_versions, key=lambda v: int(v.version))

model_uri = f"models:/{MODEL_NAME}/{latest.version}"

print(f"Downloading {model_uri}")

with tempfile.TemporaryDirectory() as tmp:
    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=tmp
    )

    shutil.copytree(
        downloaded_path,
        "model",
        dirs_exist_ok=True
    )

print("Model downloaded successfully")