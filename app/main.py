import mlflow.sklearn
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Digits Prediction API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

model = mlflow.sklearn.load_model("model")


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=64,
        max_length=64
    )


@app.post("/predict")
def predict(request: PredictionRequest):

    features = np.array(
        request.features
    ).reshape(1, 64)

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }