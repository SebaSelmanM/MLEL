from typing import List
import pandas as pd

import fastapi
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from pydantic import BaseModel, Extra
from pydantic.types import conint
from enum import Enum

# Model Import
from challenge.model import DelayModel

###############################################################################
#                 DEFINITION OF THE APP AND GLOBAL MODEL                     #
###############################################################################
app = FastAPI()
model = DelayModel()

###############################################################################
#                     VALIDATION HANDLING (422 -> 400)                       #
###############################################################################
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Converts validation errors (e.g., out-of-range values, 
    unknown fields, etc.) to HTTP 400 instead of 422.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

###############################################################################
#                     DEFINITION OF Pydantic MODELS                          #
###############################################################################
class TipoVuelo(str, Enum):
    """
    TIPOVUELO can only be "N" or "I".
    If the test sends another value, 
    Pydantic raises a validation error.
    """
    N = "N"
    I = "I"

class FlightItem(BaseModel):
    """
    Each flight: OPERA (string), TIPOVUELO ("N" or "I"), MES (1..12).
    """
    OPERA: str
    TIPOVUELO: TipoVuelo
    MES: conint(ge=1, le=12)  # integer between 1 and 12

    class Config:
        extra = Extra.forbid  # if an undeclared field arrives, error

class FlightsBatch(BaseModel):
    """
    The JSON we expect in /predict:
    {
      "flights": [ <FlightItem>, <FlightItem>, ... ]
    }
    """
    flights: List[FlightItem]

    class Config:
        extra = Extra.forbid

###############################################################################
#                                 ENDPOINTS                                   #
###############################################################################
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(batch: FlightsBatch) -> dict:
    """
    Example body:
    {
      "flights": [
        {
          "OPERA": "Argentinas",
          "TIPOVUELO": "N",
          "MES": 3
        }
      ]
    }
    """
    # Convert the list of FlightItem to a DataFrame
    df = pd.DataFrame([item.dict() for item in batch.flights])
    # Preprocess and predict
    features = model.preprocess(df)
    preds = model.predict(features)
    
    return {"predict": preds}

###############################################################################
#                             CLASS BatchPipeline                             #
###############################################################################
class BatchPipeline:
    """
    Helper class for tests, which makes requests to the API 
    using FastAPI's TestClient.
    """
    def __init__(self, app: fastapi.FastAPI):
        self.app = app
        self.client = TestClient(app)

    def predict(self, data: dict):
        return self.client.post("/predict", json=data)