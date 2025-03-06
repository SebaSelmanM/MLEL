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

# Importa tu modelo
from challenge.model import DelayModel

###############################################################################
#                 DEFINICIÓN DE LA APP Y EL MODELO GLOBAL                     #
###############################################################################
app = FastAPI()
model = DelayModel()

###############################################################################
#                     MANEJO DE VALIDACIÓN (422 -> 400)                       #
###############################################################################
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Convierte los errores de validación (por ej. valores fuera de rango, 
    campos desconocidos, etc.) en HTTP 400 en lugar de 422.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

###############################################################################
#                     DEFINICIÓN DE Pydantic MODELS                           #
###############################################################################
class TipoVuelo(str, Enum):
    """
    TIPOVUELO solo puede ser "N" o "I".
    Si el test envía otro valor, 
    Pydantic lanza error de validación.
    """
    N = "N"
    I = "I"

class FlightItem(BaseModel):
    """
    Cada vuelo: OPERA (string), TIPOVUELO ("N" o "I"), MES (1..12).
    """
    OPERA: str
    TIPOVUELO: TipoVuelo
    MES: conint(ge=1, le=12)  # entero entre 1 y 12

    class Config:
        extra = Extra.forbid  # si llega un campo no declarado, error

class FlightsBatch(BaseModel):
    """
    El JSON que esperamos en /predict:
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
    Ejemplo de body:
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
    # Convertimos la lista de FlightItem a un DataFrame
    df = pd.DataFrame([item.dict() for item in batch.flights])
    # Preprocesamos y predecimos
    features = model.preprocess(df)
    preds = model.predict(features)
    
    return {"predict": preds}

###############################################################################
#                             CLASE BatchPipeline                             #
###############################################################################
class BatchPipeline:
    """
    Clase auxiliar para los tests, que hace requests a la API 
    usando el TestClient de FastAPI.
    """
    def __init__(self, app: fastapi.FastAPI):
        self.app = app
        self.client = TestClient(app)

    def predict(self, data: dict):
        return self.client.post("/predict", json=data)
