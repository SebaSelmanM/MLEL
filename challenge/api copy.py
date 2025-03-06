#import fastapi

#app = fastapi.FastAPI()

#@app.get("/health", status_code=200)
#async def get_health() -> dict:
#    return {
#        "status": "OK"
#    }

#@app.post("/predict", status_code=200)
#async def post_predict() -> dict:
#    return

from typing import List, Optional
import fastapi
from fastapi import Request
from typing import List
import pandas as pd
from pydantic import BaseModel

# Importa tu DelayModel (ajusta la ruta según tu proyecto)
from challenge.model import DelayModel

app = fastapi.FastAPI()

# Instancia global del modelo
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Endpoint de salud, debe retornar status OK.
    """
    return {"status": "OK"}

from pydantic import BaseModel, Extra
# Define la estructura de un vuelo (ajusta campos a tu necesidad)
class FlightItem(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    # Opcional si el test no lo envía siempre

    #Fecha_I: str = "2023-01-01 07:00:00"
    #Fecha_O: str = "2023-01-01 07:30:00"
    # Agrega campos extra si hace falta
    # Campos opcionales (ej. Fecha_I, Fecha_O) si el test no los manda
    # Fecha_I: Optional[str] = None
    # Fecha_O: Optional[str] = None

    class Config:
        # Prohíbe campos que no estén listados aquí. 
        # Por defecto causaría un error 422.
        extra = Extra.forbid


# Estructura para el batch, que contiene la lista de flights
class FlightsBatch(BaseModel):
    flights: List[FlightItem]
    class Config:
        extra = Extra.forbid

@app.post("/predict", status_code=200)
#async def post_predict(items: List[FlightItem]) -> dict:
async def post_predict(batch: FlightsBatch) -> dict:
    """
    Endpoint para predecir retrasos.
    Espera un body JSON con una lista de vuelos.
    """
    # Convertir la lista de Pydantic a un DataFrame
    #df = pd.DataFrame([item.dict() for item in items])
    #df = pd.DataFrame([item.dict() for item in batch.flights])
    df = pd.DataFrame([item.dict() for item in batch.flights])
    # Preprocesar (sin target)
    features = model.preprocess(df)
    # Predecir
    preds = model.predict(features)
    # Devolver las predicciones
    #return {"predictions": preds}
    return {"predict": preds}  # Devuelve algo válido

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException

async def validation_exception_handler(request, exc):
    # Retornar 400 en vez de 422
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

import fastapi
from fastapi.testclient import TestClient

class BatchPipeline:
    def __init__(self, app: fastapi.FastAPI):
        self.app = app
        self.client = TestClient(app)

    def predict(self, data: list[dict]):
        """
        Un método que haga la request al endpoint /predict, por ejemplo.
        """
        response = self.client.post("/predict", json=data)
        return response
