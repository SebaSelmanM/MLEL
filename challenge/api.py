import fastapi

<<<<<<< Updated upstream
app = fastapi.FastAPI()
=======
from pydantic import BaseModel, Extra
from pydantic.types import conint
from enum import Enum
import uvicorn
>>>>>>> Stashed changes

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
<<<<<<< Updated upstream
async def post_predict() -> dict:
    return
=======
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

###############################################################################
#                             ENDPOINT Docker                                 #
###############################################################################

@app.get("/")
def read_root():
    return {"message": "API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
>>>>>>> Stashed changes
