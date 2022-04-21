from typing import Optional

import numpy
from DataModel import DataModel
from joblib import load
from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Up and running"}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(),
                      columns=dataModel.columns(), index=[0])
    
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return {"Predicción": numpy.array2string(result)}
