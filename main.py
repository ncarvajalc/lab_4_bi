from typing import Optional,List


import numpy
from DataModel import DataModel
from joblib import load
from fastapi import FastAPI, Query
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
    return {"Predicción": result.tolist()}

@app.post("/predictMany")
def make_predictions(dataModels : List[DataModel]):

    listModels = []

    for dataModel in dataModels:
        dictModel = dataModel.dict()
        listModels.append(dictModel)
        
    df = pd.DataFrame(listModels)
    df.columns=dataModels[0].columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return {"Predicción": result.tolist()}
