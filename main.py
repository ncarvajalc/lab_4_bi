from typing import Optional,List


import numpy
from DataModel import DataModel
from DataModelWithPredictor import DataModelWithPredictor
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
    return {"Predicción": result.tolist()[0]}

@app.post("/predictMany")
def make_predictions(dataModels : List[DataModelWithPredictor]):

    listModels = []

    for dataModel in dataModels:
        dictModel = dataModel.dict()
        listModels.append(dictModel)
        
    df = pd.DataFrame(listModels)
    df.columns=dataModels[0].columns()

    vo= "Life expectancy"

    X = df.drop(vo, axis=1)
    y= df[vo]

    model = load("assets/modelo.joblib")
    result = model.score(X,y)
    return {"R^2 del modelo": result}
