import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from source.exception import CustomException
from source.logger import logging 

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor , AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from source.utils import modelEval,saveModel

@dataclass
class modelTrainerConfig:
    modelPath = os.path.join('artifacts','model.pkl')

class modelTrainer:
    def __init__(self):
        self.modelTrainerConfig = modelTrainerConfig()

    def initiaze_modelTrainer(self,trainArray ,testArray):
        
        try:
            xTrain , yTrain , xTest , yTest = (
                trainArray[:,:-1],
                trainArray[:,-1],
                testArray[:,:-1],
                testArray[:,-1]
            )
            logging.info("initialized arrays in model trainer")
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'SVR': SVR(),
                'DecisionTree': DecisionTreeRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0),
                'XGBoost': XGBRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor()
            } 
            logging.info("initialized models in model trainer")

            r2Score={}
            for name,model in models.items():
                model.fit(xTrain,yTrain) 
                yPred =model.predict(xTest)
                mse,mae,r2 = modelEval(yTest,yPred)

                r2Score[name]=r2
            
            bestModelScore = max(sorted(r2Score.values()))
            bestModelName = max(r2Score,key=r2Score.get)
            logging.info(f"best model name:{bestModelName}, r2 score:{bestModelScore}")

            model = models[bestModelName]
            model.fit(xTrain,yTrain)
            yPred =model.predict(xTest)
            mse,mae,r2 = modelEval(yTest,yPred)

            saveModel(
                filePath=self.modelTrainerConfig.modelPath,
                obj=model
            )
            logging.info("saved model")

            return r2

        except Exception as e:
            raise CustomException(e,sys)