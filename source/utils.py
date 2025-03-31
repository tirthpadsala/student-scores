import os
import sys

from source.logger import logging
from source.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import pickle

def saveModel(filePath, obj):
    try:
        dirPath = os.path.dirname(filePath)
        os.makedirs(dirPath,exist_ok=True)
        with open(filePath,'wb') as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

def modelEval(actual, predicted):
    mse = mean_squared_error(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    r2 = r2_score(actual,predicted)
    return mse,mae,r2