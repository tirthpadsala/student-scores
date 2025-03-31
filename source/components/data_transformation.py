import os
import sys

from source.logger import logging
from source.exception import CustomException
from source.utils import saveModel

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

@dataclass
class dataTransformerConfig():
    processorObjectPath = os.path.join('artifacts','processorObject.pkl')

class dataTransformer():
    def __init__(self):
        self.transformationConfig = dataTransformerConfig()

    def dataTransformerFUNC(self):
        try:
            numericalFeatures=["reading_score","writing_score"]
            categoricalFeatures=[
                'gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course'
                ]
            numPipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler()) 
                ]
            )
            catPipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical columns scaling completed")
            logging.info("categorical columns encoding and scaling completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numPipeline,numericalFeatures),
                    ("cat_pipeline",catPipeline,categoricalFeatures)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_DataTransformstion(self,trainPath,testPath):
        try:
            trainDF = pd.read_csv(trainPath)
            testDF = pd.read_csv(testPath)

            logging.info("read of train and test is completed")
            logging.info("obtaining pre processor object")

            self.preProcessorOBJ = self.dataTransformerFUNC()

            inputTrainDF = trainDF.drop(['average_score'] , axis=1)
            targetFeatureTrain = trainDF['average_score']

            inputTestDF = testDF.drop(['average_score'] , axis=1)
            targetFeatureTest = testDF['average_score']

            logging.info("applying transformation on both trai nand test set")

            inputTrainArray = self.preProcessorOBJ.fit_transform(inputTrainDF)
            inputTestArray = self.preProcessorOBJ.transform(inputTestDF)

            trainArray = np.c_[
                inputTrainArray , np.array(targetFeatureTrain)
            ]

            testArray = np.c_[
                inputTestArray , np.array(targetFeatureTest)
            ]

            logging.info("saving preprocessing object")

            saveModel(
                filePath=self.transformationConfig.processorObjectPath,
                obj=self.preProcessorOBJ
            )

            return(
                trainArray,
                testArray,
                self.transformationConfig.processorObjectPath
            )
        except Exception as e:
            raise CustomException(e,sys)
            