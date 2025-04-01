import pandas as pd
import sys
from source.exception import CustomException
from source.utils import loadModel
import os

class predictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            modelPath = os.path.join('artifacts','model.pkl')
            preprocessorPath = os.path.join('artifacts','processorObject.pkl')
            model = loadModel(modelPath)
            preprocessor=loadModel(preprocessorPath)
            print("all models loaded")
            dataScaled = preprocessor.transform(features)
            prediction=model.predict(dataScaled)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)


class customData:
    def __init__(self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            math_score: int,
            reading_score: int,
            writing_score: int
        ):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.math_score=math_score
        self.reading_score=reading_score
        self.writing_score=writing_score

    def dataAsDataFrame(self):
        try:
            dataDict={
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "math_score":[self.math_score],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score]
            }
            return pd.DataFrame(dataDict)
        except Exception as e:
            raise CustomException(e,sys)

