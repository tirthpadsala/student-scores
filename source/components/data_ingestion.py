import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.exception import CustomException
from source.components.data_transformation import dataTransformer
from source.components.data_trainer import modelTrainer
from source.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class dataIngtestionconfig():
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class dataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngtestionconfig()
    
    def initiateDataIngestion(self):
        logging.info("entered data ingestion")
        try:
            df = pd.read_csv('notebooks/student.csv')
            logging.info("read the data")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)

            trainSet , testSet = train_test_split(df,test_size=0.2, random_state=42)
            logging.info("train test data initialized")

            trainSet.to_csv(self.ingestionConfig.train_data_path,index=False, header=True)
            testSet.to_csv(self.ingestionConfig.test_data_path,index=False, header=True) 

            logging.info("ingestion of data complted")
            
            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=dataIngestion()
    trainData , testData = obj.initiateDataIngestion()

    transformer = dataTransformer()
    trainArray , testArray,_ = transformer.initiate_DataTransformstion(trainData,testData)

    trainer = modelTrainer()
    print(trainer.initiaze_modelTrainer(trainArray=trainArray,testArray=testArray))
