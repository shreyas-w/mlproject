import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer

# creating  a data class
# dataclass helps us to initialize the class components without creating __init__

@dataclass
class DataIngestionConfig:
  train_data_path:str=os.path.join("artifacts","train.csv")
  test_data_path:str=os.path.join("artifacts","test.csv")
  raw_data_path:str=os.path.join("artifacts","data.csv")


class DataIngestion:
  def __init__(self):
    self.ingestion_config=DataIngestionConfig()
    # self.ingestion_config contains the train_data_path,test_data_path and raw_data_path

  def initiate_data_ingestion(self):
    logging.info("Entered the data ingestion method or components")
    try:
      df=pd.read_csv("F:\mlproject1(student performence predictor)\data\data\stud.csv")
      logging.info("Read dataset as dataframe")
      # from here itsef we can read the data from any source like csv file or some database

      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

      df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

      logging.info("Train Test Split initaited")

      train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

      train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

      test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

      logging.info("ingestion of the data is completed")

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )
      
      
    except Exception as e:
      raise CustomException(e,sys)


if __name__=="__main__":

  obj=DataIngestion()
  train_data,test_data=obj.initiate_data_ingestion()
  # print("path=",obj.ingestion_config.train_data_path)

  data_transformation=DataTransformation()
  train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
  # print("train_arr=",train_arr)
  # print("test_arr=",test_arr)

  modeltrainer=ModelTrainer()
  print(modeltrainer.initaite_model_trainer(train_arr,test_arr))



