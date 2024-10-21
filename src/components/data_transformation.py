import os
import sys
from exception import CustomExection
from  logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.uitls import save_object
from src.components.data_transformation import DataTransformationConfig, DataTransformation





@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:

    def __init__(self):
        self.Data_Transmation_config=DataTransfomationConfig()

    
    def get_data_transformer_obj(self):

        try:
            numerical_columns=['reading_score','writing_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']


            numerical_pipeline=Pipeline(
            
         steps=[
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler",StandardScaler(with_mean=False))
         ]

        )
        
            categorical_pipeline=Pipeline(

             steps=[
                 
                 ("imputer",SimpleImputer(strategy="most_frequent")),
                 ("one_hot_encoder",OneHotEncoder()),
                 ("scaler",StandardScaler(with_mean=False))

             ]

        )

            logging.info("Numerical columne impuerts and sclaning is  completed !")
            logging.info("categroical columne impuerts and sclaning is  completed !")

            preprocessor=ColumnTransformer(

            [
                ("numerical_pipeline",numerical_pipeline,numerical_columns),
                ("categorical_pipeline",categorical_pipeline,categorical_columns)
            ]
        )


            return preprocessor
    
        except Exception as e :
                 raise CustomExection(e,sys)
        
    
    def initiate_data_transfromations(self,train_path,test_path):

        try:
              train_df=pd.read_csv(train_path)
              test_df=pd.read_csv(test_path)
              logging.info("Read data acitive is completed!")

              preprocess_obj=self.get_data_transformer_obj()

              traget_column_name="math_score"

              input_feature_train_df=train_df.drop(columns=[traget_column_name],axis=1)
              traget_feature_train_df=train_df[traget_column_name]

              input_feature_test_df=test_df.drop(columns=[traget_column_name],axis=1)
              traget_feature_test_df=test_df[traget_column_name]

              logging.info("appying the preprocessing  objects data ")

              input_feature_train_arry=preprocess_obj.fit_transform(input_feature_train_df)
              input_feature_test_arry=preprocess_obj.transform(input_feature_test_df)


              train_arr=np.c_[
                   
                   input_feature_train_arry,np.array(traget_feature_train_df)
                   
              ]

              test_arr=np.c_[
                   input_feature_test_arry,np.array(traget_feature_test_df)
              ]
              
              logging.info("save perprocess onject")
              save_object(
                   
                 file_path=DataTransfomationConfig.preprocessor_obj_file_path,
                 obj=preprocess_obj
              )
              return (
                 test_arr,
                 train_arr,
                 self.Data_Transmation_config.preprocessor_obj_file_path
            )

        except Exception as e :
            raise CustomExection(e,sys) 