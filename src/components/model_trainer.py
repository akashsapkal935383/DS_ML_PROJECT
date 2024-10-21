
import os
import sys
from src.exception import CustomException
from  logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.uitls import save_object,evluate_models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("split the train and test data for model trainng")

            X_train,y_train,X_test,y_test=(

                train_array[:,:-1],
                train_array[:,-1],
                train_array[:,:-1],
                train_array[:,-1],

            )  

            models={

                "Linear Regression":LinearRegression(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "DecisionTree Regressor":DecisionTreeRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "RandomForest Regressor":RandomForestRegressor(),
                "catBoost Rgressor":CatBoostRegressor(),
                "XGB Regressor":XGBRegressor()


            }
             #hyperparameter tuning
            params={
                "DecisionTree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                    'max_depth':range(1,10),

                },
                "RandomForest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighbors Regressor":{
                     'n_neighbors':[5],
                    'weights':['uniform','distance']
                },

                "GradientBoosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catBoost Rgressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




            model_report:dict=evluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            

            best_model_score = max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score <0.60:
                raise CustomExection(" No model found")
            
            logging.info(f"Best found model in trainign and test dataset ")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(X_test)
            r2_score_model=r2_score(y_test,predicted)

            return r2_score_model,best_model
        
        except Exception as e:
            raise CustomExection(e,sys)