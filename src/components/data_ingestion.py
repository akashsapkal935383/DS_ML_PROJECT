import os
import sys
from ..custom_exception import CustomException  # Use relative import
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_transformation import DataTransformationConfig, DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion methods or components")

        try:
            df = pd.read_csv('src/notebook/data/StudentsPerformance.csv')
            logging.info("Read the data from dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train and test split data")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion activity is completed")

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)  # Corrected the typo
            pass
        
        
if __name__ == "__main__":
    logging.info("Logging is started here")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Models calling
    modelTrainer = ModelTrainer()
    modelTrainer.initiate_model_trainer(train_array, test_array)
    print(modelTrainer.initiate_model_trainer(train_array, test_array))  # This assumes you want to print the return value
