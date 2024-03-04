from sklearn.preprocessing import StandardScaler
import pandas as pd 
from src.StockSeer.logging import logger
import numpy as np
from src.StockSeer.config.configuration import DataTransformationConfig
import os

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def TestDataStacking(self,scaled_data,training_data_len,df):

        test_data = scaled_data[training_data_len - 100: , :]
        # Create the data sets x_test and y_test
        X_test = []
        y_test = df[training_data_len:]
        for i in range(100, len(test_data)):
            X_test.append(test_data[i-100:i, 0])
            
        # Convert the data to a numpy array
        X_test = np.array(X_test)

        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))

        np.save(os.path.join(self.config.root_dir,"X_test.npy"),X_test)
        np.save(os.path.join(self.config.root_dir,"y_test.npy"),y_test)
        logger.info("test data stacking completed")


    def TrainDataStacking(self,scaled_data,training_data_len):

        train_data = scaled_data[0:training_data_len, :]
        # Split the data into x_train and y_train data sets
        X_train = []
        y_train = []

        for i in range(100, len(train_data)):
            X_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])

        # Convert the x_train and y_train to numpy arrays 
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape the data
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        np.save(os.path.join(self.config.root_dir,"X_train.npy"),X_train)
        np.save(os.path.join(self.config.root_dir,"y_train.npy"),y_train)
        logger.info("train data stacking completed")



    def StandardScaling(self,data):
        logger.info("data scaling started")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        logger.info(f"data scaling completed and shape of data : {scaled_data.shape}")
        np.save(self.config.scaled_data_file,scaled_data)
        logger.info(f"scaled data stored at {self.config.scaled_data_file}")
        return scaled_data

    def DataTransformation(self):

        data = pd.read_csv(self.config.data_path,index_col='Date')

        scaled_data = self.StandardScaling(data)

        test_data_len = np.ceil(len(scaled_data)*0.2)
        if test_data_len > 200:
            test_data_len = 200
        train_data_len = len(scaled_data) - test_data_len

        self.TrainDataStacking(scaled_data,train_data_len)
        self.TestDataStacking(scaled_data,train_data_len,data)

        
