from tensorflow.keras.models import load_model
from src.StockSeer.logging import logger
import numpy as np
from joblib import load

class PredictionPipeline:
    def __init__(self):
        self.model = load_model("artifacts/full_model_trainer/model.h5")
        self.scaler = load("artifacts/data_transformation/scaler.joblib")
        self.scaled_data = np.load("artifacts/data_transformation/scaled_data.npy")

    def FuturePrediction(self,fdays):
        predictions = []
        self.scaled_data = self.scaled_data.reshape(1,len(self.scaled_data),1)
        logger.info("future session's prediction started")
        for i in range(fdays):

            data = self.scaled_data[:,-100:,:]

            stacked_data = data.reshape(1,100,1)

            pred = self.model.predict(stacked_data,verbose=0)

            dataset_list = list(self.scaled_data[0,:,0])
            pred = pred[0][0]

            dataset_list.append(pred)
            predictions.append(pred)

            self.scaled_data = np.array(dataset_list).reshape(1,len(dataset_list),1)
        logger.info("future session's prediction completed")
        return np.array(predictions).reshape(-1,1)
    
    def Predict(self,fdays):
        predictions = self.FuturePrediction(fdays)
        predictions = self.scaler.inverse_transform(predictions)

        return predictions