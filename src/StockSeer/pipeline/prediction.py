from tensorflow.keras.models import load_model
from src.StockSeer.logging import logger
import numpy as np

class PredictionPipeline:
    def __init__(self):
        self.model = load_model("artifacts/full_model_trainer/model.h5")
        self.scaled_data = np.load("artifacts/data_transformation/scaled_data.npy")

    def FuturePrediction(self,fdays):
        predictions = []
        self.scaled_data = self.scaled_data.reshape(1,len(self.scaled_data),1)

        for i in range(fdays):

            data = self.scaled_data[:,-100:,:]

            stacked_data = data.reshape(1,100,1)

            pred = self.model.predict(stacked_data)

            dataset_list = list(self.scaled_data[0,:,0])
            pred = pred[0][0]

            dataset_list.append(pred)
            predictions.append(pred)

            self.scaled_data = np.array(dataset_list).reshape(1,len(dataset_list),1)

        return np.array(predictions).reshape(-1,1)