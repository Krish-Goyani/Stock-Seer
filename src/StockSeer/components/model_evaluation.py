from tensorflow.keras.models import load_model
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
from src.StockSeer.utils.common import save_json
from src.StockSeer.logging import logger
from src.StockSeer.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):

        mape = mean_absolute_percentage_error(actual, pred) * 100
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)

        return mape, rmse, mae
    
    def save_results(self):
        X_test = np.load(self.config.X_test_data_path)
        y_test = np.load(self.config.y_test_data_path)
        model = load_model(self.config.model_path)
        scaler = load(self.config.scaler_file_path)
        
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
   
        (mape, rmse, mae) = self.eval_metrics(y_test,predictions)

        # Saving metrics as local
        scores = {"mape(%)": mape, "rmse": rmse, "mae": mae}
        save_json(path=Path(self.config.metric_file_name), data=scores)
