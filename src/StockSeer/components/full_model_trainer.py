from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from src.StockSeer.entity.config_entity import FullModelTrainerConfig
import tensorflow as tf
from src.StockSeer.logging import logger
import numpy as np

class FullModelTrainer:
    def __init__(self, config: FullModelTrainerConfig):
        self.config = config

    def train(self):
        np.random.seed(42)
        tf.random.set_seed(42)

        X = np.load(self.config.X_stacked_data_path)
        y = np.load(self.config.y_stacked_data_path)
        model = load_model(self.config.partial_model_name)

        lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2) 
        # Train the model
        logger.info("full model training started")
        model.fit(X, y, epochs=1, batch_size=32, callbacks=[lr_callback])
        model.save(self.config.final_model_name)
        logger.info(f"full model trainned and save at {self.config.final_model_name}")
    