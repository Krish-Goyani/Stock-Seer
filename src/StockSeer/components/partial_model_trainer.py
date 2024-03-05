from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from src.StockSeer.logging import logger
import numpy as np
from src.StockSeer.entity.config_entity import PartialModelTrainerConfig

class PartialModelTrainer:
    def __init__(self, config: PartialModelTrainerConfig):
        self.config = config

    
    def build_lstm_model(self,input_shape, lstm_units=[50, 50, 50], dense_units=1, optimizer='adam', loss='mean_squared_error'):
        model = Sequential()

        for i, units in enumerate(lstm_units):
            if i == 0:
                model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
            else:
                model.add(LSTM(units=units, return_sequences=(i < len(lstm_units)-1)))

        model.add(Dense(units=dense_units))

        model.compile(optimizer=optimizer, loss=loss)
        logger.info("model built succesfully")
        return model




    def partial_train(self):

        np.random.seed(42)
        tf.random.set_seed(42)
        # Assuming x_train.shape[1] is the time steps and 1 is the feature dimension
        X_train = np.load(self.config.X_train_data_path)
        y_train = np.load(self.config.y_train_data_path)
    
        input_shape = (X_train.shape[1], 1)
     
        # Define LSTM model with specified architecture
        lstm_model = self.build_lstm_model(input_shape=input_shape)
        lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2) 
        # Train the model
        logger.info("partial model training started")
        lstm_model.fit(X_train, y_train, epochs=1, batch_size=32, callbacks=[lr_callback])
        lstm_model.save(self.config.partial_model_name)
        logger.info(f"partial model trainned and save at {self.config.partial_model_name}")