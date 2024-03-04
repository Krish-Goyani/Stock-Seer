from src.StockSeer.config.configuration import DataIngestionConfig
from src.StockSeer.logging import logger
import yfinance as yf
import os

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):

        # Download AAPL data 
        logger.info(f"{self.config.ticker}'s data download started")
        data = yf.download(self.config.ticker)
        logger.info(f"{self.config.ticker}'s data downloaded")

        data = data.filter(['Close'])
        data = data[::-1]
        # Set filename 
        filename = 'data.csv'
        filepath = os.path.join(self.config.local_data_file, filename)

        # Check if file exists, if so delete it
        if os.path.isfile(filepath):
            os.remove(filepath)
            
        # Save downloaded data to csv    
        data.to_csv(filepath,index=True)

        logger.info(f"File downloaded and saved to: {filepath}")


    