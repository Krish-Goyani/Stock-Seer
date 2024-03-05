from src.StockSeer.config.configuration import ConfigurationManager
from src.StockSeer.components.full_model_trainer import FullModelTrainer
from src.StockSeer.logging import logger

STAGE_NAME = "Full Model Training Stage"

class FullModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        full_model_trainer_config = config.get_full_model_trainer_config()
        full_model_trainer_config = FullModelTrainer(full_model_trainer_config)
        full_model_trainer_config.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FullModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e