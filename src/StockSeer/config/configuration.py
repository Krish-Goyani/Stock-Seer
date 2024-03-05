from src.StockSeer.constants import *
from src.StockSeer.utils.common import read_yaml, create_directories
from src.StockSeer.entity.config_entity import DataIngestionConfig, DataValidationConfig,DataTransformationConfig,PartialModelTrainerConfig,ModelEvaluationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            ticker=config.ticker,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    



    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation 
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            local_data_file = config.local_data_file,
            STATUS_FILE= config.STATUS_FILE,
            all_schema= schema
        )


        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            scaled_data_file=config.scaled_data_file,
            scaler_file_path=config.scaler_file_path
        )

        return data_transformation_config
    
    def get_partial_model_trainer_config(self) -> PartialModelTrainerConfig:
        config = self.config.partial_model_trainer

        create_directories([config.root_dir])

        partial_model_trainer_config = PartialModelTrainerConfig(
            root_dir=config.root_dir,
            X_train_data_path = config.X_train_data_path,
            y_train_data_path = config.y_train_data_path,
            partial_model_name = config.partial_model_name
        )

        return partial_model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            X_test_data_path = config.X_test_data_path,
            y_test_data_path = config.y_test_data_path,
            model_path =config.model_path,
            scaler_file_path=config.scaler_file_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config