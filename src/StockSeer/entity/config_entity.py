from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    ticker: str
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    local_data_file : Path
    STATUS_FILE : str
    all_schema : dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    scaled_data_file: Path
    scaler_file_path: Path


@dataclass(frozen=True)
class PartialModelTrainerConfig:
    root_dir: Path
    X_train_data_path: Path
    y_train_data_path: Path
    partial_model_name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_path:Path
    scaler_file_path:Path
    metric_file_name: Path
