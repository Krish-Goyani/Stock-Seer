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