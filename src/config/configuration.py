from src.utils.common import read_yaml, create_directories 
from src.entity.config_entity import DataIngestionConfig , TurboProcessorConfig
from src import get_logger
from pathlib import Path
logging=get_logger()

class ConfigurationManager:
    
    def __init__(self, config_file_path: Path):
        self.config_file_path = config_file_path
        self.config = read_yaml(self.config_file_path)
        create_directories([self.config.artifacts_root]) 

    def get_data_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_config=DataIngestionConfig(
            source_url=config.source_url,
            local_data_directory=config.local_data_directory,           
        )
        return data_config

    def get_turbo_processor_config(self) -> TurboProcessorConfig:
        config = self.config.turbo_preprocessing
        processor_config = TurboProcessorConfig(
            input_dir=Path(config.input_dir),
            output_dir=Path(config.output_dir),
            T_d=config.T_d,
            T_window=config.T_window,
            eps=config.eps
        )
        return processor_config
 

