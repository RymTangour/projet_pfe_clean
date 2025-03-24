from src.config.configuration import ConfigurationManager
from src.entity.config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from pathlib import Path
from src import get_logger
logger = get_logger()



STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager(Path("config/config.yaml"))
        data_ingestion_config=config.get_data_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()




if __name__ == '__main__':
    try:
        logger.info(f"-----------------------------------------------------stage {STAGE_NAME} started-----------------------------------------------------")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"-----------------------------------------------------stage {STAGE_NAME} completed-----------------------------------------------------")
    except Exception as e:
        logger.exception(e)
        raise e
  