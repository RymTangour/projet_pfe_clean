
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import TurboProcessor
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from pathlib import Path

from src import get_logger

logger=get_logger()


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"-----------------------------------------------------stage {STAGE_NAME}-----------------------------------------------------started")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"-----------------------------------------------------stage {STAGE_NAME}-----------------------------------------------------completed")
except Exception as e:
    logger.exception(e)
    raise e
"""



STAGE_NAME = "Data Preprocessing"

try:
    logger.info(f"-----------------------------------------------------stage {STAGE_NAME} started-----------------------------------------------------")
    obj = DataPreprocessingTrainingPipeline()
    obj.main()
    logger.info(f"-----------------------------------------------------stage {STAGE_NAME} completed-----------------------------------------------------")
except Exception as e:
    logger.exception(e)
    raise e"""