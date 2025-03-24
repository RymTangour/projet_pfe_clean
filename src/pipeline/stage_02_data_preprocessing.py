
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import TurboProcessorConfig
from src.components.data_preprocessing import TurboProcessor
from pathlib import Path
from src import get_logger
logging = get_logger()



STAGE_NAME = "Data Preprocessing"


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manger=ConfigurationManager(Path("config/config.yaml"))
            processor_config=config_manger.get_turbo_processor_config()
            
            processor=TurboProcessor(processor_config)

            logging.info("Starting parallel batch processing")

            processor.run()
            logging.info("All Bathces processed successfully")

        except Exception as e:
            logging.critical(f"Fatal error{str(e)}")
            raise


if __name__ == '__main__':
    try:
        logging.info(f"-----------------------------------------------------stage {STAGE_NAME} started-----------------------------------------------------")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logging.info(f"-----------------------------------------------------stage {STAGE_NAME} completed-----------------------------------------------------")
    except Exception as e:
        logging.exception(e)
        raise e
  