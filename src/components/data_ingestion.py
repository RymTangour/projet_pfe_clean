from src.config.configuration import DataIngestionConfig
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub
import shutil
import os


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config=config
    def download_dataset(self) :
        download_path=kagglehub.dataset_download(self.config.source_url)

        os.makedirs(os.path.dirname(self.config.local_data_directory), exist_ok=True)
        for item in os.listdir(download_path):
            source_path=os.path.join(download_path, item)
            destination_path=os.path.join(self.config.local_data_directory, item)
            shutil.move(source_path, destination_path)
        print(f"Dataset moved to: {self.config.local_data_directory}")




     