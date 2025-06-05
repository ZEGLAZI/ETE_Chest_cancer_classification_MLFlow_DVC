import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self) -> str:
        """
        Fetch data from url
        """
        try:
            dataset_url = self.config.source_URL
            zip_donwload_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_donwload_dir)
            logger.info(f"Downloading data from {dataset_url} into file {zip_donwload_dir}")
            
        except Exception as e:
            raise e 
        
        
    def extract_zip_file(self):
        """
        Extracts zip file into data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
                   