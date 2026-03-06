import os
import zipfile
from cnnClassifier.utils.common import get_size
from cnnClassifier import logger
from cnnClassifier.entities.config_entity import DataIngestionConfig

import os
import shutil
import zipfile
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Fetches data from the local path defined in source_URL 
        and copies it to the artifacts directory.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Attempting to copy local file from: {self.config.source_URL}")
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            # Check if source actually exists before copying
            if os.path.exists(self.config.source_URL):
                shutil.copy(src=self.config.source_URL, dst=self.config.local_data_file)
                logger.info(f"File copied successfully to: {self.config.local_data_file}")
            else:
                logger.error(f"Source file not found at: {self.config.source_URL}")
                raise FileNotFoundError(f"File not found at {self.config.source_URL}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        logger.info(f"Unzipping file: {self.config.local_data_file} to {unzip_path}")
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
        logger.info("Unzip completed.")