from cnnClassifier.config.configuration import ConfiguarationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfiguarationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            # Re-raising ensures the outer exception block also sees the failure
            raise e

# --- MOVE THIS TO THE LEFT (NO INDENTATION) ---
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")