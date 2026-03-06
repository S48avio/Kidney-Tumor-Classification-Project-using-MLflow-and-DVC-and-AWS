from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")

STAGE_NAME = "Base Model Preparation Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")   
