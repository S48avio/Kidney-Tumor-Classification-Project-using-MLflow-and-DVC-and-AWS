from cnnClassifier.config.configuration import ConfiguarationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Base Model Preparation Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        self.config = ConfiguarationManager()
        self.prepare_base_model_config = self.config.get_prepare_base_model_config()

    def main(self):
        try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
        except Exception as e:
            logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")   


