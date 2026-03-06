from cnnClassifier.config import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger


STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.model_training_config = self.config.get_training_config()

    def main(self):
        try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            training = Training(config=self.model_training_config)
            training.train_valid_generator()
            training.train()
            training.save_model(path=self.model_training_config.trained_model_file_path, model=training.model)
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
        except Exception as e:
            logger.exception(e)
            raise e