from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfiguarationManager
from cnnClassifier import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfiguarationManager()
        self.evaluation_config = self.config.get_evaluation_config()

    def main(self):
        try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            evaluation = Evaluation(config=self.evaluation_config)
            evaluation.evaluation()
            evaluation.save_score()
            #evaluation.log_into_mlflow()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
        except Exception as e:
            logger.exception(e)
            raise e