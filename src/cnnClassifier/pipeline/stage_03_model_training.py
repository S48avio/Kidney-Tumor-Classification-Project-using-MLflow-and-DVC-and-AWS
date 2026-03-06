from cnnClassifier.config.configuration import ConfiguarationManager # FIX: Fixed the typo 'ConfiguarationManager'
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        # Initialize the config manager and fetch the training config
        self.config = ConfiguarationManager()
        self.model_training_config = self.config.get_training_config()

    def main(self):
        try:
            logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
            
            # Initialize the Training component
            training = Training(config=self.model_training_config)
            
            # 1. LOAD THE MODEL FIRST (This sets the 'self.model' attribute)
            training.get_base_model()
            
            # 2. Setup the data generators
            training.train_valid_generator()
            
            # 3. Start training
            training.train()
            
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
            
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        obj = ModelTrainingPipeline()
        obj.main()
    except Exception as e:
        raise e