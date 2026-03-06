import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf
from cnnClassifier.config.configuration import ConfiguarationManager
from cnnClassifier.entities.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        try:
            # FIX: Assign to self.model, not base_model
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights  
            )
            # Now self.model exists and can be saved
            self.save_model(path=self.config.base_model_path, model=self.model)
            
        except Exception as e:
            raise e 
    
    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False # Freeze the whole base
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # FIX: These layers must be OUTSIDE the elif block 
        # so they attach regardless of how you freeze.
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes, 
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
        
    def update_base_model(self):
        # This will now work because self.model was set in get_base_model
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)