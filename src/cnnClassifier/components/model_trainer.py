import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def setup_data_generators(self):
        datagenerator_kwargs = dict(rescale=1./255)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        if self.config.params_is_augmentration:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        eval_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "train"),
            shuffle=True,
            **dataflow_kwargs
        )

        self.test_generator = eval_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "test"),
            shuffle=False,
            **dataflow_kwargs
        )

        self.valid_generator = eval_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "valid"),
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.test_generator.samples // self.test_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.test_generator,
            validation_steps=self.validation_steps
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
