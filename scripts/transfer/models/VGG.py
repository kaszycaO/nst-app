from .abstract_model import Model

import tensorflow as tf
import os

from keras.applications.vgg19 import VGG19

class VGG(Model):

    def __init__(self, model_name: str = "vgg19.h5"):
        super().__init__(model_name)
    
    def _set_content_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return [
            "block5_conv2"
        ]
    
    def _set_style_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return [
            "block1_conv1", 
            "block2_conv1", 
            "block3_conv1", 
            "block4_conv1", 
            "block5_conv1"
        ]
    
    def _load_model(self) -> tf.keras.Sequential:
        """Load pretrained model

        :return: Loaded model
        :rtype: tf.keras.Sequential
        """
        if os.path.exists(self.model_path):
            model = super()._load_model()
        else:
            model = VGG19(include_top=False, weights='imagenet')
            model.save(self.model_path)
        return model
