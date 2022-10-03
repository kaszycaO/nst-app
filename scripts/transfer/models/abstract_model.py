import os
import tensorflow as tf

from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, model_name: str):
        self.models_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models"
        )
        self.model_name = model_name
        self.content_layers = self._set_content_layers()
        self.style_laters = self._set_style_layers()
        self.model = self._load_model()

    @abstractmethod
    def _set_content_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return NotImplementedError

    @abstractmethod
    def _set_style_layers(self)-> list[str]: 
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return NotImplementedError
    
    def _load_model(self) -> tf.keras.Sequential:
        """Load pretrained model

        :return: Loaded model
        :rtype: tf.keras.Sequential
        """
        return tf.keras.models.load_model(os.path.join(
            self.models_path,
            self.model_name
        ))
