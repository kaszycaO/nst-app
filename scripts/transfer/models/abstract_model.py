import os
import tensorflow as tf


from abc import ABC, abstractmethod
from typing import Union

class Model(ABC):

    def __init__(self, model_name: str):
        """Abstract model

        :param model_name: Model name under models directory, with extention
        :type model_name: str
        """
        self.models_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "models"
        )
        self.model_name = model_name
        self.model_path = os.path.join(
            self.models_path,
            self.model_name
        )
        self.content_layers = self._set_content_layers()
        self.style_layers = self._set_style_layers()
        self.model = self._load_model()
    
    def get_model_data(
        self
    ) -> dict[str, Union[tf.keras.Sequential, list[str]]]:

        return {
            "net": self.model,
            "clayers": self.content_layers,
            "slayers": self.style_layers
        }

    @abstractmethod
    def _set_content_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        raise NotImplementedError

    @abstractmethod
    def _set_style_layers(self)-> list[str]: 
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        raise NotImplementedError
    
    def _load_model(self) -> tf.keras.Sequential:
        """Load pretrained model

        :return: Loaded model
        :rtype: tf.keras.Sequential
        """
        return tf.keras.models.load_model(self.model_path)
