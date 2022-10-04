from typing import Dict

import tensorflow as tf

from .nst_model import NstModel
from .models import VGG, MyConv56
from .loss_functions import custom_gram, tf_gram

class ModelBuilder:

    def __init__(
        self,
        networks: Dict,
        gram_type: str
    ) -> None:
        
        self.networks = networks
        self.gram_type = gram_type




    
    @property
    def model(self):
        return self.model
    

    def _get_model(network_name):
        if network_name == "VGG":
            return VGG().get_model_data()
        elif network_name == "MYCONV_56":
            return MyConv56().get_model_data()
    
    def _prepare_models(mode):
        models = {"content": None, "style": None}
        if isinstance(mode, list):
            if len(mode) == 2:
                if mode[0] == "SAME":
                    model = get_model(mode[1])
                    models["content"] = model
                    models["style"] = model
                else:
                    models["content"] = get_model(mode[0])
                    models["style"] = get_model(mode[1])
            else:
                raise Exception("Invalid mode format! Use ['modelA', 'modelB']")
        else:
            raise Exception("Invalid mode! Type mode must be a list!")
        
        return models

    def _build(self) -> tf.keras.Sequential:
        pass

        # return NstModel(

        # )
