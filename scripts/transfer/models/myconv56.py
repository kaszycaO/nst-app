from .abstract_model import Model

class MyConv56(Model):

    def __init__(self, model_name: str = "myconv56.h5"):
        super().__init__(model_name)
    
    def _set_content_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return [
            "conv4"
        ]
    
    def _set_style_layers(self) -> list[str]:
        """Set list of layers used for content calculation 

        :return: List of layers for used model
        :rtype: list[str]
        """
        return [
            "conv1", 
            "conv2", 
            "conv3", 
        ]