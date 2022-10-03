from abstract_model import Model

class VGG(Model):

    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    def _set_content_layers(self) -> list[str]:
        return super()._set_content_layers()
        