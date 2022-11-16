from .models import VGG, MyConv56

def prepare_models(mode):
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


def get_model(network_name):
    if network_name == "VGG":
       return VGG().get_model_data()
    elif network_name == "MYCONV_56":
        return MyConv56().get_model_data()