import tensorflow as tf
import os

from keras.applications.vgg19 import VGG19

drive_path = ""

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
        content_layers = ["block5_conv2"]
        style_layers = ["block1_conv1", 
                        "block2_conv1", 
                        "block3_conv1", 
                        "block4_conv1", 
                        "block5_conv1"]
        
        return {"net": VGG19(include_top=False, weights='imagenet'), 
                "clayers": content_layers,
                "slayers": style_layers
                }
    elif "ALEX_" in network_name:
        content_layers = ["conv5"]
        style_layers = ["conv1", 
                        "conv2",
                        "conv3",
                        "conv4"]

        if network_name == "ALEX_32":
            model = tf.keras.models.load_model(os.path.join(drive_path, "models", "myconv2021_alex_32.h5"))
        elif network_name == "ALEX_42":
            model = tf.keras.models.load_model(os.path.join(drive_path, "models", "myconv2021_alex_42.h5"))
        elif network_name == "ALEX_50":
            model = tf.keras.models.load_model(os.path.join(drive_path, "models", "myconv2021_alex_50.h5"))
        elif network_name == "ALEX_TUT":
            model = tf.keras.models.load_model(os.path.join(drive_path, "models", "myconv2021_alex_tut.h5"))
        
        return {"net": model, 
                "clayers": content_layers,
                "slayers": style_layers
                }

    elif network_name == "MYCONV_56":
        content_layers = ["conv4"]
        style_layers = ["conv1", 
                        "conv2",
                        "conv3"]
        model = tf.keras.models.load_model(os.path.join(drive_path, "models", "my_own_conv2021_56.h5"))

        return {"net": model, 
                "clayers": content_layers,
                "slayers": style_layers
                }
