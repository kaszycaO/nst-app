import tensorflow as tf

from scripts.common.image_manager import tensor_to_image
from .nst_model import NstModel
from .model_manager import prepare_models

import time
from .training import train_step

def prepare_model_and_data(initial_result, content_image, style_image, 
                           gram_matrix_type):
    """Prepare initial result and NstModel
    
    :param initial_result: Initial result generation type, content or noise
    :type initial_result: str
    :param content_image: Image with content
    :type content_image: tensorflow.python.framework.ops.EagerTensor
    :param style_image: Image with style
    :type style_image: tensorflow.python.framework.ops.EagerTensor
    :param gram_matrix_type: Gram matrix implemetation type, custom or tf
    :type gram_matrix_type: str
    :return: Tuple with initial result and nst model
    :rtype: tuple

    """

    if initial_result == "noise":
        initializer = tf.random_normal_initializer()
        result_image = tf.Variable(initializer(shape=content_image.shape, dtype=tf.float32))
    elif initial_result == "content":
        result_image = tf.Variable(content_image)
        tensor_to_image(result_image)
    else:
        raise Exception(f"Invalid initial_result {initial_result}!"+
                        "Available options are: noise, content")
        
    networks = prepare_models()
    nst = NstModel(content_image, style_image, networks, gram_matrix_type)
    return result_image, nst


def train(
    result_image,
    nst,
    epochs,
    steps_per_epoch,
    display_progress=True
):
    """Main function for training"""
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(result_image, nst)
            print(".", end="")
        # clear_output(wait=True)
        print(f"Train step: {step}.")
        # if display_progress:
        #     display.display(tensor_to_image(result_image))
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    return result_image
