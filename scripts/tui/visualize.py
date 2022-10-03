from asyncio.log import logger
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from ..common.image_manager import tensor_to_image


def show_images(**data):
    """Show n images stored in data"""

    size = len(data.keys())
    _, axes = plt.subplots(1, size, figsize=(20, 10))
    for index, pair in enumerate(data.items(), 0):
        title, img = pair[0], pair[1]
        if len(img.shape) > 3:
            img = tf.squeeze(img, axis=0)
        if isinstance(axes, np.ndarray):
            ax = axes[index]
        else:
            ax = axes
        ax.set_title(title)
        ax.imshow(img)
    plt.show()


def compare_results(content_image, style_image, result_image):
    """Show image comparision

    :param content_image: Content image stored as tensor
    :type content_image: tensorflow.python.framework.ops.EagerTensor
    :param style_image: Style image stored as tensor
    :type style_image: tensorflow.python.framework.ops.EagerTensor
    :param result_image: Result image stored as tensor
    :type result_image: tensorflow.python.framework.ops.EagerTensor
    """

    logger.info("Comparing results!")
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 10))
    ax.imshow(tensor_to_image(content_image))
    ax.set_title('Content')
    ax1.imshow(tensor_to_image(style_image))
    ax1.set_title("Style")
    ax2.imshow(tensor_to_image(result_image), aspect='equal')
    ax2.set_title("Result")
    plt.show()
