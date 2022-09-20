from asyncio.log import logger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import PIL
import os


def tensor_to_image(tensor):
    """Load tensor and convert it to image

    :param tensor: Input tensor to be converted
    :type tensor: tensorflow.python.framework.ops.EagerTensor
    :return: PIL image
    :rtype: PIL.Image.Image
    """

    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    """Load and prepare input images

    :param path_to_img: Path to image
    :type path_to_img: str
    :return: Image stored as tensor
    :rtype: tensorflow.python.framework.ops.EagerTensor
    """

    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)  # convert to tensor
    # make sure that tensor has Float32 values
    img = tf.image.convert_image_dtype(img, tf.float32)  
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)

    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def save_image(image, results_dir):
    """Save image to global defined path

    :param image: Image stored as tensor
    :type image: tensorflow.python.framework.ops.EagerTensor

    """

    os.makedirs(results_dir, exist_ok=True)
    files = os.listdir(results_dir)
    file_pattern = f"-stylized.png"
    counter = 1
    filename = "n0" + file_pattern
    while filename in files:
        filename = "n" + str(counter) + file_pattern
        counter += 1

    logger.info(f"Saving image to {results_dir}/{filename}")
    tensor_to_image(image).save(
        os.path.join(results_dir, filename)
    )
