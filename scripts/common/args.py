import os
from .image_manager import load_img
import tensorflow as tf
import logging

class ArgumentsManager:

    def __init__(
        self,
        input_img_path: str,
        style_img_path: str,
        output_dir: str,
        alpha: float,
        beta: float,
    ) -> None:
        
        self.input_path = input_img_path
        self.style_path = style_img_path
        self.alpha = alpha
        self.beta = beta
        self.output_dir = os.makedirs(output_dir, exist_ok=True)

        self.initial_result = "content"
        self.gram_matrix_type = "custom"
        self.epochs = 50
        self.steps_per_epoch = 10
        self.mode = ["SAME", "VGG"]  # content and style networks are VGG19

        self.input_image = load_img(self.input_path)
        self.style_image = load_img(self.style_path)

        self._check_gpu()
    
    def _check_gpu(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            logging.warning('GPU device not found')
        else:
            logging.info('Found GPU at: {}'.format(device_name))