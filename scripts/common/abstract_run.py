from .args import ArgumentsManager
from scripts.transfer.run_transfer import prepare_model_and_data, train

import tensorflow as tf

class Run:

    def __init__(self, args: ArgumentsManager) -> None:
        self.args = args

    def run(self) -> tf.Variable:
        result_image, nst = prepare_model_and_data(
            self.args.initial_result,
            self.args.input_image, 
            self.args.style_image,
            self.args.gram_matrix_type
        )
        return train(
            result_image,
            nst,
            self.args.epochs,
            self.args.steps_per_epoch
        )

