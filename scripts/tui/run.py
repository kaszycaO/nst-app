from scripts.common.abstract_run import Run
from scripts.common.args import ArgumentsManager
from .visualize import compare_results

import tensorflow as tf

class RunTUI(Run):

    def __init__(self, args: ArgumentsManager) -> None:
        super().__init__(args)
    
    def run(self) -> tf.Variable:
        output_image = super().run()
        compare_results(
            self.args.input_image,
            self.args.style_image,
            output_image
        )