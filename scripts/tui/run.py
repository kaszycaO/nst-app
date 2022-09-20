from asyncio.log import logger
from scripts.common.abstract_run import Run
from scripts.common.args import ArgumentsManager
from scripts.common.image_manager import save_image
from .visualize import compare_results

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class RunTUI(Run):

    def __init__(self, args: ArgumentsManager) -> None:
        super().__init__(args)
    
    def run(self) -> tf.Variable:
        logger.info("TUI is running")
        output_image = super().run()
        compare_results(
            self.args.input_image,
            self.args.style_image,
            output_image
        )
        save_image(
            output_image,
            self.args.output_dir
        )
