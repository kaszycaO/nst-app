from scripts.common.inputparser import parse_arguments
from scripts.common.args import ArgumentsManager
from scripts.tui.run import RunTUI
from scripts.gui.run import RunGUI

import logging

if __name__ == "__main__":
    args = parse_arguments()
    is_tui = args.tui
    log_level = args.verbose

    logging.basicConfig(
        level=log_level,
        format=r"%(asctime)s %(funcName)s %(filename)s %(lineno)d [%(levelname)s]: %(message)s",
        encoding="utf-8",
        handlers=[
            logging.FileHandler(
                "nst.log",
                mode='w',
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Arguments: {args}")
    manager = ArgumentsManager(
        args.input_image,
        args.style_image,
        args.output,
        args.alpha,
        args.beta
    )

    RunTUI(manager).run() if is_tui else RunGUI(manager).run()
    

