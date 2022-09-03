from scripts.common.abstract_run import Run
from scripts.common.args import ArgumentsManager

class RunGUI(Run):

    def __init__(self, args: ArgumentsManager) -> None:
        super().__init__(args)