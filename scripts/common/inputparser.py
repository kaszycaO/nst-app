import argparse
from email.policy import default
import logging
import sys
import os

from typing import Optional

class InputParser(argparse.ArgumentParser):

    def error(self, message: str):
        logging.error(f"Error: {message}")
        self.print_help()
        sys.exit(2)

def check_if_path_exists(
    input_parser: InputParser,
    input_path: str
) -> Optional[str]:
    """Check if passed path exists

    :param input_parser: ArgumentParser modified instance 
    :type input_parser: InputParser
    :param input_path: Inserted path
    :type input_path: str
    :return: Input path if valid
    :rtype: str
    """

    if os.path.exists(input_path):
        return input_path
    input_parser.error(f"Input path {input_path} does not exist!")

def create_if_does_not_exist(
    input_path: str
) -> Optional[str]:
    """Create input path if it doesn't exist

    :param input_path: Inserted path
    :type input_path: str
    :return: Input path
    :rtype: str
    """

    os.makedirs(input_path, exist_ok=True)
    return input_path

def parse_arguments() -> argparse.Namespace:
    parser = InputParser()
    parser.add_argument(
        "input_image",
        help="Path to an input image",
        metavar="Input",
        type=lambda path: check_if_path_exists(parser, path)
    )
    parser.add_argument(
        "style_image",
        help="Path to a style image",
        metavar="Style",
        type=lambda path: check_if_path_exists(parser, path)
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional. Output folder, default output",
        dest="output",
        default="output",
        type=lambda path: create_if_does_not_exist(path)
    )
    parser.add_argument(
        "-a", "--alpha",
        help="Optional. Style weight, default 1e-2",
        dest="alpha",
        type=float,
        default=1e-2
    )
    parser.add_argument(
        "-b", "--beta",
        help="Optional. Content weight, default 1e4",
        dest="beta",
        type=float,
        default=1e4
    )
    parser.add_argument(
        "-t", "--tui",
        help="Optional. Run TUI instead of GUI",
        dest="tui",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "-v", "--verbose",
        help="Optional. Change log level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        default="INFO",
        dest="verbose"
    )

    return parser.parse_args()