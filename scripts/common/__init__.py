import configparser
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

config_path = pathlib.Path(
    __file__
).parent / "config.ini"


nst_config = configparser.ConfigParser()
nst_config.read(config_path)
