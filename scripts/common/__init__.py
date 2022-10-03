import configparser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

config_path = os.path.join(
    os.path.dirname(__file__), "..", "config.ini"
)

nst_config = configparser.ConfigParser()
nst_config.read(config_path)