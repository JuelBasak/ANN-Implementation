from yaml import parse
from yaml.parser import Parser
from utils.common import read_config
import argparse
from utils.data_management import get_data

def training(config_path):
    config = read_config(config_path=config_path)
    validation_data_size = config['params']['validation_data_size']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_data_size)
    print(X_test)
    

if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config.yaml')
    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
