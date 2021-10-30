import os
from utils.common import read_config
import argparse
from utils.data_management import get_data
from utils.model import create_model, save_model, save_plot
from utils.callbacks import get_callbacks


def training(config_path):
    config = read_config(config_path=config_path)

    validation_data_size = config['params']['validation_data_size']
    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    LAYER1 = config['params']['LAYER1']
    LAYER2 = config['params']['LAYER2']
    EPOCHS = config['params']['epochs']
    batch_size = config['params']['batch_size']
    ARTIFACT_DIR = config['artifacts']['artifacts_dir']
    MODEL_NAME = config['artifacts']['model_name']
    MODEL_DIR = config['artifacts']['model_dir']
    PLOT_NAME = config['artifacts']['plot_name']
    PLOT_DIR = config['artifacts']['plot_dir']

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_data_size)

    model = create_model(LOSS_FUNCTION=LOSS_FUNCTION, OPTIMIZER=OPTIMIZER, METRICS=METRICS, LAYER1=LAYER1,
                         LAYER2=LAYER2)

    VALIDATION = (X_valid, y_valid)

    CALLBACK_LIST = get_callbacks(config, X_train)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, batch_size=batch_size)

    model_dir_path = os.path.join(ARTIFACT_DIR, MODEL_DIR)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model=model, modelname=MODEL_NAME, model_dir=model_dir_path)

    plot_dir_path = os.path.join(ARTIFACT_DIR, PLOT_DIR)
    os.makedirs(plot_dir_path, exist_ok=True)
    save_plot(history, filename=PLOT_NAME, path=plot_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
