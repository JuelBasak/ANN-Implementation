import tensorflow as tf
import os
import numpy as np
import time

def get_timestamp():
    unique_name = time.strftime(f'%Y-%m-%d_%H-%M-%S')
    return unique_name


def get_callbacks(config, X_train):
    log_dir = config['logs']['log_dir']
    tensorboard_log_dir = config['logs']['tensorboard_logs']

    artifacts_dir = config['artifacts']['artifacts_dir']
    checkpoint_dir = config['artifacts']['checkpoint_dir']

    patience = config['params']['patience']
    restore_best_weights= config['params']['restore_best_weights']
    save_best_only = config['params']['save_best_only']

    dir_path = os.path.join(log_dir, tensorboard_log_dir, get_timestamp())
    file_writer = tf.summary.create_file_writer(logdir=dir_path)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1,28,28,1))
        tf.summary.image('20 Handwritten Digits Samples', images, max_outputs=25, step=0)

    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=dir_path)

    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=restore_best_weights)

    CHKPT_dir_path = os.path.join(artifacts_dir, checkpoint_dir)
    os.makedirs(CHKPT_dir_path, exist_ok=True)
    CHKPT_path = os.path.join(CHKPT_dir_path, f'{get_timestamp()}_model_chkpt.h5')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHKPT_path, save_best_only=save_best_only)


    return [tensorboard_cb, early_stopping_cb, checkpoint_cb]
