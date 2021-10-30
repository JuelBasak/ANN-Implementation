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


    dir_path = os.path.join(log_dir, tensorboard_log_dir, get_timestamp())
    
    file_writer = tf.summary.create_file_writer(logdir=dir_path)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1,28,28,1))
        tf.summary.image('20 Handwritten Digits Samples', images, max_outputs=25, step=0)
