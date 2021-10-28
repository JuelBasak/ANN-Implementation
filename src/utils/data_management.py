import tensorflow as tf


def get_data(validation_data_size):
    mnist = tf.keras.datasets.mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = mnist

    X_valid, X_train = X_train_full[:validation_data_size] / 255, X_train_full[validation_data_size:] / 255
    y_valid, y_train = y_train_full[:validation_data_size], y_train_full[validation_data_size:]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)