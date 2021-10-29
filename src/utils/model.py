import tensorflow as tf
import time
import os


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, LAYER1, LAYER2):

    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name='input_layer'),
        tf.keras.layers.Dense(LAYER1, activation='relu', name='hidden_layer_1'),
        tf.keras.layers.Dense(LAYER2, activation='relu', name='hidden_layer_2'),
        tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf


def get_unique_filename(filename):
    unique_filename = time.strftime(f'%Y-%m-%d_%H-%M-%S_{filename}')
    return unique_filename


def save_model(model, modelname, model_dir):
    unique_filename = get_unique_filename(modelname)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)
