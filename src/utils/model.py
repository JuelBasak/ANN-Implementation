import tensorflow as tf


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
