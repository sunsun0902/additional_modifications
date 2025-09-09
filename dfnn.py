import tensorflow as tf
from keras_tuner import HyperParameters

def build_dfnn_model(hp: HyperParameters, number_of_inputs: int, number_of_outputs: int):
    """
    Define the DFNN model architecture with possible hyperparameters.
    
    Parameters:
    - hp: HyperParameters object from Keras Tuner.
    - number_of_inputs: Number of input features.
    - number_of_outputs: Number of output features.
    
    Returns:
    - Keras model with the given architecture.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(number_of_inputs,)))
    model.add(
            tf.keras.layers.Dense(
                units=hp.Int(f"units_0", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
            )
    )
    if hp.Boolean("dropout_first"):
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_first_val', min_value=0.1, max_value=0.5, step=0.1)))
    
    for i in range(hp.Int("num_layers", 1, 10)):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
            )
        )
    
        if hp.Boolean("dropout_rest") & hp.Boolean("dropout_first"):
            model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_rest_val', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(number_of_outputs, activation=hp.Choice("activation", ["linear"])))
    
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss='mae',
        metrics=['mae', 'mse']
    )
    model.optimizer.lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG')
    
    return model
