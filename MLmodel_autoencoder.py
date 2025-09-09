import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout
from keras.optimizers import Adam, SGD
import keras_tuner as kt
import pytz
import json

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Define the time zone for Germany
tzInfo = pytz.timezone('Europe/Berlin')

class AutoencoderModel:
    class HyperModel(kt.HyperModel):
        def __init__(self, build_model):
            self.build_model = build_model
        
        def build(self, hp):
            return self.build_model(hp)

    def __init__(self, tuner_directory, project_name, path_model, model_name, input_shape):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.path_model = path_model
        self.model_name = model_name
        self.input_shape = input_shape
        self.encoder = None  # To store the encoder model

    def build_autoencoder_model(self, hp):
        """
        Define the autoencoder model architecture with possible hyperparameters.
        
        Parameters:
        - hp: Hyperparameters object from Keras Tuner.
        
        Returns:
        - Keras model with the given architecture.
        """
        input_layer = Input(shape=self.input_shape)

        # Encoder
        x = Conv2D(
            filters=hp.Int('encoder_conv1_filters', min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice('encoder_conv1_kernel_size', values=[3, 5, 7]),
            strides=hp.Choice('encoder_conv1_strides', values=[1, 2]),
            activation='relu',
            padding='same'
        )(input_layer)
        x = Conv2D(
            filters=hp.Int('encoder_conv2_filters', min_value=64, max_value=512, step=64),
            kernel_size=hp.Choice('encoder_conv2_kernel_size', values=[3, 5, 7]),
            strides=hp.Choice('encoder_conv2_strides', values=[1, 2]),
            activation='relu',
            padding='same'
        )(x)

        # Bottleneck
        shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
        x = Flatten()(x)
        bottleneck = Dense(
            hp.Int('bottleneck_size', min_value=16, max_value=64, step=8),
            activation=hp.Choice('bottleneck_activation', values=['relu', 'sigmoid'])
        )(x)

        # Decoder
        x = Dense(np.prod(shape_before_flattening), activation='relu')(bottleneck)
        x = Reshape(shape_before_flattening)(x)
        x = Conv2DTranspose(
            filters=hp.Int('decoder_conv1_filters', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('decoder_conv1_kernel_size', values=[3, 5, 7]),
            strides=hp.Choice('decoder_conv1_strides', values=[1, 2]),
            activation='relu',
            padding='same'
        )(x)
        output_layer = Conv2DTranspose(
            filters=self.input_shape[-1],
            kernel_size=hp.Choice('final_conv_kernel_size', values=[3, 5]),
            activation='sigmoid',
            padding='same'
        )(x)

        # Autoencoder model
        autoencoder = Model(input_layer, output_layer)

        # Compile model
        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
        if optimizer_choice == 'adam':
            optimizer = Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG'))
        else:
            optimizer = SGD(hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG'))
        
        autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse']
        )

        # Encoder model
        self.encoder = Model(input_layer, bottleneck)

        return autoencoder

    def train_or_load_model(self, train_x, epochs=100, tuning=False):
        """
        Load a trained model if available, otherwise train the model based on 
        hyperparameter tuning or run the tuner search.
        """
        # Step 1: Check if a trained model exists
        try:
            model = tf.keras.models.load_model(os.path.join(self.path_model, self.model_name))
            self.encoder = tf.keras.models.load_model(os.path.join(self.path_model, "encoder_" + self.model_name))
            print("Model loaded successfully from:", os.path.join(self.path_model, self.model_name))
            return model
        except Exception as e:
            print("No trained model found. Proceeding with hyperparameter tuning or training...")
            pass

        # Step 2: Check if best hyperparameters are available
        try:
            self.tuner = self.hyperparameter_tuner()
            best_hps = self.tuner.get_best_hyperparameters(1)[0]
            print(best_hps.values)
            model = self.build_autoencoder_model(best_hps)
            model = self.fit_model(model, train_x, epochs)
            self.encoder.save(os.path.join(self.path_model, "encoder_" + self.model_name))
            return model
        except Exception as e:
            pass

        if tuning:
            self.tuner = self.hyperparameter_tuner()
            self.tuner.search(train_x, train_x, epochs=epochs)  # Pass train_x as both input and target
            best_hps = self.tuner.get_best_hyperparameters(1)[0]
            model = self.build_autoencoder_model(best_hps)
            model = self.fit_model(model, train_x, epochs)
            self.encoder.save(os.path.join(self.path_model, "encoder_" + self.model_name))
            return model

    def hyperparameter_tuner(self):
        """
        Perform hyperparameter tuning using Keras Tuner.
        """
        self.tuner = kt.BayesianOptimization(
            hypermodel=self.HyperModel(self.build_autoencoder_model),
            objective='val_loss',
            max_trials=100,
            directory=self.tuner_directory,
            project_name=self.project_name
        )
        return self.tuner
        
    def fit_model(self, model, train_x, epochs):
        history_callback = model.fit(
            train_x, train_x,  # In autoencoder, target is the input itself
            validation_split=0.2,  # Use a portion of the training data for validation
            batch_size=32,  # Use a default value or hyperparameter-tuned value
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=15, verbose=0, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(self.path_model, self.model_name), save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)
            ]
        )

        history_dict = history_callback.history
        with open(os.path.join(self.path_model, self.model_name + '_training_history.json'), 'w') as file:
            json.dump(history_dict, file)
            
        return model
        
    def plot_loss(self, history, training_loss_name='loss', val_loss_name='val_loss'):
        plt.plot(history[training_loss_name])
        plt.plot(history[val_loss_name])
        plt.title('Loss during training')
        plt.legend(['train', 'val'], loc='upper left')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
