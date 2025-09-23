import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Conv1D, Conv1DTranspose, Flatten, Reshape, Dropout,
    BatchNormalization, ReLU, Add, GlobalAveragePooling1D, UpSampling1D, Lambda
) #Add Conv1DTranspose
from keras.optimizers import Adam, SGD
import keras_tuner as kt
import pytz
import json

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Define the time zone for Germany
tzInfo = pytz.timezone('Europe/Berlin')

def create_resnet_block(x, filters, kernel_sizes=[8, 5, 3], block_name="resnet_block"):
    """
    Create a ResNet-style convolutional block with residual connection

    Args:
        x: Input tensor
        filters: Number of filters for conv layers
        kernel_sizes: List of kernel sizes for the three conv layers
        block_name: Name prefix for layers

    Returns:
        Output tensor with residual connection
    """
    # Store input for residual connection
    shortcut = x

    # First conv layer
    x = Conv1D(filters, kernel_sizes[0], padding='same',
                name=f"{block_name}_conv1")(x)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = ReLU(name=f"{block_name}_relu1")(x)

    # Second conv layer
    x = Conv1D(filters, kernel_sizes[1], padding='same',
                name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)
    x = ReLU(name=f"{block_name}_relu2")(x)

    # Third conv layer
    x = Conv1D(filters, kernel_sizes[2], padding='same',
                name=f"{block_name}_conv3")(x)
    x = BatchNormalization(name=f"{block_name}_bn3")(x)

    # Adjust shortcut dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same',
                        name=f"{block_name}_shortcut")(shortcut)

    # Add residual connection
    x = Add(name=f"{block_name}_add")([x, shortcut])
    x = ReLU(name=f"{block_name}_relu_final")(x)

    return x

def create_resnet_decoder_block(x, filters, kernel_sizes=[8, 5, 3], block_name="resnet_decoder_block",
                               upsample=True, upsample_size=2):
    """
    Create a ResNet-style decoder block with residual connection and upsampling

    Args:
        x: Input tensor
        filters: Number of filters for conv layers
        kernel_sizes: List of kernel sizes for the three conv layers
        block_name: Name prefix for layers
        upsample: Whether to include upsampling at the beginning
        upsample_size: Size of upsampling factor

    Returns:
        Output tensor with residual connection and optional upsampling
    """
    # Optional upsampling at the beginning
    if upsample:
        x = UpSampling1D(size=upsample_size, name=f"{block_name}_upsample")(x)

    # Store input for residual connection
    shortcut = x

    # First conv layer
    x = Conv1D(filters, kernel_sizes[0], padding='same',
                name=f"{block_name}_conv1")(x)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = ReLU(name=f"{block_name}_relu1")(x)

    # Second conv layer
    x = Conv1D(filters, kernel_sizes[1], padding='same',
                name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)
    x = ReLU(name=f"{block_name}_relu2")(x)

    # Third conv layer
    x = Conv1D(filters, kernel_sizes[2], padding='same',
                name=f"{block_name}_conv3")(x)
    x = BatchNormalization(name=f"{block_name}_bn3")(x)

    # Adjust shortcut dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same',
                        name=f"{block_name}_shortcut")(shortcut)

    # Add residual connection
    x = Add(name=f"{block_name}_add")([x, shortcut])
    x = ReLU(name=f"{block_name}_relu_final")(x)

    return x

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
        Build convolutional autoencoder with ResNet-style encoder blocks using hyperparameters.

        Parameters:
        - hp: Hyperparameters object from Keras Tuner.

        Returns:
        - Keras model with ResNet architecture.
        """
        print(f"Building tunable ResNet autoencoder with input shape {self.input_shape}")

        # ==== ENCODER ====
        encoder_input = Input(shape=self.input_shape, name='weather_input')
        x = encoder_input

        # Hyperparameter for number of ResNet blocks
        num_resnet_blocks = hp.Int('num_resnet_blocks', min_value=1, max_value=5, step=1)

        # Build ResNet blocks with tunable parameters
        for i in range(num_resnet_blocks):
            filters = hp.Choice(f'filters_block_{i}', values=[16, 32, 64])
            x = create_resnet_block(x, filters=filters, kernel_sizes=[8, 5, 3],
                                   block_name=f"encoder_block{i+1}")

        # Global average pooling to compress temporal dimension
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)

        # Tunable latent dimension
        latent_dim = hp.Int('latent_dim', min_value=8, max_value=64, step=4)
        latent = Dense(latent_dim, activation='linear', name='latent_vector')(x)

        # Create encoder model
        encoder = Model(encoder_input, latent, name='encoder')

        # ==== DECODER ====
        decoder_input = Input(shape=(latent_dim,), name='latent_input')

        # Expand latent vector back to a sequence with tunable initial dimensions
        initial_length = hp.Choice('decoder_initial_length', values=[16, 32, 64])
        initial_filters = hp.Choice('decoder_initial_filters', values=[16, 32, 64])

        x = Dense(128, activation='relu', name='decoder_dense1')(decoder_input)
        x = Dense(256, activation='relu', name='decoder_dense2')(x)
        x = Dense(initial_length * initial_filters, activation='relu', name='decoder_dense3')(x)
        x = Reshape((initial_length, initial_filters), name='decoder_reshape')(x)

        # Build decoder blocks mirroring encoder structure (in reverse)
        # Calculate number of upsampling steps needed to reach target length (8760)
        current_length = initial_length
        target_length = 8760

        # Modular decoder blocks with tunable parameters
        for i in range(num_resnet_blocks):
            # Determine if we need upsampling for this block
            if current_length < target_length:
                # Calculate appropriate upsample size to get closer to target
                remaining_blocks = num_resnet_blocks - i
                remaining_ratio = target_length / current_length
                upsample_size = min(int(remaining_ratio ** (1.0 / remaining_blocks)), 4)
                upsample_size = max(upsample_size, 2)  # Minimum upsample of 2

                # Apply upsampling if needed
                if current_length * upsample_size <= target_length * 2:  # Safety check
                    current_length *= upsample_size
                    upsample = True
                else:
                    upsample = False
                    upsample_size = 1
            else:
                upsample = False
                upsample_size = 1

            # Get tunable filters for this decoder block (reverse order from encoder)
            reverse_idx = num_resnet_blocks - 1 - i
            filters = hp.Choice(f'decoder_filters_block_{i}', values=[16, 32, 64])

            x = create_resnet_decoder_block(
                x,
                filters=filters,
                kernel_sizes=[8, 5, 3],
                block_name=f"decoder_block{i+1}",
                upsample=upsample,
                upsample_size=upsample_size
            )

        # Final upsampling and adjustments to reach exact target length
        while current_length < target_length:
            if current_length * 2 <= target_length:
                x = UpSampling1D(size=2, name=f'decoder_final_upsample_{current_length}')(x)
                current_length *= 2
            else:
                break

        # Crop or pad to exact target length if needed
        if current_length != target_length:
            if current_length > target_length:
                # Crop to exact length
                x = Lambda(lambda x: x[:, :target_length, :], name='crop_to_target')(x)
            else:
                # Pad if needed (though this should be rare with proper upsampling)
                padding_needed = target_length - current_length
                x = Lambda(lambda x: tf.pad(x, [[0, 0], [0, padding_needed], [0, 0]],
                                          mode='REFLECT'), name='pad_to_target')(x)

        # Final output layer with linear activation to match input features
        decoder_output = Conv1D(self.input_shape[-1], 1, padding='same', activation='linear',
                                name='decoder_output')(x)

        # Create decoder model
        decoder = Model(decoder_input, decoder_output, name='decoder')

        # ==== COMPLETE AUTOENCODER ====
        autoencoder_output = decoder(encoder(encoder_input))
        autoencoder = Model(encoder_input, autoencoder_output, name='autoencoder')

        # Compile model with tunable optimizer and learning rate
        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG')

        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)

        autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mse']
        )

        # Store encoder model
        self.encoder = encoder

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
