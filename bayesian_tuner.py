import numpy as np
# for SHAP! we use v1
import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()
import keras_tuner as kt
from sklearn.model_selection import KFold
import os
import json
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import sys
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#print('director', os.path.dirname(os.path.dirname(__file__)))
from dfnn import build_dfnn_model
from MLmodel_autoencoder import AutoencoderModel
#from ml_hypermodels.regression.xgboost import build_xgb_regressor


# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is TensorFlow using GPU? ", tf.config.list_physical_devices('GPU'))

# Set GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# Define the time zone for Germany
tzInfo = pytz.timezone('Europe/Berlin')

class MLModel:
    class HyperModel(kt.Tuner):
        
        best_evaluation = [0]
        
        def run_trial(self, trial, x, y, epochs=100, **kwargs):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            val_losses = []
            for train_index, val_index in kf.split(x):
                train_X, val_X = x[train_index], x[val_index]
                train_Y, val_Y = y[train_index], y[val_index]
                model = self.hypermodel.build(hp=trial.hyperparameters)
                
                # Update model compilation to use MAE as loss
                if hasattr(model, 'compile'):
                    model.compile(
                        optimizer=model.optimizer,
                        loss='mae',  # Changed from 'mse' to 'mae'
                        metrics=['mae', 'mse']  # MAE first for consistency
                    )
                
                # For autoencoders, use input as target; for other models, use provided targets
                if hasattr(self, 'model_type') and getattr(self, 'model_type', '') == 'Autoencoder':
                    model.fit(train_X, train_X, shuffle=True, epochs=epochs,
                              batch_size=trial.hyperparameters.Choice('batch_size', [64, 128, 256, 512, 1024, 2048]),
                              validation_data=(val_X, val_X),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)]
                             )
                else:
                    model.fit(train_X, train_Y, shuffle=True, epochs=epochs,
                              batch_size=trial.hyperparameters.Choice('batch_size', [64, 128, 256, 512, 1024, 2048]),
                              validation_data=(val_X, val_Y),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=4, verbose=1)]
                             )
                # Evaluate and save if best model
                print("Training finished. Evaluating:")
                if hasattr(self, 'model_type') and getattr(self, 'model_type', '') == 'Autoencoder':
                    evaluation = model.evaluate(val_X, val_X)
                else:
                    evaluation = model.evaluate(val_X, val_Y)
                print("Evaluation:", evaluation)

                print(datetime.now(tz=tzInfo))
                # Use loss (index 0) for autoencoders, MAE (index 1) for other models
                if hasattr(self, 'model_type') and getattr(self, 'model_type', '') == 'Autoencoder':
                    val_losses.append(evaluation[0])  # Using loss for autoencoders
                else:
                    val_losses.append(evaluation[1])  # Using MAE for other models
                print(val_losses)
            print('val_loss:', np.mean(val_losses))
            self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
            print(trial.trial_id)
    
    class XGBHyperModel(kt.Tuner):
        
        def run_trial(self, trial, x, y, **kwargs):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            val_losses = []
            for train_index, val_index in kf.split(x):
                train_X, val_X = x[train_index], x[val_index]
                train_Y, val_Y = y[train_index], y[val_index]
                
                # Build XGBoost model with hyperparameters
                model = self.hypermodel.build(hp=trial.hyperparameters)
                
                # Train the model
                model.fit(train_X, train_Y)
                
                # Predict and calculate MAE instead of MSE
                y_pred = model.predict(val_X)
                
                # Handle multi-output regression
                if len(val_Y.shape) > 1 and val_Y.shape[1] > 1:
                    # Calculate MAE for each output and average
                    mae = np.mean([mean_absolute_error(val_Y[:, i], y_pred[:, i]) for i in range(val_Y.shape[1])])
                else:
                    mae = mean_absolute_error(val_Y, y_pred)
                
                print("Training finished. Evaluating:")
                print(f"Validation MAE: {mae}")
                print(datetime.now(tz=tzInfo))
                val_losses.append(mae)
                print(val_losses)
                
            mean_val_loss = np.mean(val_losses)
            print('val_loss:', mean_val_loss)
            self.oracle.update_trial(trial.trial_id, {'val_loss': mean_val_loss})
            print(trial.trial_id)

    def __init__(self, tuner_directory=[], project_name=[], path_model=[], model_name=[], 
                 number_of_inputs=82, number_of_outputs=1, input_shape=None, model_type='DFNN'):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.path_model = path_model
        self.model_name = model_name
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.input_shape = input_shape  # Needed for the autoencoder
        self.model_type = model_type  # Indicates the type of model to build
        self.is_xgboost = model_type == 'XGBoost'  # Flag to check if model is XGBoost

    def build_model(self, hp):
        if self.model_type == 'DFNN':
            model = build_dfnn_model(hp, self.number_of_inputs, self.number_of_outputs)
            # Update model to use MAE as loss
            model.compile(
                optimizer=model.optimizer,
                loss='mae',  # Changed from 'mse' to 'mae'
                metrics=['mae', 'mse']  # MAE first for consistency
            )
            return model
        elif self.model_type == 'Autoencoder':
            # Create an instance of AutoencoderModel and use its build method
            autoencoder_instance = AutoencoderModel(
                tuner_directory=self.tuner_directory,
                project_name=self.project_name,
                path_model=self.path_model,
                model_name=self.model_name,
                input_shape=self.input_shape
            )
            return autoencoder_instance.build_autoencoder_model(hp)
        # elif self.model_type == 'XGBoost':
        #      return build_xgb_regressor(hp)
        else:
            raise ValueError("Unsupported model type")
    
    def tune_model(self, train_x=[], train_y=[], epochs=100):
        self.tuner = self.hyperparameter_tuner()
        
        if self.is_xgboost:
            self.tuner.search(train_x, train_y)
        else:
            self.tuner.search(train_x, train_y, epochs=epochs)
 
        return print("tuner finished")
    
    def get_best_model(self, train_x=[], train_y=[], val_x=[], val_y=[]):
        self.tuner = self.hyperparameter_tuner()
        best_hps = self.tuner.get_best_hyperparameters(5)[0]
        model = self.build_model(best_hps)
        
        if not self.is_xgboost:
            print(model.summary())

        model = self.fit_model(model, train_x, train_y, val_x, val_y, best_hps)
        return model

    def hyperparameter_tuner(self):
        oracle = kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective("val_mean_absolute_error", "min"),
            max_trials=100
        )
        
        if self.is_xgboost:
            self.tuner = self.XGBHyperModel(
                hypermodel=self.build_model,
                oracle=oracle,
                overwrite=False,
                directory=self.tuner_directory,
                project_name=self.project_name
            )
        else:
            self.tuner = self.HyperModel(
                hypermodel=self.build_model,
                oracle=oracle,
                overwrite=False,
                directory=self.tuner_directory,
                project_name=self.project_name
            )
            # Pass model_type information to the tuner
            self.tuner.model_type = self.model_type
        return self.tuner
        
    def fit_model(self, model, train_x, train_y, val_x, val_y, best_hps):
        if self.is_xgboost:
            # Train XGBoost model with early stopping using MAE
            model.fit(
                train_x, train_y,
                eval_set=[(val_x, val_y)],
                eval_metric='mae',  # Use MAE as evaluation metric
                early_stopping_rounds=16,  # Match patience with DFNN
                verbose=True
            )
            
            # Save the model
            model_path = os.path.join(self.path_model, self.model_name + '.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Calculate and save metrics
            train_pred = model.predict(train_x)
            val_pred = model.predict(val_x)
            
            # Handle multi-output regression
            if len(train_y.shape) > 1 and train_y.shape[1] > 1:
                train_mae = np.mean([mean_absolute_error(train_y[:, i], train_pred[:, i]) for i in range(train_y.shape[1])])
                val_mean_absolute_error = np.mean([mean_absolute_error(val_y[:, i], val_pred[:, i]) for i in range(val_y.shape[1])])
                train_mse = np.mean([mean_squared_error(train_y[:, i], train_pred[:, i]) for i in range(train_y.shape[1])])
                val_mse = np.mean([mean_squared_error(val_y[:, i], val_pred[:, i]) for i in range(val_y.shape[1])])
            else:
                train_mae = mean_absolute_error(train_y, train_pred)
                val_mean_absolute_error = mean_absolute_error(val_y, val_pred)
                train_mse = mean_squared_error(train_y, train_pred)
                val_mse = mean_squared_error(val_y, val_pred)
            
            # Store history in same format as Keras for consistency
            history_dict = {
                'loss': [train_mae],  # Using MAE as loss
                'val_loss': [val_mean_absolute_error],
                'mae': [train_mae],
                'val_mean_absolute_error': [val_mean_absolute_error],
                'mse': [train_mse],
                'val_mse': [val_mse]
            }
            
            with open(os.path.join(self.path_model, self.model_name + '_training_history.json'), 'w') as file:
                json.dump(history_dict, file)
                
        else:
            # Train Keras model with appropriate targets
            if self.model_type == 'Autoencoder':
                # For autoencoders, target = input
                history_callback = model.fit(
                    train_x, train_x,  # Input as target
                    batch_size=best_hps.values['batch_size'],
                    epochs=1000,
                    validation_data=(val_x, val_x),  # Input as target
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',  # Monitor loss for autoencoders
                            mode="min",
                            patience=16,
                            verbose=0,
                            restore_best_weights=True
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            os.path.join(self.path_model, self.model_name),
                            save_best_only=True,
                            monitor='val_loss',  # Monitor loss for autoencoders
                            mode='min',
                            save_weights_only=False
                        )
                    ]
                )
            else:
                # For supervised models, use provided targets
                history_callback = model.fit(
                    train_x, train_y,
                    batch_size=best_hps.values['batch_size'],
                    epochs=1000,
                    validation_data=(val_x, val_y),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_mean_absolute_error',
                            mode="min",
                            patience=16,
                            verbose=0,
                            restore_best_weights=True
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            os.path.join(self.path_model, self.model_name),
                            save_best_only=True,
                            monitor='val_mean_absolute_error',
                            mode='min',
                            save_weights_only=False
                        )
                    ]
                )

            history_dict = history_callback.history

            with open(os.path.join(self.path_model, self.model_name + '_training_history.json'), 'w') as file:
                json.dump(history_dict, file)
            
        return model
        
    def plot_loss(self, history=None, 
                  training_loss_name='loss', 
                  val_loss_name='val_loss'):
        if self.is_xgboost:
            # For XGBoost, load the history from the saved file
            history_path = os.path.join(self.path_model, self.model_name + '_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as file:
                    history_dict = json.load(file)
                
                plt.figure(figsize=(10, 6))
                plt.bar(['Training MAE', 'Validation MAE'], [history_dict['mae'][0], history_dict['val_mean_absolute_error'][0]])
                plt.title('MAE for XGBoost Model')
                plt.ylabel('Mean Absolute Error')
                plt.show()
        elif history is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[training_loss_name])
            plt.plot(history.history[val_loss_name])
            plt.title('Loss During Training')
            plt.legend(['train', 'val'], loc='upper left')
            plt.ylabel('Loss (MAE)')
            plt.xlabel('epoch')
            plt.show()
