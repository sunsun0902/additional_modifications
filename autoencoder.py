"""
Weather Feature Extraction using Convolutional Autoencoder
end-to-end weather feature extraction from .epw files using a convolutional autoencoder with ResNet-style blocks.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, Add, Dense, Reshape,
    GlobalAveragePooling1D, UpSampling1D, Lambda
)
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import os
import tempfile
import warnings
import glob
warnings.filterwarnings('ignore')

# Try to import pvlib, provide fallback if not available
try:
    import pvlib
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    print("Warning: pvlib not available. Using fallback EPW parser.")

def download_sample_epw_file(output_dir="./"):
    """
    Download a sample .epw file from climate.onebuilding.org
    Returns the path to the downloaded .epw file
    """
    # URL for San Francisco weather file
    url = "http://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/CA_California/USA_CA_San.Francisco.Intl.AP.724940_TMY3.zip"

    print("Downloading sample weather file...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create temporary file for zip
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            tmp_zip.write(response.content)
            zip_path = tmp_zip.name

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Find the .epw file
        for file in os.listdir(output_dir):
            if file.endswith('.epw'):
                epw_path = os.path.join(output_dir, file)
                print(f"Downloaded weather file: {epw_path}")

                # Clean up temp zip file
                os.unlink(zip_path)
                return epw_path

    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating synthetic weather data for demonstration...")
        return create_synthetic_weather_data(output_dir)
    
def load_and_preprocess_directory(directory_path):
    """
    Loads all .epw files from a directory, preprocesses them, and stacks them into a single NumPy array.

    Args:
        directory_path (str): The path to the folder containing your .epw files.

    Returns:
        np.ndarray: A single array containing all preprocessed weather data,
                    with shape (num_files, 8760, 17).
    """
    all_weather_data = []
    # Use glob to find all files ending with .epw in the directory
    epw_files = glob.glob(os.path.join(directory_path, "*.epw"))

    if not epw_files:
        raise FileNotFoundError(f"No .epw files found in the directory: {directory_path}")

    print(f"Found {len(epw_files)} weather files to process.")

    for file_path in epw_files:
        print(f"\n--- Processing: {os.path.basename(file_path)} ---")
        # Preprocess each file
        # We only need the data, not the scaler, for the final dataset
        normalized_data, _ = preprocess_weather_data(file_path)
        all_weather_data.append(normalized_data)

    # Convert the list of arrays into a single, large NumPy array
    return np.array(all_weather_data)


def create_synthetic_weather_data(output_dir="./"):
    """
    Create synthetic weather data for demonstration purposes
    Returns path to synthetic .epw file
    """
    # Generate realistic synthetic weather data for one year (8760 hours)
    np.random.seed(42)

    # Time index for one year, hourly
    hours = np.arange(8760)
    days = hours // 24
    hour_of_day = hours % 24

    # Generate seasonal temperature pattern
    temp_air = 15 + 10 * np.sin(2 * np.pi * days / 365) + 5 * np.sin(2 * np.pi * hour_of_day / 24)+ np.random.normal(0, 2, 8760)

    # Generate correlated weather variables
    relative_humidity = 50 + 20 * np.sin(2 * np.pi * days / 365 + np.pi) + np.random.normal(0, 10,8760)
    relative_humidity = np.clip(relative_humidity, 0, 100)

    atmospheric_pressure = 101325 + 1000 * np.sin(2 * np.pi * days / 365) + np.random.normal(0,500, 8760)

    # Solar radiation (zero at night, peak during day)
    solar_mask = (hour_of_day >= 6) & (hour_of_day <= 18)
    global_horizontal_radiation = np.zeros(8760)
    global_horizontal_radiation[solar_mask] = 400 + 300 * np.sin(np.pi * (hour_of_day[solar_mask]- 6) / 12) + np.random.normal(0, 50, np.sum(solar_mask))
    global_horizontal_radiation = np.maximum(0, global_horizontal_radiation)

    # Wind speed
    wind_speed = 3 + 2 * np.sin(2 * np.pi * days / 365) + np.random.exponential(2, 8760)
    wind_speed = np.clip(wind_speed, 0, 20)

    # Create synthetic EPW-like data
    synthetic_data = pd.DataFrame({
        'year': 2023,
        'month': ((days // 30) % 12) + 1,
        'day': (days % 30) + 1,
        'hour': hour_of_day,
        'minute': 0,
        'temp_air': temp_air,
        'temp_dew': temp_air - 5 - np.random.exponential(5, 8760),
        'relative_humidity': relative_humidity,
        'atmospheric_pressure': atmospheric_pressure,
        'wind_speed': wind_speed,
        'wind_direction': np.random.uniform(0, 360, 8760),
        'global_horizontal_radiation': global_horizontal_radiation,
        'direct_normal_radiation': global_horizontal_radiation * 0.8,
        'diffuse_horizontal_radiation': global_horizontal_radiation * 0.2,
        'infrared_horizontal_radiation': 300 + 50 * np.sin(2 * np.pi * days / 365),
        'zenith_luminance': global_horizontal_radiation * 100,
        'total_sky_cover': np.random.randint(0, 11, 8760),
        'opaque_sky_cover': np.random.randint(0, 11, 8760),
        'visibility': np.random.uniform(5, 50, 8760),
        'ceiling_height': np.random.uniform(1000, 10000, 8760),
        'present_weather_observation': 0,
        'present_weather_codes': 0,
        'precipitable_water': np.random.exponential(20, 8760),
        'aerosol_optical_depth': np.random.uniform(0.1, 0.5, 8760),
        'snow_depth': 0,
        'days_since_last_snowfall': 999
    })

    # Save as CSV (simplified EPW format)
    synthetic_path = os.path.join(output_dir, "synthetic_weather.epw")
    synthetic_data.to_csv(synthetic_path, index=False)
    print(f"Created synthetic weather file: {synthetic_path}")

    return synthetic_path

def preprocess_weather_data(file_path):
    """
    Preprocess weather data from .epw file
    
    Args:
        file_path: Path to .epw file
        
    Returns:
        normalized_data: Normalized numpy array of shape (8760, 17)
        scaler: Fitted StandardScaler for later inverse transform
    """
    print("Loading and preprocessing weather data...")

    # Load EPW data using pvlib if available, otherwise use pandas
    if PVLIB_AVAILABLE:
        try:
            df, metadata = pvlib.iotools.read_epw(file_path)
            print("Loaded EPW file using pvlib")
        except Exception as e:
            print(f"pvlib loading failed: {e}, using pandas fallback")
            # Try different encodings for the fallback
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError(f"Could not read file with any encoding: {encodings}")
    else:
        # Try different encodings for pandas-only approach
        encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not read file with any encoding: {encodings}")

    print(f"Raw data shape: {df.shape}")

    # Select the 17 most relevant weather variables (based on repository analysis)
    weather_columns = [
        'temp_air', 'temp_dew', 'relative_humidity', 'atmospheric_pressure',
        'wind_speed', 'wind_direction', 'global_horizontal_radiation',
        'direct_normal_radiation', 'diffuse_horizontal_radiation',
        'infrared_horizontal_radiation', 'zenith_luminance', 'total_sky_cover',
        'opaque_sky_cover', 'visibility', 'ceiling_height',
        'precipitable_water', 'aerosol_optical_depth'
    ]

    # Handle column names that might differ
    available_columns = []
    for col in weather_columns:
        if col in df.columns:
            available_columns.append(col)
        else:
            # Try alternative names
            alternatives = {
                'temp_air': ['temperature', 'dry_bulb_temperature'],
                'temp_dew': ['dew_point_temperature'],
                'atmospheric_pressure': ['pressure', 'station_pressure'],
                'global_horizontal_radiation': ['ghi', 'global_horizontal_irradiance']
            }
            found = False
            if col in alternatives:
                for alt in alternatives[col]:
                    if alt in df.columns:
                        available_columns.append(alt)
                        found = True
                        break
            if not found:
                print(f"Warning: Column '{col}' not found, using zeros")
                df[col] = 0
                available_columns.append(col)

    # Select available weather columns (up to 17)
    weather_data = df[available_columns[:17]].copy()

    # Ensure we have exactly 17 columns
    while weather_data.shape[1] < 17:
        weather_data[f'dummy_{weather_data.shape[1]}'] = 0

    weather_data = weather_data.iloc[:, :17]  # Take first 17 columns

    # Handle missing values (replace sentinel values like 9999, 999900.0)
    weather_data = weather_data.replace([9999, 999900.0, -9999], method='ffill')
    weather_data = weather_data.fillna(method='ffill').fillna(0)

    # Ensure we have exactly 8760 hours (one year)
    if len(weather_data) > 8760:
        weather_data = weather_data.iloc[:8760]
    elif len(weather_data) < 8760:
        # Repeat data to fill year
        repeats = 8760 // len(weather_data) + 1
        weather_data = pd.concat([weather_data] * repeats, ignore_index=True)[:8760]

    print(f"Selected weather data shape: {weather_data.shape}")

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(weather_data.values)

    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

    return normalized_data, scaler

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

def build_autoencoder(input_shape=(8760, 17), latent_dim=13):
    """
    Build convolutional autoencoder with ResNet-style encoder blocks
    
    Args:
        input_shape: Shape of input weather data (time_steps, features)
        latent_dim: Dimension of latent feature vector
        
    Returns:
        autoencoder: Complete autoencoder model
        encoder: Encoder part only
        decoder: Decoder part only
    """
    print(f"Building autoencoder with input shape {input_shape}, latent dim {latent_dim}")

    # ==== ENCODER ====
    encoder_input = Input(shape=input_shape, name='weather_input')
    x = encoder_input

    # ResNet Block 1: 10 filters
    x = create_resnet_block(x, filters=10, kernel_sizes=[8, 5, 3], block_name="encoder_block1")

    # ResNet Block 2: 20 filters  
    x = create_resnet_block(x, filters=20, kernel_sizes=[8, 5, 3], block_name="encoder_block2")

    # ResNet Block 3: 20 filters
    x = create_resnet_block(x, filters=20, kernel_sizes=[8, 5, 3], block_name="encoder_block3")

    # Global average pooling to compress temporal dimension
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Final encoding layer to latent dimension
    latent = Dense(latent_dim, activation='linear', name='latent_vector')(x)

    # Create encoder model
    encoder = Model(encoder_input, latent, name='encoder')

    # ==== DECODER ====
    decoder_input = Input(shape=(latent_dim,), name='latent_input')

    # Expand latent vector back to a sequence
    # Start with a smaller sequence and upsample
    x = Dense(128, activation='relu', name='decoder_dense1')(decoder_input)
    x = Dense(256, activation='relu', name='decoder_dense2')(x)

    # Reshape to start sequence (32 timesteps, 20 channels to match encoder output)
    x = Dense(32 * 20, activation='relu', name='decoder_dense3')(x)
    x = Reshape((32, 20), name='decoder_reshape')(x)

    # Upsample back to original temporal resolution
    # 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 4096 -> 8192 -> 8760
    x = UpSampling1D(size=2, name='decoder_upsample1')(x)  # 32 -> 64
    x = Conv1D(20, 3, padding='same', activation='relu', name='decoder_conv1')(x)

    x = UpSampling1D(size=2, name='decoder_upsample2')(x)  # 64 -> 128  
    x = Conv1D(20, 3, padding='same', activation='relu', name='decoder_conv2')(x)

    x = UpSampling1D(size=2, name='decoder_upsample3')(x)  # 128 -> 256
    x = Conv1D(20, 3, padding='same', activation='relu', name='decoder_conv3')(x)

    x = UpSampling1D(size=2, name='decoder_upsample4')(x)  # 256 -> 512
    x = Conv1D(15, 3, padding='same', activation='relu', name='decoder_conv4')(x)

    x = UpSampling1D(size=2, name='decoder_upsample5')(x)  # 512 -> 1024
    x = Conv1D(15, 3, padding='same', activation='relu', name='decoder_conv5')(x)

    x = UpSampling1D(size=2, name='decoder_upsample6')(x)  # 1024 -> 2048
    x = Conv1D(15, 3, padding='same', activation='relu', name='decoder_conv6')(x)

    x = UpSampling1D(size=2, name='decoder_upsample7')(x)  # 2048 -> 4096
    x = Conv1D(15, 3, padding='same', activation='relu', name='decoder_conv7')(x)

    x = UpSampling1D(size=2, name='decoder_upsample8')(x)  # 4096 -> 8192
    x = Conv1D(17, 3, padding='same', activation='relu', name='decoder_conv8')(x)

    # Fine-tune to exact length (8760)
    x = UpSampling1D(size=2, name='decoder_upsample9')(x)  # 8192 -> 16384

    # Crop to exact target length and adjust channels
    x = Lambda(lambda x: x[:, :8760, :], name='crop_to_8760')(x)

    # Final output layer with linear activation
    decoder_output = Conv1D(17, 1, padding='same', activation='linear',
                            name='decoder_output')(x)

    # Create decoder model
    decoder = Model(decoder_input, decoder_output, name='decoder')

    # ==== COMPLETE AUTOENCODER ====
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output, name='autoencoder')

    return autoencoder, encoder, decoder

# def main():
#     """
#     Main execution function
#     """
#     print("=== Weather Feature Extraction with Convolutional Autoencoder ===\n")

#     # Step 1: Download sample weather data
#     epw_file_path = download_sample_epw_file()

#     # Step 2: Preprocess the weather data
#     weather_data, scaler = preprocess_weather_data(epw_file_path)

#     # Step 3: Reshape data for batch processing (add batch dimension)
#     weather_batch = weather_data.reshape(1, 8760, 17)
#     print(f"Training data shape: {weather_batch.shape}")

#     # Step 4: Build the autoencoder model
#     autoencoder, encoder, decoder = build_autoencoder(input_shape=(8760, 17), latent_dim=13)

#     # Print model summaries
#     print("\n=== ENCODER ARCHITECTURE ===")
#     encoder.summary()

#     print("\n=== DECODER ARCHITECTURE ===")
#     decoder.summary()

#     print("\n=== COMPLETE AUTOENCODER ARCHITECTURE ===")
#     autoencoder.summary()

#     # Step 5: Compile the autoencoder
#     autoencoder.compile(
#         optimizer='adam',
#         loss='mse',
#         metrics=['mae']
#     )

#     print("\n=== TRAINING AUTOENCODER ===")

#     # Step 6: Train the autoencoder
#     # Input and target are the same (reconstruction task)
#     history = autoencoder.fit(
#         weather_batch, weather_batch,
#         epochs=1000,
#         batch_size=1,
#         verbose=1,
#         validation_split=0.0  # No validation split since we only have one sample
#     )

#     print("\n=== FEATURE EXTRACTION ===")

#     # Step 7: Extract features using the trained encoder
#     latent_features = encoder.predict(weather_batch)

#     print(f"Original weather data shape: {weather_batch.shape}")
#     print(f"Extracted feature vector shape: {latent_features.shape}")
#     print(f"Compression ratio: {(weather_batch.size / latent_features.size):.2f}:1")

#     print(f"\nExtracted feature vector:")
#     print(f"Values: {latent_features[0]}")
#     print(f"Mean: {latent_features[0].mean():.4f}")
#     print(f"Std: {latent_features[0].std():.4f}")
#     print(f"Range: [{latent_features[0].min():.4f}, {latent_features[0].max():.4f}]")

#     # Step 8: Test reconstruction quality
#     reconstructed = autoencoder.predict(weather_batch)
#     reconstruction_error = np.mean(np.square(weather_batch - reconstructed))

#     print(f"\n=== RECONSTRUCTION QUALITY ===")
#     print(f"Mean Squared Error: {reconstruction_error:.6f}")
#     print(f"Root Mean Squared Error: {np.sqrt(reconstruction_error):.6f}")

#     # Print training history
#     print(f"\n=== TRAINING HISTORY ===")
#     final_loss = history.history['loss'][-1]
#     final_mae = history.history['mae'][-1]
#     print(f"Final training loss (MSE): {final_loss:.6f}")
#     print(f"Final training MAE: {final_mae:.6f}")

#     print("\n=== COMPLETED SUCCESSFULLY ===")
#     print("The trained encoder can now be used to extract 13-dimensional feature vectors")
#     print("from any preprocessed weather data with the same format.")

#     return encoder, autoencoder, scaler, latent_features

# if __name__ == "__main__":
#     # Run the complete pipeline
#     encoder, autoencoder, scaler, features = main()

def main():
    """
    Main execution function to train on a directory of weather files.
    """
    print("=== Weather Feature Extraction with Convolutional Autoencoder ===\n")

    # Step 1: sample weather data
    # epw_directory = "/Volumes/epb/Zentrum/02_Forschung/04_Projekte laufend/P71_StMB_KlimaLyse/4_Bearbeitung/5_Modell/Weather_files/data_filtered"
    epw_directory = "/Users/dedesun/Documents/klimalyse/Autoencoder/Autoencoder/epw_files"
    try:
        # Step 2: Load and preprocess all weather files from the directory
        weather_dataset = load_and_preprocess_directory(epw_directory)
        print(f"\nSuccessfully loaded and processed all files.")
        print(f"Final training dataset shape: {weather_dataset.shape}")
    except FileNotFoundError as e:
        print(e)
        print("Please update the 'epw_directory' variable in the script to point to your data.")
        return # Exit if no data is found

    # Step 3: Build the autoencoder model
    autoencoder, encoder, decoder = build_autoencoder(input_shape=(8760, 17), latent_dim=13)

    # Step 4: Compile the autoencoder
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    print("\n=== TRAINING AUTOENCODER ON MULTIPLE FILES ===")

    # Step 5: Train the autoencoder on the entire dataset
    # The input and target are the same (reconstruction task)
    history = autoencoder.fit(
        weather_dataset, weather_dataset,
        epochs=100,  
        batch_size=8, # Use a larger batch size for faster training
        verbose=1,
        validation_split=0.1 # Use 10% of the data for validation
    )

    print("\n=== FEATURE EXTRACTION FROM THE FIRST FILE ===")

    # Step 6: Extract features from the first file in the dataset as an example
    sample_to_encode = np.expand_dims(weather_dataset[0], axis=0) # Shape becomes (1, 8760, 17)
    latent_features = encoder.predict(sample_to_encode)

    print(f"Original sample data shape: {sample_to_encode.shape}")
    print(f"Extracted feature vector shape: {latent_features.shape}")
    print(f"\nExtracted feature vector for the first file:\n{latent_features[0]}")

    print("\n=== COMPLETED SUCCESSFULLY ===")

    return encoder, autoencoder

if __name__ == "__main__":
    # Run the complete pipeline
    encoder, autoencoder = main()

    # --- Add these lines to save the encoder ---
    print("\n=== SAVING THE TRAINED ENCODER ===")
    encoder.save('weather_encoder.keras')
    print("Encoder saved to weather_encoder.keras")
