import pandas as pd
import dask.dataframe as dd
import glob
import os
from pvlib.iotools import read_epw

def convert_epw_to_parquet(epw_directory, output_path):
    """
    Reads all EPW files in a directory using pvlib and saves them as a single Parquet file.
    """
    all_files = glob.glob(os.path.join(epw_directory, "*.epw"))
    if not all_files:
        print(f"No .epw files found in {epw_directory}")
        return

    df_list = []
    for file in all_files:
        try:
            # MODIFIED: Open the file manually with 'latin-1' encoding
            with open(file, 'r', encoding='latin-1') as f:
                # Pass the file object 'f' to pvlib instead of the filename string
                data, metadata = read_epw(f)
            
            # Add a column to identify the source of the data.
            data['location'] = metadata.get('city', os.path.basename(file))
            
            df_list.append(data)
        
        except UnicodeDecodeError:
            print(f"Could not decode file {file} with 'latin-1', skipping.")
        except Exception as e:
            print(f"An error occurred with file {file}: {e}, skipping.")
        
    if not df_list:
        print("No files were successfully processed.")
        return

    # Concatenate all DataFrames.
    master_df = pd.concat(df_list, ignore_index=False)
    
    # Move the timestamp from the index to a column called 'timestamp'
    master_df = master_df.reset_index().rename(columns={'index': 'timestamp'})
    
    master_df.to_parquet(output_path)
    print(f"Successfully created Parquet file at: {output_path}")

def process_all_epw_to_parquet_with_datacleaning(epw_directory, output_path, required_hours=8760):
    """
    Reads all EPW files, cleans them by removing sparse columns, standardizes their length,
    and saves them as a single Parquet file.
    """
    all_files = glob.glob(os.path.join(epw_directory, "*.epw"))
    if not all_files:
        print(f"No .epw files found in {epw_directory}")
        return

    # Step 1: Load all data into a single DataFrame to analyze missing values globally
    print("Loading all files to analyze data quality...")
    full_df_list = []
    locations = []
    for file in all_files:
        try:
            with open(file, 'r', encoding='latin-1') as f:
                data, metadata = read_epw(f)
            data['location'] = metadata.get('city', os.path.basename(file).replace('.epw', ''))
            full_df_list.append(data)
            locations.append(data['location'].iloc[0])
        except Exception as e:
            print(f"Could not process file {file}: {e}, skipping.")
            
    if not full_df_list:
        print("No data was loaded.")
        return
        
    master_df = pd.concat(full_df_list, ignore_index=False)

    # Step 2: Analyze and drop columns with >30% missing data
    missing_percentage = (master_df.isnull().sum() / len(master_df)) * 100
    columns_to_drop = missing_percentage[missing_percentage > 30].index
    if not columns_to_drop.empty:
        print(f"\nDropping columns with >30% missing values: {list(columns_to_drop)}")
        master_df.drop(columns=columns_to_drop, inplace=True)
    
    features_to_keep = master_df.drop(columns=['location']).columns

    # Step 3: Process each location individually to standardize length
    print("\nStandardizing length for each location to 8760 hours...")
    processed_dfs = []
    for location in locations:
        loc_df = master_df[master_df['location'] == location][features_to_keep]
        
        if len(loc_df) > required_hours:
            # Truncate if longer (e.g., leap year)
            loc_df = loc_df.iloc[:required_hours]
        elif len(loc_df) < required_hours:
            # Pad with last known value if shorter
            padding = pd.DataFrame(index=range(required_hours - len(loc_df)))
            loc_df = pd.concat([loc_df, padding]).fillna(method='ffill')

        loc_df['location'] = location # Add location identifier back
        processed_dfs.append(loc_df)

    # Step 4: Create the final DataFrame and save to Parquet
    final_df = pd.concat(processed_dfs, ignore_index=False)
    final_df = final_df.reset_index().rename(columns={'index': 'timestamp'}) # Move timestamp to a column
    final_df.to_parquet(output_path)
    print(f"\nSuccessfully created clean Parquet file at: {output_path}")

def process_epw_to_parquet_with_dask(epw_directory, output_path, required_hours=8760):
    """
    Memory-efficient EPW processing using Dask for large datasets.
    Handles the full pipeline: loading, cleaning, standardizing, and saving.
    """
    all_files = glob.glob(os.path.join(epw_directory, "*.epw"))
    if not all_files:
        print(f"No .epw files found in {epw_directory}")
        return

    print(f"Found {len(all_files)} EPW files to process...")
    
    # Step 1: Load all files as individual DataFrames first
    def load_single_epw(file_path):
        """Helper function to load a single EPW file"""
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data, metadata = read_epw(f)
            
            # Add location identifier
            location = metadata.get('city', os.path.basename(file_path).replace('.epw', ''))
            data['location'] = location
            
            # Optimize data types to reduce memory
            for col in data.select_dtypes(include=['float64']).columns:
                if col != 'location':
                    data[col] = data[col].astype('float32')
            
            return data
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # Load all files into memory-efficient format
    print("Loading EPW files...")
    dataframes = []
    for i, file in enumerate(all_files):
        print(f"Loading {i+1}/{len(all_files)}: {os.path.basename(file)}")
        df = load_single_epw(file)
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        print("No files were successfully loaded.")
        return

    # Step 2: Create Dask DataFrame with proper partitioning
    print("Creating Dask DataFrame...")
    # Use smaller partitions to manage memory better
    partition_size = max(1, len(dataframes) // 4)  # 4 partitions
    
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(pd.concat(dataframes, ignore_index=True), npartitions=4)
    
    # Step 3: Analyze missing values across the full dataset
    print("Analyzing data quality...")
    missing_counts = ddf.isnull().sum().compute()
    total_rows = len(ddf)
    missing_percentage = (missing_counts / total_rows) * 100
    
    columns_to_drop = missing_percentage[missing_percentage > 30].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >30% missing values: {columns_to_drop}")
        ddf = ddf.drop(columns=columns_to_drop)
    
    # Step 4: Process each location to standardize length
    print("Standardizing length for each location...")
    locations = ddf['location'].unique().compute()
    print(f"Processing {len(locations)} unique locations...")
    
    processed_parts = []
    for i, location in enumerate(locations):
        print(f"Processing location {i+1}/{len(locations)}: {location}")
        
        # Filter for this location and convert to pandas for processing
        loc_ddf = ddf[ddf['location'] == location]
        loc_df = loc_ddf.compute()
        
        # Drop location column for processing, we'll add it back
        features_df = loc_df.drop(columns=['location'])
        
        # Standardize to required_hours
        if len(features_df) > required_hours:
            features_df = features_df.iloc[:required_hours]
        elif len(features_df) < required_hours:
            # Pad with forward fill
            padding_rows = required_hours - len(features_df)
            last_row = features_df.iloc[-1:] if len(features_df) > 0 else None
            
            if last_row is not None:
                # Repeat last row for padding
                padding_df = pd.concat([last_row] * padding_rows, ignore_index=True)
                features_df = pd.concat([features_df, padding_df], ignore_index=True)
        
        # Add location back
        features_df['location'] = location
        processed_parts.append(features_df)
        
        # Clear memory periodically
        if i % 10 == 0:
            import gc
            gc.collect()
    
    # Step 5: Combine and save
    print("Combining processed data...")
    final_df = pd.concat(processed_parts, ignore_index=True)
    
    # Add timestamp column
    final_df = final_df.reset_index().rename(columns={'index': 'timestamp'})
    
    # Save to parquet with compression
    print("Saving to parquet...")
    final_df.to_parquet(output_path, compression='snappy', index=False)
    
    print(f"Successfully created Parquet file at: {output_path}")
    print(f"Final dataset shape: {final_df.shape}")
    
    # Memory cleanup
    del final_df, processed_parts, dataframes
    import gc
    gc.collect()

# --- USAGE ---
epw_folder = r"\\nas.ads.mwn.de\tubv\epb\zentrum\02_Forschung\04_Projekte laufend\P71_StMB_KlimaLyse\4_Bearbeitung\5_Modell\Weather_files\data_filtered"
parquet_output = r"C:\Users\go72nir\Autoencoder\Hypermodel\parquet_files\weather_data.parquet"

# Use the Dask version for better memory management
process_epw_to_parquet_with_dask(epw_folder, parquet_output)