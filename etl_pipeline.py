# ==============================================================================
# DELIVERABLE 1 & 2: THE COMPLETE PYTHON ETL PIPELINE
# File: etl_pipeline.py
# ==============================================================================

import pandas as pd
import os
import sqlite3
from sqlite3 import Error
import logging

# --- 1. SETUP LOGGING ---
# Set up a professional logging system to track the pipeline's progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 2. CONFIGURATION & PATHS ---
import os

# Base project folder
BASE_DIR = os.getcwd()

# Input files folder (where you placed your Excel & CSV files)
INPUT_PATH = os.path.join(BASE_DIR)  # since files are in the project root

# Output folder for CSV & DB
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_PATH, exist_ok=True)

DB_PATH = os.path.join(OUTPUT_PATH, 'nhs_data.db')
TABLEAU_CSV_PATH = os.path.join(OUTPUT_PATH, 'tableau_data.csv')



DB_PATH = os.path.join(OUTPUT_PATH, 'nhs_data.db')
TABLEAU_CSV_PATH = os.path.join(OUTPUT_PATH, 'tableau_data.csv')

# Configuration for reading the complex Excel files
file_config = {
    'AE_Activity.xlsx':      {'sheet_name': 'ae_attendances', 'header': 10},
    'AE_Quality_Index.xlsx':   {'sheet_name': 'AE_Quality_Index', 'header': 0},
    'Consultation.csv':        {'is_csv': True}
}

# --- 3. EXTRACT PHASE ---
def extract_data(input_path, config):
    """
    Loads all data from the source directory based on the configuration.
    Includes error handling for missing files.
    """
    logging.info("Starting [EXTRACT] phase...")
    dataframes = {}
    try:
        for filename, file_info in config.items():
            file_path = os.path.join(input_path, filename)
            logging.info(f"Loading '{filename}'...")
            if file_info.get('is_csv', False):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=file_info['sheet_name'], header=file_info['header'])
            dataframes[filename] = df
        logging.info("[EXTRACT] phase completed successfully.")
        return dataframes
    except FileNotFoundError as e:
        logging.error(f"File not found during EXTRACT phase: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during EXTRACT phase: {e}")
        return None

# --- 4. TRANSFORM PHASE ---
def transform_data(dataframes):
    """
    Cleans, reshapes, and merges the raw dataframes into a single,
    analysis-ready dataframe. Includes error handling.
    """
    logging.info("Starting [TRANSFORM] phase...")
    if dataframes is None:
        logging.error("No dataframes to transform. Aborting.")
        return None
    try:
        # Transform A&E Activity data
        df_activity = dataframes['AE_Activity.xlsx'].copy()
        df_activity.rename(columns={'Unnamed: 0': 'A&E Type'}, inplace=True)
        df_activity_long = pd.melt(df_activity, id_vars=['A&E Type'], var_name='Year', value_name='Total Attendances')
        yearly_attendance = df_activity_long.groupby('Year')['Total Attendances'].sum().reset_index()
        yearly_attendance['Year_Start'] = pd.to_datetime(yearly_attendance['Year'].str.split('-').str[0] + '-04-01')

        # Transform A&E Quality data
        df_quality = dataframes['AE_Quality_Index.xlsx'].copy()
        df_quality['Month'] = pd.to_datetime(df_quality['ATTENDANCE_MONTH'])
        df_quality['Measure_Value'] = pd.to_numeric(df_quality['MEASURE_VALUE'], errors='coerce')
        
        # Pivot quality data to get metrics as columns
        df_quality_pivot = df_quality.pivot_table(
            index=['ORG_CODE', 'ORG_NAME', 'Month'],
            columns='MEASURE_NAME',
            values='Measure_Value'
        ).reset_index()

        # Merge with yearly attendance data
        final_df = pd.merge_asof(
            df_quality_pivot.sort_values('Month'),
            yearly_attendance.sort_values('Year_Start'),
            left_on='Month',
            right_on='Year_Start',
            direction='backward'
        )
        
        logging.info("[TRANSFORM] phase completed successfully.")
        return final_df
        
    except KeyError as e:
        logging.error(f"A required column was not found during TRANSFORM phase: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during TRANSFORM phase: {e}")
        return None

# --- 5. LOAD PHASE ---
def load_data(df, db_path, tableau_path):
    """
    Loads the transformed dataframe into a SQLite database and a CSV file.
    Includes error handling for database connections.
    """
    logging.info("Starting [LOAD] phase...")
    if df is None:
        logging.error("No transformed data to load. Aborting.")
        return

    # Load to SQLite Database
    try:
        logging.info(f"Connecting to SQLite database at '{db_path}'...")
        conn = sqlite3.connect(db_path)
        # Use the table name 'quality_and_attendance'
        df.to_sql('quality_and_attendance', conn, if_exists='replace', index=False)
        conn.close()
        logging.info("Successfully loaded data into SQLite table 'quality_and_attendance'.")
    except Error as e:
        logging.error(f"Database error during LOAD phase: {e}")

    # Load to CSV for Tableau
    try:
        logging.info(f"Saving analysis-ready CSV to '{tableau_path}'...")
        df.to_csv(tableau_path, index=False)
        logging.info("Successfully saved CSV for Tableau.")
    except Exception as e:
        logging.error(f"Could not save CSV file: {e}")

# --- 6. EXECUTE THE PIPELINE ---
if __name__ == "__main__":
    logging.info("====== NHS Data ETL Pipeline: START ======")
    
    raw_data = extract_data(INPUT_PATH, file_config)
    transformed_data = transform_data(raw_data)
    load_data(transformed_data, DB_PATH, TABLEAU_CSV_PATH)
    
    logging.info("====== NHS Data ETL Pipeline: FINISHED ======")
    
    # Verify the output files
    print("\n--- Verification ---")
    if os.path.exists(DB_PATH):
        print(f"✅ SQLite Database created at: {DB_PATH}")
    if os.path.exists(TABLEAU_CSV_PATH):
        print(f"✅ CSV for Tableau created at: {TABLEAU_CSV_PATH}")