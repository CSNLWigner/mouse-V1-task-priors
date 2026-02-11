import os
import h5py
import sqlite3
import numpy as np
import pandas as pd
from .config import BASEDATAPATH

DATASET1 = {
    'CellTable_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTable.csv'),
    'TrialTable_path': os.path.join(BASEDATAPATH, 'DATASET1_TrialTable.csv'),
    'ExperimentTable_path': os.path.join(BASEDATAPATH, 'DATASET1_ExperimentTable.csv'),
    'CellTrialTable_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTrialTable.db'),
    'CellTrialTable_csv_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTrialTable'),
}

NAIVE = {
    'CellTable_path': os.path.join(BASEDATAPATH, 'Naive_CellTable.csv'),
    'TrialTable_path': os.path.join(BASEDATAPATH, 'Naive_TrialTable.csv'),
    'ExperimentTable_path': os.path.join(BASEDATAPATH, 'Naive_ExperimentTable.csv'),
    'CellTrialTable_path': os.path.join(BASEDATAPATH, 'Naive_CellTrialTable.db'),
    'CellTrialTable_csv_path': os.path.join(BASEDATAPATH, 'Naive_CellTrialTable'),
}

def query_CellTrialTable_dataframe(experiment_id, DATASET=DATASET1,):
    conn = sqlite3.connect(DATASET['CellTrialTable_path'])
    cursor = conn.cursor()
    
    query = "SELECT * FROM CellTrialTable WHERE Experiment = ?"
    cursor.execute(query, (experiment_id,))

    row = cursor.fetchone()
    column_names = [description[0] for description in cursor.description]
    df_celltrialtable = [dict(zip(column_names, row))]
    
    rows = cursor.fetchall()
    for row in rows:
        row_dict = dict(zip(column_names, row))
        df_celltrialtable.append(row_dict)
        
    conn.close()

    return pd.DataFrame(df_celltrialtable)

def check_CellTrialTable_existence(experiment_id, DATASET=DATASET1,):
    mouse_name = experiment_id[:3]
    mouse_path = os.path.join(DATASET['CellTrialTable_csv_path'], mouse_name)
    assert os.path.exists(mouse_path), 'Path for current mouse does not exist'
    
    mouse_csv_path = os.path.join(mouse_path, f'{experiment_id}.csv')
    mouse_hdf_path = os.path.join(mouse_path, f'{experiment_id}.h5')
    if os.path.isfile(mouse_csv_path) or os.path.isfile(mouse_hdf_path):
        raise FileExistsError(f"The file '{experiment_id}' already exists.")

def save_CellTrialTable_df(df_experiment, DATASET=DATASET1,):
    experiment_ids = df_experiment['Experiment'].unique()
    assert len(experiment_ids) == 1, 'Multiple experiment IDs in dataframe'

    experiment_id = experiment_ids[0]
    assert isinstance(experiment_id, str), 'Experiment ID is not string'
    
    mouse_name = experiment_id[:3]
    mouse_path = os.path.join(DATASET['CellTrialTable_csv_path'], mouse_name)
    assert os.path.exists(mouse_path), 'Path for current mouse does not exist'

    mouse_csv_path = os.path.join(mouse_path, f'{experiment_id}.csv')
    mouse_hdf_path = os.path.join(mouse_path, f'{experiment_id}.h5')
    if os.path.isfile(mouse_csv_path) or os.path.isfile(mouse_hdf_path):
        raise FileExistsError(f"The file '{experiment_id}' already exists.")
    else:
        df_experiment.to_csv(mouse_csv_path)

"""
The `retrieve_experiment_stats` and `compute_CellTrialTable_df_stats` functions
are to check parity between the queried dataframe and the expected dataframe.
"""
def retrieve_experiment_stats(experiment_id, loaded_tables,):
    df_celltable, df_trialtable = loaded_tables

    df_celltable = df_celltable[df_celltable['Experiment'] == experiment_id]
    df_trialtable = df_trialtable[df_trialtable['Experiment'] == experiment_id]

    experiment_stats = {
        'cells': len(df_celltable['Cell'].unique()),
        'trials': len(df_trialtable['Trial'].unique()),
        'rows': len(df_celltable['Cell'].unique()) * len(df_trialtable['Trial'].unique()),
    }
    return experiment_stats

def compute_CellTrialTable_df_stats(df_experiment):
    db_experiment_stats = {}
    db_experiment_stats['cells'] = len(df_experiment['Cell'].unique())
    db_experiment_stats['trials'] = len(df_experiment['Trial'].unique())
    db_experiment_stats['rows'] = len(df_experiment)
    return db_experiment_stats