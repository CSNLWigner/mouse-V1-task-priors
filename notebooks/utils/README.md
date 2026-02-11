# How was data compiled?
We received three CSVs from PO's lab:
- `DATASET1_CellTable.csv`
- `DATASET1_TrialTable.csv`
- `DATASET1_CellTrialTable.csv`

## `DATASET1_CellTable.csv`
A database of all neurons containing:
- Cell ID
- Experiment
- Preferred orientation

## `DATASET1_TrialTable.csv`
A database of all experimental trials containing:
- Trial ID
- Experiment
- Day
- Block (tuning or task)
- Licking response (binary)
- Trial outcome (Hit, Miss, CR, FA)

## `DATASET1_CellTrialTable.csv`
A database of all APrE traces (spike traces) for each neuron in every trial. This is a massive CSV with APrE traces encoded via MATLAB. To extract traces
1. we converted the CSV to an SQL database
2. created an index on experiment_ids
3. used MATLAB to decode traces
4. saved traces in H5 files grouped by experimental session

### Reruning the pipeline
To regenerate H5 files from the original CSV:
1. Run `sql_database_creation.ipynb`
2. Run `create_experiment_CSVs.ipynb`
3. Run `automaticHDF5Conversion.m` (a matlab srcipt in the `matlab/` folder)