% Define the base path to the dataset directory, uncomment accordingly for each dataset you want to process
% baseDir = 'data/DATASET1_CellTrialTable';
% experiment_ids_file = 'data/DATASET1_CellTrialTable/experiment_ids.txt';
baseDir = 'data/Naive_CellTrialTable';
experiment_ids_file = 'data/Naive_CellTrialTable/experiment_ids.txt';

% Load experiment IDs from a text file (if available)
if exist(experiment_ids_file, 'file')
    experiment_ids = importdata(experiment_ids_file);
else
    error('Experiment IDs file not found. Please provide experiment_ids.txt.');
end

% Iterate through each experiment ID and process the corresponding CSV file
for i = 1:length(experiment_ids)
    experiment_id = experiment_ids{i};  % Get the current experiment ID
    
    % Construct paths for the CSV and HDF5 files
    mouse_name = experiment_id(1:3);  % Get the first 3 characters as mouse name
    mouse_path = fullfile(baseDir, mouse_name);  % Path for the current mouse directory
    
    % Check if the directory exists
    if ~exist(mouse_path, 'dir')
        fprintf('Skipping %s: Directory does not exist.\n', experiment_id);
        continue;
    end
    
    % Define the paths for the CSV and HDF5 files
    csvFile = fullfile(mouse_path, [experiment_id, '.csv']);
    hdf5File = fullfile(mouse_path, [experiment_id, '.h5']);
    
    % Skip processing if the HDF5 file already exists
    if exist(hdf5File, 'file')
        fprintf('Skipping %s: HDF5 file already exists.\n', experiment_id);
        continue;
    end
    
    % Check if the CSV file exists
    if ~exist(csvFile, 'file')
        fprintf('Skipping %s: CSV file does not exist.\n', experiment_id);
        continue;
    end
    
    % Create the HDF5 file from the CSV file
    try
        createHDF5(csvFile, hdf5File);  % Call your function to create HDF5
        fprintf('Successfully created HDF5 for %s\n', experiment_id);
        
    catch ME
        % Catch and display any errors that occurred during HDF5 creation
        fprintf('Failed to create HDF5 for %s: %s\n', experiment_id, ME.message);
    end
end

disp('Processing completed.');