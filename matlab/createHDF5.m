function createHDF5(cellTableCSVName, hdf5FileName)
    % Step 1: Read the CSV file into a table
    data = readtable(cellTableCSVName);
    
    % Step 2: Extract the unique cells and trials
    uniqueCells = unique(data.Cell);       % Unique cells (assumed to be labeled 1-737)
    uniqueTrials = unique(data.Trial);     % Unique trials (536 trials)
    
    % Step 3: Preallocate a structure to hold the matrices for each trial
    trialMatrices = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    
    % Step 4: Iterate over each unique trial to create trial-specific matrices
    for trialNumber = 1:max(uniqueTrials)    
        % Filter the rows corresponding to the current trial
        trialData = data(data.Trial == trialNumber, :);
        
        % Determine the length of the spike trace (assumes all rows in a trial have the same length)
        sampleTrace = decodeByteString(trialData.TraceSpikes{1});  % Decode one trace to get length
        n = length(sampleTrace);  % Number of time steps
    
        % Preallocate matrix (rows are cells, columns are time steps)
        trialMatrix = NaN(max(uniqueCells), n);  % Use NaN or 0 to fill missing cells if any
    
        % Step 5: Populate the matrix for the current trial
        for rowIdx = 1:height(trialData)
            cellNumber = trialData.Cell(rowIdx);  % Get the cell number (1-737)
            spikeTrace = decodeByteString(trialData.TraceSpikes{rowIdx});  % Decode the spike trace
            trialMatrix(cellNumber, :) = spikeTrace;  % Assign the trace to the appropriate row
        end
    
        % Store the matrix for the current trial in the map
        trialMatrices(trialNumber) = trialMatrix;
    end
    
    % Step 6: Create and write to the HDF5 file    
    % Remove the file if it exists to avoid appending to old data
    if exist(hdf5FileName, 'file')
        delete(hdf5FileName);
    end
    
    % Step 7: Write each trial matrix to the HDF5 file
    for trialIdx = 1:length(uniqueTrials)
        trialNumber = uniqueTrials(trialIdx);  % Get the trial number
        trialMatrix = trialMatrices(trialNumber);  % Retrieve the corresponding matrix
    
        % Define the dataset name using the trial number
        datasetName = sprintf('/trial_%d', trialNumber);
    
        % Create and write the dataset to the HDF5 file
        h5create(hdf5FileName, datasetName, size(trialMatrix));
        h5write(hdf5FileName, datasetName, trialMatrix);
    end
    
    disp('Data has been successfully stored in the HDF5 file.');
end
