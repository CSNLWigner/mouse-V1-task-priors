function saveDecodedDataToHDF5(dataTable, columnName, h5filename)
    % Function to decode a specified column from a table and save it to an HDF5 file
    %
    % Parameters:
    %   dataTable: The input MATLAB table containing the encoded byte strings
    %   columnName: The name of the column to decode (string)
    %   h5filename: The name of the HDF5 file to write the decoded data (string)
    
    % Check if the specified column exists in the table
    if ~ismember(columnName, dataTable.Properties.VariableNames)
        error('The column "%s" does not exist in the table.', columnName);
    end
    
    % Extract the specified column from the table
    encodedColumn = dataTable.(columnName);
    
    % Determine the number of rows in the table
    numRows = height(dataTable);
    
    % Find the first non-NULL and non-empty entry to determine the decoded size
    decodedSize = [];
    for i = 1:numRows
        if ~isempty(encodedColumn{i}) && ~isequal(encodedColumn{i}, 'NULL')
            % Decode the first valid entry to determine the size of the decoded data
            sampleDecoded = decodeByteString(encodedColumn{i});
            decodedSize = size(sampleDecoded);
            break;
        end
    end
    
    % Check if a valid decoded size was found
    if isempty(decodedSize)
        error('No valid data found in column "%s" to determine decoded size.', columnName);
    end
    
    % Initialize a matrix to store the decoded data
    % Set default value as NaN for missing or NULL entries
    decodedMatrix = nan(numRows, decodedSize(1));  % Initialize with NaNs
    
    % Decode each entry in the column
    for i = 1:numRows
        if isempty(encodedColumn{i}) || isequal(encodedColumn{i}, 'NULL')
            % Skip NULL or empty entries, leave as NaN
            continue;
        else
            % Decode the byte string and store in the matrix
            decodedMatrix(i, :) = decodeByteString(encodedColumn{i});
        end
    end
    
    % Create the dataset name based on the column name (remove spaces)
    datasetName = ['/' strrep(columnName, ' ', '_')];
    
    % Write the decoded matrix to the HDF5 file
    h5create(h5filename, datasetName, size(decodedMatrix));
    h5write(h5filename, datasetName, decodedMatrix);
    
    % Display confirmation
    fprintf('Decoded data from column "%s" successfully saved to dataset "%s" in file "%s".\n', columnName, datasetName, h5filename);
    
    % Optionally display the contents of the HDF5 file
    h5disp(h5filename);
end
