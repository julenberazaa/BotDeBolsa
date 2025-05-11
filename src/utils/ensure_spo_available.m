function ensure_spo_available()
% ENSURE_SPO_AVAILABLE - Ensures the SPO function is available in the MATLAB path
%
% This function checks if the obtenerSPO function is available in the MATLAB path
% and adds the necessary directories if it's not found. This is particularly
% important for the regularized model which relies on this function.
%
% Usage:
%   ensure_spo_available();

% Check if obtenerSPO is on the path
if exist('obtenerSPO', 'file') ~= 2
    warning('obtenerSPO function not found in the MATLAB path. Attempting to locate it...');
    
    % Try to find the file
    currentDir = pwd;
    
    % Check in src/utils first (most likely location)
    utilsDir = fullfile(currentDir, 'src', 'utils');
    if exist(fullfile(utilsDir, 'obtenerSPO.m'), 'file') == 2
        addpath(utilsDir);
        fprintf('Added src/utils directory to path. obtenerSPO is now available.\n');
        return;
    end
    
    % Check in proyecto directory (alternative location)
    proyectoDir = fullfile(currentDir, 'proyecto');
    if exist(fullfile(proyectoDir, 'obtenerSPO.m'), 'file') == 2
        addpath(proyectoDir);
        fprintf('Added proyecto directory to path. obtenerSPO is now available.\n');
        return;
    end
    
    % Search for the file in subdirectories
    fprintf('Searching for obtenerSPO.m in subdirectories...\n');
    
    % Get all directories under the current directory
    allDirs = genpath(currentDir);
    dirList = strsplit(allDirs, pathsep);
    
    % Look for obtenerSPO.m in each directory
    for i = 1:length(dirList)
        if exist(fullfile(dirList{i}, 'obtenerSPO.m'), 'file') == 2
            addpath(dirList{i});
            fprintf('Found and added directory containing obtenerSPO: %s\n', dirList{i});
            return;
        end
    end
    
    % If we get here, we couldn't find the file
    error(['Could not find obtenerSPO.m in any subdirectory. ', ...
           'Please ensure this file exists and is in the MATLAB path.']);
else
    % Function is already available
    fprintf('obtenerSPO function is available in the MATLAB path.\n');
end

% Validate that the function works
try
    % Create simple test inputs
    testReturns = [0.01; 0.02; 0.005];
    testVariances = [0.0004; 0.0009; 0.0001];
    testAlpha = 0.1;
    
    % Call the function
    weights = obtenerSPO(testReturns, testVariances, testAlpha);
    
    % Check results
    if ~isempty(weights) && all(weights >= 0) && abs(sum(weights) - 1) < 0.01
        fprintf('obtenerSPO function test successful.\n');
    else
        warning('obtenerSPO function produced unexpected output. Results may be incorrect.');
    end
catch ME
    warning('Error testing obtenerSPO function: %s', ME.message);
end

end 