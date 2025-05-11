function addPathsIfNeeded()
% ADDPATHSIFNEEDED - Add necessary project paths to MATLAB path
%
% This function adds all the necessary directories to the MATLAB path
% to ensure all project components can find each other.

% Get the root directory (assuming this function is in src/utils)
currentFile = mfilename('fullpath');
[utilsDir, ~, ~] = fileparts(currentFile);
[srcDir, ~, ~] = fileparts(utilsDir);
[rootDir, ~, ~] = fileparts(srcDir);

% List of directories to add
dirsToAdd = {
    fullfile(rootDir, 'src', 'utils'),
    fullfile(rootDir, 'src', 'strategies'),
    fullfile(rootDir, 'src', 'agents'),
    fullfile(rootDir, 'src', 'data'),
    fullfile(rootDir, 'proyecto'),
    fullfile(rootDir, 'bd_stock price'),
    fullfile(rootDir, 'bd_stock price', 'stocks'),
    fullfile(rootDir, 'bd_stock price', 'etfs'),
    fullfile(rootDir, 'results'),
    fullfile(rootDir, 'results', 'figures'),
    fullfile(rootDir, 'results', 'validation'),
    fullfile(rootDir, 'results', 'simulation')
};

% Add each directory to path if it exists
for i = 1:length(dirsToAdd)
    dirPath = dirsToAdd{i};
    if exist(dirPath, 'dir')
        if ~isempty(strfind(path, dirPath))
            % Directory already in path, no need to add it
            continue;
        end
        addpath(dirPath);
        fprintf('Added to path: %s\n', dirPath);
    else
        % Try to create directory if it doesn't exist
        if contains(dirPath, 'results')
            try
                mkdir(dirPath);
                addpath(dirPath);
                fprintf('Created and added to path: %s\n', dirPath);
            catch
                warning('Could not create directory: %s', dirPath);
            end
        else
            warning('Directory does not exist: %s', dirPath);
        end
    end
end

% Ensure all paths are saved for future MATLAB sessions
try
    savepath;
    fprintf('Paths saved successfully.\n');
catch
    warning('Could not save paths permanently. You may need to run this function again in future sessions.');
end

end 