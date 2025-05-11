% STARTUP.M - Initialization script for BotDeBolsa
% This script adds all required directories to the MATLAB path
% and sets up the environment for running simulations and experiments.

fprintf('Initializing BotDeBolsa environment...\n');

% Get the root path of the project
rootPath = fileparts(mfilename('fullpath'));

% Add source directories
addpath(fullfile(rootPath, 'src', 'strategies'));
addpath(fullfile(rootPath, 'src', 'agents'));
addpath(fullfile(rootPath, 'src', 'envs'));
addpath(fullfile(rootPath, 'src', 'data', 'reader'));
addpath(fullfile(rootPath, 'src', 'utils'));

% Add experiment and test directories
addpath(fullfile(rootPath, 'experiments'));
addpath(fullfile(rootPath, 'tests'));

% Add data directories
addpath(fullfile(rootPath, 'data', 'processed'));
addpath(fullfile(rootPath, 'data', 'raw'));

% Also add the original locations for backward compatibility
addpath(fullfile(rootPath, 'proyecto'));
addpath(fullfile(rootPath, 'bd_stock price'));

fprintf('BotDeBolsa paths initialized successfully!\n');

% Display information about the available scripts
fprintf('\nAvailable experiment scripts:\n');
fprintf('  run_RSI_vs_baseline - Compare RSI strategy with Buy-and-Hold\n');
fprintf('  run_comparison_all  - Compare all available strategies\n\n');

fprintf('To run an experiment, use:\n');
fprintf('  cd experiments\n');
fprintf('  run_RSI_vs_baseline\n\n');

fprintf('To test if all modules load correctly:\n');
fprintf('  cd tests\n');
fprintf('  test_load_all\n\n'); 
 
 
 
 