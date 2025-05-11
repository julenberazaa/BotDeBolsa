% RUN_EXPERIMENT_MACD.M - Script to run MACD experiment with proper path management
% This script ensures all paths are properly set before running the MACD experiment

try
    % Run startup script to add all paths
    startup;
    
    % Change to experiments directory
    fprintf('Changing to experiments directory...\n');
    cd experiments;
    
    % Run the MACD vs Buy-and-Hold experiment
    fprintf('\nRunning MACD vs Buy-and-Hold experiment...\n\n');
    run_MACD_vs_baseline;
    
    % Return to the main directory
    cd ..;
    
    fprintf('\nExperiment completed successfully! Results saved in results/logs and results/figures directories.\n');
catch ME
    % Return to main directory if an error occurred
    try
        cd(fileparts(mfilename('fullpath')));
    catch
        % Already in the main directory or cannot determine path
    end
    
    fprintf('\nError running MACD experiment: %s\n', ME.message);
    fprintf('In file: %s, line %d\n', ME.stack(1).file, ME.stack(1).line);
    
    % Additional help if paths might be the issue
    fprintf('\nChecking if strategy file exists...\n');
    if exist('src/strategies/macd_strategy.m', 'file')
        fprintf('MACD strategy file found. Checking if it can be called...\n');
        if exist('macd_strategy', 'file') ~= 2
            fprintf('MACD strategy not properly on path. Adding path directly...\n');
            addpath('src/strategies');
            addpath('src/agents');
            addpath('src/envs');
            addpath('src/utils');
        end
    else
        fprintf('MACD strategy file not found! Please ensure it exists at src/strategies/macd_strategy.m\n');
    end
end 
 
 
 
 