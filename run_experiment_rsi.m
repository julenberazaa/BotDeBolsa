% RUN_EXPERIMENT_RSI.M - Script to run RSI experiment with proper path management
% This script ensures all paths are properly set before running the RSI experiment

% Run startup script to add all paths
startup

% Change to experiments directory
cd experiments

% Run the RSI vs Buy-and-Hold experiment
fprintf('\nRunning RSI vs Buy-and-Hold experiment...\n\n');
run_RSI_vs_baseline

% Return to the main directory
cd ..

fprintf('\nExperiment completed. Results saved in results/logs and results/figures directories.\n'); 
 
 
 
 