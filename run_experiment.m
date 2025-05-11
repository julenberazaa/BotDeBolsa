% RUN_EXPERIMENT.M - Script to run experiments with proper path management
% This script ensures all paths are properly set before running experiments

% Run startup script to add all paths
startup

% Change to experiments directory
cd experiments

% Display menu
fprintf('\nSelect an experiment to run:\n');
fprintf('  1. RSI vs Buy-and-Hold\n');
fprintf('  2. All strategies comparison\n');
fprintf('  0. Cancel\n\n');

choice = input('Enter your choice (0-2): ');

switch choice
    case 1
        fprintf('\nRunning RSI vs Buy-and-Hold experiment...\n\n');
        run_RSI_vs_baseline
    case 2
        fprintf('\nRunning all strategies comparison...\n\n');
        run_comparison_all
    case 0
        fprintf('\nExperiment canceled.\n');
    otherwise
        fprintf('\nInvalid choice.\n');
end

% Return to the main directory
cd ..

fprintf('\nExperiment completed. Results saved in results/logs and results/figures directories.\n'); 
 
 
 
 