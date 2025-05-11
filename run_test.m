% RUN_TEST.M - Script to run tests with proper path management
% This script ensures all paths are properly set before running tests

% Run startup script to add all paths
startup

% Change to tests directory
cd tests

% Run the test script
test_load_all

% Return to the main directory
cd ..

fprintf('\nTo run experiments with proper paths, use:\n');
fprintf('  run_experiment\n'); 
 
 
 
 