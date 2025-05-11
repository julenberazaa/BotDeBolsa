% Test script for the plotting function

% Add the utils directory to the path
addpath('src/utils');

% Create dummy results
results = struct();
results.Strategy1.equity = 100 + cumsum(randn(50,1));
results.Strategy1.returns = diff(results.Strategy1.equity) ./ results.Strategy1.equity(1:end-1);
results.Strategy1.sharpe = 1.2;
results.Strategy1.maxDrawdown = 0.05;
results.Strategy1.volatility = 0.1;
results.Strategy1.totalReturn = 0.15;

results.Strategy2.equity = 100 + cumsum(randn(50,1));
results.Strategy2.returns = diff(results.Strategy2.equity) ./ results.Strategy2.equity(1:end-1);
results.Strategy2.sharpe = 0.9;
results.Strategy2.maxDrawdown = 0.08;
results.Strategy2.volatility = 0.15;
results.Strategy2.totalReturn = 0.12;

% Create output directory if it doesn't exist
if ~exist('results/figures', 'dir')
    mkdir('results/figures');
end

% Call the plotting function
try
    compare_plots(results, 'results/figures');
    disp('Plotting function executed successfully!');
catch ME
    disp(['Error in plotting function: ' ME.message]);
    disp('Stack trace:');
    disp(getReport(ME));
end 
 
 
 
 