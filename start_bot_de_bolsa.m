%% BotDeBolsa - Main Entry Script
% This script initializes the trading bot environment and provides
% a menu for running different strategy simulations.

clc;
clear;
close all;

fprintf('Initializing BotDeBolsa environment...\n');

% Add core paths
addpath('src/utils');
addpath('src/strategies');
addpath('src/agents');
addpath('src/data');
addpath('proyecto');

% Create essential directories if they don't exist
dirs = {'results', 'results/logs', 'results/figures', 'results/models'};
for i = 1:length(dirs)
    if ~exist(dirs{i}, 'dir')
        mkdir(dirs{i});
        fprintf('Created directory: %s\n', dirs{i});
    end
end

% Try to initialize all paths
try
    addPathsIfNeeded();
    fprintf('BotDeBolsa paths initialized successfully!\n');
catch
    warning('Could not run addPathsIfNeeded. Some paths may be missing.');
end

% Display available experiments
fprintf('\nAvailable experiment scripts:\n');
fprintf('  run_enhanced_hybrid_strategy - Run enhanced MACD-IA hybrid strategy\n');
fprintf('  simulate_spo_vs_ia_ret      - Compare SPO with IA\n');

% Provide instructions
fprintf('\nTo run a strategy simulation, use one of these commands:\n');
fprintf('  run(''run_enhanced_hybrid_strategy'')\n');
fprintf('  run(''proyecto/simulate_spo_vs_ia_ret'')\n');

% Help section
fprintf('\nBotDeBolsa Trading System\n');
fprintf('------------------------\n');
fprintf('This system provides various trading strategies based on:\n');
fprintf('- Enhanced MACD with market regime detection\n');
fprintf('- Neural network AI complementary strategies\n');
fprintf('- Portfolio optimization techniques\n');

% Check if we're running in batch mode
isBatchMode = ~usejava('desktop') || ~feature('ShowFigureWindows');

if isBatchMode
    % In batch mode, automatically run the enhanced strategy
    fprintf('\nRunning in batch mode - automatically starting enhanced hybrid strategy...\n\n');
    run('run_enhanced_hybrid_strategy');
else
    % Interactive mode - ask user
    choice = input('\nDo you want to run the enhanced hybrid strategy now? (y/n): ', 's');
    if strcmpi(choice, 'y')
        fprintf('\nRunning enhanced hybrid MACD-IA strategy...\n\n');
        run('run_enhanced_hybrid_strategy');
    else
        fprintf('\nYou can run any strategy later using the commands above.\n');
    end
end 
 
 
 
 