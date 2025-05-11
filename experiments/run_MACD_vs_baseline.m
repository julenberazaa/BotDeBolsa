%% MACD vs Buy-and-Hold Baseline Experiment
% This script compares a MACD trading strategy with a simple Buy-and-Hold
% baseline strategy to evaluate performance differences.

% Get base path for proper file references
basePath = fileparts(fileparts(mfilename('fullpath')));

% Add paths if not already on path
if ~exist('macd_strategy', 'file')
    addpath(fullfile(basePath, 'src', 'strategies'));
    addpath(fullfile(basePath, 'src', 'agents'));
    addpath(fullfile(basePath, 'src', 'envs'));
    addpath(fullfile(basePath, 'src', 'data', 'reader'));
    addpath(fullfile(basePath, 'src', 'utils'));
    
    % Also add legacy paths for compatibility
    addpath(fullfile(basePath, 'proyecto'));
    addpath(fullfile(basePath, 'bd_stock price'));
end

%% Load price data
disp('Loading price data...');
try
    data = load(fullfile(basePath, 'data', 'processed', 'ReaderBeginingDLR.mat'));
    prices = data.RetornosMedios;
    
    % Check if dates field exists with different possible names
    if isfield(data, 'Fechas')
        dates = data.Fechas;
    elseif isfield(data, 'fechas')
        dates = data.fechas;
    elseif isfield(data, 'Dates')
        dates = data.Dates;
    elseif isfield(data, 'dates')
        dates = data.dates;
    else
        % Create dummy dates if none are found
        dates = datenum(datetime('today') - caldays(size(prices, 2):-1:1));
        warning('No dates field found in data. Using generated dates.');
    end
catch
    try
        % If specific file not found, try to load from a more general source
        data = load('ReaderBeginingDLR.mat');
        prices = data.RetornosMedios;
        
        % Check if dates field exists with different possible names
        if isfield(data, 'Fechas')
            dates = data.Fechas;
        elseif isfield(data, 'fechas')
            dates = data.fechas;
        elseif isfield(data, 'Dates')
            dates = data.Dates;
        elseif isfield(data, 'dates')
            dates = data.dates;
        else
            % Create dummy dates if none are found
            dates = datenum(datetime('today') - caldays(size(prices, 2):-1:1));
            warning('No dates field found in data. Using generated dates.');
        end
        
        % Save to the correct location
        processedDir = fullfile(basePath, 'data', 'processed');
        if ~exist(processedDir, 'dir')
            mkdir(processedDir);
        end
        copyfile('ReaderBeginingDLR.mat', fullfile(processedDir, 'ReaderBeginingDLR.mat'));
    catch err
        error('Could not load price data. Error: %s', err.message);
    end
end

%% Configure parameters
fastPeriod = 12;
slowPeriod = 26;
signalPeriod = 9;
simulationLength = 252; % Simulate for one year

%% Create agents
disp('Creating agents...');

% MACD agent
macdAgent = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod);

% Buy and Hold agent (always returns +1)
bhAgent = struct('getSignal', @(t) 1); % Simple agent that always returns buy signal

%% Run MACD strategy simulation
disp('Running MACD strategy simulation...');

% Create environment with MACD agent
try
    % Try newer signature first (with external agent)
    envMACD = PortfolioEnv(macdAgent);
catch ME
    warning('Using fallback environment initialization: %s', ME.message);
    % Fallback to old signature (no arguments)
    envMACD = PortfolioEnv();
    % Then set the agent if the method exists
    if ismethod(envMACD, 'setExternalAgent')
        envMACD.setExternalAgent(macdAgent);
    else
        warning('Cannot set external agent. MACD strategy may not work correctly.');
    end
end

envMACD.logVerbose = true;

initialPortfolioValue = 100;
histMACD = zeros(simulationLength, 1);
histMACD(1) = initialPortfolioValue;

% Create an initial action (equal weights)
initialAction = ones(envMACD.NumAssets, 1) / envMACD.NumAssets;

for t = 1:simulationLength-1
    [~, r, isDone] = step(envMACD, initialAction);
    histMACD(t+1) = histMACD(t) * (1 + r);
    if isDone
        break;
    end
end

retMACD = (histMACD(end) - histMACD(1)) / histMACD(1);
macdDailyReturns = diff(histMACD) ./ histMACD(1:end-1);
macdSharpe = mean(macdDailyReturns) / std(macdDailyReturns) * sqrt(252);
macdMaxDrawdown = max(cummax(histMACD) - histMACD) / max(histMACD);

fprintf('MACD Strategy - Final Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    retMACD*100, macdSharpe, macdMaxDrawdown*100);

%% Run Buy and Hold simulation
disp('Running Buy and Hold simulation...');

% Create environment with Buy and Hold agent
try
    % Try newer signature first (with external agent)
    envBH = PortfolioEnv(bhAgent);
catch ME
    warning('Using fallback environment initialization: %s', ME.message);
    % Fallback to old signature (no arguments)
    envBH = PortfolioEnv();
    % Then set the agent if the method exists
    if ismethod(envBH, 'setExternalAgent')
        envBH.setExternalAgent(bhAgent);
    else
        warning('Cannot set external agent. Buy and Hold strategy may not work correctly.');
    end
end

envBH.logVerbose = true;

histBH = zeros(simulationLength, 1);
histBH(1) = initialPortfolioValue;

% Create an initial action (equal weights)
initialAction = ones(envBH.NumAssets, 1) / envBH.NumAssets;

for t = 1:simulationLength-1
    [~, r, isDone] = step(envBH, initialAction);
    histBH(t+1) = histBH(t) * (1 + r);
    if isDone
        break;
    end
end

retBH = (histBH(end) - histBH(1)) / histBH(1);
bhDailyReturns = diff(histBH) ./ histBH(1:end-1);
bhSharpe = mean(bhDailyReturns) / std(bhDailyReturns) * sqrt(252);
bhMaxDrawdown = max(cummax(histBH) - histBH) / max(histBH);

fprintf('Buy & Hold Strategy - Final Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    retBH*100, bhSharpe, bhMaxDrawdown*100);

%% Save results
disp('Saving results...');

% Create struct to store results
results = struct();

% MACD results
results.MACD.equity = histMACD;
results.MACD.returns = macdDailyReturns;
results.MACD.totalReturn = retMACD;
results.MACD.sharpe = macdSharpe;
results.MACD.maxDrawdown = macdMaxDrawdown;
results.MACD.volatility = std(macdDailyReturns) * sqrt(252);

% Buy and Hold results
results.BuyHold.equity = histBH;
results.BuyHold.returns = bhDailyReturns;
results.BuyHold.totalReturn = retBH;
results.BuyHold.sharpe = bhSharpe;
results.BuyHold.maxDrawdown = bhMaxDrawdown;
results.BuyHold.volatility = std(bhDailyReturns) * sqrt(252);

% Save to file
logsDir = fullfile(basePath, 'results', 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end
save(fullfile(logsDir, 'MACD_vs_BH.mat'), 'results', 'retMACD', 'histMACD', 'retBH', 'histBH');

%% Plot results
disp('Plotting results...');
figuresDir = fullfile(basePath, 'results', 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end
compare_plots(results, figuresDir);

disp('Experiment completed!'); 
 
 
 
 