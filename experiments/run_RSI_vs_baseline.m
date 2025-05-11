%% RSI vs Buy-and-Hold Baseline Experiment
% This script compares an RSI trading strategy with a simple Buy-and-Hold
% baseline strategy to evaluate performance differences.

% Get base path for proper file references
basePath = fileparts(fileparts(mfilename('fullpath')));

% Add paths if not already on path
if ~exist('rsi_strategy', 'file')
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
rsiWindow = 14;
overbought = 70;
oversold = 30;
simulationLength = 252; % Simulate for one year

%% Create agents
disp('Creating agents...');

% RSI agent
rsiAgent = rsi_agent(prices, rsiWindow, overbought, oversold);

% Buy and Hold agent (always returns +1)
bhAgent = struct('getSignal', @(t) 1); % Simple agent that always returns buy signal

%% Run RSI strategy simulation
disp('Running RSI strategy simulation...');

% Create environment with RSI agent
try
    % Try newer signature first (with external agent)
    envRSI = PortfolioEnv(rsiAgent);
catch ME
    warning('Using fallback environment initialization: %s', ME.message);
    % Fallback to old signature (no arguments)
    envRSI = PortfolioEnv();
    % Then set the agent if the method exists
    if ismethod(envRSI, 'setExternalAgent')
        envRSI.setExternalAgent(rsiAgent);
    else
        warning('Cannot set external agent. RSI strategy may not work correctly.');
    end
end

envRSI.logVerbose = true;

initialPortfolioValue = 100;
histRSI = zeros(simulationLength, 1);
histRSI(1) = initialPortfolioValue;

for t = 1:simulationLength-1
    [~, r, isDone] = step(envRSI);
    histRSI(t+1) = histRSI(t) * (1 + r);
    if isDone
        break;
    end
end

retRSI = (histRSI(end) - histRSI(1)) / histRSI(1);
rsiDailyReturns = diff(histRSI) ./ histRSI(1:end-1);
rsiSharpe = mean(rsiDailyReturns) / std(rsiDailyReturns) * sqrt(252);
rsiMaxDrawdown = max(cummax(histRSI) - histRSI) / max(histRSI);

fprintf('RSI Strategy - Final Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    retRSI*100, rsiSharpe, rsiMaxDrawdown*100);

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

for t = 1:simulationLength-1
    [~, r, isDone] = step(envBH);
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

% RSI results
results.RSI.equity = histRSI;
results.RSI.returns = rsiDailyReturns;
results.RSI.totalReturn = retRSI;
results.RSI.sharpe = rsiSharpe;
results.RSI.maxDrawdown = rsiMaxDrawdown;
results.RSI.volatility = std(rsiDailyReturns) * sqrt(252);

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
save(fullfile(logsDir, 'RSI_vs_BH.mat'), 'results', 'retRSI', 'histRSI', 'retBH', 'histBH');

%% Plot results
disp('Plotting results...');
figuresDir = fullfile(basePath, 'results', 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end
compare_plots(results, figuresDir);

disp('Experiment completed!'); 
 
 
 