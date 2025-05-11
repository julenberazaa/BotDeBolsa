%% RSI Parameter Optimization
% This script performs a grid search to find optimal RSI strategy parameters
% (window size, overbought and oversold thresholds) that maximize Sharpe ratio.

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
catch
    try
        % If specific file not found, try to load from a more general source
        data = load('ReaderBeginingDLR.mat');
        prices = data.RetornosMedios;
    catch err
        error('Could not load price data. Error: %s', err.message);
    end
end

%% Configure optimization grid
% Define parameter ranges for grid search
windowSizes = [5, 7, 9, 11, 14, 21, 28];
overboughtLevels = [65, 70, 75, 80];
oversoldLevels = [20, 25, 30, 35];

% Initialize results container
numCombinations = length(windowSizes) * length(overboughtLevels) * length(oversoldLevels);
results = struct();
results.params = zeros(numCombinations, 3); % [window, overbought, oversold]
results.sharpe = zeros(numCombinations, 1);
results.returns = zeros(numCombinations, 1);
results.maxDrawdown = zeros(numCombinations, 1);

% Configuration
simulationLength = 252; % Simulate for one year
initialPortfolioValue = 100;

%% Run grid search
disp(['Starting grid search with ', num2str(numCombinations), ' parameter combinations...']);

combinationIdx = 1;
progressInterval = max(1, floor(numCombinations / 20)); % Show progress at ~5% intervals

for w = 1:length(windowSizes)
    window = windowSizes(w);
    
    for o = 1:length(overboughtLevels)
        overbought = overboughtLevels(o);
        
        for s = 1:length(oversoldLevels)
            oversold = oversoldLevels(s);
            
            % Skip invalid combinations (oversold must be less than overbought)
            if oversold >= overbought
                continue;
            end
            
            % Create RSI agent with current parameters
            rsiAgent = rsi_agent(prices, window, overbought, oversold);
            
            % Initialize environment
            try
                % Try newer signature first (with external agent)
                env = PortfolioEnv(rsiAgent);
            catch
                % Fallback to old signature (no arguments)
                env = PortfolioEnv();
                % Then set the agent if the method exists
                if ismethod(env, 'setExternalAgent')
                    env.setExternalAgent(rsiAgent);
                else
                    warning('Cannot set external agent. Skipping this combination.');
                    continue;
                end
            end
            
            % Run simulation
            hist = zeros(simulationLength, 1);
            hist(1) = initialPortfolioValue;
            
            for t = 1:simulationLength-1
                [~, r, isDone] = step(env);
                hist(t+1) = hist(t) * (1 + r);
                if isDone
                    break;
                end
            end
            
            % Calculate performance metrics
            totalReturn = (hist(end) - hist(1)) / hist(1);
            dailyReturns = diff(hist) ./ hist(1:end-1);
            sharpeRatio = mean(dailyReturns) / std(dailyReturns) * sqrt(252);
            maxDrawdown = max(cummax(hist) - hist) / max(hist);
            
            % Store results
            results.params(combinationIdx, :) = [window, overbought, oversold];
            results.sharpe(combinationIdx) = sharpeRatio;
            results.returns(combinationIdx) = totalReturn;
            results.maxDrawdown(combinationIdx) = maxDrawdown;
            
            % Show progress periodically
            if mod(combinationIdx, progressInterval) == 0 || combinationIdx == 1
                fprintf('Progress: %d/%d combinations (%.1f%%)\n', ...
                    combinationIdx, numCombinations, combinationIdx/numCombinations*100);
            end
            
            combinationIdx = combinationIdx + 1;
        end
    end
end

% Adjust results in case some combinations were skipped
actualCombinations = combinationIdx - 1;
results.params = results.params(1:actualCombinations, :);
results.sharpe = results.sharpe(1:actualCombinations);
results.returns = results.returns(1:actualCombinations);
results.maxDrawdown = results.maxDrawdown(1:actualCombinations);

%% Find best parameters
[maxSharpe, maxIdx] = max(results.sharpe);
bestWindow = results.params(maxIdx, 1);
bestOverbought = results.params(maxIdx, 2);
bestOversold = results.params(maxIdx, 3);
bestReturn = results.returns(maxIdx);
bestDrawdown = results.maxDrawdown(maxIdx);

fprintf('\n--- Best RSI Parameters ---\n');
fprintf('Window Size: %d\n', bestWindow);
fprintf('Overbought Threshold: %.1f\n', bestOverbought);
fprintf('Oversold Threshold: %.1f\n', bestOversold);
fprintf('Sharpe Ratio: %.4f\n', maxSharpe);
fprintf('Total Return: %.2f%%\n', bestReturn * 100);
fprintf('Max Drawdown: %.2f%%\n', bestDrawdown * 100);

%% Save results
disp('Saving optimization results...');

% Create results directory if it doesn't exist
logsDir = fullfile(basePath, 'results', 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

save(fullfile(logsDir, 'RSI_optimization_results.mat'), 'results', 'bestWindow', 'bestOverbought', 'bestOversold');

%% Visualize results
disp('Creating visualizations...');

% Create figures directory if it doesn't exist
figuresDir = fullfile(basePath, 'results', 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end

% Find unique window sizes that were tested
uniqueWindows = unique(results.params(:, 1));

% For each window size, plot a heatmap of Sharpe ratio by oversold/overbought thresholds
for i = 1:length(uniqueWindows)
    window = uniqueWindows(i);
    
    % Get data for this window
    idx = results.params(:, 1) == window;
    overboughtVals = results.params(idx, 2);
    oversoldVals = results.params(idx, 3);
    sharpeVals = results.sharpe(idx);
    
    % Create a figure
    fig = figure('Position', [100, 100, 800, 600]);
    
    % Create a scatter plot with color indicating Sharpe ratio
    scatter(overboughtVals, oversoldVals, 100, sharpeVals, 'filled');
    colorbar;
    xlabel('Overbought Threshold');
    ylabel('Oversold Threshold');
    title(sprintf('RSI Sharpe Ratio (Window Size = %d)', window));
    
    % Add text annotations with Sharpe values
    for j = 1:length(sharpeVals)
        text(overboughtVals(j), oversoldVals(j), sprintf('%.2f', sharpeVals(j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Highlight best point for this window
    [~, bestIdx] = max(sharpeVals);
    hold on;
    plot(overboughtVals(bestIdx), oversoldVals(bestIdx), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
    
    % Save figure
    saveas(fig, fullfile(figuresDir, sprintf('RSI_optimization_window_%d.png', window)));
    saveas(fig, fullfile(figuresDir, sprintf('RSI_optimization_window_%d.fig', window)));
    close(fig);
end

% Summary figure showing best Sharpe for each window size
fig = figure('Position', [100, 100, 800, 400]);
windowSharpes = zeros(length(uniqueWindows), 1);

for i = 1:length(uniqueWindows)
    window = uniqueWindows(i);
    idx = results.params(:, 1) == window;
    windowSharpes(i) = max(results.sharpe(idx));
end

bar(uniqueWindows, windowSharpes);
xlabel('RSI Window Size');
ylabel('Best Sharpe Ratio');
title('Best Sharpe Ratio by RSI Window Size');
grid on;

% Add text labels above bars
for i = 1:length(uniqueWindows)
    text(uniqueWindows(i), windowSharpes(i), sprintf('%.2f', windowSharpes(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Highlight overall best window
[~, bestWindowIdx] = max(windowSharpes);
hold on;
highlight = bar(uniqueWindows(bestWindowIdx), windowSharpes(bestWindowIdx));
highlight.FaceColor = 'r';

% Save summary figure
saveas(fig, fullfile(figuresDir, 'RSI_optimization_summary.png'));
saveas(fig, fullfile(figuresDir, 'RSI_optimization_summary.fig'));

disp('Optimization completed successfully!'); 
 
 
 
 