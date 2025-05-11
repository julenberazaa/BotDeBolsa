function [results] = run_validation_test(prices, volumes, macdConfig, filterConfig, regimeConfig, strategyOptions)
% RUN_VALIDATION_TEST - Validate hybrid MACD-IA strategy on a dataset
%
% This function runs a validation test of the hybrid MACD-IA strategy on a dataset,
% comparing performance of regular vs. regularized IA models.
%
% Inputs:
%   prices - Matrix of asset prices [assets x time]
%   volumes - Matrix of volumes [assets x time]
%   macdConfig - Structure with MACD parameters
%   filterConfig - Structure with signal filter settings
%   regimeConfig - Structure with regime detection settings
%   strategyOptions - Structure with strategy options
%
% Output:
%   results - Structure with performance metrics and test results

fprintf('Running validation test...\n');

% === Prepare Data ===
[numAssets, numSteps] = size(prices);

% Calculate returns for analysis
returns = zeros(size(prices));
for t = 2:numSteps
    returns(:, t) = (prices(:, t) - prices(:, t-1)) ./ prices(:, t-1);
end

% === Setup Simulation ===
% Define simulation range (use a portion for in-sample training, rest for out-of-sample)
trainSteps = max(1, floor(numSteps * 0.6)); % 60% for training
testStartStep = min(numSteps, trainSteps + 1);
testSteps = max(1, numSteps - trainSteps);

fprintf('In-sample period: Steps 1-%d\n', trainSteps);
fprintf('Out-of-sample period: Steps %d-%d\n', testStartStep, numSteps);

% Create MACD Agent
fprintf('Creating enhanced MACD agent...\n');
try
    enhancedMacdAgent = enhanced_macd_agent(prices, volumes, ...
        macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod, ...
        filterConfig, true); % Last param: use adaptive parameters
catch ME
    warning('Failed to create enhanced MACD agent: %s\nFalling back to basic MACD signals.', ME.message);
    basicMacdSignals = zeros(numAssets, numSteps);
    for asset = 1:numAssets
        basicMacdSignals(asset, :) = macd_strategy(prices(asset, :), ...
            macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod)';
    end
    enhancedMacdAgent = basicMacdSignals;
end

% === Train Original IA Model ===
fprintf('Training original IA model...\n');
try
    % Train on in-sample data
    trainingPrices = prices(:, 1:trainSteps);
    [iaModel, iaTrainingInfo] = train_ia_complementary(trainingPrices, macdConfig);
catch ME
    warning('Failed to train original IA model: %s\nUsing fallback model.', ME.message);
    iaModel = @(x) simple_weight_model(x, numAssets);
    iaTrainingInfo = struct('message', 'Using fallback model');
end

% === Train Regularized IA Model ===
fprintf('Training regularized IA model...\n');
try
    % Train on in-sample data
    [iaModelReg, iaRegTrainingInfo] = train_ia_complementary_regularized(trainingPrices, macdConfig);
catch ME
    warning('Failed to train regularized IA model: %s\nUsing fallback model.', ME.message);
    iaModelReg = @(x) simple_weight_model(x, numAssets);
    iaRegTrainingInfo = struct('message', 'Using fallback model');
end

% === Initialize Portfolios ===
% Initialize portfolios to track performance
valueMACD = 1;
valueIA = 1;
valueIAReg = 1;
valueHybrid = 1;
valueHybridReg = 1;
valueEqual = 1; % Equal weights baseline

% Arrays for historical values
seriesMACD = zeros(1, testSteps);
seriesIA = zeros(1, testSteps);
seriesIAReg = zeros(1, testSteps);
seriesHybrid = zeros(1, testSteps);
seriesHybridReg = zeros(1, testSteps);
seriesEqual = zeros(1, testSteps);

% Arrays for regime tracking
regimeTypes = zeros(1, testSteps);
macdWeights = zeros(1, testSteps);
iaWeights = zeros(1, testSteps);
iaRegWeights = zeros(1, testSteps);

% Arrays for weight history (for transaction cost calculation)
macdHistory = zeros(numAssets, testSteps);
iaHistory = zeros(numAssets, testSteps);
iaRegHistory = zeros(numAssets, testSteps);
hybridHistory = zeros(numAssets, testSteps);
hybridRegHistory = zeros(numAssets, testSteps);
prevMacdWeights = zeros(numAssets, 1);
prevIaWeights = zeros(numAssets, 1);
prevIaRegWeights = zeros(numAssets, 1);
prevHybridWeights = zeros(numAssets, 1);
prevHybridRegWeights = zeros(numAssets, 1);

% Equal weights for baseline strategy
equalWeights = ones(numAssets, 1) / numAssets;

% Arrays for performance tracking
dailyReturnsMACD = zeros(1, testSteps);
dailyReturnsIA = zeros(1, testSteps);
dailyReturnsIAReg = zeros(1, testSteps);
dailyReturnsHybrid = zeros(1, testSteps);
dailyReturnsHybridReg = zeros(1, testSteps);

% === Run Simulation ===
fprintf('Running simulation for %d steps...\n', testSteps);

for t = 1:testSteps
    currentStep = testStartStep + t - 1;
    
    % Get current returns
    currentReturns = returns(:, currentStep);
    
    % Get MACD-only weights
    try
        if isa(enhancedMacdAgent, 'enhanced_macd_agent')
            % Use the agent's getPortfolioWeights method
            [macdOnlyWeights, ~] = enhancedMacdAgent.getPortfolioWeights(currentStep, strategyOptions.maxPosition);
        else
            % Agent is a signal matrix
            signals = enhancedMacdAgent(:, currentStep);
            macdOnlyWeights = zeros(numAssets, 1);
            buySignals = signals == 1;
            
            if any(buySignals)
                % Allocate weights to assets with buy signals
                macdOnlyWeights(buySignals) = 1 / sum(buySignals);
            end
            
            % Apply position size limits
            for i = 1:numAssets
                if macdOnlyWeights(i) > strategyOptions.maxPosition
                    macdOnlyWeights(i) = strategyOptions.maxPosition;
                end
            end
            
            % Normalize if needed
            if sum(macdOnlyWeights) > 1
                macdOnlyWeights = macdOnlyWeights / sum(macdOnlyWeights);
            end
        end
        
        % If MACD produces no signals, use equal weights instead of all cash
        if sum(macdOnlyWeights) < 0.1
            macdOnlyWeights = equalWeights * 0.9; % 90% allocated, 10% cash
        end
        
        % Ensure at least 5% cash position for realism
        if sum(macdOnlyWeights) > 0.95
            macdOnlyWeights = macdOnlyWeights * 0.95;
        end
        
    catch
        % Fallback to equal weights on error
        macdOnlyWeights = equalWeights;
    end
    
    % Get IA-only weights for original model
    try
        windowStart = max(1, currentStep - strategyOptions.windowSize + 1);
        windowEnd = currentStep;
        window = returns(:, windowStart:windowEnd);
        
        % Normalize input for IA model
        inputVector = window(:);
        inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
        
        % Get IA predictions
        iaOnlyWeights = iaModel(inputVector);
        iaOnlyWeights = max(iaOnlyWeights, 0);
        
        if sum(iaOnlyWeights) > 0
            iaOnlyWeights = iaOnlyWeights / sum(iaOnlyWeights);
        else
            iaOnlyWeights = equalWeights;
        end
    catch
        % Fallback to equal weights on error
        iaOnlyWeights = equalWeights;
    end
    
    % Get IA-only weights for regularized model
    try
        % Same window as original model
        windowStart = max(1, currentStep - strategyOptions.windowSize + 1);
        windowEnd = currentStep;
        window = returns(:, windowStart:windowEnd);
        
        % Normalize input for IA model
        inputVector = window(:);
        inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
        
        % Get IA predictions from regularized model
        iaRegOnlyWeights = iaModelReg(inputVector);
        iaRegOnlyWeights = max(iaRegOnlyWeights, 0);
        
        if sum(iaRegOnlyWeights) > 0
            iaRegOnlyWeights = iaRegOnlyWeights / sum(iaRegOnlyWeights);
        else
            iaRegOnlyWeights = equalWeights;
        end
    catch
        % Fallback to equal weights on error
        iaRegOnlyWeights = equalWeights;
    end
    
    % Calculate hybrid weights with original IA model
    try
        % Create performance history structure for the hybrid strategy
        performanceHistory = struct();
        performanceHistory.macd = dailyReturnsMACD(1:max(1, t-1));
        performanceHistory.ia = dailyReturnsIA(1:max(1, t-1));
        
        % Add performance history to options
        strategyOptions.performanceHistory = performanceHistory;
        
        [hybridWeights, regimeInfo] = hybrid_macd_ia_strategy(prices, volumes, ...
            enhancedMacdAgent, iaModel, currentStep, strategyOptions);
    catch ME
        % On error, fallback to a simple weighted average
        warning('Hybrid strategy error: %s\nUsing simple weighted average.', ME.message);
        
        hybridWeights = 0.7 * macdOnlyWeights + 0.3 * iaOnlyWeights;
        
        % Simple regime info
        regimeInfo = struct();
        regimeInfo.type = 1;
        regimeInfo.volatility = 0;
        regimeInfo.trend = 0;
        regimeInfo.weights = struct('macdWeight', 0.7, 'iaWeight', 0.3, 'cashWeight', 0);
    end
    
    % Calculate hybrid weights with regularized IA model
    try
        % Create performance history structure for the hybrid strategy
        performanceHistory = struct();
        performanceHistory.macd = dailyReturnsMACD(1:max(1, t-1));
        performanceHistory.ia = dailyReturnsIAReg(1:max(1, t-1));
        
        % Add performance history to options
        strategyOptions.performanceHistory = performanceHistory;
        
        [hybridRegWeights, ~] = hybrid_macd_ia_strategy(prices, volumes, ...
            enhancedMacdAgent, iaModelReg, currentStep, strategyOptions);
    catch ME
        % On error, fallback to a simple weighted average
        warning('Regularized hybrid strategy error: %s\nUsing simple weighted average.', ME.message);
        
        hybridRegWeights = 0.7 * macdOnlyWeights + 0.3 * iaRegOnlyWeights;
    end
    
    % Calculate returns
    rMACD = sum(macdOnlyWeights .* currentReturns);
    rIA = sum(iaOnlyWeights .* currentReturns);
    rIAReg = sum(iaRegOnlyWeights .* currentReturns);
    rHybrid = sum(hybridWeights .* currentReturns);
    rHybridReg = sum(hybridRegWeights .* currentReturns);
    rEqual = sum(equalWeights .* currentReturns);
    
    % Apply cost factor for trading (transaction costs)
    tradingCostFactor = 0.0005; % 5 basis points per trade as a simple model
    
    % Add transaction costs based on weight changes
    if t > 1
        macdTurnover = sum(abs(macdOnlyWeights - prevMacdWeights)) / 2;
        iaTurnover = sum(abs(iaOnlyWeights - prevIaWeights)) / 2;
        iaRegTurnover = sum(abs(iaRegOnlyWeights - prevIaRegWeights)) / 2;
        hybridTurnover = sum(abs(hybridWeights - prevHybridWeights)) / 2;
        hybridRegTurnover = sum(abs(hybridRegWeights - prevHybridRegWeights)) / 2;
        
        rMACD = rMACD - (macdTurnover * tradingCostFactor);
        rIA = rIA - (iaTurnover * tradingCostFactor);
        rIAReg = rIAReg - (iaRegTurnover * tradingCostFactor);
        rHybrid = rHybrid - (hybridTurnover * tradingCostFactor);
        rHybridReg = rHybridReg - (hybridRegTurnover * tradingCostFactor);
    end
    
    % Store current weights for next iteration's cost calculation
    prevMacdWeights = macdOnlyWeights;
    prevIaWeights = iaOnlyWeights;
    prevIaRegWeights = iaRegOnlyWeights;
    prevHybridWeights = hybridWeights;
    prevHybridRegWeights = hybridRegWeights;
    
    macdHistory(:, t) = macdOnlyWeights;
    iaHistory(:, t) = iaOnlyWeights;
    iaRegHistory(:, t) = iaRegOnlyWeights;
    hybridHistory(:, t) = hybridWeights;
    hybridRegHistory(:, t) = hybridRegWeights;
    
    % Apply circuit breaker limits (max 10% daily move)
    maxDailyReturn = 0.10; % 10% max daily return
    minDailyReturn = -0.10; % -10% max daily loss
    
    rMACD = min(maxDailyReturn, max(minDailyReturn, rMACD));
    rIA = min(maxDailyReturn, max(minDailyReturn, rIA));
    rIAReg = min(maxDailyReturn, max(minDailyReturn, rIAReg));
    rHybrid = min(maxDailyReturn, max(minDailyReturn, rHybrid));
    rHybridReg = min(maxDailyReturn, max(minDailyReturn, rHybridReg));
    rEqual = min(maxDailyReturn, max(minDailyReturn, rEqual));
    
    % Update portfolio values
    valueMACD = valueMACD * (1 + rMACD);
    valueIA = valueIA * (1 + rIA);
    valueIAReg = valueIAReg * (1 + rIAReg);
    valueHybrid = valueHybrid * (1 + rHybrid);
    valueHybridReg = valueHybridReg * (1 + rHybridReg);
    valueEqual = valueEqual * (1 + rEqual);
    
    % Store values
    seriesMACD(t) = valueMACD;
    seriesIA(t) = valueIA;
    seriesIAReg(t) = valueIAReg;
    seriesHybrid(t) = valueHybrid;
    seriesHybridReg(t) = valueHybridReg;
    seriesEqual(t) = valueEqual;
    
    % Store regime information
    regimeTypes(t) = regimeInfo.type;
    macdWeights(t) = regimeInfo.weights.macdWeight;
    iaWeights(t) = regimeInfo.weights.iaWeight;
    
    % Update daily returns history
    dailyReturnsMACD(t) = rMACD;
    dailyReturnsIA(t) = rIA;
    dailyReturnsIAReg(t) = rIAReg;
    dailyReturnsHybrid(t) = rHybrid;
    dailyReturnsHybridReg(t) = rHybridReg;
    
    % Show progress
    if mod(t, ceil(testSteps/5)) == 0
        fprintf('Progress: %.1f%% complete\n', t/testSteps*100);
    end
end

% === Calculate Performance Metrics ===
fprintf('Calculating performance metrics...\n');

% Calculate returns for analysis
returnsMACD = diff([1, seriesMACD]) ./ [1, seriesMACD(1:end-1)];
returnsIA = diff([1, seriesIA]) ./ [1, seriesIA(1:end-1)];
returnsIAReg = diff([1, seriesIAReg]) ./ [1, seriesIAReg(1:end-1)];
returnsHybrid = diff([1, seriesHybrid]) ./ [1, seriesHybrid(1:end-1)];
returnsHybridReg = diff([1, seriesHybridReg]) ./ [1, seriesHybridReg(1:end-1)];
returnsEqual = diff([1, seriesEqual]) ./ [1, seriesEqual(1:end-1)];

% Risk-free rate assumption (1% annual)
riskFreeDaily = 0.01/252;

% Standard metrics
metrics = struct();

% Final returns
metrics.totalReturnMACD = (seriesMACD(end) - 1) * 100;
metrics.totalReturnIA = (seriesIA(end) - 1) * 100;
metrics.totalReturnIAReg = (seriesIAReg(end) - 1) * 100;
metrics.totalReturnHybrid = (seriesHybrid(end) - 1) * 100;
metrics.totalReturnHybridReg = (seriesHybridReg(end) - 1) * 100;
metrics.totalReturnEqual = (seriesEqual(end) - 1) * 100;

% Sharpe ratio (annualized)
metrics.sharpeMACD = (mean(returnsMACD(2:end)) - riskFreeDaily) / std(returnsMACD(2:end)) * sqrt(252);
metrics.sharpeIA = (mean(returnsIA(2:end)) - riskFreeDaily) / std(returnsIA(2:end)) * sqrt(252);
metrics.sharpeIAReg = (mean(returnsIAReg(2:end)) - riskFreeDaily) / std(returnsIAReg(2:end)) * sqrt(252);
metrics.sharpeHybrid = (mean(returnsHybrid(2:end)) - riskFreeDaily) / std(returnsHybrid(2:end)) * sqrt(252);
metrics.sharpeHybridReg = (mean(returnsHybridReg(2:end)) - riskFreeDaily) / std(returnsHybridReg(2:end)) * sqrt(252);
metrics.sharpeEqual = (mean(returnsEqual(2:end)) - riskFreeDaily) / std(returnsEqual(2:end)) * sqrt(252);

% Maximum drawdown
metrics.drawdownMACD = max(cummax(seriesMACD) - seriesMACD) / max(seriesMACD) * 100;
metrics.drawdownIA = max(cummax(seriesIA) - seriesIA) / max(seriesIA) * 100;
metrics.drawdownIAReg = max(cummax(seriesIAReg) - seriesIAReg) / max(seriesIAReg) * 100;
metrics.drawdownHybrid = max(cummax(seriesHybrid) - seriesHybrid) / max(seriesHybrid) * 100;
metrics.drawdownHybridReg = max(cummax(seriesHybridReg) - seriesHybridReg) / max(seriesHybridReg) * 100;
metrics.drawdownEqual = max(cummax(seriesEqual) - seriesEqual) / max(seriesEqual) * 100;

% Volatility (annualized)
metrics.volatilityMACD = std(returnsMACD(2:end)) * sqrt(252) * 100;
metrics.volatilityIA = std(returnsIA(2:end)) * sqrt(252) * 100;
metrics.volatilityIAReg = std(returnsIAReg(2:end)) * sqrt(252) * 100;
metrics.volatilityHybrid = std(returnsHybrid(2:end)) * sqrt(252) * 100;
metrics.volatilityHybridReg = std(returnsHybridReg(2:end)) * sqrt(252) * 100;
metrics.volatilityEqual = std(returnsEqual(2:end)) * sqrt(252) * 100;

% Win rate
metrics.winRateMACD = sum(returnsMACD(2:end) > 0) / length(returnsMACD(2:end)) * 100;
metrics.winRateIA = sum(returnsIA(2:end) > 0) / length(returnsIA(2:end)) * 100;
metrics.winRateIAReg = sum(returnsIAReg(2:end) > 0) / length(returnsIAReg(2:end)) * 100;
metrics.winRateHybrid = sum(returnsHybrid(2:end) > 0) / length(returnsHybrid(2:end)) * 100;
metrics.winRateHybridReg = sum(returnsHybridReg(2:end) > 0) / length(returnsHybridReg(2:end)) * 100;
metrics.winRateEqual = sum(returnsEqual(2:end) > 0) / length(returnsEqual(2:end)) * 100;

% Calculate average turnover
metrics.turnoverMACD = mean(sum(abs(diff(macdHistory, 1, 2)), 1)) / 2;
metrics.turnoverIA = mean(sum(abs(diff(iaHistory, 1, 2)), 1)) / 2;
metrics.turnoverIAReg = mean(sum(abs(diff(iaRegHistory, 1, 2)), 1)) / 2;
metrics.turnoverHybrid = mean(sum(abs(diff(hybridHistory, 1, 2)), 1)) / 2;
metrics.turnoverHybridReg = mean(sum(abs(diff(hybridRegHistory, 1, 2)), 1)) / 2;

% Performance by regime
regimeMetrics = struct();
for r = 1:4
    regimeDays = find(regimeTypes == r);
    if ~isempty(regimeDays)
        regimeName = {'HighVol+Trend', 'HighVol', 'Trend', 'LowVol+Trend'};
        regimeMetrics(r).name = regimeName{r};
        regimeMetrics(r).count = length(regimeDays);
        
        % Ensure regimeDays+1 doesn't exceed array bounds
        validDays = regimeDays(regimeDays+1 <= length(returnsMACD));
        
        if length(validDays) > 1
            regimeReturnsMACD = returnsMACD(validDays+1);
            regimeReturnsIA = returnsIA(validDays+1);
            regimeReturnsIAReg = returnsIAReg(validDays+1);
            regimeReturnsHybrid = returnsHybrid(validDays+1);
            regimeReturnsHybridReg = returnsHybridReg(validDays+1);
            
            regimeMetrics(r).avgReturnMACD = mean(regimeReturnsMACD) * 100;
            regimeMetrics(r).avgReturnIA = mean(regimeReturnsIA) * 100;
            regimeMetrics(r).avgReturnIAReg = mean(regimeReturnsIAReg) * 100;
            regimeMetrics(r).avgReturnHybrid = mean(regimeReturnsHybrid) * 100;
            regimeMetrics(r).avgReturnHybridReg = mean(regimeReturnsHybridReg) * 100;
            
            if length(validDays) > 5
                regimeMetrics(r).sharpeMACD = mean(regimeReturnsMACD) / std(regimeReturnsMACD) * sqrt(252);
                regimeMetrics(r).sharpeIA = mean(regimeReturnsIA) / std(regimeReturnsIA) * sqrt(252);
                regimeMetrics(r).sharpeIAReg = mean(regimeReturnsIAReg) / std(regimeReturnsIAReg) * sqrt(252);
                regimeMetrics(r).sharpeHybrid = mean(regimeReturnsHybrid) / std(regimeReturnsHybrid) * sqrt(252);
                regimeMetrics(r).sharpeHybridReg = mean(regimeReturnsHybridReg) / std(regimeReturnsHybridReg) * sqrt(252);
            else
                regimeMetrics(r).sharpeMACD = NaN;
                regimeMetrics(r).sharpeIA = NaN;
                regimeMetrics(r).sharpeIAReg = NaN;
                regimeMetrics(r).sharpeHybrid = NaN;
                regimeMetrics(r).sharpeHybridReg = NaN;
            end
        else
            regimeMetrics(r).avgReturnMACD = NaN;
            regimeMetrics(r).avgReturnIA = NaN;
            regimeMetrics(r).avgReturnIAReg = NaN;
            regimeMetrics(r).avgReturnHybrid = NaN;
            regimeMetrics(r).avgReturnHybridReg = NaN;
            regimeMetrics(r).sharpeMACD = NaN;
            regimeMetrics(r).sharpeIA = NaN;
            regimeMetrics(r).sharpeIAReg = NaN;
            regimeMetrics(r).sharpeHybrid = NaN;
            regimeMetrics(r).sharpeHybridReg = NaN;
        end
    end
end

% === Print Results ===
fprintf('\n=== VALIDATION TEST RESULTS ===\n\n');
fprintf('Strategy          | Return (%%) | Sharpe | Drawdown (%%) | Volatility (%%) | Turnover | Win Rate (%%) |\n');
fprintf('------------------|------------|--------|--------------|----------------|-----------|--------------|\n');
fprintf('Enhanced MACD     | %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnMACD, metrics.sharpeMACD, metrics.drawdownMACD, ...
    metrics.volatilityMACD, metrics.turnoverMACD, metrics.winRateMACD);
fprintf('IA Only           | %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnIA, metrics.sharpeIA, metrics.drawdownIA, ...
    metrics.volatilityIA, metrics.turnoverIA, metrics.winRateIA);
fprintf('IA Regularized    | %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnIAReg, metrics.sharpeIAReg, metrics.drawdownIAReg, ...
    metrics.volatilityIAReg, metrics.turnoverIAReg, metrics.winRateIAReg);
fprintf('Hybrid MACD-IA    | %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnHybrid, metrics.sharpeHybrid, metrics.drawdownHybrid, ...
    metrics.volatilityHybrid, metrics.turnoverHybrid, metrics.winRateHybrid);
fprintf('Hybrid Regularized| %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnHybridReg, metrics.sharpeHybridReg, metrics.drawdownHybridReg, ...
    metrics.volatilityHybridReg, metrics.turnoverHybridReg, metrics.winRateHybridReg);
fprintf('Equal Weights     | %10.2f | %6.2f | %12.2f | %14.2f | %9.3f | %12.2f |\n', ...
    metrics.totalReturnEqual, metrics.sharpeEqual, metrics.drawdownEqual, ...
    metrics.volatilityEqual, 0.0, metrics.winRateEqual);

% Calculate improvement from regularization
iaImprovement = (metrics.totalReturnIAReg - metrics.totalReturnIA) / abs(metrics.totalReturnIA) * 100;
hybridImprovement = (metrics.totalReturnHybridReg - metrics.totalReturnHybrid) / abs(metrics.totalReturnHybrid) * 100;

fprintf('\nRegularization improvements:\n');
fprintf('IA Model: %.2f%%\n', iaImprovement);
fprintf('Hybrid Model: %.2f%%\n', hybridImprovement);

% === Create Visualization ===
try
    % Figure: Strategy performance comparison
    figure('Position', [100, 100, 1200, 800]);
    
    % Equity curves
    subplot(2, 1, 1);
    hold on;
    plot(seriesMACD, 'b', 'LineWidth', 2);
    plot(seriesIA, 'g', 'LineWidth', 1);
    plot(seriesIAReg, 'g--', 'LineWidth', 2);
    plot(seriesHybrid, 'r', 'LineWidth', 1);
    plot(seriesHybridReg, 'r--', 'LineWidth', 2);
    plot(seriesEqual, 'k--', 'LineWidth', 1);
    
    title('Strategy Performance Comparison', 'FontSize', 14);
    xlabel('Trading Day', 'FontSize', 12);
    ylabel('Portfolio Value', 'FontSize', 12);
    legend({'Enhanced MACD', 'IA Only', 'IA Regularized', 'Hybrid MACD-IA', 'Hybrid Regularized', 'Equal Weights'}, ...
        'Location', 'best');
    grid on;
    
    % Drawdown comparison
    subplot(2, 1, 2);
    hold on;
    
    % Calculate drawdowns
    ddMACD = (cummax(seriesMACD) - seriesMACD) ./ cummax(seriesMACD) * 100;
    ddIA = (cummax(seriesIA) - seriesIA) ./ cummax(seriesIA) * 100;
    ddIAReg = (cummax(seriesIAReg) - seriesIAReg) ./ cummax(seriesIAReg) * 100;
    ddHybrid = (cummax(seriesHybrid) - seriesHybrid) ./ cummax(seriesHybrid) * 100;
    ddHybridReg = (cummax(seriesHybridReg) - seriesHybridReg) ./ cummax(seriesHybridReg) * 100;
    
    plot(ddMACD, 'b', 'LineWidth', 1);
    plot(ddIA, 'g', 'LineWidth', 1);
    plot(ddIAReg, 'g--', 'LineWidth', 2);
    plot(ddHybrid, 'r', 'LineWidth', 1);
    plot(ddHybridReg, 'r--', 'LineWidth', 2);
    
    title('Drawdowns', 'FontSize', 12);
    xlabel('Trading Day', 'FontSize', 10);
    ylabel('Drawdown (%)', 'FontSize', 10);
    ylim([0, max([max(ddMACD), max(ddIA), max(ddIAReg), max(ddHybrid), max(ddHybridReg)])*1.1]);
    grid on;
    legend({'MACD', 'IA', 'IA Regularized', 'Hybrid', 'Hybrid Regularized'}, 'Location', 'best');
catch ME
    warning('Error creating visualization: %s', ME.message);
end

% === Prepare Output ===
results = struct();
results.metrics = metrics;
results.seriesMACD = seriesMACD;
results.seriesIA = seriesIA;
results.seriesIAReg = seriesIAReg;
results.seriesHybrid = seriesHybrid;
results.seriesHybridReg = seriesHybridReg;
results.seriesEqual = seriesEqual;
results.regimeTypes = regimeTypes;
results.regimeMetrics = regimeMetrics;
results.macdWeights = macdWeights;
results.iaWeights = iaWeights;
results.iaHistory = iaHistory;
results.iaRegHistory = iaRegHistory;
results.macdHistory = macdHistory;
results.hybridHistory = hybridHistory;
results.hybridRegHistory = hybridRegHistory;

fprintf('\n✅ Validation test completed.\n');

end 
 
 
 
 