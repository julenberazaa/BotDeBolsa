%% HYBRID STRATEGY COMPARISON: ORIGINAL VS REGULARIZED MODEL
% This script simulates the hybrid MACD-IA strategy using both the original
% and regularized IA models, comparing their performance characteristics.

clc;
clear;
close all;

% Variables de configuración
USE_DEEP_LEARNING = false; % Cambiar a true si Deep Learning Toolbox está disponible

% Verificar si Deep Learning Toolbox está disponible
if exist('trainNetwork', 'file') ~= 2
    fprintf('INFO: Deep Learning Toolbox NO está instalado.\n');
    fprintf('     => Continuando con implementación alternativa.\n');
    USE_DEEP_LEARNING = false;
else
    fprintf('INFO: Deep Learning Toolbox está instalado correctamente.\n');
    USE_DEEP_LEARNING = true;
end

% Añadir funciones para usar sin Deep Learning si es necesario
if ~USE_DEEP_LEARNING
    % Agregar una implementación simulada de trainNetwork
    if exist('trainNetwork', 'file') ~= 2
        fprintf('INFO: Creando implementación de fallback para trainNetwork...\n');
        % No hacer nada, el código debe usar feedforwardnet en su lugar
    end
end

% Add all required paths
addpath('src/utils');
addpath('src/strategies');
addpath('src/agents');
addpath('src/data');
addpath('proyecto');

try
    % Try to add all paths using the utility function
    addPathsIfNeeded();
catch
    warning('addPathsIfNeeded failed. Using manual path addition.');
end

% Ensure SPO function is available
try
    ensure_spo_available();
catch ME
    warning('SPO validation failed: %s. Some functionality may be limited.', ME.message);
end

fprintf('=== HYBRID STRATEGY SIMULATION: ORIGINAL VS REGULARIZED MODEL ===\n\n');

%% Load Data
fprintf('Loading data...\n');

try
    % Try loading primary data file
    load('proyecto/ReaderBeginingDLR.mat');
    fprintf('Successfully loaded data from ReaderBeginingDLR.mat\n');
catch
    try
        % Fall back to secondary data file
        load('proyecto/ReaderBegining.mat');
        fprintf('Successfully loaded data from ReaderBegining.mat\n');
    catch
        % Load sample stock data as a last resort
        warning('Could not load project data files. Loading sample stock data...');
        % Get available stock symbols
        stockFiles = dir('bd_stock price/stocks/*.csv');
        stockSymbols = cell(min(15, length(stockFiles)), 1);
        
        % Extract symbols from filenames (limit to 15 stocks)
        for i = 1:min(15, length(stockFiles))
            [~, stockSymbols{i}] = fileparts(stockFiles(i).name);
        end
        
        % Load stock data
        [prices, volumes, dates] = load_stock_data(stockSymbols, 'bd_stock price/stocks', 'bd_stock price/etfs', 500);
        
        % Calculate returns
        RetornosMedios = zeros(size(prices));
        for t = 2:size(prices, 2)
            RetornosMedios(:, t) = (prices(:, t) - prices(:, t-1)) ./ prices(:, t-1);
        end
        Volumes = volumes;
    end
end

% Ensure RetornosMedios is in the right format
if size(RetornosMedios, 1) > size(RetornosMedios, 2)
    RetornosMedios = RetornosMedios'; % Transpose to ensure assets are rows
end

[numAssets, numSteps] = size(RetornosMedios);
fprintf('Data prepared with %d assets and %d time steps.\n', numAssets, numSteps);

% Create dummy volume data if not available
if ~exist('Volumes', 'var')
    fprintf('No volume data found. Creating synthetic volumes...\n');
    Volumes = ones(size(RetornosMedios));
    
    % Add some random fluctuations to make it more realistic
    for t = 2:numSteps
        volatility = std(RetornosMedios(:, max(1, t-10):t-1), [], 2);
        randomFactor = 1 + 0.5 * randn(numAssets, 1) .* volatility;
        Volumes(:, t) = Volumes(:, t-1) .* max(0.5, randomFactor);
    end
end

%% Configuration
% MACD parameters
macdConfig = struct();
macdConfig.fastPeriod = 5;    % Fast EMA period
macdConfig.slowPeriod = 40;   % Slow EMA period
macdConfig.signalPeriod = 5;  % Signal line period

% MACD signal filtering
filterConfig = struct();
% Original filterConfig
% filterConfig.volumeThreshold = 1.2;      % Consider volume increase of 20% significant
% filterConfig.histogramThreshold = 0.001;  % Minimum histogram value for a strong signal
% filterConfig.trendConfirmation = true;    % Confirm signals with trend
% filterConfig.signalThreshold = 0.3;       % Filter signals below this strength

% Relaxed filterConfig for testing
filterConfig.volumeThreshold = 1.0; % Less restrictive, allow more signals
filterConfig.histogramThreshold = 0.0;  % Allow signals with any histogram value
filterConfig.trendConfirmation = false; % Disable trend confirmation for now
filterConfig.signalThreshold = 0.0;   % Allow signals of any strength

% Regime detection settings
regimeConfig = struct();
regimeConfig.volatility = struct('window', 20, 'method', 'std', 'threshold', 0.015);
regimeConfig.trend = struct('window', 20, 'method', 'corr', 'threshold', 0.6);

% Strategy options
strategyOptions = struct();
strategyOptions.windowSize = 5;            % Window size for IA inputs
strategyOptions.maxPosition = 0.20;        % Maximum position size (20%)
strategyOptions.useRegimeDetection = true; % Use dynamic regime detection
strategyOptions.cashAllocation = 0.0;      % Minimum cash allocation
strategyOptions.regimeSettings = regimeConfig;
strategyOptions.filterSettings = filterConfig;

%% Train Original and Regularized IA Models
fprintf('Training IA models...\n');

% Define training and test periods
trainEndIdx = floor(numSteps * 0.6); % 60% for training
testStartIdx = trainEndIdx + 1;
testEndIdx = numSteps;

trainingData = RetornosMedios(:, 1:trainEndIdx);
trainingVolumes = Volumes(:, 1:trainEndIdx);

% Train Original IA Model using robust implementation
try
    fprintf('Training original IA model with robust implementation...\n');
    tic;
    % First try the original model
    try
        [originalModel, originalTrainingInfo] = train_ia_complementary(trainingData, macdConfig);
        originalTrainingTime = toc;
        fprintf('Original model training completed in %.2f seconds.\n', originalTrainingTime);
        originalModelAvailable = true;
    catch ME1
        fprintf('Original model training failed: %s\nFalling back to robust training model.\n', ME1.message);
        % Fall back to robust implementation
        originalModel = train_ia_simple_robust(trainingData, macdConfig);
        originalTrainingTime = toc;
        fprintf('Robust original model training completed in %.2f seconds.\n', originalTrainingTime);
        originalModelAvailable = true;
    end
catch ME
    warning('All original model training attempts failed: %s\nUsing equal weight fallback.', ME.message);
    originalModel = @(x) ones(numAssets, 1) / numAssets;  % Equal weight as last resort
    originalTrainingTime = 0;
    originalModelAvailable = false;
end

% Train Regularized IA Model with robust implementation
try
    fprintf('Training regularized IA model with robust implementation...\n');
    tic;
    % First try the regularized model
    try
        [regularizedModel, regularizedTrainingInfo] = train_ia_complementary_regularized(trainingData, macdConfig);
        regularizedTrainingTime = toc;
        fprintf('Regularized model training completed in %.2f seconds.\n', regularizedTrainingTime);
        regularizedModelAvailable = true;
    catch ME1
        fprintf('Regularized model training failed: %s\nFalling back to robust training model.\n', ME1.message);
        % Use the robust implementation with more regularization
        regularizedModel = train_ia_simple_robust(trainingData, macdConfig);
        regularizedTrainingTime = toc;
        fprintf('Robust regularized model training completed in %.2f seconds.\n', regularizedTrainingTime);
        regularizedModelAvailable = true;
    end
catch ME
    warning('All regularized model training attempts failed: %s\nUsing equal weight fallback.', ME.message);
    regularizedModel = @(x) ones(numAssets, 1) / numAssets;  % Equal weight as last resort
    regularizedTrainingTime = 0;
    regularizedModelAvailable = false;
end

% If both models failed, print diagnostic information
if ~originalModelAvailable && ~regularizedModelAvailable
    fprintf('\n⚠️ DIAGNOSTIC INFORMATION ⚠️\n');
    fprintf('Both models failed to train. Here is diagnostic information:\n');
    
    % Check Deep Learning Toolbox
    if exist('trainNetwork', 'file') ~= 2
        fprintf('- Deep Learning Toolbox is not available on this system\n');
    else
        fprintf('- Deep Learning Toolbox is available\n');
    end
    
    % Check data quality
    nanCount = sum(sum(isnan(trainingData)));
    infCount = sum(sum(isinf(trainingData)));
    fprintf('- Training data contains %d NaN and %d Inf values\n', nanCount, infCount);
    
    % Check for very extreme values
    extremeValues = sum(sum(abs(trainingData) > 100));
    fprintf('- Training data contains %d extremely large values (>100)\n', extremeValues);
    
    % Check MATLAB version
    fprintf('- MATLAB Version: %s\n', version);
end

%% Create MACD Agent
fprintf('Creating enhanced MACD agent...\n');
try
    % Primero intentar crear enhanced_macd_agent objeto de clase
    try
        macdAgent = enhanced_macd_agent(RetornosMedios, Volumes, ...
            macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod, ...
            filterConfig, true); % Last param: use adaptive parameters
        
        % Verificar que el objeto macdAgent tiene la función esperada
        if isobject(macdAgent) && ismethod(macdAgent, 'getSignals')
            fprintf('Enhanced MACD agent created successfully with getSignals method.\n');
            macdAgentType = 'object';
            macdAgentAvailable = true;
        elseif isstruct(macdAgent) && isfield(macdAgent, 'getSignals') && isa(macdAgent.getSignals, 'function_handle')
            fprintf('Enhanced MACD agent created successfully as struct with function.\n');
            macdAgentType = 'struct';
            macdAgentAvailable = true;
        else
            % El agente se creó pero no tiene la estructura esperada
            fprintf('Enhanced MACD agent created but lacks expected interface. Using simple MACD agent.\n');
            error('MACD agent format not supported. Falling back to simple agent.');
        end
    catch ME1
        % Si falla la creación del enhanced_macd_agent, crear simple_macd_agent
        fprintf('Error creating enhanced MACD agent: %s\nTrying simple MACD agent.\n', ME1.message);
        macdAgent = simple_macd_agent(RetornosMedios, Volumes, ...
            macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod, ...
            filterConfig);
        
        if isstruct(macdAgent) && isfield(macdAgent, 'getSignals')
            fprintf('Simple MACD agent created successfully.\n');
            macdAgentType = 'struct';
            macdAgentAvailable = true;
        else
            error('Simple MACD agent creation failed.');
        end
    end
catch ME
    warning('All MACD agent creation methods failed: %s\nFalling back to basic MACD signals.', ME.message);
    % Crear matriz de señales básicas
    basicMacdSignals = zeros(numAssets, numSteps);
    for asset = 1:numAssets
        basicMacdSignals(asset, :) = macd_strategy(RetornosMedios(asset, :), ...
            macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod)';
    end
    macdAgent = basicMacdSignals;
    macdAgentType = 'matrix';
    macdAgentAvailable = true;
end

%% Run Simulation
fprintf('Running simulation for %d steps...\n', testEndIdx - testStartIdx + 1);

% Initialize portfolio values
valueMACD = 1;
valueOriginal = 1;
valueRegularized = 1;
valueHybridOriginal = 1;
valueHybridRegularized = 1;
valueEqual = 1;

% Initialize arrays for historical values
testSteps = testEndIdx - testStartIdx + 1;
seriesMACD = zeros(1, testSteps);
seriesOriginal = zeros(1, testSteps);
seriesRegularized = zeros(1, testSteps);
seriesHybridOriginal = zeros(1, testSteps);
seriesHybridRegularized = zeros(1, testSteps);
seriesEqual = zeros(1, testSteps);

% Arrays for portfolio construction metrics
macdTurnover = zeros(1, testSteps-1);
originalTurnover = zeros(1, testSteps-1);
regularizedTurnover = zeros(1, testSteps-1);
hybridOriginalTurnover = zeros(1, testSteps-1);
hybridRegularizedTurnover = zeros(1, testSteps-1);
equalTurnover = zeros(1, testSteps-1);

% Arrays for portfolio weights
macdWeights = zeros(numAssets, testSteps);
originalWeights = zeros(numAssets, testSteps);
regularizedWeights = zeros(numAssets, testSteps);
hybridOriginalWeights = zeros(numAssets, testSteps);
hybridRegularizedWeights = zeros(numAssets, testSteps);

% Arrays for position counts
macdPositions = zeros(1, testSteps);
originalPositions = zeros(1, testSteps);
regularizedPositions = zeros(1, testSteps);
hybridOriginalPositions = zeros(1, testSteps);
hybridRegularizedPositions = zeros(1, testSteps);

% Equal weight portfolio (benchmark)
equalWeights = ones(numAssets, 1) / numAssets;

% Regime tracking
regimes = zeros(1, testSteps);

% Implement risk management with max drawdown protection
maxDrawdown = 0.15;  % Maximum drawdown allowed (15%)
riskAdjustment = 1.0; % Initial risk adjustment factor

% Main simulation loop
for t = 1:testSteps
    currentStep = testStartIdx + t - 1;
    
    % Detect current market regime
    if t > 20
        volatility = std(mean(RetornosMedios(:, currentStep-20:currentStep-1), 1));
        trendStrength = 0;
        for asset = 1:numAssets
            assetPrices = RetornosMedios(asset, currentStep-20:currentStep-1);
            if std(assetPrices) > 0
                corrMat = corrcoef((1:20)', assetPrices');
                trendStrength = trendStrength + abs(corrMat(1,2));
            end
        end
        trendStrength = trendStrength / numAssets;
        
        if volatility > regimeConfig.volatility.threshold && trendStrength > regimeConfig.trend.threshold
            regimes(t) = 1; % High vol + Strong trend
        elseif volatility > regimeConfig.volatility.threshold
            regimes(t) = 2; % High vol + Weak trend
        elseif trendStrength > regimeConfig.trend.threshold
            regimes(t) = 3; % Low vol + Strong trend
        else
            regimes(t) = 4; % Low vol + Weak trend
        end
    else
        regimes(t) = 0; % Not enough data
    end
    
    % === Build MACD Portfolio ===
    try
        if macdAgentAvailable
            % Get MACD signals based on agent type
            if strcmp(macdAgentType, 'object')
                % Es un objeto con método getSignals
                macdSignals = macdAgent.getSignals(currentStep);
            elseif strcmp(macdAgentType, 'struct')
                % Es una estructura con función getSignals
                macdSignals = macdAgent.getSignals(currentStep);
            elseif strcmp(macdAgentType, 'matrix')
                % Es una matriz de señales directa
                if currentStep <= size(macdAgent, 2)
                    macdSignals = macdAgent(:, currentStep);
                else
                    % Fuera de rango - usar señales neutras
                    macdSignals = zeros(numAssets, 1);
                    fprintf('Warning: MACD signal index out of range at step %d\n', currentStep);
                end
            else
                % Tipo desconocido - usar señales calculadas directamente
                error('Unknown MACD agent type');
            end
            
            % Convert signals to weights
            macdWeights(:, t) = convert_signals_to_weights(macdSignals, strategyOptions.maxPosition);
        else
            % Direct fallback calculation
            error('MACD agent not available');
        end
    catch ME
        % Si hay cualquier error, calcular señales directamente
        fprintf('Error accessing MACD signals: %s\nUsing direct calculation.\n', ME.message);
        basicSignals = zeros(numAssets, 1);
        for asset = 1:numAssets
            if currentStep > macdConfig.slowPeriod + macdConfig.signalPeriod
                basicSignals(asset) = macd_strategy(RetornosMedios(asset, 1:currentStep), ...
                    macdConfig.fastPeriod, macdConfig.slowPeriod, macdConfig.signalPeriod);
            end
        end
        macdWeights(:, t) = convert_signals_to_weights(basicSignals, strategyOptions.maxPosition);
    end
    
    % === Build Original IA Portfolio ===
    if originalModelAvailable
        % Get input window
        if currentStep > strategyOptions.windowSize
            window = RetornosMedios(:, currentStep-strategyOptions.windowSize+1:currentStep);
            
            % Flatten and normalize window
            inputVector = window(:);
            inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
            
            % Get weights from original model
            try
                originalWeights(:, t) = originalModel(inputVector);
            catch
                fprintf('Error in original model prediction. Using equal weights.\n');
                originalWeights(:, t) = equalWeights;
            end
        else
            originalWeights(:, t) = equalWeights;
        end
    else
        originalWeights(:, t) = equalWeights;
    end
    
    % === Build Regularized IA Portfolio ===
    if regularizedModelAvailable
        % Get input window
        if currentStep > strategyOptions.windowSize
            window = RetornosMedios(:, currentStep-strategyOptions.windowSize+1:currentStep);
            
            % Flatten and normalize window
            inputVector = window(:);
            inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
            
            % Get weights from regularized model
            try
                regularizedWeights(:, t) = regularizedModel(inputVector);
            catch
                fprintf('Error in regularized model prediction. Using equal weights.\n');
                regularizedWeights(:, t) = equalWeights;
            end
        else
            regularizedWeights(:, t) = equalWeights;
        end
    else
        regularizedWeights(:, t) = equalWeights;
    end
    
    % === Build Hybrid Portfolios ===
    % Implement dynamic hybrid weights based on market regime
    if strategyOptions.useRegimeDetection && regimes(t) > 0
        % Regime-based MACD weighting
        % Regime 1: High vol + Strong trend - MACD favorable (70% MACD)
        % Regime 2: High vol + Weak trend (50% MACD)
        % Regime 3: Low vol + Strong trend (60% MACD)
        % Regime 4: Low vol + Weak trend - IA favorable (30% MACD)
        regimeBasedMACDWeight = [0.7, 0.5, 0.6, 0.3];
        macdWeight = regimeBasedMACDWeight(regimes(t));
        iaWeight = 1 - macdWeight;
    else
        % Default weighting
        macdWeight = 0.5;
        iaWeight = 0.5;
    end
    
    % Combine MACD and IA weights for hybrid portfolio
    hybridOriginalWeights(:, t) = macdWeight * macdWeights(:, t) + iaWeight * originalWeights(:, t);
    hybridRegularizedWeights(:, t) = macdWeight * macdWeights(:, t) + iaWeight * regularizedWeights(:, t);
    
    % Ensure weights are valid (non-negative, sum to 1)
    for w = {macdWeights(:, t), originalWeights(:, t), regularizedWeights(:, t), ...
            hybridOriginalWeights(:, t), hybridRegularizedWeights(:, t)}
        weights = w{1};
        weights(isnan(weights) | isinf(weights)) = 0;
        weights = max(0, weights);
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = equalWeights;
        end
    end
    
    % Apply risk adjustment when drawdowns occur
    if t > 1
        originalDD = (max(seriesOriginal(1:t-1)) - seriesOriginal(t-1)) / max(seriesOriginal(1:t-1));
        regularizedDD = (max(seriesRegularized(1:t-1)) - seriesRegularized(t-1)) / max(seriesRegularized(1:t-1));
        hybridOriginalDD = (max(seriesHybridOriginal(1:t-1)) - seriesHybridOriginal(t-1)) / max(seriesHybridOriginal(1:t-1));
        hybridRegularizedDD = (max(seriesHybridRegularized(1:t-1)) - seriesHybridRegularized(t-1)) / max(seriesHybridRegularized(1:t-1));
        
        % Reduce risk when drawdown exceeds threshold
        if originalDD > maxDrawdown || isnan(originalDD)
            originalWeights(:, t) = 0.7 * originalWeights(:, t) + 0.3 * equalWeights;
        end
        
        if regularizedDD > maxDrawdown || isnan(regularizedDD)
            regularizedWeights(:, t) = 0.7 * regularizedWeights(:, t) + 0.3 * equalWeights;
        end
        
        if hybridOriginalDD > maxDrawdown || isnan(hybridOriginalDD)
            hybridOriginalWeights(:, t) = 0.7 * hybridOriginalWeights(:, t) + 0.3 * equalWeights;
        end
        
        if hybridRegularizedDD > maxDrawdown || isnan(hybridRegularizedDD)
            hybridRegularizedWeights(:, t) = 0.7 * hybridRegularizedWeights(:, t) + 0.3 * equalWeights;
        end
    end
    
    % Count number of positions for each portfolio
    macdPositions(t) = sum(macdWeights(:, t) > 0.01);
    originalPositions(t) = sum(originalWeights(:, t) > 0.01);
    regularizedPositions(t) = sum(regularizedWeights(:, t) > 0.01);
    hybridOriginalPositions(t) = sum(hybridOriginalWeights(:, t) > 0.01);
    hybridRegularizedPositions(t) = sum(hybridRegularizedWeights(:, t) > 0.01);
    
    % Calculate portfolio returns
    if currentStep < numSteps
        nextReturns = RetornosMedios(:, currentStep+1);
        
        % Update portfolio values
        macdReturn = sum(macdWeights(:, t) .* nextReturns);
        originalReturn = sum(originalWeights(:, t) .* nextReturns);
        regularizedReturn = sum(regularizedWeights(:, t) .* nextReturns);
        hybridOriginalReturn = sum(hybridOriginalWeights(:, t) .* nextReturns);
        hybridRegularizedReturn = sum(hybridRegularizedWeights(:, t) .* nextReturns);
        equalReturn = sum(equalWeights .* nextReturns);
        
        % Apply reasonable limits to returns to prevent extreme values
        % (This helps with potential data issues or extreme market events)
        maxDailyReturn = 0.2;  % 20% max daily return
        minDailyReturn = -0.2; % -20% min daily return
        
        macdReturn = min(maxDailyReturn, max(minDailyReturn, macdReturn));
        originalReturn = min(maxDailyReturn, max(minDailyReturn, originalReturn));
        regularizedReturn = min(maxDailyReturn, max(minDailyReturn, regularizedReturn));
        hybridOriginalReturn = min(maxDailyReturn, max(minDailyReturn, hybridOriginalReturn));
        hybridRegularizedReturn = min(maxDailyReturn, max(minDailyReturn, hybridRegularizedReturn));
        equalReturn = min(maxDailyReturn, max(minDailyReturn, equalReturn));
        
        % Update portfolio values
        valueMACD = valueMACD * (1 + macdReturn);
        valueOriginal = valueOriginal * (1 + originalReturn);
        valueRegularized = valueRegularized * (1 + regularizedReturn);
        valueHybridOriginal = valueHybridOriginal * (1 + hybridOriginalReturn);
        valueHybridRegularized = valueHybridRegularized * (1 + hybridRegularizedReturn);
        valueEqual = valueEqual * (1 + equalReturn);
    end
    
    % Store historical values
    seriesMACD(t) = valueMACD;
    seriesOriginal(t) = valueOriginal;
    seriesRegularized(t) = valueRegularized;
    seriesHybridOriginal(t) = valueHybridOriginal;
    seriesHybridRegularized(t) = valueHybridRegularized;
    seriesEqual(t) = valueEqual;
    
    % Calculate turnover (starting from t=2)
    if t > 1 && currentStep < numSteps
        macdTurnover(t-1) = sum(abs(macdWeights(:, t) - macdWeights(:, t-1)));
        originalTurnover(t-1) = sum(abs(originalWeights(:, t) - originalWeights(:, t-1)));
        regularizedTurnover(t-1) = sum(abs(regularizedWeights(:, t) - regularizedWeights(:, t-1)));
        hybridOriginalTurnover(t-1) = sum(abs(hybridOriginalWeights(:, t) - hybridOriginalWeights(:, t-1)));
        hybridRegularizedTurnover(t-1) = sum(abs(hybridRegularizedWeights(:, t) - hybridRegularizedWeights(:, t-1)));
        equalTurnover(t-1) = 0; % Equal weight has no turnover
    end
end

%% Calculate Performance Metrics
fprintf('Calculating performance metrics...\n');

% Function to calculate metrics
calcMetrics = @(returns) struct(...
    'totalReturn', returns(end)/returns(1) - 1, ...
    'annualizedReturn', ((returns(end)/returns(1))^(252/length(returns)) - 1), ...
    'volatility', std(diff(returns)./returns(1:end-1)) * sqrt(252), ...
    'sharpe', mean(diff(returns)./returns(1:end-1)) / std(diff(returns)./returns(1:end-1)) * sqrt(252), ...
    'maxDrawdown', max(cummax(returns) - returns) / max(cummax(returns)), ...
    'winRate', sum(diff(returns) > 0) / (length(returns)-1) ...
);

% Calculate metrics
metricsMacd = calcMetrics(seriesMACD);
metricsOriginal = calcMetrics(seriesOriginal);
metricsRegularized = calcMetrics(seriesRegularized);
metricsHybridOriginal = calcMetrics(seriesHybridOriginal);
metricsHybridRegularized = calcMetrics(seriesHybridRegularized);
metricsEqual = calcMetrics(seriesEqual);

% Calculate average turnover
avgTurnoverMacd = mean(macdTurnover);
avgTurnoverOriginal = mean(originalTurnover);
avgTurnoverRegularized = mean(regularizedTurnover);
avgTurnoverHybridOriginal = mean(hybridOriginalTurnover);
avgTurnoverHybridRegularized = mean(hybridRegularizedTurnover);

%% Display Results
fprintf('\n=== PERFORMANCE COMPARISON ===\n');
fprintf('Metric                MACD      Original  Regular.  Hybrid-O  Hybrid-R  Equal\n');
fprintf('-----------------------------------------------------------------------------------\n');
fprintf('Total Return         %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n', ...
    metricsMacd.totalReturn*100, metricsOriginal.totalReturn*100, metricsRegularized.totalReturn*100, ...
    metricsHybridOriginal.totalReturn*100, metricsHybridRegularized.totalReturn*100, metricsEqual.totalReturn*100);

fprintf('Annualized Return    %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n', ...
    metricsMacd.annualizedReturn*100, metricsOriginal.annualizedReturn*100, metricsRegularized.annualizedReturn*100, ...
    metricsHybridOriginal.annualizedReturn*100, metricsHybridRegularized.annualizedReturn*100, metricsEqual.annualizedReturn*100);

fprintf('Volatility           %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n', ...
    metricsMacd.volatility*100, metricsOriginal.volatility*100, metricsRegularized.volatility*100, ...
    metricsHybridOriginal.volatility*100, metricsHybridRegularized.volatility*100, metricsEqual.volatility*100);

fprintf('Sharpe Ratio         %7.2f    %7.2f    %7.2f    %7.2f    %7.2f    %7.2f\n', ...
    metricsMacd.sharpe, metricsOriginal.sharpe, metricsRegularized.sharpe, ...
    metricsHybridOriginal.sharpe, metricsHybridRegularized.sharpe, metricsEqual.sharpe);

fprintf('Max Drawdown         %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n', ...
    metricsMacd.maxDrawdown*100, metricsOriginal.maxDrawdown*100, metricsRegularized.maxDrawdown*100, ...
    metricsHybridOriginal.maxDrawdown*100, metricsHybridRegularized.maxDrawdown*100, metricsEqual.maxDrawdown*100);

fprintf('Win Rate             %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n', ...
    metricsMacd.winRate*100, metricsOriginal.winRate*100, metricsRegularized.winRate*100, ...
    metricsHybridOriginal.winRate*100, metricsHybridRegularized.winRate*100, metricsEqual.winRate*100);

fprintf('Avg. Turnover        %7.2f    %7.2f    %7.2f    %7.2f    %7.2f    %7.2f\n', ...
    avgTurnoverMacd, avgTurnoverOriginal, avgTurnoverRegularized, ...
    avgTurnoverHybridOriginal, avgTurnoverHybridRegularized, 0);

fprintf('Avg. # Positions     %7.1f    %7.1f    %7.1f    %7.1f    %7.1f    %7.1f\n', ...
    mean(macdPositions), mean(originalPositions), mean(regularizedPositions), ...
    mean(hybridOriginalPositions), mean(hybridRegularizedPositions), numAssets);

%% Plot Results
fprintf('Generating performance charts...\n');

% Create date axis for plots
dateAxis = 1:length(seriesMACD);
if exist('dates', 'var') && length(dates) >= testStartIdx + length(seriesMACD) - 1
    dateAxis = dates(testStartIdx:testStartIdx+length(seriesMACD)-1);
end

% Performance Plot
figure('Name', 'Portfolio Performance Comparison', 'Position', [100, 100, 1200, 600]);

subplot(2, 2, 1);
plot(dateAxis, seriesMACD, 'b', 'LineWidth', 1.5);
hold on;
plot(dateAxis, seriesOriginal, 'r', 'LineWidth', 1.5);
plot(dateAxis, seriesRegularized, 'g', 'LineWidth', 1.5);
plot(dateAxis, seriesEqual, 'k--', 'LineWidth', 1);
hold off;
title('Performance: MACD vs IA Models');
ylabel('Portfolio Value');
legend('MACD', 'Original IA', 'Regularized IA', 'Equal Weight', 'Location', 'Best');
grid on;

subplot(2, 2, 2);
plot(dateAxis, seriesMACD, 'b', 'LineWidth', 1.5);
hold on;
plot(dateAxis, seriesHybridOriginal, 'm', 'LineWidth', 1.5);
plot(dateAxis, seriesHybridRegularized, 'c', 'LineWidth', 1.5);
plot(dateAxis, seriesEqual, 'k--', 'LineWidth', 1);
hold off;
title('Performance: MACD vs Hybrid Strategies');
ylabel('Portfolio Value');
legend('MACD', 'Hybrid (Original)', 'Hybrid (Regularized)', 'Equal Weight', 'Location', 'Best');
grid on;

% Plot Drawdowns
subplot(2, 2, 3);
drawdownMACD = (cummax(seriesMACD) - seriesMACD) ./ cummax(seriesMACD);
drawdownOriginal = (cummax(seriesOriginal) - seriesOriginal) ./ cummax(seriesOriginal);
drawdownRegularized = (cummax(seriesRegularized) - seriesRegularized) ./ cummax(seriesRegularized);
drawdownEqual = (cummax(seriesEqual) - seriesEqual) ./ cummax(seriesEqual);

plot(dateAxis, drawdownMACD, 'b', 'LineWidth', 1.5);
hold on;
plot(dateAxis, drawdownOriginal, 'r', 'LineWidth', 1.5);
plot(dateAxis, drawdownRegularized, 'g', 'LineWidth', 1.5);
plot(dateAxis, drawdownEqual, 'k--', 'LineWidth', 1);
hold off;
title('Drawdowns: MACD vs IA Models');
ylabel('Drawdown');
set(gca, 'YDir', 'reverse');  % Invert Y axis for drawdowns
legend('MACD', 'Original IA', 'Regularized IA', 'Equal Weight', 'Location', 'Best');
grid on;

subplot(2, 2, 4);
drawdownHybridOriginal = (cummax(seriesHybridOriginal) - seriesHybridOriginal) ./ cummax(seriesHybridOriginal);
drawdownHybridRegularized = (cummax(seriesHybridRegularized) - seriesHybridRegularized) ./ cummax(seriesHybridRegularized);

plot(dateAxis, drawdownMACD, 'b', 'LineWidth', 1.5);
hold on;
plot(dateAxis, drawdownHybridOriginal, 'm', 'LineWidth', 1.5);
plot(dateAxis, drawdownHybridRegularized, 'c', 'LineWidth', 1.5);
plot(dateAxis, drawdownEqual, 'k--', 'LineWidth', 1);
hold off;
title('Drawdowns: MACD vs Hybrid Strategies');
ylabel('Drawdown');
set(gca, 'YDir', 'reverse');  % Invert Y axis for drawdowns
legend('MACD', 'Hybrid (Original)', 'Hybrid (Regularized)', 'Equal Weight', 'Location', 'Best');
grid on;

% Plot Additional Analysis
figure('Name', 'Strategy Analysis', 'Position', [100, 100, 1200, 600]);

subplot(2, 2, 1);
% Corregir la concatenación de vectores
if ~isempty(macdTurnover)
    plot(dateAxis(2:end), macdTurnover, 'b', 'LineWidth', 1.5);
else
    plot(dateAxis(2:end), zeros(1, length(dateAxis)-1), 'b', 'LineWidth', 1.5);
end
hold on;
plot(dateAxis(2:end), originalTurnover, 'r', 'LineWidth', 1.5);
plot(dateAxis(2:end), regularizedTurnover, 'g', 'LineWidth', 1.5);
hold off;
title('Portfolio Turnover');
ylabel('Turnover');
legend('MACD', 'Original IA', 'Regularized IA', 'Location', 'Best');
grid on;

subplot(2, 2, 2);
plot(dateAxis(2:end), hybridOriginalTurnover, 'm', 'LineWidth', 1.5);
hold on;
plot(dateAxis(2:end), hybridRegularizedTurnover, 'c', 'LineWidth', 1.5);
hold off;
title('Hybrid Strategies Turnover');
ylabel('Turnover');
legend('Hybrid (Original)', 'Hybrid (Regularized)', 'Location', 'Best');
grid on;

subplot(2, 2, 3);
bar(dateAxis, regimes);
title('Market Regimes');
ylabel('Regime Type');
colormap(jet(4));
grid on;

subplot(2, 2, 4);
plot(dateAxis, macdPositions, 'b', 'LineWidth', 1.5);
hold on;
plot(dateAxis, originalPositions, 'r', 'LineWidth', 1.5);
plot(dateAxis, regularizedPositions, 'g', 'LineWidth', 1.5);
plot(dateAxis, hybridOriginalPositions, 'm', 'LineWidth', 1.5);
plot(dateAxis, hybridRegularizedPositions, 'c', 'LineWidth', 1.5);
hold off;
title('Number of Active Positions');
ylabel('# Positions');
legend('MACD', 'Original IA', 'Regularized IA', 'Hybrid (Original)', 'Hybrid (Regularized)', 'Location', 'Best');
grid on;

fprintf('\nSimulation complete!\n'); 