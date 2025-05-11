function [net, trainingInfo] = train_ia_complementary(prices, macdParameters, useFallback)
% TRAIN_IA_COMPLEMENTARY - Train an AI model that complements MACD strategy
%
% Syntax: [net, trainingInfo] = train_ia_complementary(prices, macdParameters, useFallback)
%
% Description:
%   This function trains a neural network model specifically designed to complement
%   the MACD strategy by focusing on market regimes where MACD traditionally underperforms.
%   It identifies these regimes and weights them more heavily in the training data.
%
% Inputs:
%   prices - Matrix of asset prices [assets x time]
%   macdParameters - Structure with MACD parameters (optional)
%       .fastPeriod - Fast EMA period (default: 12)
%       .slowPeriod - Slow EMA period (default: 26)
%       .signalPeriod - Signal line period (default: 9)
%   useFallback - Whether to use SPO as fallback if optimization fails (default: true)
%
% Outputs:
%   net - Trained neural network
%   trainingInfo - Structure with training information and performance metrics

% === Parameter Validation ===
if nargin < 2 || isempty(macdParameters)
    macdParameters = struct();
end

if ~isfield(macdParameters, 'fastPeriod')
    macdParameters.fastPeriod = 12;
end

if ~isfield(macdParameters, 'slowPeriod')
    macdParameters.slowPeriod = 26;
end

if ~isfield(macdParameters, 'signalPeriod')
    macdParameters.signalPeriod = 9;
end

if nargin < 3 || isempty(useFallback)
    useFallback = true;
end

% === Data Preparation ===
fprintf('Preparing data for IA-Complementary training...\n');

% Ensure prices are in the right format (assets x time)
if size(prices, 1) > size(prices, 2)
    prices = prices'; % Transpose if in column format
end

[numAssets, numSteps] = size(prices);

% Define lookback window for inputs
windowSize = 5;
if numSteps <= windowSize
    error('Not enough data for training. Minimum required: %d steps.', windowSize+1);
end

% Calculate returns for both inputs and target optimization
returns = zeros(size(prices));
for t = 2:numSteps
    returns(:, t) = (prices(:, t) - prices(:, t-1)) ./ prices(:, t-1);
end

% === Generate MACD Signals ===
fprintf('Calculating MACD signals for all assets...\n');
macdSignals = zeros(numAssets, numSteps);
for asset = 1:numAssets
    macdSignals(asset, :) = macd_strategy(prices(asset, :), ...
        macdParameters.fastPeriod, macdParameters.slowPeriod, macdParameters.signalPeriod)';
end

% === Market Regime Detection ===
fprintf('Detecting market regimes...\n');
regimeTypes = zeros(1, numSteps);
volatilityWindow = 20;
trendWindow = 20;

% We need at least windowSize + max(volatilityWindow, trendWindow) data points
if numSteps <= windowSize + max(volatilityWindow, trendWindow)
    warning('Not enough data for reliable regime detection. Using simple detection.');
    volatilityWindow = floor(numSteps/4);
    trendWindow = floor(numSteps/4);
end

% Detect market regimes for each time step
for t = max(volatilityWindow, trendWindow)+1:numSteps
    % Calculate volatility (std of returns)
    volIdx = max(1, t-volatilityWindow):t;
    volatility = mean(std(returns(:, volIdx), 0, 2));
    
    % Calculate trend strength (correlation with time)
    trendIdx = max(1, t-trendWindow):t;
    trend = 0;
    for asset = 1:numAssets
        assetPrices = prices(asset, trendIdx);
        timeVec = 1:length(trendIdx);
        if std(assetPrices) > 0
            corrMat = corrcoef(timeVec, assetPrices);
            trend = trend + abs(corrMat(1,2));
        end
    end
    trend = trend / numAssets;
    
    % Classify market regime
    highVol = volatility >= 0.015;
    strongTrend = trend >= 0.6;
    
    if highVol && strongTrend
        regimeTypes(t) = 1; % High vol + Strong trend - MACD favorable
    elseif highVol && ~strongTrend
        regimeTypes(t) = 2; % High vol + Weak trend
    elseif ~highVol && strongTrend
        regimeTypes(t) = 3; % Low vol + Strong trend
    else
        regimeTypes(t) = 4; % Low vol + Weak trend - IA favorable
    end
end

% Fill in initial values
regimeTypes(1:max(volatilityWindow, trendWindow)) = regimeTypes(max(volatilityWindow, trendWindow)+1);

% === Generate Training Data ===
fprintf('Generating training dataset...\n');

% Initialize arrays for inputs and targets
X = [];
Y = [];
Regimes = [];
Steps = [];

% Define sample weights - emphasize regimes where MACD underperforms
regimeWeights = [0.5, 1.5, 1.0, 2.0]; % Weights for regimes 1-4

% Generate training samples
for t = 1:numSteps-windowSize
    % Get current window of returns
    window = returns(:, t:t+windowSize-1);
    
    % Calculate average returns and volatility for target step
    targetIdx = t + windowSize;
    avgReturns = mean(returns(:, max(1, targetIdx-10):targetIdx-1), 2);
    varReturns = var(returns(:, max(1, targetIdx-10):targetIdx-1), 0, 2);
    
    % Get MACD signals for this step
    macdSignalsCurrent = macdSignals(:, targetIdx);
    
    % Get target returns
    targetReturns = returns(:, targetIdx);
    
    % Get current regime
    currentRegime = regimeTypes(targetIdx);
    
    try
        % Optimize portfolio using SPO
        % This is the "ideal" target based on having perfect knowledge
        % We'll use this as our training target
        wOpt = obtenerSPO(avgReturns, varReturns, 0.1);
        
        % For regimes where MACD underperforms, adjust the target
        % to emphasize different assets than what MACD would choose
        if currentRegime == 2 || currentRegime == 4
            % For MACD-unfavorable regimes, put more weight on assets MACD ignores
            % and less weight on assets MACD favors
            for i = 1:numAssets
                if macdSignalsCurrent(i) == 1
                    % MACD buy signal - reduce weight slightly
                    wOpt(i) = wOpt(i) * 0.8;
                elseif macdSignalsCurrent(i) == 0
                    % MACD neutral - keep weight as is
                    % No change
                elseif macdSignalsCurrent(i) == -1
                    % MACD sell signal but asset has good characteristics
                    % If expected return is positive, increase weight
                    if avgReturns(i) > 0 && varReturns(i) < mean(varReturns)
                        wOpt(i) = wOpt(i) * 1.5 + 0.02;
                    end
                end
            end
            
            % Clean up and normalize
            wOpt = max(wOpt, 0);
            if sum(wOpt) > 0
                wOpt = wOpt / sum(wOpt);
            else
                wOpt = ones(numAssets, 1) / numAssets;
            end
        end
        
        % Store valid sample
        if ~any(isnan(wOpt)) && all(wOpt >= 0) && abs(sum(wOpt) - 1) < 0.01
            % Flatten and normalize the window input
            inputVector = window(:);
            
            % Handle potential NaNs/Infs in inputVector before normalization
            if any(isinf(inputVector))
                fprintf('INFO train_ia_complementary: Infs found in inputVector for sample before X. Replacing with 0.\n');
                inputVector(isinf(inputVector)) = 0;
            end
            if any(isnan(inputVector))
                fprintf('INFO train_ia_complementary: NaNs found in inputVector for sample before X. Replacing with 0.\n');
                inputVector(isnan(inputVector)) = 0;
            end
            
            inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
            
            % Handle potential NaNs/Infs in inputVector AFTER normalization (e.g. if std was 0)
            if any(isinf(inputVector))
                fprintf('INFO train_ia_complementary: Infs found in inputVector for sample AFTER norm. Replacing with 0.\n');
                inputVector(isinf(inputVector)) = 0;
            end
            if any(isnan(inputVector))
                fprintf('INFO train_ia_complementary: NaNs found in inputVector for sample AFTER norm. Replacing with 0.\n');
                inputVector(isnan(inputVector)) = 0;
            end
            
            % Store input, target, regime, and step
            X = [X, inputVector];
            Y = [Y, wOpt];
            Regimes = [Regimes, currentRegime];
            Steps = [Steps, targetIdx];
        end
        
    catch
        % If SPO optimization fails, try a simpler alternative
        if useFallback
            % Simple alternative: weight inversely proportional to volatility
            invVol = 1 ./ (varReturns + 1e-10);
            wAlt = invVol / sum(invVol);
            
            % Apply the same regime-based adjustments
            if currentRegime == 2 || currentRegime == 4
                for i = 1:numAssets
                    if macdSignalsCurrent(i) == 1
                        wAlt(i) = wAlt(i) * 0.8;
                    elseif macdSignalsCurrent(i) == -1
                        if avgReturns(i) > 0 && varReturns(i) < mean(varReturns)
                            wAlt(i) = wAlt(i) * 1.5 + 0.02;
                        end
                    end
                end
                
                % Clean up and normalize
                wAlt = max(wAlt, 0);
                if sum(wAlt) > 0
                    wAlt = wAlt / sum(wAlt);
                else
                    wAlt = ones(numAssets, 1) / numAssets;
                end
            end
            
            % Store valid sample with fallback weights
            if ~any(isnan(wAlt)) && all(wAlt >= 0) && abs(sum(wAlt) - 1) < 0.01
                % Flatten and normalize the window input
                inputVector = window(:);

                % Handle potential NaNs/Infs in inputVector before normalization
                if any(isinf(inputVector))
                    fprintf('INFO train_ia_complementary: Infs found in inputVector (fallback) for sample before X. Replacing with 0.\n');
                    inputVector(isinf(inputVector)) = 0;
                end
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_complementary: NaNs found in inputVector (fallback) for sample before X. Replacing with 0.\n');
                    inputVector(isnan(inputVector)) = 0;
                end
                
                inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);

                % Handle potential NaNs/Infs in inputVector AFTER normalization (e.g. if std was 0)
                if any(isinf(inputVector))
                    fprintf('INFO train_ia_complementary: Infs found in inputVector (fallback) for sample AFTER norm. Replacing with 0.\n');
                    inputVector(isinf(inputVector)) = 0;
                end
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_complementary: NaNs found in inputVector (fallback) for sample AFTER norm. Replacing with 0.\n');
                    inputVector(isnan(inputVector)) = 0;
                end
                
                % Store input, target, regime, and step
                X = [X, inputVector];
                Y = [Y, wAlt];
                Regimes = [Regimes, currentRegime];
                Steps = [Steps, targetIdx];
            end
        end
    end
end

% === Sample Weighting Based on Regimes ===
fprintf('Applying regime-based sample weights...\n');
sampleWeights = zeros(1, size(X, 2));
for i = 1:length(sampleWeights)
    sampleWeights(i) = regimeWeights(Regimes(i));
end

% Normalize weights to average of 1
sampleWeights = sampleWeights / mean(sampleWeights);

% === Neural Network Training ===
fprintf('Training neural network...\n');
numSamples = size(X, 2);

if numSamples < 10
    error('Not enough valid samples for training (%d). Check data quality.', numSamples);
end

fprintf('Total training samples: %d\n', numSamples);

% Setup network architecture
layers = [
    featureInputLayer(size(X, 1), "Name", "input")
    fullyConnectedLayer(128, "Name", "fc1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.3, "Name", "dropout1")
    fullyConnectedLayer(64, "Name", "fc2")
    reluLayer("Name", "relu2")
    fullyConnectedLayer(numAssets, "Name", "output")
    regressionLayer("Name", "regression")
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', min(32, floor(numSamples/3)), ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X(:, 1:3:end)', Y(:, 1:3:end)'}, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Apply sample weights - oversample important regimes
if ~all(sampleWeights == 1)
    % Create weighted training data by repeating samples based on weights
    weightedX = [];
    weightedY = [];
    
    for i = 1:length(sampleWeights)
        % Determine how many times to repeat this sample
        repeats = max(1, round(sampleWeights(i)));
        
        % Repeat sample accordingly
        for j = 1:repeats
            weightedX = [weightedX, X(:, i)];
            weightedY = [weightedY, Y(:, i)];
        end
    end
    
    % Use the weighted dataset
    trainingX = weightedX';
    trainingY = weightedY';
else
    % Use original dataset
    trainingX = X';
    trainingY = Y';
end

% Train the network
net = trainNetwork(trainingX, trainingY, layers, options);

% === Performance Evaluation ===
fprintf('Evaluating model performance...\n');

% Predict weights for all samples
predictedWeights = predict(net, X');
predictedWeights = predictedWeights';

% Calculate prediction error
predictionError = mean(sum((Y - predictedWeights).^2, 1));

% Calculate expected returns based on targets and predictions
expectedReturnTarget = zeros(1, size(Y, 2));
expectedReturnPrediction = zeros(1, size(Y, 2));

for i = 1:size(Y, 2)
    step = Steps(i);
    if step < size(returns, 2)
        nextReturns = returns(:, step+1);
        expectedReturnTarget(i) = sum(Y(:, i) .* nextReturns);
        expectedReturnPrediction(i) = sum(predictedWeights(:, i) .* nextReturns);
    end
end

% Calculate performance metrics by regime
regimeStats = struct();
for r = 1:4
    regimeIdx = Regimes == r;
    if any(regimeIdx)
        regimeStats(r).count = sum(regimeIdx);
        regimeStats(r).error = mean(sum((Y(:, regimeIdx) - predictedWeights(:, regimeIdx)).^2, 1));
        
        validIdxTarget = ~isnan(expectedReturnTarget(regimeIdx));
        validIdxPred = ~isnan(expectedReturnPrediction(regimeIdx));
        validIdx = validIdxTarget & validIdxPred;
        
        if any(validIdx)
            regimeStats(r).targetReturn = mean(expectedReturnTarget(regimeIdx(validIdx)));
            regimeStats(r).predReturn = mean(expectedReturnPrediction(regimeIdx(validIdx)));
        else
            regimeStats(r).targetReturn = NaN;
            regimeStats(r).predReturn = NaN;
        end
    else
        regimeStats(r).count = 0;
        regimeStats(r).error = NaN;
        regimeStats(r).targetReturn = NaN;
        regimeStats(r).predReturn = NaN;
    end
end

% Store training information
trainingInfo = struct();
trainingInfo.numSamples = numSamples;
trainingInfo.predictionError = predictionError;
trainingInfo.expectedReturnTarget = mean(expectedReturnTarget(~isnan(expectedReturnTarget)));
trainingInfo.expectedReturnPrediction = mean(expectedReturnPrediction(~isnan(expectedReturnPrediction)));
trainingInfo.regimeStats = regimeStats;
trainingInfo.regimeCounts = [sum(Regimes == 1), sum(Regimes == 2), sum(Regimes == 3), sum(Regimes == 4)];

% Display summary
fprintf('\n=== Training Results ===\n');
fprintf('Total samples: %d\n', numSamples);
fprintf('Overall prediction error: %.6f\n', predictionError);
fprintf('Expected return (target): %.2f%%\n', trainingInfo.expectedReturnTarget * 100);
fprintf('Expected return (prediction): %.2f%%\n', trainingInfo.expectedReturnPrediction * 100);
fprintf('\nSamples by regime: [%d, %d, %d, %d]\n', trainingInfo.regimeCounts);

fprintf('\n=== Regime-Specific Performance ===\n');
for r = 1:4
    if regimeStats(r).count > 0
        fprintf('Regime %d (%d samples):\n', r, regimeStats(r).count);
        fprintf('  Prediction error: %.6f\n', regimeStats(r).error);
        fprintf('  Target return: %.2f%%\n', regimeStats(r).targetReturn * 100);
        fprintf('  Predicted return: %.2f%%\n', regimeStats(r).predReturn * 100);
    end
end

fprintf('\n✅ Complementary IA model training completed.\n');

end 
 
 
 
 