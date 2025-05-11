function [net, trainingInfo] = train_ia_complementary_regularized(prices, macdParameters, useFallback)
% TRAIN_IA_COMPLEMENTARY_REGULARIZED - Train an AI model with regularization that complements MACD strategy
%
% Syntax: [net, trainingInfo] = train_ia_complementary_regularized(prices, macdParameters, useFallback)
%
% Description:
%   This function trains a neural network model with regularization techniques specifically 
%   designed to complement the MACD strategy by focusing on market regimes where MACD 
%   traditionally underperforms. It includes L2 regularization, dropout, and early stopping
%   to produce a more robust model and prevent overfitting.
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
fprintf('Preparing data for Regularized IA-Complementary training...\n');

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
fprintf('Detecting market regimes with adaptive thresholds...\n');
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
    
    % Use adaptive thresholds instead of fixed values
    if t > volatilityWindow*2
        % Calculate adaptive threshold based on historical volatility distribution
        volHistory = zeros(1, volatilityWindow);
        for i = 1:volatilityWindow
            histIdx = max(1, t-volatilityWindow-i+1):t-i;
            volHistory(i) = mean(std(returns(:, histIdx), 0, 2));
        end
        highVolThreshold = quantile(volHistory, 0.75); % 75th percentile as threshold
    else
        highVolThreshold = 0.015; % Default threshold if not enough history
    end
    
    if t > trendWindow*2
        % Calculate adaptive threshold based on historical trend distribution
        trendHistory = zeros(1, trendWindow);
        for i = 1:trendWindow
            histIdx = max(1, t-trendWindow-i+1):t-i;
            trendVal = 0;
            for asset = 1:numAssets
                assetPrices = prices(asset, histIdx);
                timeVec = 1:length(histIdx);
                if std(assetPrices) > 0
                    corrMat = corrcoef(timeVec, assetPrices);
                    trendVal = trendVal + abs(corrMat(1,2));
                end
            end
            trendHistory(i) = trendVal / numAssets;
        end
        strongTrendThreshold = quantile(trendHistory, 0.75); % 75th percentile
    else
        strongTrendThreshold = 0.6; % Default threshold if not enough history
    end
    
    % Classify market regime
    highVol = volatility >= highVolThreshold;
    strongTrend = trend >= strongTrendThreshold;
    
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

% === Generate Training Data with Enhanced Diversification ===
fprintf('Generating regularized training dataset...\n');

% Initialize arrays for inputs and targets
X = [];
Y = [];
Regimes = [];
Steps = [];

% Define sample weights - emphasize regimes where MACD underperforms
regimeWeights = [0.5, 1.5, 1.0, 2.0]; % Weights for regimes 1-4

% Generate training samples with more focus on robustness
for t = 10:numSteps-windowSize-5 % Leave some data for validation
    % Get current window of returns
    window = returns(:, t:t+windowSize-1);
    
    % Calculate average returns and volatility for target step
    targetIdx = t + windowSize;
    
    % Use a longer lookback for more robust estimates
    lookback = 15; % Use 15 days for more stable estimates
    avgReturns = mean(returns(:, max(1, targetIdx-lookback):targetIdx-1), 2);
    varReturns = var(returns(:, max(1, targetIdx-lookback):targetIdx-1), 0, 2);
    
    % Add noise to returns for regularization (data augmentation)
    noiseLevel = 0.0005; % Small amount of noise
    avgReturnsNoisy = avgReturns + noiseLevel * randn(size(avgReturns)) .* avgReturns;
    
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
        wOpt = obtenerSPO(avgReturnsNoisy, varReturns, 0.1);
        
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
        
        % Apply L1 regularization to targets - promote sparsity
        % This helps create more focused portfolios
        l1Factor = 0.02;
        threshold = l1Factor * mean(wOpt(wOpt > 0));
        wOpt(wOpt < threshold) = 0;
        
        if sum(wOpt) > 0
            wOpt = wOpt / sum(wOpt);
        else
            wOpt = ones(numAssets, 1) / numAssets;
        end
        
        % Store valid sample
        if ~any(isnan(wOpt)) && all(wOpt >= 0) && abs(sum(wOpt) - 1) < 0.01
            % Add data augmentation by creating slightly varied inputs
            for aug = 1:3
                % Create augmented input with small variations
                if aug > 1
                    augWindow = window + 0.001 * randn(size(window)) .* window;
                else
                    augWindow = window;
                end
                
                % Flatten and normalize the window input
                inputVector = augWindow(:);

                % Handle potential NaNs/Infs in inputVector before normalization
                if any(isinf(inputVector))
                    fprintf('INFO train_ia_complementary_regularized: Infs found in inputVector (aug %d) for sample before X. Replacing with 0.\n', aug);
                    inputVector(isinf(inputVector)) = 0;
                end
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_complementary_regularized: NaNs found in inputVector (aug %d) for sample before X. Replacing with 0.\n', aug);
                    inputVector(isnan(inputVector)) = 0;
                end

                inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);

                % Handle potential NaNs/Infs in inputVector AFTER normalization
                if any(isinf(inputVector))
                    fprintf('INFO train_ia_complementary_regularized: Infs found in inputVector (aug %d) for sample AFTER norm. Replacing with 0.\n', aug);
                    inputVector(isinf(inputVector)) = 0;
                end
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_complementary_regularized: NaNs found in inputVector (aug %d) for sample AFTER norm. Replacing with 0.\n', aug);
                    inputVector(isnan(inputVector)) = 0;
                end
                
                % Store input, target, regime, and step
                X = [X, inputVector];
                Y = [Y, wOpt];
                Regimes = [Regimes, currentRegime];
                Steps = [Steps, targetIdx];
            end
        end
        
    catch
        % If SPO optimization fails, try a simpler alternative with inverse volatility weighting
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
            
            % Apply L1 regularization to targets
            l1Factor = 0.02;
            threshold = l1Factor * mean(wAlt(wAlt > 0));
            wAlt(wAlt < threshold) = 0;
            
            if sum(wAlt) > 0
                wAlt = wAlt / sum(wAlt);
            else
                wAlt = ones(numAssets, 1) / numAssets;
            end
            
            % Store valid sample with fallback weights
            if ~any(isnan(wAlt)) && all(wAlt >= 0) && abs(sum(wAlt) - 1) < 0.01
                % Create augmented data
                for aug = 1:3
                    % Create augmented input with small variations
                    if aug > 1
                        augWindow = window + 0.001 * randn(size(window)) .* window;
                    else
                        augWindow = window;
                    end
                    
                    % Flatten and normalize the window input
                    inputVector = augWindow(:);

                    % Handle potential NaNs/Infs in inputVector before normalization
                    if any(isinf(inputVector))
                        fprintf('INFO train_ia_complementary_regularized: Infs found in inputVector (fallback aug %d) for sample before X. Replacing with 0.\n', aug);
                        inputVector(isinf(inputVector)) = 0;
                    end
                    if any(isnan(inputVector))
                        fprintf('INFO train_ia_complementary_regularized: NaNs found in inputVector (fallback aug %d) for sample before X. Replacing with 0.\n', aug);
                        inputVector(isnan(inputVector)) = 0;
                    end

                    inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);

                    % Handle potential NaNs/Infs in inputVector AFTER normalization
                    if any(isinf(inputVector))
                        fprintf('INFO train_ia_complementary_regularized: Infs found in inputVector (fallback aug %d) for sample AFTER norm. Replacing with 0.\n', aug);
                        inputVector(isinf(inputVector)) = 0;
                    end
                    if any(isnan(inputVector))
                        fprintf('INFO train_ia_complementary_regularized: NaNs found in inputVector (fallback aug %d) for sample AFTER norm. Replacing with 0.\n', aug);
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
end

% === Sample Weighting Based on Regimes with Robust Balancing ===
fprintf('Applying enhanced regime-based sample weights...\n');

% Count samples by regime
regimeCounts = zeros(1, 4);
for r = 1:4
    regimeCounts(r) = sum(Regimes == r);
end
fprintf('Samples by regime before balancing: [%d, %d, %d, %d]\n', regimeCounts);

% Balance training samples across regimes to avoid over-representation
% This is important for robustness in different market conditions
maxSamplesPerRegime = min(250, max(regimeCounts));
balancedX = [];
balancedY = [];
balancedRegimes = [];
balancedSteps = [];

for r = 1:4
    regimeIdx = find(Regimes == r);
    
    if ~isempty(regimeIdx)
        % If we have more samples than the maximum, sample randomly
        if length(regimeIdx) > maxSamplesPerRegime
            selectedIdx = regimeIdx(randperm(length(regimeIdx), maxSamplesPerRegime));
        else
            selectedIdx = regimeIdx;
        end
        
        % Add selected samples to balanced dataset
        balancedX = [balancedX, X(:, selectedIdx)];
        balancedY = [balancedY, Y(:, selectedIdx)];
        balancedRegimes = [balancedRegimes, Regimes(selectedIdx)];
        balancedSteps = [balancedSteps, Steps(selectedIdx)];
    end
end

% Replace original dataset with balanced dataset
X = balancedX;
Y = balancedY;
Regimes = balancedRegimes;
Steps = balancedSteps;

% Recalculate regime counts
regimeCounts = zeros(1, 4);
for r = 1:4
    regimeCounts(r) = sum(Regimes == r);
end
fprintf('Samples by regime after balancing: [%d, %d, %d, %d]\n', regimeCounts);

% Calculate sample weights based on regime weights
sampleWeights = zeros(1, size(X, 2));
for i = 1:length(sampleWeights)
    sampleWeights(i) = regimeWeights(Regimes(i));
end

% Normalize weights to average of 1
sampleWeights = sampleWeights / mean(sampleWeights);

% === Neural Network Training with Regularization ===
fprintf('Training neural network with regularization techniques...\n');
numSamples = size(X, 2);

if numSamples < 20
    error('Not enough valid samples for training (%d). Check data quality.', numSamples);
end

fprintf('Total training samples: %d\n', numSamples);

% Split data into training and validation sets
cv = cvpartition(numSamples, 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = ~idxTrain;

XTrain = X(:, idxTrain);
YTrain = Y(:, idxTrain);
XVal = X(:, idxVal);
YVal = Y(:, idxVal);

% Setup network architecture with regularization (L2 weight decay and dropouts)
layers = [
    featureInputLayer(size(X, 1), "Name", "input")
    
    fullyConnectedLayer(128, "Name", "fc1", ...
                       "WeightL2Factor", 0.001, ... % L2 regularization
                       "BiasL2Factor", 0.001)
    batchNormalizationLayer("Name", "bn1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.4, "Name", "dropout1") % Higher dropout for regularization
    
    fullyConnectedLayer(64, "Name", "fc2", ...
                       "WeightL2Factor", 0.001, ...
                       "BiasL2Factor", 0.001)
    batchNormalizationLayer("Name", "bn2")
    reluLayer("Name", "relu2")
    dropoutLayer(0.3, "Name", "dropout2")
    
    fullyConnectedLayer(numAssets, "Name", "output", ...
                       "WeightL2Factor", 0.0005, ...
                       "BiasL2Factor", 0.0005)
    regressionLayer("Name", "regression")
];

% Training options with early stopping
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', min(32, floor(numSamples/4)), ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal', YVal'}, ...
    'ValidationFrequency', 10, ...
    'ValidationPatience', 20, ... % Early stopping
    'Verbose', false, ...
    'Plots', 'training-progress');

% Apply sample weights through weighted sampling
if ~all(sampleWeights == 1)
    % Create weighted training data by sampling based on weights
    numTrainSamples = sum(idxTrain);
    weightedIndices = [];
    
    % Create sampling distribution
    sampleWeightsTrain = sampleWeights(idxTrain);
    sampleProbs = sampleWeightsTrain / sum(sampleWeightsTrain);
    
    % Sample with replacement based on weights
    sampledIndices = datasample(1:numTrainSamples, 2*numTrainSamples, 'Weights', sampleProbs);
    
    % Use the sampled data
    trainingX = XTrain(:, sampledIndices)';
    trainingY = YTrain(:, sampledIndices)';
else
    % Use original dataset
    trainingX = XTrain';
    trainingY = YTrain';
end

% Train the network
net = trainNetwork(trainingX, trainingY, layers, options);

% === Performance Evaluation with Robust Metrics ===
fprintf('Evaluating model performance with robust metrics...\n');

% Predict weights for all samples
predictedWeights = predict(net, X');
predictedWeights = predictedWeights';

% Calculate prediction error with regularization penalty
predictionError = mean(sum((Y - predictedWeights).^2, 1));

% Calculate expected returns based on targets and predictions
expectedReturnTarget = zeros(1, size(Y, 2));
expectedReturnPrediction = zeros(1, size(Y, 2));

% Calculate portfolio concentration (Herfindahl index)
concentrationTarget = zeros(1, size(Y, 2));
concentrationPrediction = zeros(1, size(Y, 2));

for i = 1:size(Y, 2)
    step = Steps(i);
    if step < size(returns, 2)
        nextReturns = returns(:, step+1);
        expectedReturnTarget(i) = sum(Y(:, i) .* nextReturns);
        expectedReturnPrediction(i) = sum(predictedWeights(:, i) .* nextReturns);
    end
    
    % Calculate concentration - lower is better (more diversified)
    concentrationTarget(i) = sum(Y(:, i).^2);
    concentrationPrediction(i) = sum(predictedWeights(:, i).^2);
end

% Calculate portfolio turnover - measure of stability
turnover = zeros(1, size(Y, 2)-1);
for i = 1:size(Y, 2)-1
    turnover(i) = sum(abs(predictedWeights(:, i+1) - predictedWeights(:, i)))/2;
end

% Calculate performance metrics by regime with robust statistics
regimeStats = struct();
for r = 1:4
    regimeIdx = Regimes == r;
    if any(regimeIdx)
        regimeStats(r).count = sum(regimeIdx);
        regimeStats(r).error = median(sum((Y(:, regimeIdx) - predictedWeights(:, regimeIdx)).^2, 1));
        
        validIdxTarget = ~isnan(expectedReturnTarget(regimeIdx));
        validIdxPred = ~isnan(expectedReturnPrediction(regimeIdx));
        validIdx = validIdxTarget & validIdxPred;
        
        if any(validIdx)
            targetReturns = expectedReturnTarget(regimeIdx(validIdx));
            predReturns = expectedReturnPrediction(regimeIdx(validIdx));
            
            regimeStats(r).targetReturn = median(targetReturns);
            regimeStats(r).predReturn = median(predReturns);
            
            % Calculate robust Sharpe ratio (using MAD instead of std)
            if length(targetReturns) > 5
                regimeStats(r).targetSharpe = median(targetReturns) / (1.4826 * median(abs(targetReturns - median(targetReturns))));
                regimeStats(r).predSharpe = median(predReturns) / (1.4826 * median(abs(predReturns - median(predReturns))));
            else
                regimeStats(r).targetSharpe = NaN;
                regimeStats(r).predSharpe = NaN;
            end
            
            % Calculate concentration
            regimeStats(r).targetConcentration = median(concentrationTarget(regimeIdx(validIdx)));
            regimeStats(r).predConcentration = median(concentrationPrediction(regimeIdx(validIdx)));
        else
            regimeStats(r).targetReturn = NaN;
            regimeStats(r).predReturn = NaN;
            regimeStats(r).targetSharpe = NaN;
            regimeStats(r).predSharpe = NaN;
            regimeStats(r).targetConcentration = NaN;
            regimeStats(r).predConcentration = NaN;
        end
    else
        regimeStats(r).count = 0;
        regimeStats(r).error = NaN;
        regimeStats(r).targetReturn = NaN;
        regimeStats(r).predReturn = NaN;
        regimeStats(r).targetSharpe = NaN;
        regimeStats(r).predSharpe = NaN;
        regimeStats(r).targetConcentration = NaN;
        regimeStats(r).predConcentration = NaN;
    end
end

% Store training information with robust metrics
trainingInfo = struct();
trainingInfo.numSamples = numSamples;
trainingInfo.predictionError = predictionError;
trainingInfo.expectedReturnTarget = median(expectedReturnTarget(~isnan(expectedReturnTarget)));
trainingInfo.expectedReturnPrediction = median(expectedReturnPrediction(~isnan(expectedReturnPrediction)));
trainingInfo.regimeStats = regimeStats;
trainingInfo.regimeCounts = [sum(Regimes == 1), sum(Regimes == 2), sum(Regimes == 3), sum(Regimes == 4)];
trainingInfo.averageTurnover = mean(turnover);
trainingInfo.averageConcentrationTarget = mean(concentrationTarget);
trainingInfo.averageConcentrationPrediction = mean(concentrationPrediction);

% Display summary
fprintf('\n=== Training Results for Regularized Model ===\n');
fprintf('Total samples: %d\n', numSamples);
fprintf('Overall prediction error: %.6f\n', predictionError);
fprintf('Expected return (target): %.2f%%\n', trainingInfo.expectedReturnTarget * 100);
fprintf('Expected return (prediction): %.2f%%\n', trainingInfo.expectedReturnPrediction * 100);
fprintf('Average portfolio concentration (target): %.4f\n', trainingInfo.averageConcentrationTarget);
fprintf('Average portfolio concentration (prediction): %.4f\n', trainingInfo.averageConcentrationPrediction);
fprintf('Average portfolio turnover: %.4f\n', trainingInfo.averageTurnover);
fprintf('\nSamples by regime: [%d, %d, %d, %d]\n', trainingInfo.regimeCounts);

fprintf('\n=== Regime-Specific Performance ===\n');
for r = 1:4
    if regimeStats(r).count > 0
        fprintf('Regime %d (%d samples):\n', r, regimeStats(r).count);
        fprintf('  Prediction error: %.6f\n', regimeStats(r).error);
        fprintf('  Target return: %.2f%%\n', regimeStats(r).targetReturn * 100);
        fprintf('  Predicted return: %.2f%%\n', regimeStats(r).predReturn * 100);
        if ~isnan(regimeStats(r).targetSharpe)
            fprintf('  Target Sharpe: %.2f\n', regimeStats(r).targetSharpe);
            fprintf('  Predicted Sharpe: %.2f\n', regimeStats(r).predSharpe);
        end
        fprintf('  Target concentration: %.4f\n', regimeStats(r).targetConcentration);
        fprintf('  Pred concentration: %.4f\n', regimeStats(r).predConcentration);
    end
end

fprintf('\n✅ Regularized complementary IA model training completed.\n');

end 
 
 
 
 