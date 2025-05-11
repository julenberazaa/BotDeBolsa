function [weights, regimeInfo] = hybrid_macd_ia_strategy(prices, volumes, macdAgent, iaModel, currentStep, options)
% HYBRID_MACD_IA_STRATEGY - Combine MACD and IA signals with market regime detection
%
% Syntax: [weights, regimeInfo] = hybrid_macd_ia_strategy(prices, volumes, macdAgent, iaModel, currentStep, options)
%
% Description:
%   This function combines MACD and IA-based signals using dynamic weighting based on
%   market regime detection. It provides an optimal allocation of assets that changes
%   based on market conditions, leveraging the strengths of both strategies.
%
% Inputs:
%   prices - Matrix of asset prices [assets x time]
%   volumes - Matrix of trading volumes [assets x time] (optional)
%   macdAgent - Enhanced MACD agent object or standard MACD signals
%   iaModel - Neural network model for IA signals
%   currentStep - Current time step index
%   options - Structure with strategy options (optional)
%       .windowSize - Input window size for IA model (default: 5)
%       .maxPosition - Maximum weight for a single position (default: 0.20)
%       .useRegimeDetection - Whether to use regime detection (default: true)
%       .cashAllocation - Minimum cash allocation (default: 0.0)
%       .regimeSettings - Settings for regime detection (optional)
%       .filterSettings - Signal filter settings (optional)
%
% Outputs:
%   weights - Calculated portfolio weights [assets x 1]
%   regimeInfo - Information about the detected market regime

% === Parameter Validation ===
if nargin < 6 || isempty(options)
    options = struct();
end

if ~isfield(options, 'windowSize')
    options.windowSize = 5;
end

if ~isfield(options, 'maxPosition')
    options.maxPosition = 0.20;
end

if ~isfield(options, 'useRegimeDetection')
    options.useRegimeDetection = true;
end

if ~isfield(options, 'cashAllocation')
    options.cashAllocation = 0.0;
end

% Ensure cash allocation is in valid range
options.cashAllocation = max(0, min(1, options.cashAllocation));

% Ensure prices are in the right format (assets x time)
if size(prices, 1) > size(prices, 2)
    prices = prices'; % Transpose if in column format
end

[numAssets, numSteps] = size(prices);

% Validate current step
if currentStep > numSteps
    error('Current step (%d) exceeds available data length (%d).', currentStep, numSteps);
end

if currentStep <= options.windowSize
    warning('Current step (%d) is less than or equal to window size (%d). Results may be unreliable.', ...
        currentStep, options.windowSize);
end

% === Market Regime Detection ===
if options.useRegimeDetection
    if isfield(options, 'regimeSettings')
        [regimeType, volatility, trend, regimeWeights] = market_regime_detector(prices(:, 1:currentStep), ...
            options.regimeSettings.volatility, options.regimeSettings.trend);
    else
        [regimeType, volatility, trend, regimeWeights] = market_regime_detector(prices(:, 1:currentStep));
    end
else
    % Default weights without regime detection
    regimeType = 0;
    volatility = 0;
    trend = 0;
    regimeWeights = struct('macdWeight', 0.7, 'iaWeight', 0.3, 'cashWeight', options.cashAllocation);
end

% Store regime information for output
regimeInfo = struct();
regimeInfo.type = regimeType;
regimeInfo.volatility = volatility;
regimeInfo.trend = trend;
regimeInfo.weights = regimeWeights;

% === Get MACD Signals/Weights ===
macdWeights = zeros(numAssets, 1);

% Check if macdAgent is an enhanced_macd_agent object
if isa(macdAgent, 'enhanced_macd_agent')
    % Use the agent's getPortfolioWeights method
    [macdWeights, macdCash] = macdAgent.getPortfolioWeights(currentStep, options.maxPosition);
elseif isnumeric(macdAgent) && size(macdAgent, 1) == numAssets
    % macdAgent is a signal matrix [assets x time]
    signals = macdAgent(:, currentStep);
    
    % Convert signals to weights
    buySignals = signals == 1;
    if any(buySignals)
        % Allocate weights to assets with buy signals
        macdWeights(buySignals) = 1 / sum(buySignals);
    end
    
    % Ensure no position exceeds maximum size
    for i = 1:numAssets
        if macdWeights(i) > options.maxPosition
            macdWeights(i) = options.maxPosition;
        end
    end
    
    % Calculate cash position
    macdCash = 1 - sum(macdWeights);
    
    % Normalize weights if sum exceeds 1
    if sum(macdWeights) > 0
        macdWeights = macdWeights / sum(macdWeights) * (1 - macdCash);
    end
else
    error('Invalid macdAgent. Must be enhanced_macd_agent object or signal matrix.');
end

% === Get IA Signals/Weights ===
iaWeights = zeros(numAssets, 1);

% Prepare input for IA model
windowStart = max(1, currentStep - options.windowSize + 1);
windowEnd = currentStep;

if windowEnd - windowStart + 1 < options.windowSize
    % Not enough data for full window, use available data
    window = prices(:, windowStart:windowEnd);
    % Pad with zeros if necessary
    if size(window, 2) < options.windowSize
        padding = zeros(numAssets, options.windowSize - size(window, 2));
        window = [padding, window];
    end
else
    window = prices(:, windowStart:windowEnd);
end

% Calculate returns for the window
returns = zeros(size(window));
for t = 2:size(window, 2)
    returns(:, t) = (window(:, t) - window(:, t-1)) ./ window(:, t-1);
end

% Normalize input for IA model
inputVector = returns(:);
inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);

% Get IA predictions
try
    if isa(iaModel, 'function_handle')
        % If iaModel is a function handle, call it directly
        iaWeights = iaModel(inputVector);
    else
        % Otherwise assume it's a neural network and use predict
        iaWeights = predict(iaModel, inputVector')';
    end
    
    % Apply position size limits
    for i = 1:numAssets
        if iaWeights(i) > options.maxPosition
            iaWeights(i) = options.maxPosition;
        end
    end
    
    % Normalize weights
    if sum(iaWeights) > 0
        iaWeights = iaWeights / sum(iaWeights);
    else
        % If all weights are zero, use equal weights
        iaWeights = ones(numAssets, 1) / numAssets;
    end
catch
    warning('IA model prediction failed. Using equal weights.');
    iaWeights = ones(numAssets, 1) / numAssets;
end

% === Combine MACD and IA Weights ===
% Apply the regime-based weights to determine the final allocation
% But also consider recent performance of each strategy

% Check if we have historical performance data
if isfield(options, 'performanceHistory') && ~isempty(options.performanceHistory)
    % Extract recent performance data
    macdPerf = options.performanceHistory.macd;
    iaPerf = options.performanceHistory.ia;
    
    % Calculate relative performance (who's doing better recently)
    if ~isempty(macdPerf) && ~isempty(iaPerf)
        % Use last 10 days or whatever is available
        lookback = min(10, length(macdPerf));
        recentMacdPerf = mean(macdPerf(end-lookback+1:end));
        recentIaPerf = mean(iaPerf(end-lookback+1:end));
        
        % Calculate a performance factor (range 0-1)
        if recentMacdPerf > 0 && recentIaPerf > 0
            % Both positive - weight by relative performance
            totalPerf = recentMacdPerf + recentIaPerf;
            perfFactor = recentMacdPerf / totalPerf;
        elseif recentMacdPerf > 0
            % Only MACD positive - favor it more
            perfFactor = 0.8;
        elseif recentIaPerf > 0
            % Only IA positive - favor it more
            perfFactor = 0.2;
        else
            % Both negative - stick with regime weights
            perfFactor = 0.5;
        end
        
        % Adjust regime weights with performance factor (50% regime, 50% recent performance)
        origMacdWeight = regimeWeights.macdWeight;
        perfAdjustedMacdWeight = perfFactor;
        
        % Blend original regime weight with performance-based weight
        regimeWeights.macdWeight = 0.5 * origMacdWeight + 0.5 * perfAdjustedMacdWeight;
        regimeWeights.iaWeight = 1 - regimeWeights.macdWeight - regimeWeights.cashWeight;
    end
end

% Apply adjusted weights
macdContribution = macdWeights * regimeWeights.macdWeight;
iaContribution = iaWeights * regimeWeights.iaWeight;

% Calculate combined weights before cash allocation
combinedWeights = macdContribution + iaContribution;

% Apply diversification factor based on signal agreement
signalAgreement = zeros(numAssets, 1);
for i = 1:numAssets
    % Check if both strategies agree on this asset
    if macdWeights(i) > 0.05 && iaWeights(i) > 0.05
        signalAgreement(i) = 1; % Both strategies agree to buy
    elseif macdWeights(i) < 0.01 && iaWeights(i) < 0.01
        signalAgreement(i) = 1; % Both strategies agree to avoid
    else
        signalAgreement(i) = 0; % Disagreement
    end
end

% Increase weights for assets where strategies agree
agreementFactor = 1.2; % 20% boost on agreement
for i = 1:numAssets
    if signalAgreement(i) == 1 && combinedWeights(i) > 0.01
        combinedWeights(i) = combinedWeights(i) * agreementFactor;
    end
end

% Apply maximum position constraint
for i = 1:numAssets
    if combinedWeights(i) > options.maxPosition
        combinedWeights(i) = options.maxPosition;
    end
end

% Normalize combined weights
if sum(combinedWeights) > 0
    totalWeight = 1 - regimeWeights.cashWeight;
    combinedWeights = combinedWeights / sum(combinedWeights) * totalWeight;
end

% === Signal Conflict Resolution ===
% If MACD and IA signals conflict significantly for an asset, reduce exposure
for i = 1:numAssets
    % Check if one signal is strongly positive and the other is strongly negative
    if (macdWeights(i) > 0.1 && iaWeights(i) < 0.02) || ...
       (iaWeights(i) > 0.1 && macdWeights(i) < 0.02)
        % Signal conflict detected - reduce position
        combinedWeights(i) = combinedWeights(i) * 0.7;
    end
end

% Final renormalization
if sum(combinedWeights) > 0
    totalWeight = 1 - regimeWeights.cashWeight;
    combinedWeights = combinedWeights / sum(combinedWeights) * totalWeight;
end

% Return final weights
weights = combinedWeights;

end 
 
 
 
 