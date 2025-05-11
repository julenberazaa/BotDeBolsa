classdef enhanced_macd_agent < handle
    % ENHANCED_MACD_AGENT - Agent implementation for the Enhanced MACD trading strategy
    %
    % This class provides an interface for using the enhanced MACD strategy in
    % portfolio optimization and simulation environments. It includes additional
    % features like regime detection and adaptive parameters.
    
    properties
        prices
        volumes
        signals
        macdLine
        signalLine
        histogram
        signalStrength
        regimeTypes
        
        fastPeriod
        slowPeriod
        signalPeriod
        useAdaptiveParams
        filterSettings
    end
    
    methods
        function obj = enhanced_macd_agent(prices, volumes, fastPeriod, slowPeriod, signalPeriod, filterSettings, useAdaptiveParams)
            % Constructor for Enhanced MACD Agent
            %
            % Inputs:
            %    prices - Matrix or vector of asset prices
            %    volumes - Matrix or vector of trading volumes (optional)
            %    fastPeriod - Fast EMA period
            %    slowPeriod - Slow EMA period
            %    signalPeriod - Signal line EMA period
            %    filterSettings - Structure with filter settings
            %    useAdaptiveParams - Whether to adapt parameters based on market regime
            
            if nargin < 2 || isempty(volumes)
                volumes = [];
            end
            
            if nargin < 3 || isempty(fastPeriod)
                fastPeriod = 12;
            end
            
            if nargin < 4 || isempty(slowPeriod)
                slowPeriod = 26;
            end
            
            if nargin < 5 || isempty(signalPeriod)
                signalPeriod = 9;
            end
            
            if nargin < 6 || isempty(filterSettings)
                filterSettings = struct(...
                    'volumeThreshold', 1.5, ...
                    'histogramThreshold', 0.001, ...
                    'trendConfirmation', true, ...
                    'signalThreshold', 0.3);
            end
            
            if nargin < 7 || isempty(useAdaptiveParams)
                useAdaptiveParams = false;
            end
            
            obj.prices = prices;
            obj.volumes = volumes;
            obj.fastPeriod = fastPeriod;
            obj.slowPeriod = slowPeriod;
            obj.signalPeriod = signalPeriod;
            obj.filterSettings = filterSettings;
            obj.useAdaptiveParams = useAdaptiveParams;
            
            % Compute MACD signals for all assets
            if size(prices, 1) > size(prices, 2)
                % Single asset as column vector
                [obj.signals, obj.macdLine, obj.signalLine, obj.histogram, obj.signalStrength] = ...
                    enhanced_macd_strategy(prices, volumes, fastPeriod, slowPeriod, signalPeriod, filterSettings);
                obj.regimeTypes = obj.detectMarketRegime();
            else
                % Multiple assets as rows
                numAssets = size(prices, 1);
                numPeriods = size(prices, 2);
                
                obj.signals = zeros(numAssets, numPeriods);
                obj.macdLine = zeros(numAssets, numPeriods);
                obj.signalLine = zeros(numAssets, numPeriods);
                obj.histogram = zeros(numAssets, numPeriods);
                obj.signalStrength = zeros(numAssets, numPeriods);
                obj.regimeTypes = zeros(numAssets, numPeriods);
                
                for asset = 1:numAssets
                    assetPrices = prices(asset, :)';
                    
                    % Extract volumes for this asset if available
                    assetVolumes = [];
                    if ~isempty(volumes) && size(volumes, 1) == numAssets
                        assetVolumes = volumes(asset, :)';
                    end
                    
                    % Determine MACD parameters (adaptive or fixed)
                    assetFast = fastPeriod;
                    assetSlow = slowPeriod;
                    assetSignal = signalPeriod;
                    
                    if obj.useAdaptiveParams
                        [assetFast, assetSlow, assetSignal] = obj.getAdaptiveParameters(assetPrices);
                    end
                    
                    % Calculate MACD for this asset
                    [obj.signals(asset,:), obj.macdLine(asset,:), obj.signalLine(asset,:), ...
                     obj.histogram(asset,:), obj.signalStrength(asset,:)] = ...
                        enhanced_macd_strategy(assetPrices, assetVolumes, assetFast, assetSlow, assetSignal, filterSettings);
                    
                    % Detect market regime
                    obj.regimeTypes(asset,:) = obj.detectMarketRegime(assetPrices);
                end
            end
        end
        
        function signal = getSignal(obj, t, assetIdx)
            % Get the trading signal for time step t and asset index
            %
            % Inputs:
            %    t - Time step (index)
            %    assetIdx - Asset index (default: 1)
            %
            % Outputs:
            %    signal - Trading signal: 1 (buy), -1 (sell), 0 (hold)
            
            if nargin < 3 || isempty(assetIdx)
                assetIdx = 1;
            end
            
            % Return the precomputed signal for this time step and asset
            if t > 0 && size(obj.signals, 2) >= t && assetIdx <= size(obj.signals, 1)
                signal = obj.signals(assetIdx, t);
            else
                warning('Time step or asset index out of range, returning 0');
                signal = 0;
            end
        end
        
        function signals = getSignals(obj, t)
            % Get all asset signals at time step t (alias for getSignalMatrix)
            %
            % Inputs:
            %    t - Time step (index)
            %
            % Outputs:
            %    signals - Vector of signals for all assets
            
            signals = obj.getSignalMatrix(t);
        end
        
        function signals = getSignalMatrix(obj, t)
            % Get all asset signals at time step t
            %
            % Inputs:
            %    t - Time step (index)
            %
            % Outputs:
            %    signals - Vector of signals for all assets
            
            if t > 0 && t <= size(obj.signals, 2)
                signals = obj.signals(:, t);
            else
                warning('Time step out of range, returning zeros');
                signals = zeros(size(obj.signals, 1), 1);
            end
        end
        
        function [weights, cash] = getPortfolioWeights(obj, t, maxPositionSize, diversificationFactor)
            % Get the recommended portfolio weights based on signals
            %
            % Inputs:
            %    t - Time step (index)
            %    maxPositionSize - Maximum weight for a single position (default: 0.2)
            %    diversificationFactor - How much to diversify (0-1, default: 0.7)
            %
            % Outputs:
            %    weights - Vector of portfolio weights
            %    cash - Cash position weight
            
            if nargin < 3 || isempty(maxPositionSize)
                maxPositionSize = 0.2;
            end
            
            if nargin < 4 || isempty(diversificationFactor)
                diversificationFactor = 0.7;
            end
            
            numAssets = size(obj.signals, 1);
            weights = zeros(numAssets, 1);
            
            % Get current signals
            signals = obj.getSignalMatrix(t);
            strength = obj.signalStrength(:, min(t, end));
            
            % Find buy signals
            buySignals = signals == 1;
            
            if any(buySignals)
                % Weight by signal strength if available
                if ~isempty(strength)
                    relativeStrength = strength(buySignals) / sum(strength(buySignals));
                    totalWeight = diversificationFactor;
                    baseWeights = totalWeight * relativeStrength;
                    
                    % Apply position size limits
                    for i = 1:length(baseWeights)
                        if baseWeights(i) > maxPositionSize
                            baseWeights(i) = maxPositionSize;
                        end
                    end
                    
                    % Normalize weights
                    adjustedWeight = sum(baseWeights);
                    if adjustedWeight > 0
                        baseWeights = baseWeights * (totalWeight / adjustedWeight);
                    end
                    
                    % Assign weights to assets with buy signals
                    idx = find(buySignals);
                    for i = 1:length(idx)
                        weights(idx(i)) = baseWeights(i);
                    end
                else
                    % Equal weight distribution if no strength info
                    weights(buySignals) = diversificationFactor / sum(buySignals);
                end
            end
            
            % Ensure no position exceeds maximum size
            for i = 1:numAssets
                if weights(i) > maxPositionSize
                    weights(i) = maxPositionSize;
                end
            end
            
            % Calculate cash position
            cash = 1 - sum(weights);
            
            % Final normalization
            if sum(weights) > 0
                weights = weights / sum(weights) * (1 - cash);
            end
        end
        
        function [regimeTypes] = detectMarketRegime(obj, prices)
            % Detect market regime (trend/volatility characteristics)
            %
            % Outputs:
            %    regimeTypes - Regime classification (1-4)
            %        1: High volatility + Strong trend (MACD favorable)
            %        2: High volatility + Weak trend
            %        3: Low volatility + Strong trend
            %        4: Low volatility + Weak trend (IA favorable)
            
            if nargin < 2 || isempty(prices)
                prices = obj.prices;
            end
            
            if size(prices, 1) > size(prices, 2)
                prices = prices'; % Ensure prices are in row format
            end
            
            % Initialize
            numPeriods = size(prices, 2);
            regimeTypes = zeros(1, numPeriods);
            windowSize = min(20, floor(numPeriods/5));
            
            if windowSize < 5
                % Not enough data for reliable detection
                regimeTypes(:) = 1; % Default to regime 1
                return;
            end
            
            % Calculate volatility and trend for each window
            for t = windowSize+1:numPeriods
                % Get price window
                window = prices(:, t-windowSize+1:t);
                
                % Calculate volatility (standard deviation of returns)
                returns = diff(window, 1, 2) ./ window(:, 1:end-1);
                volatility = mean(std(returns, 0, 2));
                
                % Calculate trend strength (absolute correlation with time)
                timeVector = 1:size(window, 2);
                trendStrength = 0;
                
                for asset = 1:size(window, 1)
                    assetPrices = window(asset, :);
                    if std(assetPrices) > 0
                        corrMatrix = corrcoef(timeVector, assetPrices);
                        trendStrength = trendStrength + abs(corrMatrix(1, 2));
                    end
                end
                trendStrength = trendStrength / size(window, 1);
                
                % Classify regime based on volatility and trend
                highVolThreshold = 0.015;
                highTrendThreshold = 0.6;
                
                if volatility > highVolThreshold && trendStrength > highTrendThreshold
                    regimeTypes(t) = 1; % High vol + Strong trend
                elseif volatility > highVolThreshold
                    regimeTypes(t) = 2; % High vol + Weak trend
                elseif trendStrength > highTrendThreshold
                    regimeTypes(t) = 3; % Low vol + Strong trend
                else
                    regimeTypes(t) = 4; % Low vol + Weak trend
                end
            end
            
            % Fill in initial values
            regimeTypes(1:windowSize) = regimeTypes(windowSize+1);
        end
        
        function [fastPeriod, slowPeriod, signalPeriod] = getAdaptiveParameters(obj, prices)
            % Get adaptive MACD parameters based on price characteristics
            
            if nargin < 2 || isempty(prices)
                prices = obj.prices;
            end
            
            if size(prices, 1) > size(prices, 2)
                prices = prices'; % Ensure prices are in row format
            end
            
            % Default parameters
            fastPeriod = obj.fastPeriod;
            slowPeriod = obj.slowPeriod;
            signalPeriod = obj.signalPeriod;
            
            % Calculate price volatility
            if length(prices) > 30
                returns = diff(prices) ./ prices(1:end-1);
                volatility = std(returns);
                
                % Adjust parameters based on volatility
                if volatility > 0.02 % High volatility
                    % Use shorter periods for faster response in volatile markets
                    fastPeriod = max(5, floor(fastPeriod * 0.8));
                    slowPeriod = max(15, floor(slowPeriod * 0.8));
                    signalPeriod = max(5, floor(signalPeriod * 0.8));
                elseif volatility < 0.008 % Low volatility
                    % Use longer periods for smoother signals in quiet markets
                    fastPeriod = min(20, ceil(fastPeriod * 1.2));
                    slowPeriod = min(40, ceil(slowPeriod * 1.2));
                    signalPeriod = min(15, ceil(signalPeriod * 1.2));
                end
            end
        end
        
        function setData(obj, prices, volumes)
            % Update the price and volume data and recompute signals
            %
            % Inputs:
            %    prices - New matrix or vector of asset prices
            %    volumes - New matrix or vector of volumes (optional)
            
            obj.prices = prices;
            
            if nargin > 2
                obj.volumes = volumes;
            end
            
            % Recompute signals with new data
            if size(prices, 1) > size(prices, 2)
                % Single asset as column vector
                [obj.signals, obj.macdLine, obj.signalLine, obj.histogram, obj.signalStrength] = ...
                    enhanced_macd_strategy(prices, obj.volumes, obj.fastPeriod, obj.slowPeriod, obj.signalPeriod, obj.filterSettings);
                obj.regimeTypes = obj.detectMarketRegime();
            else
                % Multiple assets as rows
                numAssets = size(prices, 1);
                numPeriods = size(prices, 2);
                
                obj.signals = zeros(numAssets, numPeriods);
                obj.macdLine = zeros(numAssets, numPeriods);
                obj.signalLine = zeros(numAssets, numPeriods);
                obj.histogram = zeros(numAssets, numPeriods);
                obj.signalStrength = zeros(numAssets, numPeriods);
                obj.regimeTypes = zeros(numAssets, numPeriods);
                
                for asset = 1:numAssets
                    assetPrices = prices(asset, :)';
                    
                    % Extract volumes for this asset if available
                    assetVolumes = [];
                    if ~isempty(obj.volumes) && size(obj.volumes, 1) == numAssets
                        assetVolumes = obj.volumes(asset, :)';
                    end
                    
                    % Determine MACD parameters (adaptive or fixed)
                    assetFast = obj.fastPeriod;
                    assetSlow = obj.slowPeriod;
                    assetSignal = obj.signalPeriod;
                    
                    if obj.useAdaptiveParams
                        [assetFast, assetSlow, assetSignal] = obj.getAdaptiveParameters(assetPrices);
                    end
                    
                    % Calculate MACD for this asset
                    [obj.signals(asset,:), obj.macdLine(asset,:), obj.signalLine(asset,:), ...
                     obj.histogram(asset,:), obj.signalStrength(asset,:)] = ...
                        enhanced_macd_strategy(assetPrices, assetVolumes, assetFast, assetSlow, assetSignal, obj.filterSettings);
                    
                    % Detect market regime
                    obj.regimeTypes(asset,:) = obj.detectMarketRegime(assetPrices);
                end
            end
        end
    end
end 
 
 