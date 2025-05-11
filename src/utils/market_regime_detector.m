function [regimeType, volatility, trend, regimeWeights] = market_regime_detector(prices, volatilitySettings, trendSettings)
% MARKET_REGIME_DETECTOR - Utility function to detect market regimes
%
% Syntax: [regimeType, volatility, trend, regimeWeights] = market_regime_detector(prices, volatilitySettings, trendSettings)
%
% Description:
%   This function analyzes price data to classify the current market regime based on volatility
%   and trend characteristics. It can be used to adjust trading strategies dynamically
%   based on different market conditions.
%
% Inputs:
%   prices - Matrix of asset prices [assets x time] or vector for single asset
%   volatilitySettings - Structure with volatility calculation settings (optional)
%       .window - Lookback window for volatility calculation (default: 20)
%       .method - Method: 'std' (standard deviation), 'range' (high-low), 'atr' (average true range)
%       .threshold - Volatility threshold for high/low classification (default: 0.015)
%   trendSettings - Structure with trend calculation settings (optional)
%       .window - Lookback window for trend calculation (default: 20) 
%       .method - Method: 'corr' (correlation), 'slope' (linear regression), 'ma' (moving avg)
%       .threshold - Trend strength threshold for classification (default: 0.6)
%
% Outputs:
%   regimeType - Market regime classification:
%       1: High volatility + Strong trend - Trend following (MACD favorable)
%       2: High volatility + Weak trend - Risk-off (reduce exposure) 
%       3: Low volatility + Strong trend - Trend following with higher allocation
%       4: Low volatility + Weak trend - Mean reversion (IA favorable)
%   volatility - Calculated market volatility value
%   trend - Calculated market trend strength
%   regimeWeights - Recommended strategy weights for different regimes:
%       .macdWeight - Weight for MACD strategy (0-1)
%       .iaWeight - Weight for IA strategy (0-1)
%       .cashWeight - Weight for cash (0-1)

% === Parameter Validation ===
% Default settings
if nargin < 2 || isempty(volatilitySettings)
    volatilitySettings = struct();
end

if nargin < 3 || isempty(trendSettings)
    trendSettings = struct();
end

% Set defaults for volatility settings
if ~isfield(volatilitySettings, 'window')
    volatilitySettings.window = 20;
end

if ~isfield(volatilitySettings, 'method')
    volatilitySettings.method = 'std';
end

if ~isfield(volatilitySettings, 'threshold')
    volatilitySettings.threshold = 0.015;
end

% Set defaults for trend settings
if ~isfield(trendSettings, 'window')
    trendSettings.window = 20;
end

if ~isfield(trendSettings, 'method')
    trendSettings.method = 'corr';
end

if ~isfield(trendSettings, 'threshold')
    trendSettings.threshold = 0.6;
end

% === Data Preparation ===
% Ensure prices is in the right format (rows = assets, columns = time)
if size(prices, 1) > size(prices, 2)
    prices = prices'; % Single asset as row vector
end

numAssets = size(prices, 1);
numPeriods = size(prices, 2);

% Ensure we have enough data
volWindow = volatilitySettings.window;
trendWindow = trendSettings.window;
requiredPeriods = max(volWindow, trendWindow) + 1;

if numPeriods < requiredPeriods
    warning('Not enough data for reliable regime detection. Using default regime 1.');
    regimeType = 1;
    volatility = 0;
    trend = 0;
    regimeWeights = struct('macdWeight', 0.85, 'iaWeight', 0.15, 'cashWeight', 0);
    return;
end

% === Volatility Calculation ===
volatility = 0;
volWindowActual = min(volWindow, numPeriods-1);

% Historical volatility collection for adaptive thresholds
allVolatilities = [];

for asset = 1:numAssets
    assetPrices = prices(asset, :);
    
    % Use the entire price history for establishing baseline
    returns = diff(assetPrices) ./ assetPrices(1:end-1);
    
    % Store all calculated volatilities for adaptive thresholding
    if ~isempty(returns)
        assetVol = movstd(returns, volWindowActual);
        allVolatilities = [allVolatilities; assetVol];
    end
    
    % Current window volatility
    assetPricesWindow = prices(asset, end-volWindowActual:end);
    
    switch lower(volatilitySettings.method)
        case 'std'
            % Standard deviation of returns
            returnsWindow = diff(assetPricesWindow) ./ assetPricesWindow(1:end-1);
            assetVol = std(returnsWindow);
        
        case 'range'
            % High-Low range as percentage of average price
            ranges = (max(assetPricesWindow) - min(assetPricesWindow)) / mean(assetPricesWindow);
            assetVol = ranges;
            
        case 'atr'
            % Approximate Average True Range
            highLow = diff(assetPricesWindow);
            atr = mean(abs(highLow));
            assetVol = atr / mean(assetPricesWindow);
            
        otherwise
            % Default to standard deviation
            returnsWindow = diff(assetPricesWindow) ./ assetPricesWindow(1:end-1);
            assetVol = std(returnsWindow);
    end
    
    volatility = volatility + assetVol;
end

% Average volatility across assets
volatility = volatility / numAssets;

% Adaptive threshold calculation - use historical distribution
if ~isempty(allVolatilities)
    % Use quantiles for more robust threshold setting
    volatilitySettings.adaptiveThreshold = quantile(allVolatilities(:), 0.75);
else
    % Default to static threshold if no historical data
    volatilitySettings.adaptiveThreshold = volatilitySettings.threshold;
end

% === Trend Calculation with Similar Improvements ===
trend = 0;
trendWindowActual = min(trendWindow, numPeriods);

% Historical trend strength collection for adaptive thresholds
allTrends = [];

for asset = 1:numAssets
    % Calculate trend for the entire price history
    for t = trendWindowActual:numPeriods
        window = prices(asset, max(1, t-trendWindowActual+1):t);
        timeVec = 1:length(window);
        
        if std(window) > 0
            corrMatrix = corrcoef(timeVec, window);
            trendVal = abs(corrMatrix(1,2));
            allTrends = [allTrends, trendVal];
        end
    end
    
    % Current window trend calculation
    assetPrices = prices(asset, end-trendWindowActual+1:end);
    
    switch lower(trendSettings.method)
        case 'corr'
            % Correlation with time
            timeVec = 1:length(assetPrices);
            if std(assetPrices) > 0
                corrMatrix = corrcoef(timeVec, assetPrices);
                assetTrend = abs(corrMatrix(1,2));
            else
                assetTrend = 0;
            end
            
        case 'slope'
            % Linear regression slope
            timeVec = 1:length(assetPrices);
            if std(assetPrices) > 0
                p = polyfit(timeVec, assetPrices, 1);
                slope = p(1);
                % Normalize slope to 0-1 range (slope * periods / mean price)
                assetTrend = abs(slope * length(assetPrices) / mean(assetPrices));
                assetTrend = min(1, assetTrend); % Cap at 1
            else
                assetTrend = 0;
            end
            
        case 'ma'
            % Moving average crossovers
            if length(assetPrices) >= 10
                shortMA = mean(assetPrices(end-min(5,length(assetPrices)-1):end));
                longMA = mean(assetPrices);
                % Normalize the difference
                assetTrend = abs(shortMA - longMA) / longMA;
                assetTrend = min(1, assetTrend * 10); % Scale and cap
            else
                assetTrend = 0;
            end
            
        otherwise
            % Default to correlation
            timeVec = 1:length(assetPrices);
            if std(assetPrices) > 0
                corrMatrix = corrcoef(timeVec, assetPrices);
                assetTrend = abs(corrMatrix(1,2));
            else
                assetTrend = 0;
            end
    end
    
    trend = trend + assetTrend;
end

% Average trend across assets
trend = trend / numAssets;

% Adaptive threshold calculation for trend
if ~isempty(allTrends)
    % Use quantiles for more robust threshold setting  
    trendSettings.adaptiveThreshold = quantile(allTrends, 0.65);
else
    % Default to static threshold if no historical data
    trendSettings.adaptiveThreshold = trendSettings.threshold;
end

% === Regime Classification with Adaptive Thresholds ===
% Determine regime type based on volatility and trend strength
highVol = volatility >= volatilitySettings.adaptiveThreshold;
strongTrend = trend >= trendSettings.adaptiveThreshold;

% Add some randomization to prevent getting stuck in one regime
if rand() < 0.05 % 5% chance to reassess using different thresholds
    randomFactor = 0.8 + 0.4 * rand(); % 0.8 to 1.2 random factor
    highVol = volatility >= (volatilitySettings.adaptiveThreshold * randomFactor);
    strongTrend = trend >= (trendSettings.adaptiveThreshold * randomFactor);
end

if highVol && strongTrend
    regimeType = 1; % High vol + Strong trend - Trend following (MACD favorable)
elseif highVol && ~strongTrend
    regimeType = 2; % High vol + Weak trend - Risk-off (reduce exposure)
elseif ~highVol && strongTrend
    regimeType = 3; % Low vol + Strong trend - Trend following with higher allocation
else
    regimeType = 4; % Low vol + Weak trend - Mean reversion (IA favorable)
end

% === Calculate Strategy Weights ===
% Base weights for each regime
regimeWeights = struct();

switch regimeType
    case 1
        % High vol + Strong trend - MACD dominant
        regimeWeights.macdWeight = 0.9;
        regimeWeights.iaWeight = 0.1;
        regimeWeights.cashWeight = 0;
    case 2
        % High vol + Weak trend - More cautious, higher cash
        regimeWeights.macdWeight = 0.6;
        regimeWeights.iaWeight = 0.2;
        regimeWeights.cashWeight = 0.2;
    case 3
        % Low vol + Strong trend - MACD with some IA
        regimeWeights.macdWeight = 0.7;
        regimeWeights.iaWeight = 0.3;
        regimeWeights.cashWeight = 0;
    case 4
        % Low vol + Weak trend - IA favorable
        regimeWeights.macdWeight = 0.4;
        regimeWeights.iaWeight = 0.6;
        regimeWeights.cashWeight = 0;
end

% Continuous adjustment - Fine-tune weights based on actual volatility and trend values
% This makes transitions between regimes smoother
if regimeType == 1 || regimeType == 3
    % In trend-following regimes, adjust MACD weight based on trend strength
    trendStrengthFactor = min(1, trend / trendSettings.threshold);
    regimeWeights.macdWeight = regimeWeights.macdWeight * trendStrengthFactor + (1 - trendStrengthFactor) * 0.6;
    regimeWeights.iaWeight = 1 - regimeWeights.macdWeight - regimeWeights.cashWeight;
elseif regimeType == 2
    % In high volatility with weak trend, increase cash based on volatility
    volatilityFactor = min(1, volatility / (volatilitySettings.threshold * 1.5));
    regimeWeights.cashWeight = 0.2 + volatilityFactor * 0.3; % Up to 50% cash in extreme volatility
    regimeWeights.macdWeight = (1 - regimeWeights.cashWeight) * 0.6;
    regimeWeights.iaWeight = 1 - regimeWeights.macdWeight - regimeWeights.cashWeight;
elseif regimeType == 4
    % In mean reversion regime, adjust IA weight based on lack of trend
    trendWeaknessFactor = max(0, 1 - (trend / trendSettings.threshold));
    regimeWeights.iaWeight = 0.4 + trendWeaknessFactor * 0.2;
    regimeWeights.macdWeight = 1 - regimeWeights.iaWeight;
end

end 
 
 
 
 