function [signals, macdLine, signalLine, histogram, signalStrength] = enhanced_macd_strategy(prices, volumes, fastPeriod, slowPeriod, signalPeriod, filterSettings)
% ENHANCED_MACD_STRATEGY - Improved MACD strategy with signal filtering and volume confirmation
%
% Syntax:  [signals, macdLine, signalLine, histogram, signalStrength] = 
%             enhanced_macd_strategy(prices, volumes, fastPeriod, slowPeriod, signalPeriod, filterSettings)
%
% Inputs:
%    prices - Vector of asset prices
%    volumes - Vector of trading volumes (optional, can be empty)
%    fastPeriod - Fast EMA period (default: 12)
%    slowPeriod - Slow EMA period (default: 26)
%    signalPeriod - Signal line EMA period (default: 9)
%    filterSettings - Structure with filter settings (optional)
%        .volumeThreshold - Volume increase percentage threshold (default: 1.5)
%        .histogramThreshold - Minimum histogram value for strong signal (default: 0.001)
%        .trendConfirmation - Whether to confirm with price trend (default: true)
%        .signalThreshold - Filter weak signals below this strength (default: 0.3)
%
% Outputs:
%    signals - Trading signals: 1 (buy), -1 (sell), 0 (hold)
%    macdLine - MACD line values (fast EMA - slow EMA)
%    signalLine - Signal line values
%    histogram - Histogram values (MACD line - signal line)
%    signalStrength - Signal strength values (0-1)

% Set default values if not provided
if nargin < 3 || isempty(fastPeriod)
    fastPeriod = 12;
end

if nargin < 4 || isempty(slowPeriod)
    slowPeriod = 26;
end

if nargin < 5 || isempty(signalPeriod)
    signalPeriod = 9;
end

% Default filter settings
if nargin < 6 || isempty(filterSettings)
    filterSettings = struct();
end

if ~isfield(filterSettings, 'volumeThreshold')
    filterSettings.volumeThreshold = 1.5;
end

if ~isfield(filterSettings, 'histogramThreshold')
    filterSettings.histogramThreshold = 0.001;
end

if ~isfield(filterSettings, 'trendConfirmation')
    filterSettings.trendConfirmation = true;
end

if ~isfield(filterSettings, 'signalThreshold')
    filterSettings.signalThreshold = 0.3;
end

% Ensure prices is a column vector
prices = prices(:);
n = length(prices);

% Handle volume data (if provided)
hasVolume = false;
if nargin >= 2 && ~isempty(volumes)
    volumes = volumes(:);
    if length(volumes) == n
        hasVolume = true;
    else
        warning('Volume data length does not match price data. Volume confirmation disabled.');
    end
end

% Preallocate arrays
signals = zeros(n, 1);
signalStrength = zeros(n, 1);
fastEMA = zeros(n, 1);
slowEMA = zeros(n, 1);
macdLine = zeros(n, 1);
signalLine = zeros(n, 1);
histogram = zeros(n, 1);

% Calculate fast EMA
alpha_fast = 2 / (fastPeriod + 1);
fastEMA(1) = prices(1);
for t = 2:n
    fastEMA(t) = alpha_fast * prices(t) + (1 - alpha_fast) * fastEMA(t-1);
end

% Calculate slow EMA
alpha_slow = 2 / (slowPeriod + 1);
slowEMA(1) = prices(1);
for t = 2:n
    slowEMA(t) = alpha_slow * prices(t) + (1 - alpha_slow) * slowEMA(t-1);
end

% Calculate MACD line (fast EMA - slow EMA)
macdLine = fastEMA - slowEMA;

% Calculate signal line (EMA of MACD line)
alpha_signal = 2 / (signalPeriod + 1);
signalLine(1) = macdLine(1);
for t = 2:n
    signalLine(t) = alpha_signal * macdLine(t) + (1 - alpha_signal) * signalLine(t-1);
end

% Calculate histogram (MACD line - signal line)
histogram = macdLine - signalLine;

% Calculate price trend (for trend confirmation)
if filterSettings.trendConfirmation
    priceMA = zeros(n, 1);
    maPeriod = min(20, floor(n/4)); % Short-term moving average period
    priceMA(1:maPeriod) = prices(1:maPeriod);
    
    for t = maPeriod+1:n
        priceMA(t) = mean(prices(t-maPeriod+1:t));
    end
    
    % Trend direction: 1 = uptrend, -1 = downtrend, 0 = neutral
    trendDirection = zeros(n, 1);
    for t = maPeriod+1:n
        if prices(t) > priceMA(t)
            trendDirection(t) = 1;
        elseif prices(t) < priceMA(t)
            trendDirection(t) = -1;
        end
    end
end

% Calculate volume trend (if volume data is available)
if hasVolume
    volMA = zeros(n, 1);
    volPeriod = min(10, floor(n/5));
    volMA(1:volPeriod) = volumes(1:volPeriod);
    
    for t = volPeriod+1:n
        volMA(t) = mean(volumes(t-volPeriod+1:t));
    end
    
    % Volume spike detection: true if current volume is above average
    volumeSpike = zeros(n, 1);
    for t = volPeriod+1:n
        if volumes(t) > volMA(t) * filterSettings.volumeThreshold
            volumeSpike(t) = 1;
        end
    end
end

% Generate signals based on MACD crossovers with enhanced filtering
for t = 2:n
    % Determine signal strength based on histogram magnitude
    if histogram(t) ~= 0
        % Normalize signal strength to a 0-1 scale
        signalStrength(t) = min(1, abs(histogram(t)) / filterSettings.histogramThreshold);
    end
    
    % Basic MACD crossover signals
    if macdLine(t) > signalLine(t) && macdLine(t-1) <= signalLine(t-1)
        % Bullish crossover (MACD line crosses above signal line)
        signals(t) = 1;  % Buy signal
    elseif macdLine(t) < signalLine(t) && macdLine(t-1) >= signalLine(t-1)
        % Bearish crossover (MACD line crosses below signal line)
        signals(t) = -1; % Sell signal
    end
    
    % Additional zero line crossover signals (can be enabled/disabled)
    if macdLine(t) > 0 && macdLine(t-1) <= 0
        % MACD crossing above zero line - additional buy confirmation
        if signals(t) ~= -1  % Don't override a sell signal from main crossover
            signals(t) = 1;
        end
    elseif macdLine(t) < 0 && macdLine(t-1) >= 0
        % MACD crossing below zero line - additional sell confirmation
        if signals(t) ~= 1  % Don't override a buy signal from main crossover
            signals(t) = -1;
        end
    end
    
    % Apply signal strength filter
    if abs(signalStrength(t)) < filterSettings.signalThreshold
        signals(t) = 0;  % Filter out weak signals
    end
    
    % Apply volume confirmation (if available)
    if hasVolume && signals(t) ~= 0
        if volumeSpike(t) == 0
            signals(t) = 0;  % Filter out signals without volume confirmation
        end
    end
    
    % Apply trend confirmation (if enabled)
    if filterSettings.trendConfirmation && signals(t) ~= 0
        if (signals(t) == 1 && trendDirection(t) == -1) || ...
           (signals(t) == -1 && trendDirection(t) == 1)
            signals(t) = 0;  % Filter signals that go against the trend
        end
    end
end

end 
 
 
 
 