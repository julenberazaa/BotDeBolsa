function [signals] = macd_strategy(prices, fastPeriod, slowPeriod, signalPeriod)
% MACD_STRATEGY - Basic MACD strategy for compatibility with enhanced models
%
% Syntax:  [signals] = macd_strategy(prices, fastPeriod, slowPeriod, signalPeriod)
%
% Inputs:
%    prices - Vector of asset prices
%    fastPeriod - Fast EMA period (default: 12)
%    slowPeriod - Slow EMA period (default: 26)
%    signalPeriod - Signal line EMA period (default: 9)
%
% Outputs:
%    signals - Trading signals: 1 (buy), -1 (sell), 0 (hold)

% Set default values if not provided
if nargin < 2 || isempty(fastPeriod)
    fastPeriod = 12;
end

if nargin < 3 || isempty(slowPeriod)
    slowPeriod = 26;
end

if nargin < 4 || isempty(signalPeriod)
    signalPeriod = 9;
end

% Ensure prices is a column vector
prices = prices(:);
n = length(prices);

% Preallocate arrays
signals = zeros(n, 1);
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

% Generate signals based on MACD crossovers
for t = 2:n
    % Bullish crossover (MACD line crosses above signal line)
    if macdLine(t) > signalLine(t) && macdLine(t-1) <= signalLine(t-1)
        signals(t) = 1;  % Buy signal
    % Bearish crossover (MACD line crosses below signal line)  
    elseif macdLine(t) < signalLine(t) && macdLine(t-1) >= signalLine(t-1)
        signals(t) = -1; % Sell signal
    end
end

end 
 
 
 
 