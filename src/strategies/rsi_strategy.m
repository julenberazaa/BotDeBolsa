function [signals] = rsi_strategy(prices, window, overbought, oversold)
% RSI_STRATEGY - Relative Strength Index strategy for generating trading signals
%
% Syntax:  [signals] = rsi_strategy(prices, window, overbought, oversold)
%
% Inputs:
%    prices - Vector of asset prices
%    window - RSI calculation window (default: 14)
%    overbought - Overbought threshold (default: 70)
%    oversold - Oversold threshold (default: 30)
%
% Outputs:
%    signals - Trading signals: 1 (buy), -1 (sell), 0 (hold)
%

% Set default values if not provided
if nargin < 2
    window = 14;
end

if nargin < 3
    overbought = 70;
end

if nargin < 4
    oversold = 30;
end

% Ensure prices is a column vector
prices = prices(:);

% Calculate price changes
changes = diff(prices);
n = length(changes);

% Preallocate arrays
signals = zeros(length(prices), 1);
rsi = zeros(length(prices), 1);

% We need at least window+1 prices to calculate RSI
if length(prices) <= window
    warning('Not enough data to calculate RSI');
    return;
end

% Calculate first RSI
gains = zeros(window, 1);
losses = zeros(window, 1);

for i = 1:window
    if changes(i) > 0
        gains(i) = changes(i);
    elseif changes(i) < 0
        losses(i) = abs(changes(i));
    end
end

avgGain = mean(gains);
avgLoss = mean(losses);

% Avoid division by zero
if avgLoss == 0
    rsi(window+1) = 100;
else
    rs = avgGain / avgLoss;
    rsi(window+1) = 100 - (100 / (1 + rs));
end

% Calculate remaining RSI values
for i = window+1:n
    if changes(i) > 0
        avgGain = (avgGain * (window-1) + changes(i)) / window;
        avgLoss = (avgLoss * (window-1)) / window;
    elseif changes(i) < 0
        avgGain = (avgGain * (window-1)) / window;
        avgLoss = (avgLoss * (window-1) + abs(changes(i))) / window;
    else
        avgGain = (avgGain * (window-1)) / window;
        avgLoss = (avgLoss * (window-1)) / window;
    end
    
    % Avoid division by zero
    if avgLoss == 0
        rsi(i+1) = 100;
    else
        rs = avgGain / avgLoss;
        rsi(i+1) = 100 - (100 / (1 + rs));
    end
    
    % Generate signals based on RSI values
    if rsi(i+1) < oversold
        signals(i+1) = 1;  % Buy signal
    elseif rsi(i+1) > overbought
        signals(i+1) = -1; % Sell signal
    end
end

end 
 
 
 