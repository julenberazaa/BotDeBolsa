function [weights] = convert_signals_to_weights(signals, maxPosition)
% CONVERT_SIGNALS_TO_WEIGHTS - Convert MACD signals to portfolio weights
%
% This function takes MACD signals (-1, 0, 1) and converts them to portfolio weights
% respecting the maximum position size constraint.
%
% Inputs:
%   signals - Vector of MACD signals for each asset [-1, 0, 1]
%   maxPosition - Maximum position size for any single asset (default: 0.2)
%
% Output:
%   weights - Portfolio weights (same length as signals)

% Input validation
if nargin < 2 || isempty(maxPosition)
    maxPosition = 0.2; % Default 20% maximum position size
end

% Ensure signals is a column vector
signals = signals(:);
numAssets = length(signals);

% Initialize weights
weights = zeros(numAssets, 1);

% Count buy signals (signal = 1)
buySignals = (signals == 1);
numBuys = sum(buySignals);

if numBuys > 0
    % Allocate to buy signals
    weights(buySignals) = 1 / numBuys;
    
    % Apply position size limits
    for i = 1:numAssets
        if weights(i) > maxPosition
            weights(i) = maxPosition;
        end
    end
end

% Normalize weights to sum to 1
if sum(weights) > 0
    weights = weights / sum(weights);
end

end 