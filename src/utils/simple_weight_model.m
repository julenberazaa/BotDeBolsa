function [weights] = simple_weight_model(inputVector, numAssets)
% SIMPLE_WEIGHT_MODEL - Simple fallback model for portfolio weights
%
% This function provides a simple model to generate portfolio weights
% when the main IA models are unavailable or fail. It implements a basic
% inverse volatility weighting approach.
%
% Inputs:
%   inputVector - Input vector (from a window of returns)
%   numAssets - Number of assets in the portfolio
%
% Output:
%   weights - Portfolio weights (numAssets x 1 vector)

% Reshape input to get the original window structure
% Assuming inputVector is a flattened 'assets x time steps' matrix
if numel(inputVector) ~= numAssets && mod(numel(inputVector), numAssets) == 0
    timeSteps = numel(inputVector) / numAssets;
    window = reshape(inputVector, numAssets, timeSteps);
else
    % If reshaping is not possible, create random weights
    weights = rand(numAssets, 1);
    weights = weights / sum(weights);
    return;
end

% Calculate volatility for each asset
volEstimates = std(window, 0, 2);

% Replace zeros or NaNs with the mean volatility
meanVol = mean(volEstimates(~isnan(volEstimates) & volEstimates > 0));
if isnan(meanVol) || meanVol <= 0
    meanVol = 0.01; % Fallback value
end
volEstimates(isnan(volEstimates) | volEstimates <= 0) = meanVol;

% Inverse volatility weighting
invVol = 1 ./ volEstimates;
weights = invVol / sum(invVol);

% Ensure no NaNs or negative values
weights(isnan(weights)) = 1/numAssets;
weights(weights < 0) = 0;

% Normalize weights to sum to 1
if sum(weights) > 0
    weights = weights / sum(weights);
else
    weights = ones(numAssets, 1) / numAssets;
end

end 