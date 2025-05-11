function [w] = obtenerSPO(rMedios, vRetornos, alpha)
% OBTENERSPO - Simplified Stochastic Portfolio Optimization
%
% This is a simplified version of the SPO algorithm that optimizes
% portfolio weights based on return and risk characteristics.
%
% Inputs:
%   rMedios - Vector of expected returns for each asset
%   vRetornos - Vector of return variances for each asset
%   alpha - Risk aversion parameter (default: 0.1)
%
% Output:
%   w - Optimized portfolio weights
%
% Usage:
%   weights = obtenerSPO(expectedReturns, variances, 0.1);
%
% Note: This function is required by both original and regularized IA models

% Input validation with improved error handling
if nargin < 3 || isempty(alpha)
    alpha = 0.1;
end

% Ensure inputs are column vectors
rMedios = rMedios(:);
vRetornos = vRetornos(:);

if length(rMedios) ~= length(vRetornos)
    error('Return and variance vectors must have the same length');
end

% Handle NaN or Inf values in inputs
if any(isnan(rMedios)) || any(isinf(rMedios)) || any(isnan(vRetornos)) || any(isinf(vRetornos))
    warning('NaN or Inf values detected in inputs. Replacing with reasonable values.');
    rMedios(isnan(rMedios) | isinf(rMedios)) = 0;
    vRetornos(isnan(vRetornos)) = mean(vRetornos(~isnan(vRetornos)));
    vRetornos(isinf(vRetornos)) = max(vRetornos(~isinf(vRetornos))) * 10;
end

% Ensure all variances are positive
vRetornos = max(vRetornos, 1e-6);

% If we have few assets, just use the PSO approach
numAssets = length(rMedios);

if numAssets <= 5
    % For very few assets, use a simple approach
    w = simple_weight_optimization(rMedios, vRetornos, alpha);
    return;
end

% Parameters for the PSO algorithm
nP = 100;                  % Number of particles
nIterations = 50;          % Number of iterations
w_inertia = 0.1;           % Inertia
phi1Max = 0.2;             % Personal best influence
phi2Max = 0.2;             % Global best influence

% Initialize particles
x = zeros(numAssets, nP);   % Particle positions
v = randn(numAssets, nP) * 0.1; % Particle velocities
costes = inf(1, nP);         % Costs

% Initialize with valid random weights
for i = 1:nP
    x(:, i) = rand(numAssets, 1);
    x(:, i) = x(:, i) / sum(x(:, i));
    costes(i) = calcularCoste(x(:, i), alpha, rMedios, vRetornos);
end

% Initialize best positions
xOptimo = x;
costesOptimos = costes;
[costeOptimoGlobal, idxBest] = min(costesOptimos);
xOptimoGlobal = xOptimo(:, idxBest);

% Main PSO loop
try
    for t = 1:nIterations
        for i = 1:nP
            % Update velocity
            phi1 = rand * phi1Max;
            phi2 = rand * phi2Max;
            v(:, i) = w_inertia*v(:, i) + phi1*(xOptimo(:, i) - x(:, i)) + phi2*(xOptimoGlobal - x(:, i));
            
            % Update position
            x(:, i) = x(:, i) + v(:, i);
            
            % Enforce constraints
            x(:, i) = max(0, x(:, i));  % Non-negative weights
            x(:, i) = x(:, i) / sum(x(:, i));  % Sum to 1
            
            % Calculate cost
            coste = calcularCoste(x(:, i), alpha, rMedios, vRetornos);
            
            % Update best positions
            if coste < costesOptimos(i)
                costesOptimos(i) = coste;
                xOptimo(:, i) = x(:, i);
                
                if coste < costeOptimoGlobal
                    costeOptimoGlobal = coste;
                    xOptimoGlobal = x(:, i);
                end
            end
        end
    end
catch ME
    warning('SPO optimization failed: %s. Using fallback method.', ME.message);
    % Fallback to inverse variance weighting if PSO fails
    invVol = 1 ./ (vRetornos + 1e-6);
    xOptimoGlobal = invVol / sum(invVol);
end

% Enforce minimum weight threshold (remove very small allocations)
minWeight = 0.01;
xOptimoGlobal(xOptimoGlobal < minWeight) = 0;

% Re-normalize
if sum(xOptimoGlobal) > 0
    xOptimoGlobal = xOptimoGlobal / sum(xOptimoGlobal);
else
    % Fallback to equal weights if all weights are zeroed
    xOptimoGlobal = ones(numAssets, 1) / numAssets;
end

% Return optimized weights
w = xOptimoGlobal;

end

% Cost function
function [cost] = calcularCoste(w, alpha, rMedios, vRetornos)
    % Calculate cost function for SPO
    % Objective: Minimize risk adjusted by expected return
    % Higher alpha values prioritize return over risk
    
    w = w(:);
    rMedios = rMedios(:);
    vRetornos = vRetornos(:);
    
    % Calculate portfolio variance
    portfolioVar = sum((w.^2) .* vRetornos);
    
    % Calculate expected return
    expectedReturn = w' * rMedios;
    
    % Calculate cost (lower is better)
    cost = portfolioVar - alpha * expectedReturn;
    
    % Return very high cost for invalid portfolios
    if any(w < 0) || any(w > 1) || abs(sum(w) - 1) > 0.01
        cost = 1e6;
    end
end

% Simple weight optimization for small asset numbers
function [w] = simple_weight_optimization(rMedios, vRetornos, alpha)
    numAssets = length(rMedios);
    
    % For very few assets, enumerate more possibilities
    steps = 5;  % Steps between 0 and 1
    
    if numAssets == 1
        w = 1;
        return;
    elseif numAssets == 2
        % For 2 assets, just try different allocations
        bestCost = inf;
        bestW = [0.5; 0.5];
        
        for i = 0:steps
            w1 = i/steps;
            w2 = 1 - w1;
            w = [w1; w2];
            cost = calcularCoste(w, alpha, rMedios, vRetornos);
            
            if cost < bestCost
                bestCost = cost;
                bestW = w;
            end
        end
        
        w = bestW;
    else
        % For 3-5 assets, use inverse variance weighting with return adjustment
        invVar = 1 ./ (vRetornos + 1e-10);
        returnFactor = max(0, rMedios);
        
        % Adjust weights by returns
        combinedWeight = invVar .* (1 + alpha * returnFactor);
        
        % Normalize
        w = combinedWeight / sum(combinedWeight);
    end
end 