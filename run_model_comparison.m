%% MODEL COMPARISON: ORIGINAL VS REGULARIZED IA MODEL
% This script compares the original and regularized IA models, analyzing their
% performance characteristics, portfolio construction, and robustness.

clc;
clear;
close all;

% Add all required paths
addpath('src/utils');
addpath('src/strategies');
addpath('src/agents');
addpath('src/data');
addpath('proyecto');

try
    % Try to add all paths using the utility function
    addPathsIfNeeded();
catch
    warning('addPathsIfNeeded failed. Using manual path addition.');
end

fprintf('=== COMPARING ORIGINAL VS REGULARIZED IA MODELS ===\n\n');

%% Load Data
fprintf('Loading data...\n');

% Try different data files if one fails
dataFiles = {'proyecto/ReaderBeginingDLR.mat', 'proyecto/ReaderBegining.mat'};
dataLoaded = false;

for i = 1:length(dataFiles)
    try
        load(dataFiles{i});
        dataLoaded = true;
        fprintf('Successfully loaded data from %s\n', dataFiles{i});
        break;
    catch
        warning('Failed to load %s. Trying next file...', dataFiles{i});
    end
end

if ~dataLoaded
    error('Failed to load any data file. Please check data files exist and are valid.');
end

% Check if RetornosMedios exists or needs to be created from another variable
if ~exist('RetornosMedios', 'var')
    % Try to find another suitable variable
    if exist('returnData', 'var')
        RetornosMedios = returnData;
    elseif exist('returns', 'var')
        RetornosMedios = returns;
    elseif exist('StockReturns', 'var')
        RetornosMedios = StockReturns;
    elseif exist('prices', 'var')
        % Calculate returns from prices
        fprintf('Calculating returns from prices...\n');
        RetornosMedios = zeros(size(prices));
        for t = 2:size(prices, 2)
            RetornosMedios(:, t) = (prices(:, t) - prices(:, t-1)) ./ prices(:, t-1);
        end
    else
        error('Could not find or create RetornosMedios. Please check data structure.');
    end
end

% Ensure RetornosMedios is in the right format
if size(RetornosMedios, 1) > size(RetornosMedios, 2)
    RetornosMedios = RetornosMedios'; % Transpose to ensure assets are rows
end

[numAssets, numSteps] = size(RetornosMedios);
fprintf('Data prepared with %d assets and %d time steps.\n', numAssets, numSteps);

% Create dummy volume data if not available
try 
    if ~exist('Volumes', 'var')
        % Generate synthetic volume data
        fprintf('No volume data found. Creating synthetic volumes...\n');
        Volumes = ones(size(RetornosMedios));
        
        % Add some random fluctuations to make it more realistic
        for t = 2:numSteps
            volatility = std(RetornosMedios(:, max(1, t-10):t-1), [], 2);
            randomFactor = 1 + 0.5 * randn(numAssets, 1) .* volatility;
            Volumes(:, t) = Volumes(:, t-1) .* max(0.5, randomFactor);
        end
    end
catch
    % Just use constant volumes if anything goes wrong
    warning('Error creating synthetic volumes. Using constant volumes.');
    Volumes = ones(size(RetornosMedios));
end

%% Configuration
% MACD parameters
macdConfig = struct();
macdConfig.fastPeriod = 5;    % Optimal from previous results
macdConfig.slowPeriod = 40;   % Optimal from previous results
macdConfig.signalPeriod = 5;  % Optimal from previous results

% Strategy options
strategyOptions = struct();
strategyOptions.windowSize = 5;           % Window size for IA inputs
strategyOptions.maxPosition = 0.20;       % Maximum position size (20%)

%% Create Enhanced MACD Agent
fprintf('Creating enhanced MACD agent...\n');

% Define training and test periods
trainEndIdx = floor(numSteps * 0.6); % 60% for training
testStartIdx = trainEndIdx + 1;
testEndIdx = numSteps;

trainingData = RetornosMedios(:, 1:trainEndIdx);
trainingVolumes = Volumes(:, 1:trainEndIdx);

%% Train Original IA Model
fprintf('Training original IA model...\n');
try
    tic;
    [originalModel, originalTrainingInfo] = train_ia_complementary(trainingData, macdConfig);
    originalTrainingTime = toc;
    fprintf('Original model training completed in %.2f seconds.\n', originalTrainingTime);
catch ME
    warning('Original model training failed: %s\nUsing fallback.', ME.message);
    originalModel = @(x) simple_weight_model(x, numAssets);
    originalTrainingInfo = struct('message', 'Using fallback model');
    originalTrainingTime = 0;
end

%% Train Regularized IA Model
fprintf('Training regularized IA model...\n');
try
    tic;
    [regularizedModel, regularizedTrainingInfo] = train_ia_complementary_regularized(trainingData, macdConfig);
    regularizedTrainingTime = toc;
    fprintf('Regularized model training completed in %.2f seconds.\n', regularizedTrainingTime);
catch ME
    warning('Regularized model training failed: %s\nUsing fallback.', ME.message);
    regularizedModel = @(x) simple_weight_model(x, numAssets);
    regularizedTrainingInfo = struct('message', 'Using fallback model');
    regularizedTrainingTime = 0;
end

%% Compare Training Performance
fprintf('\nTraining Performance Comparison:\n');
fprintf('Metric                  | Original Model | Regularized Model\n');
fprintf('------------------------|----------------|------------------\n');

if isstruct(originalTrainingInfo) && isstruct(regularizedTrainingInfo)
    fprintf('Training samples         | %14d | %16d\n', ...
        originalTrainingInfo.numSamples, regularizedTrainingInfo.numSamples);
    fprintf('Prediction error         | %14.6f | %16.6f\n', ...
        originalTrainingInfo.predictionError, regularizedTrainingInfo.predictionError);
    fprintf('Expected return (target) | %14.2f%% | %16.2f%%\n', ...
        originalTrainingInfo.expectedReturnTarget*100, regularizedTrainingInfo.expectedReturnTarget*100);
    fprintf('Expected return (pred)   | %14.2f%% | %16.2f%%\n', ...
        originalTrainingInfo.expectedReturnPrediction*100, regularizedTrainingInfo.expectedReturnPrediction*100);
    
    if isfield(regularizedTrainingInfo, 'averageTurnover')
        fprintf('Average turnover        | %14s | %16.4f\n', ...
            'N/A', regularizedTrainingInfo.averageTurnover);
    end
    
    if isfield(regularizedTrainingInfo, 'averageConcentrationTarget')
        fprintf('Portfolio concentration | %14s | %16.4f\n', ...
            'N/A', regularizedTrainingInfo.averageConcentrationPrediction);
    end
    
    fprintf('Training time (seconds) | %14.2f | %16.2f\n', ...
        originalTrainingTime, regularizedTrainingTime);
else
    fprintf('Training information not available for comparison.\n');
end

%% Compare Models on Test Data
fprintf('\nComparing models on test data...\n');

% Prepare test data
testData = RetornosMedios(:, testStartIdx:testEndIdx);
testVolumes = Volumes(:, testStartIdx:testEndIdx);
numTestSteps = size(testData, 2);

% Initialize arrays for portfolio weights and performance metrics
originalWeights = zeros(numAssets, numTestSteps);
regularizedWeights = zeros(numAssets, numTestSteps);
originalReturns = zeros(1, numTestSteps);
regularizedReturns = zeros(1, numTestSteps);
originalTurnover = zeros(1, numTestSteps-1);
regularizedTurnover = zeros(1, numTestSteps-1);
originalConcentration = zeros(1, numTestSteps);
regularizedConcentration = zeros(1, numTestSteps);

% Process each test step
for t = 1:numTestSteps
    % Skip the first few steps where we can't form a full window
    if t < strategyOptions.windowSize
        continue;
    end
    
    % Create input window
    windowStart = max(1, t - strategyOptions.windowSize + 1);
    windowEnd = t;
    window = testData(:, windowStart:windowEnd);
    
    % Normalize window data
    inputVector = window(:);
    inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10);
    
    % Get predictions from both models
    try
        originalPrediction = originalModel(inputVector);
        regularizedPrediction = regularizedModel(inputVector);
        
        % Ensure positive weights that sum to 1
        originalPrediction = max(0, originalPrediction);
        regularizedPrediction = max(0, regularizedPrediction);
        
        if sum(originalPrediction) > 0
            originalPrediction = originalPrediction / sum(originalPrediction);
        else
            originalPrediction = ones(numAssets, 1) / numAssets;
        end
        
        if sum(regularizedPrediction) > 0
            regularizedPrediction = regularizedPrediction / sum(regularizedPrediction);
        else
            regularizedPrediction = ones(numAssets, 1) / numAssets;
        end
        
        % Store weights
        originalWeights(:, t) = originalPrediction;
        regularizedWeights(:, t) = regularizedPrediction;
        
        % Calculate performance metrics
        if t < numTestSteps
            % For each model, calculate return for the next step
            actualReturns = testData(:, t+1);
            originalReturns(t) = sum(originalPrediction .* actualReturns);
            regularizedReturns(t) = sum(regularizedPrediction .* actualReturns);
        end
        
        % Calculate portfolio concentration (Herfindahl index)
        originalConcentration(t) = sum(originalPrediction.^2);
        regularizedConcentration(t) = sum(regularizedPrediction.^2);
        
        % Calculate turnover (if not first step)
        if t > 1
            originalTurnover(t-1) = sum(abs(originalWeights(:, t) - originalWeights(:, t-1)))/2;
            regularizedTurnover(t-1) = sum(abs(regularizedWeights(:, t) - regularizedWeights(:, t-1)))/2;
        end
    catch ME
        warning('Error processing step %d: %s', t, ME.message);
    end
end

%% Calculate Comparison Metrics
fprintf('Calculating comparison metrics...\n');

% Portfolio performance
avgOriginalReturn = mean(originalReturns(~isnan(originalReturns))) * 100;
avgRegularizedReturn = mean(regularizedReturns(~isnan(regularizedReturns))) * 100;

stdOriginalReturn = std(originalReturns(~isnan(originalReturns))) * 100;
stdRegularizedReturn = std(regularizedReturns(~isnan(regularizedReturns))) * 100;

sharpeOriginal = mean(originalReturns(~isnan(originalReturns))) / std(originalReturns(~isnan(originalReturns))) * sqrt(252);
sharpeRegularized = mean(regularizedReturns(~isnan(regularizedReturns))) / std(regularizedReturns(~isnan(regularizedReturns))) * sqrt(252);

% Portfolio construction
avgOriginalTurnover = mean(originalTurnover(~isnan(originalTurnover)));
avgRegularizedTurnover = mean(regularizedTurnover(~isnan(regularizedTurnover)));

avgOriginalConcentration = mean(originalConcentration(~isnan(originalConcentration)));
avgRegularizedConcentration = mean(regularizedConcentration(~isnan(regularizedConcentration)));

% Calculate number of active positions
originalActivePositions = zeros(1, numTestSteps);
regularizedActivePositions = zeros(1, numTestSteps);

for t = 1:numTestSteps
    originalActivePositions(t) = sum(originalWeights(:, t) > 0.01);
    regularizedActivePositions(t) = sum(regularizedWeights(:, t) > 0.01);
end

avgOriginalActivePositions = mean(originalActivePositions(~isnan(originalActivePositions)));
avgRegularizedActivePositions = mean(regularizedActivePositions(~isnan(regularizedActivePositions)));

%% Display Comparison Results
fprintf('\n=== MODEL COMPARISON RESULTS ===\n\n');
fprintf('Performance Metrics          | Original Model | Regularized Model | Difference\n');
fprintf('------------------------------|----------------|------------------|------------\n');
fprintf('Average Daily Return         | %14.4f%% | %16.4f%% | %10.4f%%\n', ...
    avgOriginalReturn, avgRegularizedReturn, avgRegularizedReturn-avgOriginalReturn);
fprintf('Return Volatility            | %14.4f%% | %16.4f%% | %10.4f%%\n', ...
    stdOriginalReturn, stdRegularizedReturn, stdRegularizedReturn-stdOriginalReturn);
fprintf('Sharpe Ratio (annualized)    | %14.4f | %16.4f | %10.4f\n', ...
    sharpeOriginal, sharpeRegularized, sharpeRegularized-sharpeOriginal);

fprintf('\nPortfolio Construction Metrics | Original Model | Regularized Model | Difference\n');
fprintf('------------------------------|----------------|------------------|------------\n');
fprintf('Average Turnover             | %14.4f | %16.4f | %10.4f\n', ...
    avgOriginalTurnover, avgRegularizedTurnover, avgRegularizedTurnover-avgOriginalTurnover);
fprintf('Portfolio Concentration      | %14.4f | %16.4f | %10.4f\n', ...
    avgOriginalConcentration, avgRegularizedConcentration, avgRegularizedConcentration-avgOriginalConcentration);
fprintf('Average Active Positions     | %14.1f | %16.1f | %10.1f\n', ...
    avgOriginalActivePositions, avgRegularizedActivePositions, avgRegularizedActivePositions-avgOriginalActivePositions);

% Calculate correlation between model returns
returnCorrelation = corrcoef(originalReturns(~isnan(originalReturns) & ~isnan(regularizedReturns)), ...
                           regularizedReturns(~isnan(originalReturns) & ~isnan(regularizedReturns)));
fprintf('\nReturn Correlation: %.4f\n', returnCorrelation(1,2));

% Calculate agreement on position direction
positionAgreement = zeros(1, numTestSteps);
for t = 1:numTestSteps
    % Count positions where both models agree
    agreement = sum((originalWeights(:, t) > 0.01) & (regularizedWeights(:, t) > 0.01));
    totalPositions = sum((originalWeights(:, t) > 0.01) | (regularizedWeights(:, t) > 0.01));
    
    if totalPositions > 0
        positionAgreement(t) = agreement / totalPositions;
    else
        positionAgreement(t) = NaN;
    end
end

avgPositionAgreement = mean(positionAgreement(~isnan(positionAgreement))) * 100;
fprintf('Position Agreement: %.2f%%\n', avgPositionAgreement);

% Calculate improvement from regularization
if avgOriginalReturn ~= 0
    returnImprovement = (avgRegularizedReturn - avgOriginalReturn) / abs(avgOriginalReturn) * 100;
    fprintf('Return Improvement: %.2f%%\n', returnImprovement);
end

if avgOriginalTurnover ~= 0
    turnoverImprovement = (avgRegularizedTurnover - avgOriginalTurnover) / abs(avgOriginalTurnover) * 100;
    fprintf('Turnover Change: %.2f%%\n', turnoverImprovement);
end

if avgOriginalConcentration ~= 0
    concentrationImprovement = (avgRegularizedConcentration - avgOriginalConcentration) / abs(avgOriginalConcentration) * 100;
    fprintf('Concentration Change: %.2f%%\n', concentrationImprovement);
end

%% Visualizations
fprintf('\nCreating visualizations...\n');

% Create figure directory if it doesn't exist
figDir = 'results/figures';
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Figure 1: Cumulative Returns
figure('Position', [100, 100, 1000, 600]);
subplot(2, 1, 1);
cumOriginal = cumprod(1 + originalReturns);
cumRegularized = cumprod(1 + regularizedReturns);
plot(cumOriginal, 'b', 'LineWidth', 1.5);
hold on;
plot(cumRegularized, 'r', 'LineWidth', 1.5);
title('Cumulative Returns Comparison', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
legend({'Original Model', 'Regularized Model'}, 'Location', 'best');
grid on;

% Drawdown comparison
subplot(2, 1, 2);
ddOriginal = (cummax(cumOriginal) - cumOriginal) ./ cummax(cumOriginal) * 100;
ddRegularized = (cummax(cumRegularized) - cumRegularized) ./ cummax(cumRegularized) * 100;
plot(ddOriginal, 'b', 'LineWidth', 1.5);
hold on;
plot(ddRegularized, 'r', 'LineWidth', 1.5);
title('Drawdown Comparison', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Drawdown (%)', 'FontSize', 12);
legend({'Original Model', 'Regularized Model'}, 'Location', 'best');
grid on;

saveas(gcf, fullfile(figDir, 'model_returns_comparison.png'));

% Figure 2: Portfolio Construction Metrics
figure('Position', [100, 100, 1000, 800]);
subplot(3, 1, 1);
plot(originalTurnover, 'b', 'LineWidth', 1);
hold on;
plot(regularizedTurnover, 'r', 'LineWidth', 1);
title('Portfolio Turnover', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Turnover', 'FontSize', 12);
legend({'Original Model', 'Regularized Model'}, 'Location', 'best');
grid on;

subplot(3, 1, 2);
plot(originalConcentration, 'b', 'LineWidth', 1);
hold on;
plot(regularizedConcentration, 'r', 'LineWidth', 1);
title('Portfolio Concentration', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Herfindahl Index', 'FontSize', 12);
legend({'Original Model', 'Regularized Model'}, 'Location', 'best');
grid on;

subplot(3, 1, 3);
plot(originalActivePositions, 'b', 'LineWidth', 1);
hold on;
plot(regularizedActivePositions, 'r', 'LineWidth', 1);
title('Active Positions', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Number of Positions', 'FontSize', 12);
legend({'Original Model', 'Regularized Model'}, 'Location', 'best');
grid on;

saveas(gcf, fullfile(figDir, 'model_portfolio_metrics.png'));

% Figure 3: Weight Distribution Comparison
figure('Position', [100, 100, 1000, 600]);

% Select a sample point halfway through testing
sampleIdx = floor(numTestSteps/2);
originalSample = originalWeights(:, sampleIdx);
regularizedSample = regularizedWeights(:, sampleIdx);

% Sort by weight for better visualization
[sortedOriginal, idxOriginal] = sort(originalSample, 'descend');
[sortedRegularized, idxRegularized] = sort(regularizedSample, 'descend');

% Display top 10 positions or fewer if less assets
numToShow = min(10, numAssets);

subplot(1, 2, 1);
bar(sortedOriginal(1:numToShow), 'FaceColor', [0.3 0.6 0.9]);
title('Original Model - Top Positions', 'FontSize', 14);
xlabel('Asset Rank', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
grid on;

subplot(1, 2, 2);
bar(sortedRegularized(1:numToShow), 'FaceColor', [0.9 0.3 0.3]);
title('Regularized Model - Top Positions', 'FontSize', 14);
xlabel('Asset Rank', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
grid on;

sgtitle(sprintf('Portfolio Weight Distribution Comparison (Day %d)', sampleIdx), 'FontSize', 16);
saveas(gcf, fullfile(figDir, 'model_weight_distribution.png'));

% Figure 4: Position Agreement Over Time
figure('Position', [100, 100, 1000, 400]);
plot(positionAgreement * 100, 'k', 'LineWidth', 1.5);
hold on;
plot(ones(size(positionAgreement)) * avgPositionAgreement, 'r--', 'LineWidth', 1);
title('Position Agreement Between Models', 'FontSize', 14);
xlabel('Trading Day', 'FontSize', 12);
ylabel('Agreement (%)', 'FontSize', 12);
grid on;
legend({'Daily Agreement', 'Average Agreement'}, 'Location', 'best');
saveas(gcf, fullfile(figDir, 'model_position_agreement.png'));

fprintf('\n✅ Model comparison completed.\n');
fprintf('Results visualizations saved to %s\n', figDir); 
 
 
 
 