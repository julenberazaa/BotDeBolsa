%% EXTENDED VALIDATION OF HYBRID MACD-IA STRATEGY
% This script runs extended validation of the hybrid MACD-IA strategy
% against multiple stock datasets, comparing the original and regularized models.

clc;
clear;
close all;

% Add all required paths
addpath('src/utils');
addpath('src/strategies');
addpath('src/agents');
addpath('src/data');
addpath('proyecto');
addpath('bd_stock price');
addpath('bd_stock price/stocks');
addpath('bd_stock price/etfs');

try
    % Try to add all paths using the utility function
    addPathsIfNeeded();
catch
    warning('addPathsIfNeeded failed. Using manual path addition.');
end

%% Configuration
fprintf('=== EXTENDED VALIDATION OF HYBRID MACD-IA STRATEGY ===\n\n');

% MACD parameters
macdConfig = struct();
macdConfig.fastPeriod = 5;    % Optimal from previous results
macdConfig.slowPeriod = 40;   % Optimal from previous results
macdConfig.signalPeriod = 5;  % Optimal from previous results

% MACD signal filtering
filterConfig = struct();
filterConfig.volumeThreshold = 1.2;      % Consider volume increase of 20% significant
filterConfig.histogramThreshold = 0.001;  % Minimum histogram value for a strong signal
filterConfig.trendConfirmation = true;    % Confirm signals with trend
filterConfig.signalThreshold = 0.3;       % Filter signals below this strength

% Regime detection settings
regimeConfig = struct();
regimeConfig.volatility = struct('window', 20, 'method', 'std', 'threshold', 0.015);
regimeConfig.trend = struct('window', 20, 'method', 'corr', 'threshold', 0.6);

% Strategy options
strategyOptions = struct();
strategyOptions.windowSize = 5;           % Window size for IA inputs
strategyOptions.maxPosition = 0.20;       % Maximum position size (20%)
strategyOptions.useRegimeDetection = true; % Use dynamic regime detection
strategyOptions.cashAllocation = 0.0;     % Minimum cash allocation
strategyOptions.regimeSettings = regimeConfig;
strategyOptions.filterSettings = filterConfig;

%% Find Available Stock Data Files
fprintf('Scanning for available stock data...\n');

% Get all CSV files in the stocks directory
stockFiles = dir('bd_stock price/stocks/*.csv');
stockSymbols = cell(length(stockFiles), 1);

% Extract symbols from filenames
for i = 1:length(stockFiles)
    [~, stockSymbols{i}] = fileparts(stockFiles(i).name);
end

fprintf('Found %d stock symbols for validation.\n', length(stockSymbols));

% Determine which ETFs to use
etfFiles = dir('bd_stock price/etfs/*.csv');
etfSymbols = cell(length(etfFiles), 1);

% Extract symbols from filenames
for i = 1:length(etfFiles)
    [~, etfSymbols{i}] = fileparts(etfFiles(i).name);
end

fprintf('Found %d ETF symbols for validation.\n', length(etfSymbols));

%% Create Test Datasets
fprintf('Creating test datasets...\n');

% Number of datasets to create
numDatasets = 5;
datasetsInfo = cell(numDatasets, 1);

% For each dataset, select 10-20 random assets
for d = 1:numDatasets
    % Determine size of this dataset
    datasetSize = randi([10, 20]);
    
    % Initialize dataset info
    datasetsInfo{d} = struct();
    datasetsInfo{d}.name = sprintf('Dataset_%d', d);
    datasetsInfo{d}.size = datasetSize;
    
    % Decide if we include ETFs (30% chance)
    includeETFs = (rand() < 0.3);
    
    % Select symbols
    if includeETFs
        % Determine mix of stocks and ETFs
        numETFs = max(1, round(datasetSize * 0.3));
        numStocks = datasetSize - numETFs;
        
        % Randomly select stocks and ETFs
        stockIndices = randperm(length(stockSymbols), min(numStocks, length(stockSymbols)));
        etfIndices = randperm(length(etfSymbols), min(numETFs, length(etfSymbols)));
        
        % Combine symbols
        selectedStocks = stockSymbols(stockIndices);
        selectedETFs = etfSymbols(etfIndices);
        selectedSymbols = [selectedStocks; selectedETFs];
        
        datasetsInfo{d}.symbols = selectedSymbols(1:min(datasetSize, length(selectedSymbols)));
        datasetsInfo{d}.includesETFs = true;
    else
        % Only stocks
        stockIndices = randperm(length(stockSymbols), min(datasetSize, length(stockSymbols)));
        datasetsInfo{d}.symbols = stockSymbols(stockIndices);
        datasetsInfo{d}.includesETFs = false;
    end
    
    fprintf('Created dataset %d with %d assets%s\n', d, length(datasetsInfo{d}.symbols), ...
        datasetsInfo{d}.includesETFs ? ' (including ETFs)' : '');
end

%% Process Datasets and Run Tests
fprintf('\nRunning validation tests on multiple datasets...\n');

% Arrays to store results across datasets
totalReturnMACD = zeros(1, numDatasets);
totalReturnIA = zeros(1, numDatasets);
totalReturnIAReg = zeros(1, numDatasets);
totalReturnHybrid = zeros(1, numDatasets);
totalReturnHybridReg = zeros(1, numDatasets);
totalReturnEqual = zeros(1, numDatasets);

sharpeMACD = zeros(1, numDatasets);
sharpeIA = zeros(1, numDatasets);
sharpeIAReg = zeros(1, numDatasets);
sharpeHybrid = zeros(1, numDatasets);
sharpeHybridReg = zeros(1, numDatasets);
sharpeEqual = zeros(1, numDatasets);

drawdownMACD = zeros(1, numDatasets);
drawdownIA = zeros(1, numDatasets);
drawdownIAReg = zeros(1, numDatasets);
drawdownHybrid = zeros(1, numDatasets);
drawdownHybridReg = zeros(1, numDatasets);
drawdownEqual = zeros(1, numDatasets);

% Create structure to store detailed results
detailedResults = struct();

% Process each dataset
for d = 1:numDatasets
    fprintf('\n=== Processing Dataset %d: %s ===\n', d, datasetsInfo{d}.name);
    
    % Load data for selected symbols
    symbols = datasetsInfo{d}.symbols;
    numAssets = length(symbols);
    
    % Initialize arrays to store price and volume data
    allPrices = cell(numAssets, 1);
    allVolumes = cell(numAssets, 1);
    allDates = cell(numAssets, 1);
    
    % Load data for each symbol
    for i = 1:numAssets
        symbol = symbols{i};
        
        % Determine file path
        if datasetsInfo{d}.includesETFs && i > datasetsInfo{d}.size - sum(datasetsInfo{d}.includesETFs)
            filePath = fullfile('bd_stock price/etfs', [symbol '.csv']);
        else
            filePath = fullfile('bd_stock price/stocks', [symbol '.csv']);
        end
        
        % Load data if file exists
        if exist(filePath, 'file')
            try
                % Read CSV file
                data = readtable(filePath);
                
                % Extract columns
                if width(data) >= 6 % Standard format with Date,Open,High,Low,Close,Volume
                    dates = data{:, 1};
                    prices = data{:, 5}; % Use Close price
                    volumes = data{:, 6};
                elseif width(data) >= 5 % Format with Date,Open,High,Low,Close
                    dates = data{:, 1};
                    prices = data{:, 5}; % Use Close price
                    volumes = ones(size(prices)); % Create dummy volume data
                elseif width(data) >= 2 % Minimal format with Date,Price
                    dates = data{:, 1};
                    prices = data{:, 2};
                    volumes = ones(size(prices)); % Create dummy volume data
                else
                    warning('Invalid data format for %s. Skipping.', symbol);
                    continue;
                end
                
                % Store data
                allPrices{i} = prices;
                allVolumes{i} = volumes;
                allDates{i} = dates;
                
                fprintf('Loaded data for %s: %d data points\n', symbol, length(prices));
            catch
                warning('Error loading data for %s. Skipping.', symbol);
            end
        else
            warning('File not found for %s. Skipping.', symbol);
        end
    end
    
    % Remove empty cells (failed loads)
    validIdx = ~cellfun(@isempty, allPrices);
    allPrices = allPrices(validIdx);
    allVolumes = allVolumes(validIdx);
    allDates = allDates(validIdx);
    symbols = symbols(validIdx);
    
    numAssets = length(allPrices);
    
    if numAssets < 5
        warning('Not enough valid assets in dataset %d. Skipping.', d);
        continue;
    end
    
    % Standardize data to common date range
    % Find common date range (assuming dates are datetime objects)
    try
        % Identify overlapping date range
        startDates = cellfun(@(x) x(1), allDates);
        endDates = cellfun(@(x) x(end), allDates);
        
        commonStartDate = max(startDates);
        commonEndDate = min(endDates);
        
        if commonStartDate >= commonEndDate
            warning('No common date range in dataset %d. Skipping.', d);
            continue;
        end
        
        % Create standardized price and volume matrices
        matrixLength = 500; % Use 500 data points
        startIdx = max(1, length(commonStartDate:commonEndDate) - matrixLength + 1);
        
        prices = zeros(numAssets, matrixLength);
        volumes = zeros(numAssets, matrixLength);
        
        for i = 1:numAssets
            % Find indices corresponding to common date range
            dateIdx = find(allDates{i} >= commonStartDate & allDates{i} <= commonEndDate);
            
            if length(dateIdx) >= matrixLength
                % If we have enough data, use the last matrixLength points
                prices(i, :) = allPrices{i}(dateIdx(end-matrixLength+1:end));
                volumes(i, :) = allVolumes{i}(dateIdx(end-matrixLength+1:end));
            else
                % Pad with NaNs if not enough data
                prices(i, end-length(dateIdx)+1:end) = allPrices{i}(dateIdx);
                volumes(i, end-length(dateIdx)+1:end) = allVolumes{i}(dateIdx);
            end
        end
        
        % Replace NaNs and zeros with interpolated values
        for i = 1:numAssets
            nanIdx = isnan(prices(i, :)) | prices(i, :) == 0;
            if any(nanIdx) && ~all(nanIdx)
                validIdx = find(~nanIdx);
                interpIdx = find(nanIdx);
                
                prices(i, nanIdx) = interp1(validIdx, prices(i, validIdx), interpIdx, 'linear', 'extrap');
                volumes(i, nanIdx) = interp1(validIdx, volumes(i, validIdx), interpIdx, 'linear', 'extrap');
            end
        end
        
        % Store dataset information
        datasetsInfo{d}.prices = prices;
        datasetsInfo{d}.volumes = volumes;
        datasetsInfo{d}.validSymbols = symbols;
        
        fprintf('Processed dataset with %d assets and %d time steps.\n', numAssets, matrixLength);
        
        % Run validation test for this dataset
        [results] = run_validation_test(prices, volumes, macdConfig, filterConfig, regimeConfig, strategyOptions);
        
        % Store results
        totalReturnMACD(d) = results.metrics.totalReturnMACD;
        totalReturnIA(d) = results.metrics.totalReturnIA;
        totalReturnIAReg(d) = results.metrics.totalReturnIAReg;
        totalReturnHybrid(d) = results.metrics.totalReturnHybrid;
        totalReturnHybridReg(d) = results.metrics.totalReturnHybridReg;
        totalReturnEqual(d) = results.metrics.totalReturnEqual;
        
        sharpeMACD(d) = results.metrics.sharpeMACD;
        sharpeIA(d) = results.metrics.sharpeIA;
        sharpeIAReg(d) = results.metrics.sharpeIAReg;
        sharpeHybrid(d) = results.metrics.sharpeHybrid;
        sharpeHybridReg(d) = results.metrics.sharpeHybridReg;
        sharpeEqual(d) = results.metrics.sharpeEqual;
        
        drawdownMACD(d) = results.metrics.drawdownMACD;
        drawdownIA(d) = results.metrics.drawdownIA;
        drawdownIAReg(d) = results.metrics.drawdownIAReg;
        drawdownHybrid(d) = results.metrics.drawdownHybrid;
        drawdownHybridReg(d) = results.metrics.drawdownHybridReg;
        drawdownEqual(d) = results.metrics.drawdownEqual;
        
        % Store detailed results
        detailedResults.(sprintf('dataset_%d', d)) = results;
    catch ME
        warning('Error processing dataset %d: %s', d, ME.message);
    end
end

%% Summarize Results
fprintf('\n=== VALIDATION RESULTS SUMMARY ===\n\n');

% Average metrics across datasets
avgTotalReturnMACD = mean(totalReturnMACD(totalReturnMACD ~= 0));
avgTotalReturnIA = mean(totalReturnIA(totalReturnIA ~= 0));
avgTotalReturnIAReg = mean(totalReturnIAReg(totalReturnIAReg ~= 0));
avgTotalReturnHybrid = mean(totalReturnHybrid(totalReturnHybrid ~= 0));
avgTotalReturnHybridReg = mean(totalReturnHybridReg(totalReturnHybridReg ~= 0));
avgTotalReturnEqual = mean(totalReturnEqual(totalReturnEqual ~= 0));

avgSharpeMACD = mean(sharpeMACD(sharpeMACD ~= 0));
avgSharpeIA = mean(sharpeIA(sharpeIA ~= 0));
avgSharpeIAReg = mean(sharpeIAReg(sharpeIAReg ~= 0));
avgSharpeHybrid = mean(sharpeHybrid(sharpeHybrid ~= 0));
avgSharpeHybridReg = mean(sharpeHybridReg(sharpeHybridReg ~= 0));
avgSharpeEqual = mean(sharpeEqual(sharpeEqual ~= 0));

avgDrawdownMACD = mean(drawdownMACD(drawdownMACD ~= 0));
avgDrawdownIA = mean(drawdownIA(drawdownIA ~= 0));
avgDrawdownIAReg = mean(drawdownIAReg(drawdownIAReg ~= 0));
avgDrawdownHybrid = mean(drawdownHybrid(drawdownHybrid ~= 0));
avgDrawdownHybridReg = mean(drawdownHybridReg(drawdownHybridReg ~= 0));
avgDrawdownEqual = mean(drawdownEqual(drawdownEqual ~= 0));

% Display results table
fprintf('Strategy             | Return (%%) | Sharpe  | Drawdown (%%) |\n');
fprintf('---------------------|------------|---------|---------------|\n');
fprintf('Enhanced MACD        | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnMACD, avgSharpeMACD, avgDrawdownMACD);
fprintf('IA Only              | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnIA, avgSharpeIA, avgDrawdownIA);
fprintf('IA Regularized       | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnIAReg, avgSharpeIAReg, avgDrawdownIAReg);
fprintf('Hybrid MACD-IA       | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnHybrid, avgSharpeHybrid, avgDrawdownHybrid);
fprintf('Hybrid MACD-IA Reg   | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnHybridReg, avgSharpeHybridReg, avgDrawdownHybridReg);
fprintf('Equal Weights        | %10.2f | %7.2f | %12.2f |\n', avgTotalReturnEqual, avgSharpeEqual, avgDrawdownEqual);

% Improvement percentage of regularized models over non-regularized
if avgTotalReturnIA ~= 0
    iaImprovement = (avgTotalReturnIAReg - avgTotalReturnIA) / abs(avgTotalReturnIA) * 100;
    fprintf('\nRegularized IA improves return by: %.2f%%\n', iaImprovement);
end

if avgTotalReturnHybrid ~= 0
    hybridImprovement = (avgTotalReturnHybridReg - avgTotalReturnHybrid) / abs(avgTotalReturnHybrid) * 100;
    fprintf('Regularized Hybrid improves return by: %.2f%%\n', hybridImprovement);
end

% Save results
fprintf('\nSaving validation results...\n');

results = struct();
results.datasetInfo = datasetsInfo;
results.detailedResults = detailedResults;
results.summary = struct(...
    'totalReturnMACD', totalReturnMACD, ...
    'totalReturnIA', totalReturnIA, ...
    'totalReturnIAReg', totalReturnIAReg, ...
    'totalReturnHybrid', totalReturnHybrid, ...
    'totalReturnHybridReg', totalReturnHybridReg, ...
    'totalReturnEqual', totalReturnEqual, ...
    'sharpeMACD', sharpeMACD, ...
    'sharpeIA', sharpeIA, ...
    'sharpeIAReg', sharpeIAReg, ...
    'sharpeHybrid', sharpeHybrid, ...
    'sharpeHybridReg', sharpeHybridReg, ...
    'sharpeEqual', sharpeEqual, ...
    'drawdownMACD', drawdownMACD, ...
    'drawdownIA', drawdownIA, ...
    'drawdownIAReg', drawdownIAReg, ...
    'drawdownHybrid', drawdownHybrid, ...
    'drawdownHybridReg', drawdownHybridReg, ...
    'drawdownEqual', drawdownEqual, ...
    'avgTotalReturnMACD', avgTotalReturnMACD, ...
    'avgTotalReturnIA', avgTotalReturnIA, ...
    'avgTotalReturnIAReg', avgTotalReturnIAReg, ...
    'avgTotalReturnHybrid', avgTotalReturnHybrid, ...
    'avgTotalReturnHybridReg', avgTotalReturnHybridReg, ...
    'avgTotalReturnEqual', avgTotalReturnEqual, ...
    'avgSharpeMACD', avgSharpeMACD, ...
    'avgSharpeIA', avgSharpeIA, ...
    'avgSharpeIAReg', avgSharpeIAReg, ...
    'avgSharpeHybrid', avgSharpeHybrid, ...
    'avgSharpeHybridReg', avgSharpeHybridReg, ...
    'avgSharpeEqual', avgSharpeEqual, ...
    'avgDrawdownMACD', avgDrawdownMACD, ...
    'avgDrawdownIA', avgDrawdownIA, ...
    'avgDrawdownIAReg', avgDrawdownIAReg, ...
    'avgDrawdownHybrid', avgDrawdownHybrid, ...
    'avgDrawdownHybridReg', avgDrawdownHybridReg, ...
    'avgDrawdownEqual', avgDrawdownEqual ...
);

% Create directories if they don't exist
logsDir = 'results/validation';
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

% Save results
save(fullfile(logsDir, 'extended_validation_results.mat'), 'results');

fprintf('\n✅ Extended validation of hybrid MACD-IA strategy completed.\n'); 
 
 
 
 