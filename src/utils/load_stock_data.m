function [prices, volumes, dates] = load_stock_data(symbolList, stockDir, etfDir, maxSteps)
% LOAD_STOCK_DATA - Load stock price data from CSV files
%
% This function loads price and volume data for a list of symbols from CSV files.
%
% Inputs:
%   symbolList - Cell array of stock/ETF symbols to load
%   stockDir - Directory containing stock CSV files (default: 'bd_stock price/stocks')
%   etfDir - Directory containing ETF CSV files (default: 'bd_stock price/etfs')
%   maxSteps - Maximum number of time steps to return (default: all available)
%
% Outputs:
%   prices - Matrix of prices [assets x time]
%   volumes - Matrix of volumes [assets x time]
%   dates - Cell array of dates [assets x time]

% Set default parameters
if nargin < 2 || isempty(stockDir)
    stockDir = 'bd_stock price/stocks';
end

if nargin < 3 || isempty(etfDir)
    etfDir = 'bd_stock price/etfs';
end

if nargin < 4
    maxSteps = Inf;
end

% Initialize arrays to store data
numSymbols = length(symbolList);
allPrices = cell(numSymbols, 1);
allVolumes = cell(numSymbols, 1);
allDates = cell(numSymbols, 1);

fprintf('Loading data for %d symbols...\n', numSymbols);

% Load data for each symbol
for i = 1:numSymbols
    symbol = symbolList{i};
    
    % Try stock directory first, then ETF directory
    stockFilePath = fullfile(stockDir, [symbol '.csv']);
    etfFilePath = fullfile(etfDir, [symbol '.csv']);
    
    filePath = '';
    if exist(stockFilePath, 'file')
        filePath = stockFilePath;
    elseif exist(etfFilePath, 'file')
        filePath = etfFilePath;
    end
    
    % Load data if file exists
    if ~isempty(filePath)
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
        catch ME
            warning('Error loading data for %s: %s. Skipping.', symbol, ME.message);
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
loadedSymbols = symbolList(validIdx);

numAssets = length(allPrices);
fprintf('Successfully loaded data for %d symbols.\n', numAssets);

if numAssets == 0
    error('No valid data loaded for any symbols.');
end

% Find the common date range
try
    % Convert dates to datetime if they aren't already
    if ~isa(allDates{1}, 'datetime')
        for i = 1:numAssets
            allDates{i} = datetime(allDates{i}, 'InputFormat', 'yyyy-MM-dd', 'ConvertFrom', 'string');
        end
    end
    
    % Find common date range
    startDates = cellfun(@(x) x(1), allDates);
    endDates = cellfun(@(x) x(end), allDates);
    
    commonStartDate = max(startDates);
    commonEndDate = min(endDates);
    
    fprintf('Common date range: %s to %s\n', string(commonStartDate), string(commonEndDate));
    
    % Create standardized price and volume matrices
    desiredLength = min(maxSteps, datenum(commonEndDate) - datenum(commonStartDate) + 1);
    if desiredLength <= 0
        error('No common date range across symbols.');
    end
    
    prices = zeros(numAssets, desiredLength);
    volumes = zeros(numAssets, desiredLength);
    dates = cell(1, desiredLength);
    
    % Convert to datetime array for easier processing
    dateArray = commonStartDate:days(1):commonEndDate;
    dateArray = dateArray(1:min(length(dateArray), desiredLength));
    
    % Fill in the data matrices
    for i = 1:numAssets
        % Find indices corresponding to common date range
        [~, assetIdxs, dateIdxs] = intersect(allDates{i}, dateArray);
        
        % Fill in data at the appropriate indices
        prices(i, dateIdxs) = allPrices{i}(assetIdxs);
        volumes(i, dateIdxs) = allVolumes{i}(assetIdxs);
    end
    
    % Convert dateArray to cell array of strings
    for i = 1:length(dateArray)
        dates{i} = string(dateArray(i));
    end
    
    % Replace NaNs with interpolated values
    for i = 1:numAssets
        % Get valid price indices
        validIdx = ~isnan(prices(i, :)) & prices(i, :) ~= 0;
        
        if sum(validIdx) > 1
            % Create interpolation function
            xValid = find(validIdx);
            yValid = prices(i, validIdx);
            
            % Find NaN indices
            nanIdx = find(~validIdx);
            
            % Interpolate NaN values
            if ~isempty(nanIdx) && ~isempty(xValid)
                prices(i, nanIdx) = interp1(xValid, yValid, nanIdx, 'linear', 'extrap');
                
                % Interpolate volumes as well
                yValidVol = volumes(i, validIdx);
                volumes(i, nanIdx) = interp1(xValid, yValidVol, nanIdx, 'linear', 'extrap');
            end
        end
    end
    
    fprintf('Processed data with %d assets and %d time steps.\n', numAssets, desiredLength);
catch ME
    % Fallback method for incompatible date formats
    warning('Date standardization failed: %s. Using simplified approach.', ME.message);
    
    % Get lengths of each time series
    lengths = cellfun(@length, allPrices);
    desiredLength = min(min(lengths), maxSteps);
    
    % Create output matrices using the last desiredLength points
    prices = zeros(numAssets, desiredLength);
    volumes = zeros(numAssets, desiredLength);
    dates = cell(1, desiredLength);
    
    for i = 1:numAssets
        % Use the most recent data points
        dataLength = length(allPrices{i});
        startIdx = max(1, dataLength - desiredLength + 1);
        endIdx = dataLength;
        
        % Number of points to extract
        pointsToExtract = min(desiredLength, endIdx - startIdx + 1);
        
        % Extract data
        prices(i, end-pointsToExtract+1:end) = allPrices{i}(endIdx-pointsToExtract+1:endIdx);
        volumes(i, end-pointsToExtract+1:end) = allVolumes{i}(endIdx-pointsToExtract+1:endIdx);
        
        % Use dates from the first valid asset
        if i == 1
            dateEnd = length(allDates{i});
            dateStart = max(1, dateEnd - desiredLength + 1);
            datePoints = min(desiredLength, dateEnd - dateStart + 1);
            
            for j = 1:datePoints
                dates{end-datePoints+j} = string(allDates{i}(dateEnd-datePoints+j));
            end
        end
    end
    
    fprintf('Processed data with %d assets and %d time steps (simplified method).\n', numAssets, desiredLength);
end

% Return final data
fprintf('Data loading complete.\n');
end 
 
 
 
 