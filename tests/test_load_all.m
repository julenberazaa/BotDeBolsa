%% test_load_all.m
% This script tests that all main components of the BotDeBolsa project
% can be loaded without errors. It doesn't run simulations, just verifies
% that classes and functions can be initialized.

disp('BotDeBolsa - Testing Module Loading');
disp('==================================');

% Get the root path of the project (parent of tests directory)
rootPath = fileparts(fileparts(mfilename('fullpath')));

% Add all paths with proper path separators
addpath(fullfile(rootPath, 'src', 'strategies'));
addpath(fullfile(rootPath, 'src', 'agents'));
addpath(fullfile(rootPath, 'src', 'envs'));
addpath(fullfile(rootPath, 'src', 'data', 'reader'));
addpath(fullfile(rootPath, 'src', 'utils'));
addpath(fullfile(rootPath, 'data', 'processed'));

% Also add the original locations for backward compatibility
addpath(fullfile(rootPath, 'proyecto'));
addpath(fullfile(rootPath, 'bd_stock price'));

% Initialize error counter
errorCount = 0;
warningCount = 0;

%% Test PortfolioEnv loading
try
    disp('Testing PortfolioEnv...');
    env = PortfolioEnv();
    disp('  ✓ PortfolioEnv loaded successfully');
catch ME
    errorCount = errorCount + 1;
    disp(['  ✗ Error loading PortfolioEnv: ' ME.message]);
end

%% Test RSI strategy loading
try
    disp('Testing RSI strategy...');
    % Create dummy prices with appropriate dimensions for the RSI function
    dummyPrices = cumsum(randn(10, 100)/100) + 100;
    signals = rsi_strategy(dummyPrices(1,:)', 14);
    disp('  ✓ RSI strategy loaded successfully');
catch ME
    errorCount = errorCount + 1;
    disp(['  ✗ Error loading RSI strategy: ' ME.message]);
end

%% Test RSI agent loading
try
    disp('Testing RSI agent...');
    % Create dummy prices with appropriate dimensions
    dummyPrices = cumsum(randn(10, 100)/100) + 100;
    agent = rsi_agent(dummyPrices(1,:)', 14);
    signal = agent.getSignal(15);  % Get signal for a valid time step (after RSI window)
    disp('  ✓ RSI agent loaded successfully');
catch ME
    errorCount = errorCount + 1;
    disp(['  ✗ Error loading RSI agent: ' ME.message]);
end

%% Test plotting module
try
    disp('Testing plotting module...');
    % Create dummy results
    dummyResults = struct();
    dummyResults.Strategy1.equity = 100 + cumsum(randn(50,1));
    dummyResults.Strategy1.returns = diff(dummyResults.Strategy1.equity) ./ dummyResults.Strategy1.equity(1:end-1);
    dummyResults.Strategy1.sharpe = 1.2;
    dummyResults.Strategy1.maxDrawdown = 0.05;
    dummyResults.Strategy1.volatility = 0.1;
    dummyResults.Strategy1.totalReturn = 0.15;
    
    % Create a temporary directory for output
    tempDir = fullfile(tempdir, 'botdebolsa_test');
    if ~exist(tempDir, 'dir')
        mkdir(tempDir);
    end
    
    % Suppress figure display and output during testing
    oldState = warning('off', 'all');
    try
        % Call plotting function with evaluation to suppress output
        evalc('compare_plots(dummyResults, tempDir);');
        disp('  ✓ Plotting module loaded successfully');
    catch ME
        errorCount = errorCount + 1;
        disp(['  ✗ Error in plotting module: ' ME.message]);
    end
    warning(oldState);
end

%% Test data availability
try
    disp('Checking data reader modules...');
    dataFound = false;
    
    % Check multiple possible locations for data files
    dataLocations = {
        fullfile(rootPath, 'data', 'processed', 'ReaderBeginingDLR.mat'),
        fullfile(rootPath, 'src', 'data', 'reader', 'ReaderBeginingDLR.mat'),
        fullfile(rootPath, 'proyecto', 'ReaderBeginingDLR.mat'),
        fullfile(rootPath, 'bd_stock price', 'ReaderBeginingDLR.mat'),
        'ReaderBeginingDLR.mat'
    };
    
    for i = 1:length(dataLocations)
        if exist(dataLocations{i}, 'file')
            disp(['  ✓ Data file found at: ' dataLocations{i}]);
            dataFound = true;
            break;
        end
    end
    
    if ~dataFound
        warningCount = warningCount + 1;
        disp('  ! Warning: Data files not found in any expected location');
    end
catch ME
    warningCount = warningCount + 1;
    disp(['  ! Warning checking data files: ' ME.message]);
end

%% Summary
disp('==================================');
disp(['Test completed with ' num2str(errorCount) ' errors and ' num2str(warningCount) ' warnings.']);

if errorCount == 0
    disp('✓ All modules loaded successfully!');
else
    disp('✗ Some modules failed to load.');
end 
 
 
 
 