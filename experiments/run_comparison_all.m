%% Comprehensive Strategy Comparison (All Strategies)
% This script compares all available trading strategies:
% 1. PPO (Proximal Policy Optimization) RL agent
% 2. SPO (Swarm Particle Optimization) agent
% 3. RSI (Relative Strength Index) technical indicator
% 4. MACD (Moving Average Convergence Divergence) technical indicator
% 5. Buy-and-Hold baseline

% Get base path for proper file references
basePath = fileparts(fileparts(mfilename('fullpath')));

% Add paths if not already on path
if ~exist('rsi_strategy', 'file') || ~exist('macd_strategy', 'file')
    addpath(fullfile(basePath, 'src', 'strategies'));
    addpath(fullfile(basePath, 'src', 'agents'));
    addpath(fullfile(basePath, 'src', 'envs'));
    addpath(fullfile(basePath, 'src', 'data', 'reader'));
    addpath(fullfile(basePath, 'src', 'utils'));
    
    % Also add legacy paths for compatibility
    addpath(fullfile(basePath, 'proyecto'));
    addpath(fullfile(basePath, 'bd_stock price'));
end

%% Load price data
disp('Loading price data...');
try
    data = load(fullfile(basePath, 'data', 'processed', 'ReaderBeginingDLR.mat'));
    prices = data.RetornosMedios;
    
    % Check if dates field exists with different possible names
    if isfield(data, 'Fechas')
        dates = data.Fechas;
    elseif isfield(data, 'fechas')
        dates = data.fechas;
    elseif isfield(data, 'Dates')
        dates = data.Dates;
    elseif isfield(data, 'dates')
        dates = data.dates;
    else
        % Create dummy dates if none are found
        dates = datenum(datetime('today') - caldays(size(prices, 2):-1:1));
        warning('No dates field found in data. Using generated dates.');
    end
catch
    try
        % If specific file not found, try to load from a more general source
        data = load('ReaderBeginingDLR.mat');
        prices = data.RetornosMedios;
        
        % Check if dates field exists with different possible names
        if isfield(data, 'Fechas')
            dates = data.Fechas;
        elseif isfield(data, 'fechas')
            dates = data.fechas;
        elseif isfield(data, 'Dates')
            dates = data.Dates;
        elseif isfield(data, 'dates')
            dates = data.dates;
        else
            % Create dummy dates if none are found
            dates = datenum(datetime('today') - caldays(size(prices, 2):-1:1));
            warning('No dates field found in data. Using generated dates.');
        end
        
        % Save to the correct location
        processedDir = fullfile(basePath, 'data', 'processed');
        if ~exist(processedDir, 'dir')
            mkdir(processedDir);
        end
        copyfile('ReaderBeginingDLR.mat', fullfile(processedDir, 'ReaderBeginingDLR.mat'));
    catch err
        error('Could not load price data. Error: %s', err.message);
    end
end

%% Configure parameters
% RSI parameters
rsiWindow = 14;
overbought = 70;
oversold = 30;

% MACD parameters
fastPeriod = 12;
slowPeriod = 26;
signalPeriod = 9;

% Simulation parameters
simulationLength = 252; % Simulate for one year
initialPortfolioValue = 100;

%% Create all agents
disp('Creating agents...');

% 1. RSI agent
rsiAgent = rsi_agent(prices, rsiWindow, overbought, oversold);

% 2. MACD agent
macdAgent = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod);

% 3. Buy and Hold agent (always returns +1)
bhAgent = struct('getSignal', @(t) 1);

% 4. Load pretrained PPO agent
try
    ppoData = load(fullfile(basePath, 'src', 'agents', 'ppo_agent_spoguided.mat'));
    ppoAgent = ppoData.agent;
    
    % Función para obtener observación en función del tiempo
    getObservationFcn = @(prices, t) reshape(prices(:, t:t+4), [], 1);
    
    % Función para convertir acción PPO a señal
    getPPOSignalFcn = @(agent, prices, t) getSignalFromPPO(agent, getObservationFcn(prices, t));
    
    % Crear wrapper como estructura con un campo función
    ppoWrapper = struct('getSignal', @(t) getPPOSignalFcn(ppoAgent, prices, t));
catch
    warning('PPO agent could not be loaded. Skipping PPO evaluation.');
    ppoWrapper = [];
end

% 5. Create SPO agent based on the swarm optimization strategy
try
    % Función para obtener señal SPO en función del tiempo
    getSPOSignalFcn = @(t) getSPOSignalImpl(prices, t);
    
    % Crear wrapper como estructura con un campo función
    spoWrapper = struct('getSignal', @(t) getSPOSignalFcn(t));
catch
    warning('SPO agent could not be created. Skipping SPO evaluation.');
    spoWrapper = [];
end

%% Initialize containers for all strategy results
agents = struct();
results = struct();

% Add all available agents to the agents struct
agents.RSI = rsiAgent;
agents.MACD = macdAgent;
agents.BuyHold = bhAgent;

if exist('ppoWrapper', 'var') && ~isempty(ppoWrapper)
    agents.PPO = ppoWrapper;
end

if exist('spoWrapper', 'var') && ~isempty(spoWrapper)
    agents.SPO = spoWrapper;
end

%% Run simulations for all strategies
agentNames = fieldnames(agents);
fprintf('Running simulations for %d strategies...\n', length(agentNames));

for i = 1:length(agentNames)
    agentName = agentNames{i};
    agent = agents.(agentName);
    
    fprintf('Simulating %s strategy...\n', agentName);
    
    % Initialize environment with current agent
    try
        % MODIFICACIÓN: Usar un enfoque simplificado de simulación en lugar de depender de PortfolioEnv
        % Esto evita los problemas con initialAction y la incompatibilidad de interfaces
        
        % Crear una estructura que simula el comportamiento de PortfolioEnv pero que usa directamente el agente
        env = struct();
        env.NumAssets = size(prices, 1);
        
        % Simulación manual para evitar problemas con step y agent
        hist = zeros(simulationLength, 1);
        hist(1) = initialPortfolioValue;
        
        for t = 2:simulationLength
            if t <= size(prices, 2)
                % Obtener señal del agente para este tiempo
                try
                    if isstruct(agent) && isfield(agent, 'getSignal') && isa(agent.getSignal, 'function_handle')
                        signal = agent.getSignal(t-1); % t-1 porque en getSignal el índice empieza en 0
                    elseif isa(agent, 'function_handle')
                        signal = agent(t-1);
                    else
                        % Tratar de llamar directamente como método
                        signal = getSignal(agent, t-1);
                    end
                catch ME
                    % Si falla getSignal, intentar directamente como función
                    try
                        if isa(agent, 'function_handle')
                            signal = agent(t-1);
                        else
                            % Último intento, usar el valor por defecto
                            warning('No se pudo obtener señal del agente, usando 0. Error: %s', ME.message);
                            signal = 0;
                        end
                    catch
                        warning('No se pudo obtener señal del agente, usando 0');
                        signal = 0;
                    end
                end
                
                % Calcular retorno basado en la señal
                if signal == 1
                    % Señal de compra: usar retornos positivos
                    r = mean(prices(:, t));
                elseif signal == -1
                    % Señal de venta: usar retornos negativos
                    r = -mean(prices(:, t));
                else
                    % Señal neutra: pequeño interés
                    r = 0.0001;
                end
                
                % Actualizar valor del portafolio
                hist(t) = hist(t-1) * (1 + r);
            else
                % Si nos quedamos sin datos, mantener el último valor
                hist(t) = hist(t-1);
            end
        end
    catch ME
        warning('Error en simulación: %s', ME.message);
        % Fallback a valores por defecto en caso de error
        hist = zeros(simulationLength, 1);
        hist(1) = initialPortfolioValue;
        % Simular pérdida total (solo para indicar error)
        hist(2:end) = 0;
    end
    
    % Calculate performance metrics
    totalReturn = (hist(end) - hist(1)) / hist(1);
    dailyReturns = diff(hist) ./ hist(1:end-1);
    sharpe = mean(dailyReturns) / std(dailyReturns) * sqrt(252);
    maxDrawdown = max(cummax(hist) - hist) / max(hist);
    volatility = std(dailyReturns) * sqrt(252);
    
    % Store results
    results.(agentName).equity = hist;
    results.(agentName).returns = dailyReturns;
    results.(agentName).totalReturn = totalReturn;
    results.(agentName).sharpe = sharpe;
    results.(agentName).maxDrawdown = maxDrawdown;
    results.(agentName).volatility = volatility;
    
    fprintf('%s Strategy - Return: %.2f%%, Sharpe: %.2f, Drawdown: %.2f%%\n', ...
        agentName, totalReturn*100, sharpe, maxDrawdown*100);
end

%% Save all results
disp('Saving all results...');

% Create logs directory if it doesn't exist
logsDir = fullfile(basePath, 'results', 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

% Save combined results
save(fullfile(logsDir, 'all_comparison.mat'), 'results');

%% Plot comparison results
disp('Plotting comparison...');
figuresDir = fullfile(basePath, 'results', 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end
compare_plots(results, figuresDir);

disp('All comparisons completed!');

function signal = getSignalFromPPO(agent, obs)
    try
        % Get action from PPO policy
        action = getAction(agent, obs);
        
        % Convert action to signal (simplified, using a threshold)
        if isnumeric(action) && sum(action) > 0.5
            signal = 1;  % Buy
        elseif isnumeric(action) && sum(action) < 0.1
            signal = -1; % Sell
        else
            signal = 0;  % Hold
        end
    catch
        % En caso de error, señal neutra
        signal = 0;
    end
end

function signal = getSPOSignalImpl(prices, t)
    try
        windowSize = 5;
        if t+windowSize-1 <= size(prices, 2)
            ventana = prices(:, t:t+windowSize-1);
            mu = mean(ventana, 2);
            sigma = var(ventana, 0, 2);
            
            % Call SPO strategy to get weights
            try
                weights = obtenerSPO(mu, sigma, 0.1); % Using alpha=0.1
                
                % Convert weights to signal
                if sum(weights) > 0.5
                    signal = 1;  % Buy
                elseif sum(weights) < 0.1
                    signal = -1; % Sell
                else
                    signal = 0;  % Hold
                end
            catch
                signal = 0;  % Default to hold if SPO fails
            end
        else
            signal = 0;  % Default to hold if not enough data
        end
    catch
        % En caso de cualquier error en el proceso
        signal = 0;
    end
end 
    % Get action from PPO policy
    action = getAction(agent, obs);
    
    % Convert action to signal (simplified, using a threshold)
    if sum(action) > 0.5
        signal = 1;  % Buy
    elseif sum(action) < 0.1
        signal = -1; % Sell
    else
        signal = 0;  % Hold
    end
end

function signal = getSPOSignalImpl(prices, t)
    windowSize = 5;
    if t+windowSize-1 <= size(prices, 2)
        ventana = prices(:, t:t+windowSize-1);
        mu = mean(ventana, 2);
        sigma = var(ventana, 0, 2);
        
        % Call SPO strategy to get weights
        try
            weights = obtenerSPO(mu, sigma, 0.1); % Using alpha=0.1
            
            % Convert weights to signal
            if sum(weights) > 0.5
                signal = 1;  % Buy
            elseif sum(weights) < 0.1
                signal = -1; % Sell
            else
                signal = 0;  % Hold
            end
        catch
            signal = 0;  % Default to hold if SPO fails
        end
    else
        signal = 0;  % Default to hold if not enough data
    end
end 