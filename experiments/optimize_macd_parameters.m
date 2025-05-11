%% MACD Parameter Optimization
% This script performs a grid search to find optimal MACD strategy parameters
% (fast period, slow period, and signal period) that maximize Sharpe ratio.

% Get base path for proper file references
basePath = fileparts(fileparts(mfilename('fullpath')));

% Add paths if not already on path
if ~exist('macd_strategy', 'file')
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
catch
    try
        % If specific file not found, try to load from a more general source
        data = load('ReaderBeginingDLR.mat');
        prices = data.RetornosMedios;
    catch err
        error('Could not load price data. Error: %s', err.message);
    end
end

%% Configure optimization grid
% Define parameter ranges for grid search
fastPeriods = [5, 8, 12, 15, 20];
slowPeriods = [15, 20, 26, 30, 40];
signalPeriods = [5, 7, 9, 12];

% Initialize results container
numCombinations = length(fastPeriods) * length(slowPeriods) * length(signalPeriods);
results = struct();
results.params = zeros(numCombinations, 3); % [fast, slow, signal]
results.sharpe = zeros(numCombinations, 1);
results.returns = zeros(numCombinations, 1);
results.maxDrawdown = zeros(numCombinations, 1);

% Configuration
simulationLength = 252; % Simulate for one year
initialPortfolioValue = 100;

%% Run grid search
disp(['Starting grid search with ', num2str(numCombinations), ' parameter combinations...']);

combinationIdx = 1;
progressInterval = max(1, floor(numCombinations / 20)); % Show progress at ~5% intervals

for f = 1:length(fastPeriods)
    fastPeriod = fastPeriods(f);
    
    for s = 1:length(slowPeriods)
        slowPeriod = slowPeriods(s);
        
        % Skip invalid combinations (fast must be smaller than slow)
        if fastPeriod >= slowPeriod
            continue;
        end
        
        for g = 1:length(signalPeriods)
            signalPeriod = signalPeriods(g);
            
            % Create MACD agent with current parameters
            macdAgent = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod);
            
            % Initialize environment
            try
                % MODIFICACIÓN: Usar función anónima en lugar de función anidada
                simulateStepFn = @(unused1, unused2) simulateStepImpl(macdAgent, fastPeriod, prices, simulationLength);
                env = struct('step', simulateStepFn);
            catch err
                warning('Error al inicializar entorno: %s. Usando implementación alternativa.', err.message);
                
                % Fallback extremadamente simple si lo anterior falla
                hist = zeros(simulationLength, 1);
                hist(1) = initialPortfolioValue;
                
                % Simular retornos basados directamente en las señales MACD
                for simT = 2:simulationLength
                    t = min(simT, size(prices, 2) - 1);
                    signal = macdAgent.getSignal(t);
                    
                    if signal == 1
                        hist(simT) = hist(simT-1) * (1 + mean(prices(:, t+1)));
                    elseif signal == -1
                        hist(simT) = hist(simT-1) * (1 - mean(prices(:, t+1)));
                    else
                        hist(simT) = hist(simT-1) * 1.001; % Pequeño interés por efectivo
                    end
                end
                
                % Calcular performance metrics desde la simulación fallback
                totalReturn = (hist(end) - hist(1)) / hist(1);
                dailyReturns = diff(hist) ./ hist(1:end-1);
                sharpeRatio = mean(dailyReturns) / std(dailyReturns) * sqrt(252);
                maxDrawdown = max(cummax(hist) - hist) / max(hist);
                
                % Almacenar resultados y continuar con la siguiente combinación
                results.params(combinationIdx, :) = [fastPeriod, slowPeriod, signalPeriod];
                results.sharpe(combinationIdx) = sharpeRatio;
                results.returns(combinationIdx) = totalReturn;
                results.maxDrawdown(combinationIdx) = maxDrawdown;
                
                combinationIdx = combinationIdx + 1;
                continue; % Saltar al siguiente ciclo del bucle
            end
            
            % Run simulation
            hist = zeros(simulationLength, 1);
            hist(1) = initialPortfolioValue;
            
            for t = 1:simulationLength-1
                try
                    % Llamamos a la función step con los argumentos requeridos
                    % Pasamos dos argumentos vacíos ya que nuestra función simulateStep no los usa
                    [~, r, isDone] = env.step([], []);
                    hist(t+1) = hist(t) * (1 + r);
                    if isDone
                        break;
                    end
                catch stepErr
                    warning('Error en step: %s. Continuando con siguiente iteración.', stepErr.message);
                    % Si falla, mantener el valor anterior
                    hist(t+1) = hist(t);
                end
            end
            
            % Calculate performance metrics
            totalReturn = (hist(end) - hist(1)) / hist(1);
            dailyReturns = diff(hist) ./ hist(1:end-1);
            sharpeRatio = mean(dailyReturns) / std(dailyReturns) * sqrt(252);
            maxDrawdown = max(cummax(hist) - hist) / max(hist);
            
            % Store results
            results.params(combinationIdx, :) = [fastPeriod, slowPeriod, signalPeriod];
            results.sharpe(combinationIdx) = sharpeRatio;
            results.returns(combinationIdx) = totalReturn;
            results.maxDrawdown(combinationIdx) = maxDrawdown;
            
            % Show progress periodically
            if mod(combinationIdx, progressInterval) == 0 || combinationIdx == 1
                fprintf('Progress: %d/%d combinations (%.1f%%)\n', ...
                    combinationIdx, numCombinations, combinationIdx/numCombinations*100);
            end
            
            combinationIdx = combinationIdx + 1;
        end
    end
end

% Adjust results in case some combinations were skipped
actualCombinations = combinationIdx - 1;
results.params = results.params(1:actualCombinations, :);
results.sharpe = results.sharpe(1:actualCombinations);
results.returns = results.returns(1:actualCombinations);
results.maxDrawdown = results.maxDrawdown(1:actualCombinations);

%% Find best parameters
[maxSharpe, maxIdx] = max(results.sharpe);
bestFastPeriod = results.params(maxIdx, 1);
bestSlowPeriod = results.params(maxIdx, 2);
bestSignalPeriod = results.params(maxIdx, 3);
bestReturn = results.returns(maxIdx);
bestDrawdown = results.maxDrawdown(maxIdx);

fprintf('\n--- Best MACD Parameters ---\n');
fprintf('Fast Period: %d\n', bestFastPeriod);
fprintf('Slow Period: %d\n', bestSlowPeriod);
fprintf('Signal Period: %d\n', bestSignalPeriod);
fprintf('Sharpe Ratio: %.4f\n', maxSharpe);
fprintf('Total Return: %.2f%%\n', bestReturn * 100);
fprintf('Max Drawdown: %.2f%%\n', bestDrawdown * 100);

%% Save results
disp('Saving optimization results...');

% Create results directory if it doesn't exist
logsDir = fullfile(basePath, 'results', 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

save(fullfile(logsDir, 'MACD_optimization_results.mat'), 'results', 'bestFastPeriod', 'bestSlowPeriod', 'bestSignalPeriod');

%% Visualize results
disp('Creating visualizations...');

% Create figures directory if it doesn't exist
figuresDir = fullfile(basePath, 'results', 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end

% Find unique fast periods that were tested
uniqueFastPeriods = unique(results.params(:, 1));

% For each fast period, plot a heatmap of Sharpe ratio by slow/signal periods
for i = 1:length(uniqueFastPeriods)
    fastPeriod = uniqueFastPeriods(i);
    
    % Get data for this fast period
    idx = results.params(:, 1) == fastPeriod;
    slowPeriodVals = results.params(idx, 2);
    signalPeriodVals = results.params(idx, 3);
    sharpeVals = results.sharpe(idx);
    
    % Create a figure
    fig = figure('Position', [100, 100, 800, 600]);
    
    % Create a scatter plot with color indicating Sharpe ratio
    scatter(slowPeriodVals, signalPeriodVals, 100, sharpeVals, 'filled');
    colorbar;
    xlabel('Slow Period');
    ylabel('Signal Period');
    title(sprintf('MACD Sharpe Ratio (Fast Period = %d)', fastPeriod));
    
    % Add text annotations with Sharpe values
    for j = 1:length(sharpeVals)
        text(slowPeriodVals(j), signalPeriodVals(j), sprintf('%.2f', sharpeVals(j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Highlight best point for this fast period
    [~, bestIdx] = max(sharpeVals);
    hold on;
    plot(slowPeriodVals(bestIdx), signalPeriodVals(bestIdx), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
    
    % Save figure
    saveas(fig, fullfile(figuresDir, sprintf('MACD_optimization_fast_%d.png', fastPeriod)));
    saveas(fig, fullfile(figuresDir, sprintf('MACD_optimization_fast_%d.fig', fastPeriod)));
    close(fig);
end

% Summary figure showing best Sharpe for each fast period
fig = figure('Position', [100, 100, 800, 400]);
fastPeriodSharpes = zeros(length(uniqueFastPeriods), 1);

for i = 1:length(uniqueFastPeriods)
    fastPeriod = uniqueFastPeriods(i);
    idx = results.params(:, 1) == fastPeriod;
    fastPeriodSharpes(i) = max(results.sharpe(idx));
end

bar(uniqueFastPeriods, fastPeriodSharpes);
xlabel('MACD Fast Period');
ylabel('Best Sharpe Ratio');
title('Best Sharpe Ratio by MACD Fast Period');
grid on;

% Add text labels above bars
for i = 1:length(uniqueFastPeriods)
    text(uniqueFastPeriods(i), fastPeriodSharpes(i), sprintf('%.2f', fastPeriodSharpes(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Highlight overall best fast period
[~, bestFastPeriodIdx] = max(fastPeriodSharpes);
hold on;
highlight = bar(uniqueFastPeriods(bestFastPeriodIdx), fastPeriodSharpes(bestFastPeriodIdx));
highlight.FaceColor = 'r';

% Save summary figure
saveas(fig, fullfile(figuresDir, 'MACD_optimization_summary.png'));
saveas(fig, fullfile(figuresDir, 'MACD_optimization_summary.fig'));

disp('Optimization completed successfully!');

% Función de implementación para simulateStep
function [newState, reward, isDone] = simulateStepImpl(macdAgent, fastPeriod, prices, simulationLength)
    % Esta es una implementación simplificada de step() que usa
    % directamente el agente MACD en lugar de depender del entorno
    
    % Obtener un índice aleatorio para simular un paso de tiempo
    t = min(randi([fastPeriod+1, size(prices, 2)-1]), simulationLength);
    
    % Obtener señal del agente MACD
    try
        signal = macdAgent.getSignal(t);
    catch
        % Si falla getSignal, usar un valor por defecto
        signal = 0;
    end
    
    % Calcular retorno basado en la señal
    if t+1 <= size(prices, 2)
        if signal == 1
            % Señal de compra: usar retornos del siguiente período
            reward = mean(prices(:, t+1));
        elseif signal == -1
            % Señal de venta: usar el negativo de los retornos
            reward = -mean(prices(:, t+1));
        else
            % Señal neutra: retorno pequeño (equivalente a efectivo)
            reward = 0.001;
        end
    else
        % Si no hay datos futuros disponibles
        reward = 0;
    end
    
    % Simplemente devolver el mismo estado y que nunca termine
    newState = t;
    isDone = false;
end 
disp('Optimization completed successfully!'); 