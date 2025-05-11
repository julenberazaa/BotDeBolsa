clc;
clear;
close all;

% Asegurar que tenemos las rutas adecuadas
addPathsIfNeeded();

% Cargar datos
try
    load('ReaderBeginingDLR.mat');
catch
    % Intentar cargar desde ubicación alternativa
    try
        projPath = fileparts(fileparts(mfilename('fullpath')));
        load(fullfile(projPath, 'data', 'processed', 'ReaderBeginingDLR.mat'));
    catch err
        error('No se pudo cargar ReaderBeginingDLR.mat: %s', err.message);
    end
end

% === PARÁMETROS ===
windowSize = 5;
% Parámetros MACD
fastPeriod = 12;
slowPeriod = 26;
signalPeriod = 9;
% Parámetros RSI
rsiPeriod = 14;
overbought = 70;
oversold = 30;

[numAssets, numSteps] = size(RetornosMedios);
steps = numSteps - windowSize;

% Inicialización de valores de portafolio
valueMACD = 1;
valueRSI = 1;
valueBuyHold = 1;

% Pesos para estrategia Buy & Hold
weightsBuyHold = ones(numAssets, 1) / numAssets;

% Arrays para guardar historias de valores
seriesMACD = zeros(1, steps);
seriesRSI = zeros(1, steps);
seriesBuyHold = zeros(1, steps);

% Calcular señales MACD para todos los activos
fprintf('Calculando señales MACD para %d activos...\n', numAssets);
macdSignals = zeros(numAssets, numSteps);
for asset = 1:numAssets
    macdSignals(asset, :) = macd_strategy(RetornosMedios(asset, :), fastPeriod, slowPeriod, signalPeriod)';
end

% Calcular señales RSI para todos los activos
fprintf('Calculando señales RSI para %d activos...\n', numAssets);
rsiSignals = zeros(numAssets, numSteps);
for asset = 1:numAssets
    rsiSignals(asset, :) = rsi_strategy(RetornosMedios(asset, :), rsiPeriod, overbought, oversold)';
end

% Simulación paso a paso
fprintf('Iniciando simulación para %d pasos...\n', steps);
for t = 1:steps
    % Obtener ventana de retornos actuales
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    
    % Obtener señales MACD actuales
    currentMacdSignals = macdSignals(:, t + windowSize);
    
    % Obtener señales RSI actuales
    currentRsiSignals = rsiSignals(:, t + windowSize);
    
    % Calcular pesos basados en MACD
    pesosMACD = zeros(numAssets, 1);
    buySignalsMACD = currentMacdSignals == 1;
    if any(buySignalsMACD)
        % Si hay señales de compra, invertir solo en esos activos
        pesosMACD(buySignalsMACD) = 0.6 / sum(buySignalsMACD);
        
        % Añadir diversificación con activos neutros
        neutralSignalsMACD = currentMacdSignals == 0;
        if any(neutralSignalsMACD)
            pesosMACD(neutralSignalsMACD) = 0.3 / sum(neutralSignalsMACD);
        end
    else
        % Si no hay señales de compra, usar pesos iguales para señales neutras (0)
        neutralSignalsMACD = currentMacdSignals == 0;
        if any(neutralSignalsMACD)
            pesosMACD(neutralSignalsMACD) = 0.5 / sum(neutralSignalsMACD);
        else
            % Si todo son señales de venta, mantener efectivo (todos ceros)
        end
    end
    
    % Asegurar que la suma de pesos no excede 1
    if sum(pesosMACD) > 1
        pesosMACD = pesosMACD / sum(pesosMACD);
    end
    
    % Calcular pesos basados en RSI
    pesosRSI = zeros(numAssets, 1);
    buySignalsRSI = currentRsiSignals == 1;
    if any(buySignalsRSI)
        % Si hay señales de compra, invertir solo en esos activos
        pesosRSI(buySignalsRSI) = 0.6 / sum(buySignalsRSI);
        
        % Añadir diversificación con activos neutros
        neutralSignalsRSI = currentRsiSignals == 0;
        if any(neutralSignalsRSI)
            pesosRSI(neutralSignalsRSI) = 0.3 / sum(neutralSignalsRSI);
        end
    else
        % Si no hay señales de compra, usar pesos iguales para señales neutras (0)
        neutralSignalsRSI = currentRsiSignals == 0;
        if any(neutralSignalsRSI)
            pesosRSI(neutralSignalsRSI) = 0.5 / sum(neutralSignalsRSI);
        else
            % Si todo son señales de venta, mantener efectivo (todos ceros)
        end
    end
    
    % MODIFICACIÓN: Asegurar que la suma de pesos no excede 1
    if sum(pesosRSI) > 1
        pesosRSI = pesosRSI / sum(pesosRSI);
    end
    
    % Obtener retorno para el siguiente periodo
    retorno = RetornosMedios(:, t + windowSize);
    
    % Calcular retornos del portafolio
    rMACD = sum(pesosMACD .* retorno);
    rRSI = sum(pesosRSI .* retorno);
    rBuyHold = sum(weightsBuyHold .* retorno);
    
    % Limitar pérdidas extremas (protección)
    rMACD = max(rMACD, -0.95);
    rRSI = max(rRSI, -0.95);
    rBuyHold = max(rBuyHold, -0.95);
    
    % Actualizar valores de portafolio
    valueMACD = valueMACD * (1 + rMACD);
    valueRSI = valueRSI * (1 + rRSI);
    valueBuyHold = valueBuyHold * (1 + rBuyHold);
    
    % Guardar historias
    seriesMACD(t) = valueMACD;
    seriesRSI(t) = valueRSI;
    seriesBuyHold(t) = valueBuyHold;
    
    % Mostrar progreso
    if mod(t, ceil(steps/10)) == 0
        fprintf('Progreso: %.1f%% completado\n', t/steps*100);
    end
end

% Calcular métricas de rendimiento
finalReturnMACD = (valueMACD - 1) * 100;
finalReturnRSI = (valueRSI - 1) * 100;
finalReturnBuyHold = (valueBuyHold - 1) * 100;

dailyReturnsMACD = diff([1, seriesMACD]) ./ [1, seriesMACD(1:end-1)];
dailyReturnsRSI = diff([1, seriesRSI]) ./ [1, seriesRSI(1:end-1)];
dailyReturnsBuyHold = diff([1, seriesBuyHold]) ./ [1, seriesBuyHold(1:end-1)];

% Calcular Sharpe Ratio (asumiendo 252 días de trading al año)
sharpeMACD = mean(dailyReturnsMACD(2:end)) / std(dailyReturnsMACD(2:end)) * sqrt(252);
sharpeRSI = mean(dailyReturnsRSI(2:end)) / std(dailyReturnsRSI(2:end)) * sqrt(252);
sharpeBuyHold = mean(dailyReturnsBuyHold(2:end)) / std(dailyReturnsBuyHold(2:end)) * sqrt(252);

% Calcular Drawdown máximo
drawdownMACD = calculateMaxDrawdown(seriesMACD);
drawdownRSI = calculateMaxDrawdown(seriesRSI);
drawdownBuyHold = calculateMaxDrawdown(seriesBuyHold);

% === Mostrar resultados ===
fprintf('\n=== RESULTADOS DE LA SIMULACIÓN ===\n');
fprintf('Estrategia MACD:          Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnMACD, sharpeMACD, drawdownMACD*100);
fprintf('Estrategia RSI:           Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnRSI, sharpeRSI, drawdownRSI*100);
fprintf('Estrategia Buy & Hold:    Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnBuyHold, sharpeBuyHold, drawdownBuyHold*100);

% === Gráfico comparativo ===
figure;
plot(seriesMACD, 'b', 'LineWidth', 2); hold on;
plot(seriesRSI, 'g', 'LineWidth', 2);
plot(seriesBuyHold, 'r-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('MACD vs RSI vs Buy & Hold');
legend('MACD', 'RSI', 'Buy & Hold');
grid on;

% Guardar gráfico
saveFigurePath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'figures');
if ~exist(saveFigurePath, 'dir')
    mkdir(saveFigurePath);
end
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_rsi_comparison.png'));
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_rsi_comparison.fig'));

fprintf("✅ Simulación completada: MACD vs RSI vs Buy & Hold.\n");
fprintf("Gráficos guardados en: %s\n", saveFigurePath);

% === Funciones auxiliares ===
function maxDrawdown = calculateMaxDrawdown(equityCurve)
    % Calcula el máximo drawdown de una curva de equity
    peakValue = equityCurve(1);
    maxDrawdown = 0;
    
    for i = 2:length(equityCurve)
        if equityCurve(i) > peakValue
            peakValue = equityCurve(i);
        else
            drawdown = (peakValue - equityCurve(i)) / peakValue;
            if drawdown > maxDrawdown
                maxDrawdown = drawdown;
            end
        end
    end
end

function addPathsIfNeeded()
    % Añadir rutas si no están en el path
    if ~exist('macd_strategy', 'file') || ~exist('rsi_strategy', 'file')
        fprintf('Añadiendo rutas necesarias...\n');
        
        % Obtener ruta del proyecto
        projPath = fileparts(fileparts(mfilename('fullpath')));
        
        % Añadir rutas de src
        addpath(fullfile(projPath, 'src', 'strategies'));
        addpath(fullfile(projPath, 'src', 'agents'));
        addpath(fullfile(projPath, 'src', 'utils'));
        
        % Verificar si las rutas se añadieron correctamente
        if ~exist('macd_strategy', 'file')
            error('No se pudo añadir macd_strategy al path. Asegúrate de que el archivo existe.');
        end
        if ~exist('rsi_strategy', 'file')
            error('No se pudo añadir rsi_strategy al path. Asegúrate de que el archivo existe.');
        end
    end
end 