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
alpha = 0.1;                        % Parámetro para SPO
fastPeriod = 12;                    % Periodo rápido para MACD
slowPeriod = 26;                    % Periodo lento para MACD
signalPeriod = 9;                   % Periodo de señal para MACD
[numAssets, numSteps] = size(RetornosMedios);
steps = numSteps - windowSize;

% Inicialización de valores de portafolio
valueMACD = 1;
valueSPO = 1;
valueEqual = 1;

% Pesos para estrategia de igual ponderación
weightsEqual = ones(numAssets, 1) / numAssets;

% Arrays para guardar historias de valores
seriesMACD = zeros(1, steps);
seriesSPO = zeros(1, steps);
seriesEqual = zeros(1, steps);

% Calcular señales MACD para todos los activos
fprintf('Calculando señales MACD para %d activos...\n', numAssets);
macdSignals = zeros(numAssets, numSteps);
for asset = 1:numAssets
    macdSignals(asset, :) = macd_strategy(RetornosMedios(asset, :), fastPeriod, slowPeriod, signalPeriod)';
end

% Simulación paso a paso
fprintf('Iniciando simulación para %d pasos...\n', steps);
for t = 1:steps
    % Obtener ventana de retornos actuales
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    RetornosPromedio = mean(ventana, 2);
    VarianzaRetornos = var(ventana, 0, 2);
    
    % Obtener señales MACD actuales
    currentSignals = macdSignals(:, t + windowSize);
    
    % Calcular pesos basados en MACD
    % Estrategia: invertir en activos con señal de compra (1)
    % Evitar activos con señal de venta (-1)
    % Distribuir igualmente entre activos con señal de compra
    pesosMACD = zeros(numAssets, 1);
    buySignals = currentSignals == 1;
    if any(buySignals)
        % Si hay señales de compra, invertir solo en esos activos
        % MODIFICACIÓN: Limitar exposición máxima al 60% del portafolio
        pesosMACD(buySignals) = 0.6 / sum(buySignals);
        
        % MODIFICACIÓN: Añadir diversificación con activos neutros
        neutralSignals = currentSignals == 0;
        if any(neutralSignals)
            pesosMACD(neutralSignals) = 0.3 / sum(neutralSignals);
        end
    else
        % Si no hay señales de compra, usar pesos iguales para señales neutras (0)
        neutralSignals = currentSignals == 0;
        if any(neutralSignals)
            pesosMACD(neutralSignals) = 0.5 / sum(neutralSignals);
        else
            % Si todo son señales de venta, mantener efectivo (todos ceros)
        end
    end
    
    % MODIFICACIÓN: Asegurar que la suma de pesos no excede 1
    if sum(pesosMACD) > 1
        pesosMACD = pesosMACD / sum(pesosMACD);
    end
    
    % Calcular pesos SPO
    try
        pesosSPO = obtenerSPO(RetornosPromedio, VarianzaRetornos, alpha);
    catch
        % En caso de error, usar pesos iguales
        pesosSPO = weightsEqual;
    end
    
    % MODIFICACIÓN: Normalizar pesos SPO para asegurar que suman 1
    if sum(pesosSPO) > 0
        pesosSPO = pesosSPO / sum(pesosSPO);
    end
    
    % Obtener retorno para el siguiente periodo
    retorno = RetornosMedios(:, t + windowSize);
    
    % Calcular retornos del portafolio
    rMACD = sum(pesosMACD .* retorno);
    rSPO = sum(pesosSPO .* retorno);
    rEqual = sum(weightsEqual .* retorno);
    
    % Limitar pérdidas extremas (protección)
    rMACD = max(rMACD, -0.95);
    rSPO = max(rSPO, -0.95);
    rEqual = max(rEqual, -0.95);
    
    % Actualizar valores de portafolio
    valueMACD = valueMACD * (1 + rMACD);
    valueSPO = valueSPO * (1 + rSPO);
    valueEqual = valueEqual * (1 + rEqual);
    
    % Guardar historias
    seriesMACD(t) = valueMACD;
    seriesSPO(t) = valueSPO;
    seriesEqual(t) = valueEqual;
    
    % Mostrar progreso
    if mod(t, ceil(steps/10)) == 0
        fprintf('Progreso: %.1f%% completado\n', t/steps*100);
    end
end

% Calcular métricas de rendimiento
finalReturnMACD = (valueMACD - 1) * 100;
finalReturnSPO = (valueSPO - 1) * 100;
finalReturnEqual = (valueEqual - 1) * 100;

dailyReturnsMACD = diff([1, seriesMACD]) ./ [1, seriesMACD(1:end-1)];
dailyReturnsSPO = diff([1, seriesSPO]) ./ [1, seriesSPO(1:end-1)];
dailyReturnsEqual = diff([1, seriesEqual]) ./ [1, seriesEqual(1:end-1)];

% Calcular Sharpe Ratio (asumiendo 252 días de trading al año)
sharpeMACD = mean(dailyReturnsMACD(2:end)) / std(dailyReturnsMACD(2:end)) * sqrt(252);
sharpeSPO = mean(dailyReturnsSPO(2:end)) / std(dailyReturnsSPO(2:end)) * sqrt(252);
sharpeEqual = mean(dailyReturnsEqual(2:end)) / std(dailyReturnsEqual(2:end)) * sqrt(252);

% Calcular Drawdown máximo
drawdownMACD = calculateMaxDrawdown(seriesMACD);
drawdownSPO = calculateMaxDrawdown(seriesSPO);
drawdownEqual = calculateMaxDrawdown(seriesEqual);

% === Mostrar resultados ===
fprintf('\n=== RESULTADOS DE LA SIMULACIÓN ===\n');
fprintf('Estrategia MACD:          Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnMACD, sharpeMACD, drawdownMACD*100);
fprintf('Estrategia SPO:           Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnSPO, sharpeSPO, drawdownSPO*100);
fprintf('Estrategia Igual Peso:    Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnEqual, sharpeEqual, drawdownEqual*100);

% === Gráfico comparativo ===
figure;
plot(seriesMACD, 'b', 'LineWidth', 2); hold on;
plot(seriesSPO, 'g', 'LineWidth', 2);
plot(seriesEqual, 'r-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('MACD vs SPO vs Igual Peso');
legend('MACD', 'SPO', 'Igual Peso');
grid on;

% Guardar gráfico
saveFigurePath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'figures');
if ~exist(saveFigurePath, 'dir')
    mkdir(saveFigurePath);
end
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_others_comparison.png'));
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_others_comparison.fig'));

fprintf("✅ Simulación completada: MACD vs SPO vs igual peso.\n");
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
    if ~exist('macd_strategy', 'file')
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
    end
end 