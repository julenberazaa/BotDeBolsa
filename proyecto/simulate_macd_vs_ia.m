clc;
clear;
close all;

% Asegurar que tenemos las rutas adecuadas
addPathsIfNeeded();

% Cargar datos
try
    load('ReaderBeginingDLR.mat');
    load('spo_ajustada_ret.mat');  % Red entrenada con sesgo hacia retorno
catch
    % Intentar cargar desde ubicación alternativa
    try
        projPath = fileparts(fileparts(mfilename('fullpath')));
        load(fullfile(projPath, 'data', 'processed', 'ReaderBeginingDLR.mat'));
        load(fullfile(projPath, 'data', 'processed', 'spo_ajustada_ret.mat'));
    catch err
        error('No se pudieron cargar los datos necesarios: %s', err.message);
    end
end

% === PARÁMETROS ===
windowSize = 5;
% Parámetros MACD
fastPeriod = 12;
slowPeriod = 26;
signalPeriod = 9;

[numAssets, numSteps] = size(RetornosMedios);
steps = numSteps - windowSize;

% Inicialización de valores de portafolio
valueMACD = 1;
valueIA = 1;
valueIAMACD = 1; % Nuevo: Estrategia híbrida IA-MACD

% Arrays para guardar historias de valores
seriesMACD = zeros(1, steps);
seriesIA = zeros(1, steps);
seriesIAMACD = zeros(1, steps); % Para estrategia híbrida

% Arrays para registro de régimen de mercado
volatilidad_hist = zeros(1, steps);
tendencia_hist = zeros(1, steps);
proporcionMACD_hist = zeros(1, steps);

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
    currentMacdSignals = macdSignals(:, t + windowSize);
    
    % Calcular pesos basados en MACD - versión modificada más conservadora
    pesosMACD = zeros(numAssets, 1);
    buySignals = currentMacdSignals == 1;
    if any(buySignals)
        % MODIFICACIÓN: Reducir exposición al 40% y añadir más diversificación
        pesosMACD(buySignals) = 0.4 / sum(buySignals);
        
        % Diversificar con señales neutras
        neutralSignals = currentMacdSignals == 0;
        if any(neutralSignals)
            pesosMACD(neutralSignals) = 0.2 / sum(neutralSignals);
        end
        
        % MODIFICACIÓN: Mantener algo de efectivo para reducir la volatilidad
        % El 40% restante queda como efectivo (pesos = 0)
    else
        % Si no hay señales de compra, usar señales neutras con menor exposición
        neutralSignals = currentMacdSignals == 0;
        if any(neutralSignals)
            % MODIFICACIÓN: Reducir exposición al 30% para señales neutras
            pesosMACD(neutralSignals) = 0.3 / sum(neutralSignals);
        end
    end
    
    % MODIFICACIÓN: Añadir límite de posición por activo para evitar concentración
    maxPosicionPorActivo = 0.15; % 15% como máximo por activo
    for i = 1:numAssets
        if pesosMACD(i) > maxPosicionPorActivo
            pesosMACD(i) = maxPosicionPorActivo;
        end
    end
    
    % Renormalizar pesos después de aplicar límites
    if sum(pesosMACD) > 0
        pesosMACD = pesosMACD / sum(pesosMACD);
    end
    
    % Preparar input para la red neuronal
    inputIA = normalize(ventana(:), 1);  % Normalizar igual que en entrenamiento
    
    % Obtener predicciones de la red neuronal
    try
        pesosIA = predict(net, inputIA');  % entrada como fila
        pesosIA = pesosIA';
        
        % Normalizar pesos IA
        pesosIA = max(pesosIA, 0);
        pesosIA = pesosIA / (sum(pesosIA) + 1e-10);
        
        % MODIFICACIÓN: Mejorar estrategia IA con criterios adicionales
        % 1. Limitar exposición máxima por activo
        maxPesoIA = 0.20; % Máximo 20% por activo
        for i = 1:numAssets
            if pesosIA(i) > maxPesoIA
                pesosIA(i) = maxPesoIA;
            end
        end
        
        % 2. Filtrar activos con retornos promedio negativos
        for i = 1:numAssets
            if RetornosPromedio(i) < -0.01 % Si el retorno medio es muy negativo
                pesosIA(i) = pesosIA(i) * 0.5; % Reducir su peso a la mitad
            end
        end
        
        % 3. Aumentar pesos de activos con menor volatilidad
        volNormalizada = VarianzaRetornos / max(VarianzaRetornos + 1e-10);
        for i = 1:numAssets
            % Bonificar activos de baja volatilidad
            if volNormalizada(i) < 0.3 && pesosIA(i) > 1/numAssets
                pesosIA(i) = pesosIA(i) * 1.2;
            end
        end
        
        % Volver a normalizar después de ajustes
        pesosIA = pesosIA / sum(pesosIA);
    catch
        % En caso de error, usar pesos iguales
        pesosIA = ones(numAssets, 1) / numAssets;
    end
    
    % === DETECTOR DE RÉGIMEN DE MERCADO ===
    % Detectamos el tipo de mercado para ajustar dinámicamente el balance entre MACD e IA
    
    % Calcular volatilidad reciente usando los últimos 20 pasos (o menos si no hay suficientes)
    periodoVolatilidad = min(20, t);
    if t >= periodoVolatilidad
        % Volatilidad como desviación estándar de retornos
        volatilidades = zeros(numAssets, 1);
        for asset = 1:numAssets
            ventanaVol = RetornosMedios(asset, max(1, t+windowSize-periodoVolatilidad):t+windowSize-1);
            volatilidades(asset) = std(ventanaVol);
        end
        volatilidadPromedio = mean(volatilidades);
        
        % Identificar tendencia (direccionalidad del mercado)
        tendencias = zeros(numAssets, 1);
        for asset = 1:numAssets
            ventanaTrend = RetornosMedios(asset, max(1, t+windowSize-periodoVolatilidad):t+windowSize-1);
            % Calcular correlación entre tiempo y precios (aproximación de tendencia)
            tiempos = 1:length(ventanaTrend);
            if length(ventanaTrend) > 3
                coefCorr = corrcoef(tiempos, ventanaTrend);
                tendencias(asset) = abs(coefCorr(1,2)); % Valor absoluto de correlación
            else
                tendencias(asset) = 0;
            end
        end
        tendenciaPromedio = mean(tendencias);
        
        % Ajustar proporción MACD/IA basado en régimen detectado
        % Alta volatilidad + alta tendencia = favorable para MACD
        % Baja volatilidad o baja tendencia = favorable para IA
        
        % Umbrales de decisión (ajustables)
        umbralVolatilidadAlta = 0.015;
        umbralTendenciaAlta = 0.6;
        
        if volatilidadPromedio > umbralVolatilidadAlta && tendenciaPromedio > umbralTendenciaAlta
            % Mercado con tendencia clara y volátil: MACD domina (90%)
            proporcionMACD = 0.9;
        elseif volatilidadPromedio <= umbralVolatilidadAlta && tendenciaPromedio <= umbralTendenciaAlta
            % Mercado lateral o de baja volatilidad: IA tiene más peso (40%)
            proporcionMACD = 0.6;
        else
            % Condiciones mixtas: mantener balance estándar (80%)
            proporcionMACD = 0.8;
        end
    else
        % No hay suficientes datos, usar la proporción estándar
        proporcionMACD = 0.8;
    end
    
    % === IMPLEMENTACIÓN DE LA ESTRATEGIA HÍBRIDA IA-MACD ===
    % Integrar señales de MACD e IA usando la proporción determinada por el régimen de mercado
    
    % 1. Partir de la base MACD según proporción detectada
    pesosIAMACD = pesosMACD * proporcionMACD;
    restoPorAsignar = 1 - proporcionMACD;
    
    % 2. Identificar los mejores picks de IA que no contradigan a MACD
    [valoresIA, indicesIA] = sort(pesosIA, 'descend');
    topNIA = min(3, numAssets); % Considerar top 3 activos de IA
    
    % Asignar resto de la asignación a los mejores picks de IA que no contradigan MACD
    pesosPorIA = zeros(numAssets, 1);
    
    for i = 1:topNIA
        idx = indicesIA(i);
        % Solo aumentar peso si MACD no da señal de venta para este activo
        if currentMacdSignals(idx) >= 0 && pesosIA(idx) > 1/numAssets && restoPorAsignar > 0
            % Asignar peso proporcional a su ranking en IA
            asignacion = min(restoPorAsignar, 0.15); % máximo 15% por activo
            pesosPorIA(idx) = asignacion;
            restoPorAsignar = restoPorAsignar - asignacion;
        end
    end
    
    % 3. Si queda peso por asignar, distribuirlo entre los activos con mejores señales MACD
    if restoPorAsignar > 0.01 % Si queda más del 1% por asignar
        [~, indicesMACD] = sort(pesosMACD, 'descend');
        for i = 1:numAssets
            idx = indicesMACD(i);
            if currentMacdSignals(idx) == 1 && restoPorAsignar > 0
                incremento = min(restoPorAsignar, 0.05); % incrementos de 5%
                pesosIAMACD(idx) = pesosIAMACD(idx) + incremento;
                restoPorAsignar = restoPorAsignar - incremento;
            end
        end
    end
    
    % 4. Combinar ambas asignaciones
    pesosIAMACD = pesosIAMACD + pesosPorIA;
    
    % 5. Aplicar límite de exposición y diversificación
    maxPosicionPorActivo = 0.20; % Máximo 20% por activo
    for i = 1:numAssets
        if pesosIAMACD(i) > maxPosicionPorActivo
            pesosIAMACD(i) = maxPosicionPorActivo;
        end
    end
    
    % 6. Normalizar para asegurar que la suma es 1
    if sum(pesosIAMACD) > 0
        pesosIAMACD = pesosIAMACD / sum(pesosIAMACD);
    else
        pesosIAMACD = ones(numAssets, 1) / numAssets;
    end
    
    % Obtener retorno para el siguiente periodo
    retorno = RetornosMedios(:, t + windowSize);
    
    % Inicializar vectores de pesos anteriores para calcular costos de transacción
    if t == 1
        pesosMACD_anterior = zeros(size(pesosMACD));
        pesosIA_anterior = zeros(size(pesosIA));
        pesosIAMACD_anterior = zeros(size(pesosIAMACD));
    end
    
    % Calcular costos de transacción para cada estrategia
    costoTransaccion = 0.001; % 0.1% por operación
    
    cambiosPesosMACD = sum(abs(pesosMACD - pesosMACD_anterior)) / 2;
    cambiosPesosIA = sum(abs(pesosIA - pesosIA_anterior)) / 2;
    cambiosPesosIAMACD = sum(abs(pesosIAMACD - pesosIAMACD_anterior)) / 2;
    
    penalizacionMACD = cambiosPesosMACD * costoTransaccion;
    penalizacionIA = cambiosPesosIA * costoTransaccion;
    penalizacionIAMACD = cambiosPesosIAMACD * costoTransaccion;
    
    % Aplicar costos de transacción a los retornos
    rMACD = sum(pesosMACD .* retorno) - penalizacionMACD;
    rIA = sum(pesosIA .* retorno) - penalizacionIA;
    rIAMACD = sum(pesosIAMACD .* retorno) - penalizacionIAMACD;
    
    % Limitar pérdidas extremas (protección)
    rMACD = max(rMACD, -0.95);
    rIA = max(rIA, -0.95);
    rIAMACD = max(rIAMACD, -0.95);
    
    % Actualizar valores de portafolio
    valueMACD = valueMACD * (1 + rMACD);
    valueIA = valueIA * (1 + rIA);
    valueIAMACD = valueIAMACD * (1 + rIAMACD);
    
    % Guardar historias
    seriesMACD(t) = valueMACD;
    seriesIA(t) = valueIA;
    seriesIAMACD(t) = valueIAMACD;
    
    % Guardar pesos actuales para el siguiente período
    pesosMACD_anterior = pesosMACD;
    pesosIA_anterior = pesosIA;
    pesosIAMACD_anterior = pesosIAMACD;
    
    % Registrar valores para análisis posterior
    volatilidad_hist(t) = volatilidadPromedio;
    tendencia_hist(t) = tendenciaPromedio;
    proporcionMACD_hist(t) = proporcionMACD;
    
    % Mostrar progreso
    if mod(t, ceil(steps/10)) == 0
        fprintf('Progreso: %.1f%% completado\n', t/steps*100);
    end
end

% Calcular métricas de rendimiento
finalReturnMACD = (valueMACD - 1) * 100;
finalReturnIA = (valueIA - 1) * 100;
finalReturnIAMACD = (valueIAMACD - 1) * 100;

dailyReturnsMACD = diff([1, seriesMACD]) ./ [1, seriesMACD(1:end-1)];
dailyReturnsIA = diff([1, seriesIA]) ./ [1, seriesIA(1:end-1)];
dailyReturnsIAMACD = diff([1, seriesIAMACD]) ./ [1, seriesIAMACD(1:end-1)];

% Calcular Sharpe Ratio (asumiendo 252 días de trading al año)
sharpeMACD = mean(dailyReturnsMACD(2:end)) / std(dailyReturnsMACD(2:end)) * sqrt(252);
sharpeIA = mean(dailyReturnsIA(2:end)) / std(dailyReturnsIA(2:end)) * sqrt(252);
sharpeIAMACD = mean(dailyReturnsIAMACD(2:end)) / std(dailyReturnsIAMACD(2:end)) * sqrt(252);

% Calcular Drawdown máximo
drawdownMACD = calculateMaxDrawdown(seriesMACD);
drawdownIA = calculateMaxDrawdown(seriesIA);
drawdownIAMACD = calculateMaxDrawdown(seriesIAMACD);

% === Mostrar resultados ===
fprintf('\n=== RESULTADOS DE LA SIMULACIÓN ===\n');
fprintf('Estrategia MACD:         Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnMACD, sharpeMACD, drawdownMACD*100);
fprintf('Estrategia IA:           Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnIA, sharpeIA, drawdownIA*100);
fprintf('Estrategia IA-MACD:      Retorno: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    finalReturnIAMACD, sharpeIAMACD, drawdownIAMACD*100);

% === Gráfico comparativo ===
figure;
plot(seriesMACD, 'b', 'LineWidth', 2); hold on;
plot(seriesIA, 'g', 'LineWidth', 2);
plot(seriesIAMACD, 'r-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('MACD vs IA vs IA-MACD');
legend('MACD', 'IA', 'IA-MACD');
grid on;

% Guardar gráfico
saveFigurePath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'figures');
if ~exist(saveFigurePath, 'dir')
    mkdir(saveFigurePath);
end
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_ia_comparison.png'));
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_ia_comparison.fig'));

% === Gráfico de régimen de mercado ===
figure;
subplot(3,1,1);
plot(volatilidad_hist, 'b', 'LineWidth', 1.5);
title('Volatilidad Promedio del Mercado');
xlabel('Paso'); ylabel('Volatilidad');
grid on;

subplot(3,1,2);
plot(tendencia_hist, 'g', 'LineWidth', 1.5);
title('Tendencia Promedio del Mercado');
xlabel('Paso'); ylabel('Tendencia (|corr|)');
grid on;

subplot(3,1,3);
plot(proporcionMACD_hist, 'r', 'LineWidth', 1.5);
title('Proporción Asignada a MACD');
xlabel('Paso'); ylabel('Proporción');
ylim([0.5, 1.0]);
grid on;

saveas(gcf, fullfile(saveFigurePath, 'macd_vs_ia_regimen_mercado.png'));
saveas(gcf, fullfile(saveFigurePath, 'macd_vs_ia_regimen_mercado.fig'));

fprintf("✅ Simulación completada: MACD vs IA.\n");
fprintf("✅ Nueva estrategia híbrida IA-MACD implementada para proyecto de IA.\n");
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