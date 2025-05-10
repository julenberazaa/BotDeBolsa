
clc;
clear;
close all;

load("ReaderBeginingDLR.mat");  % Carga: RetornosMedios [Activos x Tiempos]

% === PREPROCESADO DE DATOS ===
RetornosPromedio = mean(RetornosMedios, 2);         % [Activos x 1]
RetornosMedios = RetornosPromedio(:);               % Forzar columna
VarianzaRetornos = VarianzaRetornos(:);

% Eliminar datos con NaNs
validos = ~isnan(RetornosMedios);
RetornosMedios = RetornosMedios(validos);
VarianzaRetornos = VarianzaRetornos(validos);

% === PARÁMETROS DEL ALGORITMO SPO ===
nP = 500;                          % Nº de partículas
nVar = length(RetornosMedios);   % Nº de activos
nIteraciones = 100;               % Nº de iteraciones
w = 0.1;                          % Inercia
phi1Max = 0.2;
phi2Max = 0.2;
alphaValue = 0.1;

% === INICIALIZACIÓN ===
costes = inf(1, nP);
x = zeros(nVar, nP);
v = random("Uniform", -0.1, 0.1, nVar, nP);
costeGlobalHist = zeros(nIteraciones, 1);

for i = 1:nP
    while true
        xcand = rand(nVar, 1);
        xcand = xcand / sum(xcand);
        coste = calcularCoste(xcand, alphaValue, RetornosMedios, VarianzaRetornos);
        if isfinite(coste)
            x(:, i) = xcand;
            costes(i) = coste;
            break;
        end
    end
end

xOptimo = x;
costesOptimos = costes;
[costeOptimoGlobal, ref] = min(costesOptimos);
xOptimoGlobal = xOptimo(:, ref);

% === BUCLE PRINCIPAL ===
for t = 1:nIteraciones
    for i = 1:nP
        phi1 = rand * phi1Max;
        phi2 = rand * phi2Max;

        v(:, i) = w*v(:, i) + phi1*(xOptimo(:, i) - x(:, i)) + phi2*(xOptimoGlobal - x(:, i));
        x(:, i) = x(:, i) + v(:, i);

        x(:, i) = max(0, x(:, i));  % Eliminar negativos
        x(:, i) = x(:, i) / sum(x(:, i) + 1e-10);  % Normalizar

        coste = calcularCoste(x(:, i), alphaValue, RetornosMedios, VarianzaRetornos);

        if coste < costesOptimos(i)
            costesOptimos(i) = coste;
            xOptimo(:, i) = x(:, i);

            if coste < costeOptimoGlobal
                costeOptimoGlobal = coste;
                xOptimoGlobal = x(:, i);
            end
        end
    end

    costeGlobalHist(t) = costeOptimoGlobal;

    % === VISUALIZACIÓN DEL PROCESO (OPCIONAL) ===
    figure(1); clf;
    subplot(1, 2, 1);
    plot(x(1,:), x(2,:), 'xr');
    grid on;
    xlabel('x_1'); ylabel('x_2');
    title('Trayectoria de las Partículas');

    subplot(1, 2, 2);
    plot(1:t, costeGlobalHist(1:t), '-ob');
    grid on;
    xlabel('Iteración'); ylabel('Coste óptimo global');
    title('Evolución del Coste');
    drawnow;
end

% === VISUALIZACIÓN FINAL ===
figure;
bar(xOptimoGlobal);
xlabel('Activo'); ylabel('Peso óptimo asignado');
title('Pesos del Portafolio Óptimo (SPO)');
grid on;

figure;
plot(1:nIteraciones, costeGlobalHist, '-o', 'LineWidth', 2);
xlabel('Iteración'); ylabel('Coste óptimo global');
title('Evolución del Coste del Portafolio');
grid on;


% === FUNCIÓN DE COSTE ===
function [M] = calcularCoste(w, alpha, rMedios, vRetornos)
    w = w(:);
    rMedios = rMedios(:);
    vRetornos = vRetornos(:);

    if ~(length(w) == length(rMedios) && length(w) == length(vRetornos))
        error('Dimensiones incompatibles.');
    end

    M = sum((w.^2) .* vRetornos) - alpha * (w' * rMedios);

    if any(w < 0) || any(w > 1) || sum(w) > 1
        M = inf;
    end
end

% === ANÁLISIS DE REGRESIÓN RENDIMIENTO-RIESGO ===
figure;

% Crear gráfico de dispersión de activos (rendimiento vs riesgo)
scatter(VarianzaRetornos, RetornosMedios, 50, xOptimoGlobal, 'filled');
hold on;

% Añadir regresión lineal
coef = polyfit(VarianzaRetornos, RetornosMedios, 1);
x_reg = linspace(min(VarianzaRetornos), max(VarianzaRetornos), 100);
y_reg = polyval(coef, x_reg);
plot(x_reg, y_reg, 'r--', 'LineWidth', 2);

% Calcular y mostrar ratio de Sharpe para cada activo
ratioSharpe = RetornosMedios ./ sqrt(VarianzaRetornos);
[ratioSharpeOrdenado, indicesSharpe] = sort(ratioSharpe, 'descend');

% Anotar los mejores activos según ratio de Sharpe
for i = 1:min(5, length(ratioSharpe))
    text(VarianzaRetornos(indicesSharpe(i)), RetornosMedios(indicesSharpe(i)), ...
         [' ' num2str(indicesSharpe(i))], 'FontWeight', 'bold');
end

% Personalizar gráfico
colorbar('Description', 'Peso en portafolio');
xlabel('Varianza (Riesgo)');
ylabel('Retorno Medio');
title('Análisis Rendimiento-Riesgo con Pesos Óptimos');
grid on;

% Añadir cuadrante de "inversiones ideales" (alto rendimiento, bajo riesgo)
mediana_riesgo = median(VarianzaRetornos);
mediana_retorno = median(RetornosMedios);
plot([min(VarianzaRetornos) mediana_riesgo mediana_riesgo min(VarianzaRetornos) min(VarianzaRetornos)], ...
     [mediana_retorno mediana_retorno max(RetornosMedios)*1.1 max(RetornosMedios)*1.1 mediana_retorno], ...
     'g-', 'LineWidth', 1.5);
text(min(VarianzaRetornos)*1.1, max(RetornosMedios), 'ZONA ÓPTIMA', 'Color', 'g', 'FontWeight', 'bold');

% === GRÁFICO DE EFICIENCIA (RATIO SHARPE) ===
figure;
bar(ratioSharpe);
hold on;
plot(1:length(ratioSharpe), ones(size(ratioSharpe))*mean(ratioSharpe), 'r--', 'LineWidth', 2);
xlabel('Activo');
ylabel('Ratio Retorno/Riesgo');
title('Eficiencia de cada Activo (Ratio de Sharpe)');
grid on;

% === ANÁLISIS DE CONTRIBUCIÓN AL PORTAFOLIO ===
figure;
% Contribución al rendimiento vs contribución al riesgo
contribRendimiento = xOptimoGlobal .* RetornosMedios;
contribRiesgo = (xOptimoGlobal.^2) .* VarianzaRetornos;
scatter(contribRiesgo, contribRendimiento, 70, xOptimoGlobal, 'filled');
hold on;

% Línea de regresión de contribuciones
coef_contrib = polyfit(contribRiesgo, contribRendimiento, 1);
x_contrib = linspace(min(contribRiesgo), max(contribRiesgo), 100);
y_contrib = polyval(coef_contrib, x_contrib);
plot(x_contrib, y_contrib, 'r--', 'LineWidth', 2);

% Anotar activos con mayor contribución
[~, idxContrib] = sort(contribRendimiento ./ (contribRiesgo + eps), 'descend');
for i = 1:min(5, length(idxContrib))
    text(contribRiesgo(idxContrib(i)), contribRendimiento(idxContrib(i)), ...
         [' ' num2str(idxContrib(i))], 'FontWeight', 'bold');
end

colorbar('Description', 'Peso en portafolio');
xlabel('Contribución al Riesgo');
ylabel('Contribución al Rendimiento');
title('Contribución de cada Activo al Portafolio Óptimo');
grid on;

% === GRÁFICA DE EVALUACIÓN INTEGRAL DE ACTIVOS ===
figure;
tiledlayout(2,1);

% Panel superior: Pesos óptimos vs Ratio de Sharpe
ax1 = nexttile;
yyaxis left
bar(xOptimoGlobal, 'FaceColor', [0.3 0.6 0.9]);
ylabel('Peso Óptimo');
ylim([0 max(xOptimoGlobal)*1.1]);

yyaxis right
plot(ratioSharpe, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
ylabel('Ratio de Sharpe');
grid on;
title('Evaluación Integral de Activos para Inversión');
xlabel('Activo');
legend('Peso Asignado', 'Ratio de Sharpe', 'Location', 'NorthWest');

% Panel inferior: Score compuesto (combinación normalizada de métricas)
ax2 = nexttile;
% Normalizar métricas al rango [0,1]
sharpeNorm = (ratioSharpe - min(ratioSharpe))/(max(ratioSharpe)-min(ratioSharpe));
contribRatio = contribRendimiento./(contribRiesgo + eps);
contribNorm = (contribRatio - min(contribRatio))/(max(contribRatio)-min(contribRatio));
% Métricas compuestas
scoreCompuesto = 0.4*sharpeNorm + 0.3*contribNorm + 0.3*xOptimoGlobal/max(xOptimoGlobal);
bar(scoreCompuesto, 'FaceColor', [0.2 0.7 0.3]);
ylabel('Score Compuesto (0-1)');
xlabel('Activo');
grid on;
title('Score Integral: Combinación de Ratio Sharpe, Contribución y Peso Asignado');

% Añadir etiquetas a los 3 mejores activos
[scoreOrdenado, idxScore] = sort(scoreCompuesto, 'descend');
for i = 1:3
    text(idxScore(i), scoreCompuesto(idxScore(i))+0.05, ...
        ['\bf{#' num2str(i) '}'], 'HorizontalAlignment', 'center');
end