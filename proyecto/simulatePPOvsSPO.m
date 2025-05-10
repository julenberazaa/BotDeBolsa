clc;
clear;
close all;

% === Cargar datos y modelos ===
load('ReaderBeginingDLR.mat');
load('ppo_agent_spoguided.mat', 'agent');

% === Parámetros ===
windowSize = 5;
alpha = 0.1;
[numAssets, numSteps] = size(RetornosMedios);
steps = numSteps - windowSize;

valuePPO = 1;
valueSPO = 1;
valueEqual = 1;

weightsEqual = ones(numAssets, 1) / numAssets;
seriesPPO = zeros(1, steps);
seriesSPO = zeros(1, steps);
seriesEqual = zeros(1, steps);

for t = 1:steps
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    obs = ventana(:);

    % === Acción del PPO ===
    action = getAction(agent, obs);
    if isstruct(action)
        action = action{1};
    elseif iscell(action)
        action = action{1};
    end
    action = double(action);
    action = max(action, 0);
    action = action / (sum(action) + 1e-10);
    pesosPPO = action;

    % === Acción del SPO ===
    mu = mean(ventana, 2);
    sigma = var(ventana, 0, 2);
    try
        pesosSPO = obtenerSPO(mu, sigma, alpha);
    catch
        pesosSPO = weightsEqual;
    end

    % === Retornos reales ===
    retorno = RetornosMedios(:, t + windowSize);
    
    rPPO = sum(pesosPPO .* retorno);
    rSPO = sum(pesosSPO .* retorno);
    rEqual = sum(weightsEqual .* retorno);

    valuePPO = valuePPO * (1 + rPPO);
    valueSPO = valueSPO * (1 + rSPO);
    valueEqual = valueEqual * (1 + rEqual);

    seriesPPO(t) = valuePPO;
    seriesSPO(t) = valueSPO;
    seriesEqual(t) = valueEqual;
end

% === Gráfico comparativo ===
figure;
plot(seriesPPO, 'b', 'LineWidth', 2); hold on;
plot(seriesSPO, 'g--', 'LineWidth', 2);
plot(seriesEqual, 'r-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('Comparación: PPO vs SPO vs Pesos Iguales');
legend('PPO','SPO','Igual Peso');
grid on;
