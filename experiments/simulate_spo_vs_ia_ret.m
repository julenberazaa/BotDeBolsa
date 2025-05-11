clc;
clear;
close all;

load('ReaderBeginingDLR.mat');
load('spo_ajustada_ret.mat');  % Red entrenada con sesgo hacia retorno

% === PARÁMETROS ===
windowSize = 5;
alpha = 0.1;
[numAssets, numSteps] = size(RetornosMedios);
steps = numSteps - windowSize;

valueIA = 1;
valueSPO = 1;
valueEqual = 1;

weightsEqual = ones(numAssets, 1) / numAssets;

seriesIA = zeros(1, steps);
seriesSPO = zeros(1, steps);
seriesEqual = zeros(1, steps);

for t = 1:steps
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    RetornosPromedio = mean(ventana, 2);
    VarianzaRetornos = var(ventana, 0, 2);

    try
        pesosSPO = obtenerSPO(RetornosPromedio, VarianzaRetornos, alpha);
    catch
        pesosSPO = weightsEqual;
    end

    inputIA = normalize(ventana(:), 1);  % Normalizar igual que en entrenamiento

    try
        pesosIA = predict(net, inputIA');  % entrada como fila
        pesosIA = pesosIA';
    catch
        pesosIA = weightsEqual;
    end

    % Normalizar pesos IA
    pesosIA = max(pesosIA, 0);
    pesosIA = pesosIA / (sum(pesosIA) + 1e-10);

    retorno = RetornosMedios(:, t + windowSize);

    rSPO = sum(pesosSPO .* retorno);
    rIA = sum(pesosIA .* retorno);
    rEqual = sum(weightsEqual .* retorno);

    rSPO = max(rSPO, -0.95);
    rIA = max(rIA, -0.95);
    rEqual = max(rEqual, -0.95);

    valueSPO = valueSPO * (1 + rSPO);
    valueIA = valueIA * (1 + rIA);
    valueEqual = valueEqual * (1 + rEqual);

    seriesSPO(t) = valueSPO;
    seriesIA(t) = valueIA;
    seriesEqual(t) = valueEqual;
end

% === Gráfico comparativo ===
figure;
plot(seriesSPO, 'g', 'LineWidth', 2); hold on;
plot(seriesIA, 'b--', 'LineWidth', 2);
plot(seriesEqual, 'r-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('SPO Real vs IA Ajustada Retorno vs Igual Peso');
legend('SPO', 'IA Ajustada', 'Igual Peso');
grid on;

fprintf("✅ Simulación completada: IA ajustada vs SPO vs baseline.\n");
