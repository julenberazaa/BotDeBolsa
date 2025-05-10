clc;
clear;
close all;

load('ReaderBeginingDLR.mat');

% === PARÁMETROS ===
windowSize = 5;
alpha = 0.1;
gamma = 0.2;  % factor de ajuste hacia retorno
[numAssets, numSteps] = size(RetornosMedios);
numSamples = numSteps - windowSize;

X = [];
Y = [];

fprintf("⏳ Generando dataset ajustado SPO + retorno...\n");

for t = 1:numSamples
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    RetornosPromedio = mean(ventana, 2);
    VarianzaRetornos = var(ventana, 0, 2);

    try
        wOpt = obtenerSPO(RetornosPromedio, VarianzaRetornos, alpha);
        
        % Ajuste hacia retorno esperado
        ajuste = gamma * (RetornosPromedio - mean(RetornosPromedio));
        wAdj = wOpt + ajuste;

        % Limpiar negativos y normalizar
        wAdj = max(wAdj, 0);
        wAdj = wAdj / (sum(wAdj) + 1e-10);

        if ~any(isnan(wAdj)) && all(wAdj >= 0) && abs(sum(wAdj) - 1) < 0.01
            X = [X, ventana(:)];
            Y = [Y, wAdj];
        end
    catch
        continue;
    end
end

fprintf("✅ Muestras válidas generadas: %d\n", size(X, 2));
if isempty(X)
    error("❌ No se generaron muestras válidas.");
end

% === NORMALIZAR ENTRADAS ===
X = normalize(X, 2);

% === ARQUITECTURA IA ===
inputs = X;
targets = Y;

layers = [
    featureInputLayer(size(inputs,1), "Name", "input")
    fullyConnectedLayer(128, "Name", "fc1")
    reluLayer("Name", "relu1")
    fullyConnectedLayer(64, "Name", "fc2")
    reluLayer("Name", "relu2")
    fullyConnectedLayer(numAssets, "Name", "output")
    regressionLayer("Name", "regression")
];

options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 16, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

XTrain = inputs';
YTrain = targets';

fprintf("🚀 Entrenando red IA ajustada hacia retorno...\n");
net = trainNetwork(XTrain, YTrain, layers, options);

save('spo_ajustada_ret.mat', 'net');
fprintf("✅ Red IA ajustada guardada como 'spo_ajustada_ret.mat'\n");

% === VISUALIZAR SALIDAS ===
YHat = predict(net, XTrain)';
figure;
imagesc(YHat);
colorbar;
title('Predicción de pesos IA ajustada hacia retorno');
xlabel('Muestra'); ylabel('Activo');
