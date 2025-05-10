clc;
clear;
close all;

load('ReaderBeginingDLR.mat');

% === PARAMETROS ===
windowSize = 5;
alpha = 0.1;
[numAssets, numSteps] = size(RetornosMedios);
numSamples = numSteps - windowSize;

X = [];
Y = [];

fprintf("⏳ Generando dataset IA a partir de SPO...\n");

for t = 1:numSamples
    ventana = RetornosMedios(:, t:t + windowSize - 1);
    mu = mean(ventana, 2);
    sigma = var(ventana, 0, 2);

    try
        wOpt = obtenerSPO(mu, sigma, alpha);
        if ~any(isnan(wOpt)) && all(wOpt >= 0) && abs(sum(wOpt) - 1) < 0.01
            X = [X, ventana(:)];
            Y = [Y, wOpt];
        end
    catch
        continue;
    end
end

fprintf("✅ Muestras válidas generadas: %d\n", size(X, 2));

if isempty(X)
    error("❌ SPO no generó ningún conjunto de pesos válido.");
end

% === NORMALIZAR entradas (por fila) ===
X = normalize(X, 2);

% === ARCHITECTURA IA ===
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
    'MaxEpochs', 300, ...
    'MiniBatchSize', 64, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

XTrain = inputs';
YTrain = targets';

fprintf("🚀 Entrenando red IA profunda para imitar SPO...\n");
net = trainNetwork(XTrain, YTrain, layers, options);

% === GUARDAR ===
save('spo_imitator_deep.mat', 'net');
fprintf("✅ Red guardada como 'spo_imitator_deep.mat'\n");

% === VALIDACION VISUAL ===
YHat = predict(net, XTrain)';
figure;
imagesc(YHat);
colorbar;
title('Predicción de pesos por la red IA');
xlabel('Muestra'); ylabel('Activo');
