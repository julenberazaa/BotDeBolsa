function [model] = train_ia_simple_robust(prices, macdConfig)
% TRAIN_IA_SIMPLE_ROBUST - Versión simplificada y robusta del entrenamiento IA
%
% Esta función entrena un modelo de IA simplificado para complementar la estrategia MACD
% con mejor manejo de errores y compatibilidad con diferentes versiones de MATLAB.
%
% Inputs:
%   prices - Matriz de precios de activos [activos x tiempo]
%   macdConfig - Estructura con parámetros MACD (opcional)
%
% Output:
%   model - Función handle del modelo entrenado (o fallback si falla)

fprintf('Iniciando entrenamiento de modelo IA robusto...\n');

% === Validación de parámetros ===
if nargin < 2 || isempty(macdConfig)
    macdConfig = struct();
end

if ~isfield(macdConfig, 'fastPeriod')
    macdConfig.fastPeriod = 12;
end

if ~isfield(macdConfig, 'slowPeriod')
    macdConfig.slowPeriod = 26;
end

if ~isfield(macdConfig, 'signalPeriod')
    macdConfig.signalPeriod = 9;
end

% === Preparación de datos ===
fprintf('Preparando datos para entrenamiento...\n');

% Asegurar que los precios están en el formato correcto (activos x tiempo)
if size(prices, 1) > size(prices, 2)
    prices = prices'; % Transponer si está en formato columna
end

[numAssets, numSteps] = size(prices);

% Definir ventana para inputs
windowSize = 5;
if numSteps <= windowSize
    error('No hay suficientes datos para entrenamiento. Mínimo requerido: %d pasos.', windowSize+1);
end

% Calcular retornos
returns = zeros(size(prices));
for t = 2:numSteps
    returns(:, t) = (prices(:, t) - prices(:, t-1)) ./ prices(:, t-1);
end

% Intentar entrenar con el mejor método disponible
try
    % === Generar conjunto de entrenamiento ===
    fprintf('Generando conjunto de datos de entrenamiento...\n');
    X = [];
    Y = [];
    
    % Generar muestras de entrenamiento
    trainRange = windowSize+10:numSteps-5; % Dejar algunos datos para validación
    
    for t = trainRange
        % Obtener ventana actual de retornos
        window = returns(:, t-windowSize+1:t);
        
        % Calcular retornos promedio y varianza para el paso objetivo
        targetIdx = t + 1;
        avgReturns = mean(returns(:, max(1, targetIdx-10):targetIdx-1), 2);
        varReturns = var(returns(:, max(1, targetIdx-10):targetIdx-1), 0, 2);
        
        try
            % Intentar optimización SPO para target
            wOpt = obtenerSPO(avgReturns, varReturns, 0.1);
            
            if ~any(isnan(wOpt)) && all(wOpt >= 0) && abs(sum(wOpt) - 1) < 0.01
                % Aplanar y normalizar la ventana de entrada
                inputVector = window(:);

                % --- BEGIN CRITICAL NaN/Inf HANDLING for inputVector ---
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_simple_robust: NaNs found in inputVector for sample before X (t=%d). Replacing with 0.\n', t);
                    inputVector(isnan(inputVector)) = 0;
                end
                if any(isinf(inputVector))
                     fprintf('INFO train_ia_simple_robust: Infs found in inputVector for sample before X (t=%d). Replacing with 0.\n', t);
                    inputVector(isinf(inputVector)) = 0;
                end
                % --- END CRITICAL NaN/Inf HANDLING for inputVector ---
                
                inputVector = (inputVector - mean(inputVector)) / (std(inputVector) + 1e-10); % Normalization
                
                % --- BEGIN CRITICAL NaN/Inf HANDLING for inputVector AFTER NORMALIZATION ---
                % Check again after normalization, as division by zero std could create NaNs/Infs
                if any(isnan(inputVector))
                    fprintf('INFO train_ia_simple_robust: NaNs found in inputVector AFTER normalization (t=%d). Replacing with 0.\n', t);
                    inputVector(isnan(inputVector)) = 0;
                end
                if any(isinf(inputVector))
                     fprintf('INFO train_ia_simple_robust: Infs found in inputVector AFTER normalization (t=%d). Replacing with 0.\n', t);
                    inputVector(isinf(inputVector)) = 0;
                end
                % --- END CRITICAL NaN/Inf HANDLING for inputVector AFTER NORMALIZATION ---

                % Almacenar entrada y target
                X = [X, inputVector];
                Y = [Y, wOpt];
            end
        catch ME_SPO
            % Ignorar errores de optimización SPO y continuar, pero registrarlo
            fprintf('WARNING train_ia_simple_robust: SPO optimization failed for t=%d. Message: %s. Skipping sample.\n', t, ME_SPO.message);
            continue;
        end
    end
    
    numSamples = size(X, 2);
    
    if numSamples < 10
        error('No hay suficientes muestras válidas para entrenamiento (%d). Comprueba la calidad de los datos.', numSamples);
    end
    
    fprintf('Total de muestras de entrenamiento: %d\n', numSamples);
    
    % === Entrenamiento de red neuronal ===
    fprintf('Entrenando red neuronal...\n');
    
    % Intentar entrenar con la mejor opción disponible
    if exist('trainNetwork', 'file') == 2
        % Si Deep Learning Toolbox está disponible, usarlo
        try
            % Configuración para entrenamiento con Deep Learning Toolbox
            X_train_dl = X'; % Data for trainNetwork (samples as rows)
            Y_train_dl = Y'; % Data for trainNetwork (samples as rows)

            fprintf('DEBUG trainNetwork: Size X_train_dl: %s, Size Y_train_dl: %s\n', mat2str(size(X_train_dl)), mat2str(size(Y_train_dl)));
            fprintf('DEBUG trainNetwork: Class X_train_dl: %s, Class Y_train_dl: %s\n', class(X_train_dl), class(Y_train_dl));
            fprintf('DEBUG trainNetwork: Any NaN in X_train_dl: %d, Any Inf in X_train_dl: %d\n', any(isnan(X_train_dl(:))), any(isinf(X_train_dl(:))));
            fprintf('DEBUG trainNetwork: Any NaN in Y_train_dl: %d, Any Inf in Y_train_dl: %d\n', any(isnan(Y_train_dl(:))), any(isinf(Y_train_dl(:))));
            if ~isempty(Y_train_dl) && size(Y_train_dl,2) == numAssets
                row_sums_Y_train_dl = sum(Y_train_dl,2);
                fprintf('DEBUG trainNetwork: Min/Max Y_train_dl row sums: %.4f / %.4f (expected ~1 for softmax targets)\n', min(row_sums_Y_train_dl), max(row_sums_Y_train_dl));
                 if any(abs(row_sums_Y_train_dl - 1) > 1e-3)
                    fprintf('WARNING trainNetwork: Some Y_train_dl rows do not sum to 1!\n');
                end
            else
                 fprintf('DEBUG trainNetwork: Y_train_dl is empty or wrong number of columns for row sum check.\n');
            end

            % Temporarily Simplified layers for testing trainNetwork stability
            fprintf('DEBUG trainNetwork: Using temporarily simplified layers for stability testing.\n');
            
            % Define layers individually for layerGraph construction
            inputLayer = featureInputLayer(size(X,1), 'Normalization', 'zscore', 'Name', 'input');
            fcLayer = fullyConnectedLayer(numAssets, 'Name', 'fc');
            smLayer = softmaxLayer('Name', 'softmax');
            regLayer = regressionLayer('Name', 'output'); % Explicit output layer

            % Create Layer Graph
            lgraph = layerGraph();
            lgraph = addLayers(lgraph, inputLayer);
            lgraph = addLayers(lgraph, fcLayer);
            lgraph = addLayers(lgraph, smLayer);
            lgraph = addLayers(lgraph, regLayer);

            % Connect Layers
            lgraph = connectLayers(lgraph, 'input', 'fc');
            lgraph = connectLayers(lgraph, 'fc', 'softmax');
            lgraph = connectLayers(lgraph, 'softmax', 'output');
            
            options = trainingOptions('adam', ...
                'MaxEpochs', 50, ... % Reduced epochs for faster testing
                'MiniBatchSize', min(32, floor(numSamples/4)), ...
                'InitialLearnRate', 0.001, ...
                'Shuffle', 'every-epoch', ...
                'Verbose', false, ...
                'Plots', 'none'); % Disable plots for faster testing
            
            % Entrenar la red
            fprintf('DEBUG trainNetwork: Attempting to train with trainNetwork using lgraph...\n');
            net = trainNetwork(X_train_dl, Y_train_dl, lgraph, options);
            
            % Crear función handle para el modelo
            model = @(x) predictWithNet(net, x, numAssets);
            
            fprintf('Entrenamiento completado con Deep Learning Toolbox.\n');
            return; % Salir de la función ya que el entrenamiento fue exitoso
        catch ME1
            fprintf('Error en entrenamiento con Deep Learning Toolbox: %s\n', ME1.message);
            fprintf('Stack trace for ME1:\n%s\n', getReport(ME1, 'extended', 'hyperlinks','off'));
            fprintf('Intentando con método alternativo (feedforwardnet)...\n');
            % Continuar e intentar con feedforwardnet
        end
    end
    
    % Si Deep Learning Toolbox no está disponible o falló, usar feedforwardnet
    try
        % Intenta con una configuración básica para versiones antiguas o sin DL Toolbox
        net = feedforwardnet([64, 32]);
        net.trainParam.epochs = 100;
        net.trainParam.lr = 0.01;
        net = train(net, X, Y);
        
        % Crear función handle para el modelo
        model = @(x) predictWithFeedforward(net, x, numAssets);
        
        fprintf('Entrenamiento completado con red feedforward básica.\n');
        return; % Salir de la función ya que el entrenamiento fue exitoso
    catch ME2
        fprintf('Error en entrenamiento con red básica: %s\n', ME2.message);
        error('No se pudo entrenar ninguna red neuronal. %s', ME2.message);
    end
catch ME
    fprintf('Error en entrenamiento: %s\n', ME.message);
    fprintf('Usando modelo de inversión por volatilidad como alternativa.\n');
    
    % Crear modelo de inversión por volatilidad como fallback
    model = @(x) robust_inv_vol_model(x, numAssets);
end

fprintf('✅ Proceso de entrenamiento IA finalizado.\n');
end

% Función auxiliar para predicción con red moderna
function weights = predictWithNet(net, x, numAssets)
    try
        % Asegurar que la entrada está normalizada
        x = (x - mean(x)) / (std(x) + 1e-10);
        
        % Predecir con la red
        weights = predict(net, x');
        weights = weights';
        
        % Asegurar pesos no negativos que suman 1
        weights = max(weights, 0);
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = ones(numAssets, 1) / numAssets;
        end
    catch
        % Si hay error, usar pesos iguales
        weights = ones(numAssets, 1) / numAssets;
    end
end

% Función auxiliar para predicción con red feedforward
function weights = predictWithFeedforward(net, x, numAssets)
    try
        % Asegurar que la entrada está normalizada
        x = (x - mean(x)) / (std(x) + 1e-10);
        
        % Predecir con la red
        weights = net(x);
        
        % Asegurar pesos no negativos que suman 1
        weights = max(weights, 0);
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = ones(numAssets, 1) / numAssets;
        end
    catch
        % Si hay error, usar pesos iguales
        weights = ones(numAssets, 1) / numAssets;
    end
end

% Modelo de inversión por volatilidad mejorado
function weights = robust_inv_vol_model(inputVector, numAssets)
    try
        % Reshape la entrada para obtener la estructura original
        if numel(inputVector) ~= numAssets && mod(numel(inputVector), numAssets) == 0
            timeSteps = numel(inputVector) / numAssets;
            window = reshape(inputVector, numAssets, timeSteps);
        else
            % Si no es posible el reshape, usar pesos iguales
            weights = ones(numAssets, 1) / numAssets;
            return;
        end
        
        % Calcular volatilidad para cada activo con algoritmo robusto
        volEstimates = zeros(numAssets, 1);
        for i = 1:numAssets
            % Usar estimación robusta de volatilidad
            returns = diff(window(i, :));
            if ~isempty(returns) && ~all(isnan(returns))
                % Excluir valores extremos
                validReturns = returns(~isnan(returns) & abs(returns) < 5*std(returns(~isnan(returns))));
                if ~isempty(validReturns)
                    volEstimates(i) = std(validReturns) + 1e-6;
                else
                    volEstimates(i) = 0.01;
                end
            else
                volEstimates(i) = 0.01;
            end
        end
        
        % Asignar posiciones más conservadoras
        invVol = 1 ./ volEstimates;
        weights = invVol / sum(invVol);
        
        % Limitar posiciones individuales (máx 15% por activo)
        maxWeight = 0.15;
        for i = 1:numAssets
            if weights(i) > maxWeight
                excess = weights(i) - maxWeight;
                weights(i) = maxWeight;
                
                % Redistribuir el exceso
                remainingIdx = true(numAssets, 1);
                remainingIdx(i) = false;
                if any(remainingIdx)
                    weights(remainingIdx) = weights(remainingIdx) + excess * weights(remainingIdx)/sum(weights(remainingIdx));
                end
            end
        end
        
        % Asegurar que no hay NaNs y los pesos son positivos
        weights(isnan(weights)) = 1/numAssets;
        weights = max(weights, 0);
        
        % Normalizar para que sumen 1
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = ones(numAssets, 1) / numAssets;
        end
        
    catch
        % En caso de error, volver a pesos iguales
        weights = ones(numAssets, 1) / numAssets;
    end
end 