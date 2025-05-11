function agent = simple_macd_agent(prices, volumes, fastPeriod, slowPeriod, signalPeriod, filterConfig)
% SIMPLE_MACD_AGENT - Versión simplificada del MACD agent para usar como fallback
%
% Esta función crea una estructura simple con interfaz compatible con el enhanced_macd_agent
% pero sin usar características avanzadas de objetos.
%
% Inputs:
%   prices - Matriz de precios (activos x tiempo)
%   volumes - Matriz de volúmenes (opcional)
%   fastPeriod - Periodo rápido del MACD
%   slowPeriod - Periodo lento del MACD
%   signalPeriod - Periodo de la línea de señal
%   filterConfig - Configuración de filtros (opcional)
%
% Output:
%   agent - Estructura con los siguientes campos:
%       .signals - Matriz de señales MACD
%       .getSignals - Función para obtener señales en un tiempo específico

% Validar parámetros
if nargin < 2 || isempty(volumes)
    volumes = [];
end

if nargin < 3 || isempty(fastPeriod)
    fastPeriod = 12;
end

if nargin < 4 || isempty(slowPeriod)
    slowPeriod = 26;
end

if nargin < 5 || isempty(signalPeriod)
    signalPeriod = 9;
end

if nargin < 6 || isempty(filterConfig)
    filterConfig = struct(...
        'volumeThreshold', 1.5, ...
        'histogramThreshold', 0.001, ...
        'trendConfirmation', true, ...
        'signalThreshold', 0.3);
end

% Asegurar que los precios están en formato correcto (activos x tiempo)
if size(prices, 1) > size(prices, 2)
    prices = prices';
end

% Calcular señales MACD para todos los activos
[numAssets, numSteps] = size(prices);
signals = zeros(numAssets, numSteps);

for asset = 1:numAssets
    assetPrices = prices(asset, :);
    
    % Obtener volúmenes si están disponibles
    if ~isempty(volumes) && size(volumes, 1) >= asset
        assetVolumes = volumes(asset, :);
    else
        assetVolumes = [];
    end
    
    % Calcular señales MACD
    try
        % Intentar usar MACD estrategia mejorada si está disponible
        if exist('enhanced_macd_strategy', 'file') == 2
            signals(asset, :) = enhanced_macd_strategy(assetPrices', assetVolumes', ...
                fastPeriod, slowPeriod, signalPeriod, filterConfig)';
        else
            % Usar MACD básico como fallback
            signals(asset, :) = macd_strategy(assetPrices, ...
                fastPeriod, slowPeriod, signalPeriod)';
        end
    catch
        % Si hay error, usar MACD básico
        signals(asset, :) = macd_strategy(assetPrices, ...
            fastPeriod, slowPeriod, signalPeriod)';
    end
end

% Crear estructura de agent compatible
agent = struct();
agent.signals = signals;
agent.getSignals = @(t) getSignalsAt(signals, t);

% Función interna para obtener señales en tiempo t
function sigs = getSignalsAt(signalMatrix, t)
    if t > 0 && t <= size(signalMatrix, 2)
        sigs = signalMatrix(:, t);
    else
        % Si el tiempo está fuera de rango, devolver ceros
        sigs = zeros(size(signalMatrix, 1), 1);
    end
end

end 