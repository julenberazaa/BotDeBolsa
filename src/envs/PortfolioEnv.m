classdef PortfolioEnv < rl.env.MATLABEnvironment
    properties
        Retornos
        NumAssets
        CurrentStep
        WindowSize = 5
        MaxSteps
        alpha = 0.1
        beta = 0.01
        logVerbose = false
        externalAgent  % Added external agent property
    end

    properties(Access = protected)
        IsDone = false
    end

    methods
        function this = PortfolioEnv(externalAgent)
            % Constructor for Portfolio Environment
            %
            % Inputs:
            %   externalAgent (optional) - External agent with getSignal method
            
            % Try to load data from different possible locations
            try
                % First try data in src/data/reader
                dummyData = load(fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'reader', 'ReaderBeginingDLR.mat'));
            catch
                try
                    % Then try in data/processed folder
                    dummyData = load(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data', 'processed', 'ReaderBeginingDLR.mat'));
                catch
                    try
                        % Then try in the original project location
                        dummyData = load('ReaderBeginingDLR.mat');
                    catch err
                        error('Could not find ReaderBeginingDLR.mat. Error: %s', err.message);
                    end
                end
            end
            
            numAssets = size(dummyData.RetornosMedios, 1);
            windowSize = 5;

            obsSpec = rlNumericSpec([numAssets * windowSize 1], ...
                'LowerLimit', -inf, 'UpperLimit', inf);
            obsSpec.Name = 'VentanaRetornos';

            actSpec = rlNumericSpec([numAssets 1], ...
                'LowerLimit', 0, 'UpperLimit', 1);
            actSpec.Name = 'PesosPortafolio';

            this = this@rl.env.MATLABEnvironment(obsSpec, actSpec);

            this.Retornos = dummyData.RetornosMedios;
            this.NumAssets = numAssets;
            this.WindowSize = windowSize;
            this.MaxSteps = size(this.Retornos, 2) - windowSize;
            
            % Set external agent if provided
            if nargin > 0
                this.externalAgent = externalAgent;
            else
                this.externalAgent = [];
            end

            reset(this);
        end

        function [obs, reward, isDone, log] = step(this, action)
            log = [];
            
            % Use external agent if available
            if ~isempty(this.externalAgent) && ismethod(this.externalAgent, 'getSignal')
                % Get signal from external agent
                signal = this.externalAgent.getSignal(this.CurrentStep);
                
                % Convert signal to action based on the signal: 
                % 1 (buy), -1 (sell), 0 (hold)
                if signal == 1  % Buy signal
                    equalWeight = ones(this.NumAssets, 1) / this.NumAssets;
                    action = equalWeight;
                elseif signal == -1  % Sell signal
                    action = zeros(this.NumAssets, 1);
                else  % Hold signal - use previous action if available
                    if nargin < 2 || isempty(action)
                        action = ones(this.NumAssets, 1) / this.NumAssets;
                    end
                end
            elseif nargin < 2 || isempty(action)
                % Default action if none provided
                action = ones(this.NumAssets, 1) / this.NumAssets;
            end

            action = action / (sum(action) + 1e-10);
            ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
            retorno = this.Retornos(:, this.CurrentStep + this.WindowSize);

            r = sum(action .* retorno);
            v = sum((action .^ 2) .* var(ventana, 0, 2));
            diversificacion = -sum(action .^ 2);
            rawReward = r - this.alpha * v + this.beta * diversificacion;

            try
                mu = mean(ventana, 2);
                sigma = var(ventana, 0, 2);
                spoPesos = obtenerSPO(mu, sigma, this.alpha);
                distancia = norm(action - spoPesos);
                rawReward = rawReward - 0.2 * distancia;
            catch
                distancia = 0;
            end

            reward = tanh(5 * rawReward);
            if ~isfinite(reward) || ~isscalar(reward)
                reward = 0;
            end

            this.CurrentStep = this.CurrentStep + 1;
            obs = this.getObservation();
            isDone = (this.CurrentStep >= this.MaxSteps);
            this.IsDone = isDone;
        end

        function obs = reset(this)
            this.CurrentStep = 1 + randi([0, this.MaxSteps - 1]);
            this.IsDone = false;
            obs = this.getObservation();
        end
        
        function setExternalAgent(this, agent)
            % Set a new external agent
            %
            % Inputs:
            %   agent - External agent with getSignal method
            
            this.externalAgent = agent;
        end
    end

    methods (Access = private)
        function obs = getObservation(this)
            ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
            obs = ventana(:);
        end
    end
end

function wOpt = obtenerSPO(mu, varianzas, alpha)
    n = length(mu);
    nP = 200;
    nIt = 50;
    w = 0.1;
    phi1Max = 0.2; phi2Max = 0.2;

    v = randn(n, nP) * 0.01;
    x = zeros(n, nP);
    c = inf(1, nP);

    for i = 1:nP
        x0 = rand(n, 1); x0 = x0 / sum(x0);
        cost = coste(x0);
        x(:, i) = x0;
        c(i) = cost;
    end

    xOpt = x;
    cOpt = c;
    [~, idx] = min(cOpt);
    xG = xOpt(:, idx);

    for t = 1:nIt
        for i = 1:nP
            phi1 = rand * phi1Max;
            phi2 = rand * phi2Max;
            v(:, i) = w*v(:, i) + phi1*(xOpt(:, i) - x(:, i)) + phi2*(xG - x(:, i));
            x(:, i) = x(:, i) + v(:, i);
            x(:, i) = max(0, x(:, i));
            x(:, i) = x(:, i) / sum(x(:, i) + 1e-10);
            cost = coste(x(:, i));
            if cost < cOpt(i)
                cOpt(i) = cost;
                xOpt(:, i) = x(:, i);
                if cost < coste(xG)
                    xG = x(:, i);
                end
            end
        end
    end

    wOpt = xG;

    function m = coste(w)
        m = sum((w.^2) .* varianzas) - alpha * (w' * mu);
        if any(w < 0) || any(w > 1) || sum(w) > 1.01
            m = inf;
        end
    end
end

% classdef PortfolioEnv < rl.env.MATLABEnvironment
%     properties
%         Retornos
%         NumAssets
%         CurrentStep
%         WindowSize = 5
%         MaxSteps
%         alpha = 0.1
%         beta = 0.01
%         logVerbose = false
%     end
% 
%     properties(Access = protected)
%         IsDone = false
%     end
% 
%     methods
%         function this = PortfolioEnv()
%             % === Cargar datos antes de usar 'this' ===
%             dummyData = load('ReaderBeginingDLR.mat');
%             numAssets = size(dummyData.RetornosMedios, 1);
%             windowSize = 5;
% 
%             % Crear specs de observación y acción
%             obsSpec = rlNumericSpec([numAssets * windowSize 1], ...
%                 'LowerLimit', -inf, 'UpperLimit', inf);
%             obsSpec.Name = 'VentanaRetornos';
% 
%             actSpec = rlNumericSpec([numAssets 1], ...
%                 'LowerLimit', 0, 'UpperLimit', 1);
%             actSpec.Name = 'PesosPortafolio';
% 
%             % === Llamar al constructor del padre ===
%             this = this@rl.env.MATLABEnvironment(obsSpec, actSpec);
% 
%             % Inicializar propiedades del entorno
%             this.Retornos = dummyData.RetornosMedios;
%             this.NumAssets = numAssets;
%             this.WindowSize = windowSize;
%             this.MaxSteps = size(this.Retornos, 2) - windowSize;
% 
%             reset(this);
%         end
% 
%         function [obs, reward, isDone, log] = step(this, action)
%             log = [];
% 
%             % Normalizar acción
%             action = action / (sum(action) + 1e-10);
% 
%             % Obtener retorno del siguiente paso
%             retorno = this.Retornos(:, this.CurrentStep + this.WindowSize);
% 
%             % Calcular retorno ponderado
%             r = sum(action .* retorno);
% 
%             % Calcular riesgo con varianza ponderada
%             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
%             v = sum((action .^ 2) .* var(ventana, 0, 2));
% 
%             % Penalizar concentración excesiva (diversificación)
%             diversificacion = -sum(action .^ 2);
% 
%             % Recompensa con función suave
%             rawReward = r - this.alpha * v + this.beta * diversificacion;
%             reward = tanh(5 * rawReward);  % Escalar recompensas
% 
%             if ~isfinite(reward) || ~isscalar(reward)
%                 reward = 0;
%             end
% 
%             % Avanzar el paso
%             this.CurrentStep = this.CurrentStep + 1;
%             obs = this.getObservation();
% 
%             isDone = (this.CurrentStep >= this.MaxSteps);
%             this.IsDone = isDone;
%         end
% 
%         function obs = reset(this)
%             this.CurrentStep = 1 + randi([0, this.MaxSteps - 1]);
%             this.IsDone = false;
%             obs = this.getObservation();
%         end
%     end
% 
%     methods (Access = private)
%         function obs = getObservation(this)
%             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
%             obs = ventana(:);
%         end
%     end
% end
% 
% % classdef PortfolioEnv < rl.env.MATLABEnvironment
% %     properties
% %         Retornos                      % Matriz de retornos históricos
% %         NumAssets                     % Número de activos
% %         CurrentStep                   % Paso actual
% %         WindowSize = 5                % Tamaño de la ventana de observación
% %         MaxSteps                      % Máximo de pasos por episodio
% %         alpha = 0.1                   % Penalización por riesgo
% %         beta = 0.01                   % Bonificación por diversificación
% %     end
% % 
% %     properties(Access = protected)
% %         IsDone = false                % Bandera de fin de episodio
% %     end
% % 
% %     methods
% %         function this = PortfolioEnv()
% %             % === Cargar datos y definir specs SIN USAR 'this' ===
% %             data = load('ReaderBeginingDLR.mat');
% %             retornos = data.RetornosMedios;
% %             numAssets = size(retornos, 1);
% %             windowSize = 5;
% %             maxSteps = size(retornos, 2) - windowSize;
% % 
% %             obsSpec = rlNumericSpec([numAssets * windowSize 1], ...
% %                 'LowerLimit', -inf, 'UpperLimit', inf);
% %             obsSpec.Name = 'VentanaRetornos';
% % 
% %             actSpec = rlNumericSpec([numAssets 1], ...
% %                 'LowerLimit', 0, 'UpperLimit', 1);
% %             actSpec.Name = 'PesosPortafolio';
% % 
% %             % === Llamar primero al constructor del padre ===
% %             this = this@rl.env.MATLABEnvironment(obsSpec, actSpec);
% % 
% %             % === Ahora sí se puede usar 'this' ===
% %             this.Retornos = retornos;
% %             this.NumAssets = numAssets;
% %             this.WindowSize = windowSize;
% %             this.MaxSteps = maxSteps;
% % 
% %             reset(this);
% %         end
% % 
% %         function [obs, reward, isDone, log] = step(this, action)
% %             log = [];
% % 
% %             % Normalizar acción para que sume 1
% %             action = action / (sum(action) + 1e-10);
% % 
% %             % Retorno actual
% %             retorno = this.Retornos(:, this.CurrentStep + this.WindowSize);
% % 
% %             % Ventana usada para cálculo de riesgo
% %             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
% % 
% %             % Calcular recompensa
% %             r = sum(action .* retorno);
% %             exceso = r - mean(retorno);
% %             riesgo = sum((action .^ 2) .* var(ventana, 0, 2));
% %             diversidad = -std(action);
% % 
% %             reward = exceso - this.alpha * riesgo + this.beta * diversidad;
% %             reward = min(max(reward, -10), 10);  % Limitar reward
% % 
% %             if ~isfinite(reward) || ~isscalar(reward)
% %                 reward = 0;
% %             end
% % 
% %             this.CurrentStep = this.CurrentStep + 1;
% %             obs = this.getObservation();
% %             isDone = (this.CurrentStep >= this.MaxSteps);
% %             this.IsDone = isDone;
% %         end
% % 
% %         function obs = reset(this)
% %             this.CurrentStep = 1 + randi([0, this.MaxSteps - 1]);
% %             this.IsDone = false;
% %             obs = this.getObservation();
% %         end
% %     end
% % 
% %     methods (Access = private)
% %         function obs = getObservation(this)
% %             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
% %             obs = ventana(:);
% %         end
% %     end
% % end
% % 
% % 
% % % classdef PortfolioEnv < rl.env.MATLABEnvironment
% % %     properties
% % %         Retornos
% % %         NumAssets
% % %         CurrentStep
% % %         WindowSize = 5
% % %         MaxSteps
% % %         alpha = 0.1  % penalización por riesgo
% % %         beta = 0.01  % bonificación por diversificación
% % %         logVerbose = false
% % %     end
% % % 
% % %     properties(Access = protected)
% % %         IsDone = false
% % %     end
% % % 
% % %     methods
% % %         function this = PortfolioEnv()
% % %             dummyData = load('ReaderBeginingDLR.mat');
% % %             numAssets = size(dummyData.RetornosMedios, 1);
% % %             windowSize = 5;
% % % 
% % %             obsSpec = rlNumericSpec([numAssets * windowSize 1], ...
% % %                 'LowerLimit', -inf, 'UpperLimit', inf);
% % %             obsSpec.Name = 'VentanaRetornos';
% % % 
% % %             actSpec = rlNumericSpec([numAssets 1], ...
% % %                 'LowerLimit', 0, 'UpperLimit', 1);
% % %             actSpec.Name = 'PesosPortafolio';
% % % 
% % %             this = this@rl.env.MATLABEnvironment(obsSpec, actSpec);
% % % 
% % %             this.Retornos = dummyData.RetornosMedios;
% % %             this.NumAssets = numAssets;
% % %             this.WindowSize = windowSize;
% % %             this.MaxSteps = size(this.Retornos, 2) - this.WindowSize;
% % % 
% % %             reset(this);
% % %         end
% % % 
% % %         function [obs, reward, isDone, log] = step(this, action)
% % %             log = [];
% % % 
% % %             action = action / (sum(action) + 1e-10);
% % % 
% % %             retorno = this.Retornos(:, this.CurrentStep + this.WindowSize);
% % %             r = sum(action .* retorno);
% % % 
% % %             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
% % %             v = sum((action .^ 2) .* var(ventana, 0, 2));
% % % 
% % %             diversificacion = -sum(action .^ 2);  % Penaliza concentración
% % % 
% % %             rawReward = r - this.alpha * v + this.beta * diversificacion;
% % % 
% % %             reward = tanh(5 * rawReward);  % Escalar para mayor estabilidad
% % % 
% % %             if ~isfinite(reward) || ~isscalar(reward)
% % %                 reward = 0;
% % %             end
% % % 
% % %             if this.logVerbose
% % %                 fprintf("Paso %d | Retorno: %.4f | Riesgo: %.4f | Diversif.: %.4f | Reward: %.4f\n", ...
% % %                     this.CurrentStep, r, v, diversificacion, reward);
% % %             end
% % % 
% % %             this.CurrentStep = this.CurrentStep + 1;
% % %             obs = this.getObservation();
% % %             isDone = (this.CurrentStep >= this.MaxSteps);
% % %             this.IsDone = isDone;
% % %         end
% % % 
% % %         function obs = reset(this)
% % %             this.CurrentStep = 1 + randi([0, this.MaxSteps - 1]);
% % %             this.IsDone = false;
% % %             obs = this.getObservation();
% % %         end
% % %     end
% % % 
% % %     methods (Access = private)
% % %         function obs = getObservation(this)
% % %             ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
% % %             obs = ventana(:);
% % %         end
% % %     end
% % % end
