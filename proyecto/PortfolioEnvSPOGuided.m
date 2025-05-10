classdef PortfolioEnvSPOGuided < rl.env.MATLABEnvironment
    properties
        Retornos
        NumAssets
        CurrentStep
        WindowSize = 5
        MaxSteps
        alpha = 0.1
        beta = 0.01
        gamma = 0.5  % penalizacion por desviarse de SPO
    end

    properties(Access = protected)
        IsDone = false
    end

    methods
        function this = PortfolioEnvSPOGuided()
            data = load('ReaderBeginingDLR.mat');
            numAssets = size(data.RetornosMedios, 1);

            obsSpec = rlNumericSpec([numAssets * 5 1], 'LowerLimit', -inf, 'UpperLimit', inf);
            actSpec = rlNumericSpec([numAssets 1], 'LowerLimit', 0, 'UpperLimit', 1);

            this = this@rl.env.MATLABEnvironment(obsSpec, actSpec);

            this.Retornos = data.RetornosMedios;
            this.NumAssets = numAssets;
            this.MaxSteps = size(this.Retornos, 2) - this.WindowSize;

            reset(this);
        end

        function [obs, reward, isDone, log] = step(this, action)
            log = [];

            action = action / (sum(action) + 1e-10);
            ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
            retorno = this.Retornos(:, this.CurrentStep + this.WindowSize);

            r = sum(action .* retorno);
            v = sum((action .^ 2) .* var(ventana, 0, 2));
            diversificacion = -sum(action .^ 2);
            rawReward = r - this.alpha * v + this.beta * diversificacion;

            % === Penalizacion por desviarse del SPO ===
            mu = mean(ventana, 2);
            sigma = var(ventana, 0, 2);
            try
                pesosSPO = obtenerSPO(mu, sigma, this.alpha);
                distancia = norm(action - pesosSPO);
                rawReward = rawReward - this.gamma * distancia;
            catch
                % Si SPO falla, no penaliza
            end

            reward = tanh(5 * rawReward);

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
    end

    methods (Access = private)
        function obs = getObservation(this)
            ventana = this.Retornos(:, this.CurrentStep:this.CurrentStep + this.WindowSize - 1);
            obs = ventana(:);
        end
    end
end
