% === Simulación del agente entrenado + SPO + igual peso ===

env = PortfolioEnv();
load('trainedAgent.mat', 'agent');

obs = reset(env);
episodeSteps = env.MaxSteps;
numAssets = env.NumAssets;

valueAgent = 1;
valueEqual = 1;
valueSPO = 1;

weightsEqual = ones(numAssets, 1) / numAssets;

valueSeriesAgent = zeros(1, episodeSteps);
valueSeriesEqual = zeros(1, episodeSteps);
valueSeriesSPO = zeros(1, episodeSteps);
actionsAgent = zeros(numAssets, episodeSteps);
actionsSPO = zeros(numAssets, episodeSteps);
rewards = zeros(1, episodeSteps);

for t = 1:episodeSteps
    rawAction = getAction(agent, obs);
    if isstruct(rawAction)
        action = rawAction.mean;
    else
        action = rawAction;
    end

    if ~isnumeric(action) || any(isnan(action)) || sum(action) == 0
        action = weightsEqual;
    else
        action = action / (sum(action) + 1e-10);
    end
    actionsAgent(:, t) = action;

    [obs, reward, isDone, ~] = step(env, action);
    rewards(t) = reward;

    ventana = env.Retornos(:, env.CurrentStep - env.WindowSize:env.CurrentStep - 1);
    retorno = env.Retornos(:, env.CurrentStep + env.WindowSize - 1);

    try
        mu = mean(ventana, 2);
        sigma = var(ventana, 0, 2);
        spoAction = obtenerSPO(mu, sigma, env.alpha);
    catch
        spoAction = weightsEqual;
    end

    actionsSPO(:, t) = spoAction;

    retAgent = sum(action .* retorno);
    retEqual = sum(weightsEqual .* retorno);
    retSPO = sum(spoAction .* retorno);

    retAgent = max(retAgent, -0.95);
    retEqual = max(retEqual, -0.95);
    retSPO = max(retSPO, -0.95);

    valueAgent = valueAgent * (1 + retAgent);
    valueEqual = valueEqual * (1 + retEqual);
    valueSPO = valueSPO * (1 + retSPO);

    valueSeriesAgent(t) = valueAgent;
    valueSeriesEqual(t) = valueEqual;
    valueSeriesSPO(t) = valueSPO;

    fprintf("Paso %3d | Reward: %+6.4f | Valor PPO: %.4f | SPO: %.4f\n", ...
        t, reward, valueAgent, valueSPO);

    if valueAgent < 0.01 || isDone
        break;
    end
end

% === Gráfico de valor del portafolio ===
figure;
plot(valueSeriesAgent(1:t), 'b', 'LineWidth', 2); hold on;
plot(valueSeriesEqual(1:t), 'r--', 'LineWidth', 2);
plot(valueSeriesSPO(1:t), 'g-.', 'LineWidth', 2);
xlabel('Paso'); ylabel('Valor del Portafolio');
title('Evolución: Agente RL vs Igual Peso vs SPO');
legend('Agente RL', 'Igual Peso', 'SPO');
grid on;

% === Gráfico de acciones SPO vs PPO ===
figure;
subplot(2,1,1);
area(actionsAgent(:, 1:t)');
title('Pesos del Agente RL');
ylabel('Peso');

subplot(2,1,2);
area(actionsSPO(:, 1:t)');
title('Pesos del SPO');
xlabel('Paso'); ylabel('Peso');

% === Guardar resultados ===
resultados.valorAgente = valueSeriesAgent(1:t);
resultados.valorSPO = valueSeriesSPO(1:t);
resultados.valorEqual = valueSeriesEqual(1:t);
resultados.actionsAgent = actionsAgent(:, 1:t);
resultados.actionsSPO = actionsSPO(:, 1:t);
save('resultadosSPO_vs_PPO.mat', 'resultados');
fprintf("✅ Resultados guardados en 'resultadosSPO_vs_PPO.mat'\n");

% % === Simulación del agente entrenado + baseline pasivo ===
% 
% env = PortfolioEnv();
% load('trainedAgent.mat', 'agent');
% 
% obs = reset(env);
% episodeSteps = env.MaxSteps;
% numAssets = env.NumAssets;
% 
% valueAgent = 1;
% valuePassive = 1;
% equalWeights = ones(numAssets, 1) / numAssets;
% 
% valueSeriesAgent = zeros(1, episodeSteps);
% valueSeriesPassive = zeros(1, episodeSteps);
% actionsAgent = zeros(numAssets, episodeSteps);
% rewards = zeros(1, episodeSteps);
% 
% for t = 1:episodeSteps
%     % Acción del agente
%     rawAction = getAction(agent, obs);
%     if isstruct(rawAction)
%         action = rawAction.mean;
%     else
%         action = rawAction;
%     end
% 
%     % Normalizar
%     if ~isnumeric(action) || any(isnan(action)) || sum(action) == 0
%         action = equalWeights;
%     else
%         action = action / (sum(action) + 1e-10);
%     end
%     actionsAgent(:, t) = action;
% 
%     [obs, reward, isDone, ~] = step(env, action);
%     rewards(t) = reward;
% 
%     retorno = env.Retornos(:, env.CurrentStep + env.WindowSize - 1);
% 
%     % Actualizar valores
%     retAgente = sum(action .* retorno);
%     retAgente = max(retAgente, -0.95);
%     valueAgent = valueAgent * (1 + retAgente);
%     valueSeriesAgent(t) = valueAgent;
% 
%     retPasivo = sum(equalWeights .* retorno);
%     valuePassive = valuePassive * (1 + retPasivo);
%     valueSeriesPassive(t) = valuePassive;
% 
%     fprintf("Paso %3d | Reward: %+6.4f | Valor: %.4f\n", ...
%         t, reward, valueAgent);
%     disp("Acción: " + mat2str(action', 3));
% 
%     if valueAgent < 0.01 || isDone
%         fprintf("⚠️ Portafolio terminó en paso %d\n", t);
%         break;
%     end
% end
% 
% % Gráficos
% figure;
% plot(valueSeriesAgent(1:t), 'b', 'LineWidth', 2); hold on;
% plot(valueSeriesPassive(1:t), 'r--', 'LineWidth', 2);
% xlabel('Paso'); ylabel('Valor del Portafolio');
% title('Agente RL vs Igual Peso');
% legend('Agente', 'Pasivo'); grid on;
% 
% figure;
% area(actionsAgent(:, 1:t)');
% xlabel('Paso'); ylabel('Peso');
% title('Pesos del Agente');
% legend(arrayfun(@(x) sprintf('Activo %d', x), 1:numAssets, 'UniformOutput', false));
% grid on;
% 
% % Guardar resultados
% resultados = struct();
% resultados.valorAgente = valueSeriesAgent(1:t);
% resultados.valorPasivo = valueSeriesPassive(1:t);
% resultados.rewards = rewards(1:t);
% resultados.actions = actionsAgent(:, 1:t);
% save('resultadosSimulacion.mat', 'resultados');
% fprintf("✅ Resultados guardados en 'resultadosSimulacion.mat'\n");
% 
% % % === Simulación del agente entrenado + estrategia pasiva ===
% % 
% % % Cargar entorno y agente
% % env = PortfolioEnv();
% % load('trainedAgent.mat', 'agent');
% % 
% % obs = reset(env);
% % episodeSteps = env.MaxSteps;
% % numAssets = env.NumAssets;
% % 
% % % Inicialización
% % valueAgent = 1;
% % valuePassive = 1;
% % 
% % valueSeriesAgent = zeros(1, episodeSteps);
% % valueSeriesPassive = zeros(1, episodeSteps);
% % actionsAgent = zeros(numAssets, episodeSteps);
% % rewards = zeros(1, episodeSteps);
% % 
% % equalWeights = ones(numAssets, 1) / numAssets;
% % 
% % for t = 1:episodeSteps
% %     % === Acción del agente ===
% %     rawAction = getAction(agent, obs);
% %     if isstruct(rawAction)
% %         action = rawAction.mean;
% %     else
% %         action = rawAction;
% %     end
% % 
% %     % Normalizar
% %     if ~isnumeric(action) || any(isnan(action)) || sum(action) == 0
% %         action = equalWeights;
% %     else
% %         action = action / (sum(action) + 1e-10);
% %     end
% %     actionsAgent(:, t) = action;
% % 
% %     % Tomar paso
% %     [obs, reward, isDone, ~] = step(env, action);
% %     rewards(t) = reward;
% % 
% %     % === Retornos y evolución ===
% %     retorno = env.Retornos(:, env.CurrentStep + env.WindowSize - 1);
% % 
% %     % Valor agente (limitando colapso)
% %     retAgente = sum(action .* retorno);
% %     retAgente = max(retAgente, -0.95);
% %     valueAgent = valueAgent * (1 + retAgente);
% %     valueSeriesAgent(t) = valueAgent;
% % 
% %     % Valor pasivo
% %     retPasivo = sum(equalWeights .* retorno);
% %     valuePassive = valuePassive * (1 + retPasivo);
% %     valueSeriesPassive(t) = valuePassive;
% % 
% %     % Logs
% %     fprintf("Paso %3d | Reward: %+6.4f | RetAgente: %+6.4f | Valor: %.4f\n", ...
% %         t, reward, retAgente, valueAgent);
% %     disp("Acción agente: " + mat2str(action', 3));
% % 
% %     % Salida por quiebra o final
% %     if valueAgent < 0.01 || isDone
% %         fprintf("⚠️ Portafolio agente terminó en paso %d\n", t);
% %         break;
% %     end
% % end
% % 
% % % === Gráfico de evolución de valores ===
% % figure;
% % plot(valueSeriesAgent(1:t), 'b', 'LineWidth', 2); hold on;
% % plot(valueSeriesPassive(1:t), 'r--', 'LineWidth', 2);
% % xlabel('Paso'); ylabel('Valor del Portafolio');
% % title('📉 Evolución: Agente vs Pasivo');
% % legend('Agente RL', 'Pasivo (igual pesos)');
% % grid on;
% % 
% % % === Gráfico de asignación de pesos ===
% % figure;
% % area(actionsAgent(:, 1:t)');
% % xlabel('Paso'); ylabel('Peso');
% % title('📊 Pesos Asignados por el Agente');
% % legend(arrayfun(@(x) sprintf('Activo %d', x), 1:numAssets, 'UniformOutput', false));
% % grid on;
% % 
% % % === Guardado de resultados ===
% % resultados = struct();
% % resultados.valorAgente = valueSeriesAgent(1:t);
% % resultados.valorPasivo = valueSeriesPassive(1:t);
% % resultados.rewards = rewards(1:t);
% % resultados.actions = actionsAgent(:, 1:t);
% % save('resultadosSimulacion.mat', 'resultados');
% % fprintf("✅ Resultados guardados en 'resultadosSimulacion.mat'\n");
