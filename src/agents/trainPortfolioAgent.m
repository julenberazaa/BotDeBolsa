clc;
clear;
close all;

% === Crear entorno ===
env = PortfolioEnv();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% === Red del actor ===
inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'state');

commonPath = [
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
];

meanPath = fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean');

stdPath = [
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'std')
    softplusLayer('Name', 'std_out')
];

lgActor = layerGraph();
lgActor = addLayers(lgActor, inputLayer);
lgActor = addLayers(lgActor, commonPath);
lgActor = addLayers(lgActor, meanPath);
lgActor = addLayers(lgActor, stdPath);

lgActor = connectLayers(lgActor, 'state', 'fc1');
lgActor = connectLayers(lgActor, 'relu2', 'mean');
lgActor = connectLayers(lgActor, 'relu2', 'std');

actorNet = dlnetwork(lgActor);
actor = rlStochasticActorRepresentation(actorNet, obsInfo, actInfo);

criticLayers = [
    featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
];

criticNet = dlnetwork(layerGraph(criticLayers));
critic = rlValueFunction(criticNet, obsInfo);

% === Configuración del agente PPO ===
agentOptions = rlPPOAgentOptions( ...
    'SampleTime', 1, ...
    'ExperienceHorizon', 128, ...
    'ClipFactor', 0.2, ...
    'EntropyLossWeight', 0.01, ...
    'MiniBatchSize', 64, ...
    'NumEpoch', 10, ...
    'AdvantageEstimateMethod', 'gae', ...
    'GAEFactor', 0.95, ...
    'DiscountFactor', 0.99, ...
    'ActorOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-4), ...
    'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-3));

agent = rlPPOAgent(actor, critic, agentOptions);

% === Opciones de entrenamiento ===
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'Plots', "training-progress", ...
    'StopTrainingCriteria','EpisodeCount', ...
    'StopTrainingValue', 1000);

% === Entrenamiento ===
fprintf("Entrenando agente PPO...\n");
trainingStats = train(agent, env, trainOpts);
save('trainedAgent.mat', 'agent');
fprintf("✅ Agente entrenado y guardado como 'trainedAgent.mat'.\n");


% % === ENTRENAMIENTO PPO MEJORADO PARA PortfolioEnv ===
% 
% clc;
% clear;
% close all;
% 
% % === ENTORNO ===
% env = PortfolioEnv();
% obsInfo = getObservationInfo(env);
% actInfo = getActionInfo(env);
% 
% %% === RED DEL ACTOR CON SOFTMAX ===
% inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'state');
% 
% commonPath = [
%     fullyConnectedLayer(128, 'Name', 'fc1')
%     reluLayer('Name', 'relu1')
%     fullyConnectedLayer(128, 'Name', 'fc2')
%     reluLayer('Name', 'relu2')
% ];
% 
% meanPath = [
%     fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean_fc')
%     softmaxLayer('Name', 'mean_logits')
% ];
% 
% stdPath = [
%     fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'std_fc')
%     softplusLayer('Name', 'std_out')
% ];
% 
% lg = layerGraph(inputLayer);
% lg = addLayers(lg, commonPath);
% lg = addLayers(lg, meanPath);
% lg = addLayers(lg, stdPath);
% lg = connectLayers(lg, 'state', 'fc1');
% lg = connectLayers(lg, 'relu2', 'mean_fc');
% lg = connectLayers(lg, 'relu2', 'std_fc');
% 
% dlnet = dlnetwork(lg);
% 
% actor = rlContinuousGaussianActor(dlnet, obsInfo, actInfo, ...
%     'ObservationInputNames', 'state', ...
%     'ActionMeanOutputNames', 'mean_logits', ...
%     'ActionStandardDeviationOutputNames', 'std_out');
% 
% %% === RED DEL CRITICO ===
% criticLG = [
%     featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
%     fullyConnectedLayer(64, 'Name', 'fc1')
%     reluLayer('Name', 'relu1')
%     fullyConnectedLayer(64, 'Name', 'fc2')
%     reluLayer('Name', 'relu2')
%     fullyConnectedLayer(1, 'Name', 'value')
% ];
% criticNet = dlnetwork(layerGraph(criticLG));
% critic = rlValueFunction(criticNet, obsInfo);
% 
% %% === OPCIONES DEL AGENTE ===
% agentOptions = rlPPOAgentOptions( ...
%     'SampleTime', 1, ...
%     'ExperienceHorizon', 256, ...
%     'ClipFactor', 0.1, ...
%     'EntropyLossWeight', 0.5, ...
%     'MiniBatchSize', 64, ...
%     'NumEpoch', 5, ...
%     'AdvantageEstimateMethod', 'gae', ...
%     'GAEFactor', 0.95, ...
%     'DiscountFactor', 0.99, ...
%     'ActorOptimizerOptions', rlOptimizerOptions('LearnRate',1e-4), ...
%     'CriticOptimizerOptions', rlOptimizerOptions('LearnRate',1e-3));
% 
% agent = rlPPOAgent(actor, critic, agentOptions);
% 
% %% === OPCIONES DE ENTRENAMIENTO ===
% trainOpts = rlTrainingOptions( ...
%     'MaxEpisodes', 1500, ...
%     'MaxStepsPerEpisode', env.MaxSteps, ...
%     'ScoreAveragingWindowLength', 30, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress', ...
%     'StopTrainingCriteria','EpisodeCount', ...
%     'StopTrainingValue', 1500);
% 
% %% === ENTRENAMIENTO ===
% fprintf("\n🚀 Entrenando agente PPO mejorado...\n");
% trainingStats = train(agent, env, trainOpts);
% 
% %% === GUARDADO ===
% save('trainedAgent.mat','agent');
% fprintf("\n✅ Agente guardado en 'trainedAgent.mat'\n");
% 
% % clc;
% % clear;
% % close all;
% % 
% % % === Cargar entorno personalizado y datos ===
% % env = PortfolioEnv();  % Asegúrate de que PortfolioEnv.m esté en el path
% % obsInfo = getObservationInfo(env);
% % actInfo = getActionInfo(env);
% % 
% % %% === RED DEL ACTOR MEJORADA ===
% % inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'state');
% % 
% % commonPath = [
% %     fullyConnectedLayer(128, 'Name', 'fc1')
% %     reluLayer('Name', 'relu1')
% %     fullyConnectedLayer(128, 'Name', 'fc2')
% %     reluLayer('Name', 'relu2')
% %     fullyConnectedLayer(64, 'Name', 'fc3')
% %     reluLayer('Name', 'relu3')
% % ];
% % 
% % meanPath = fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean');
% % 
% % stdPath = [
% %     fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'std')
% %     softplusLayer('Name', 'std_out')
% % ];
% % 
% % lgActor = layerGraph();
% % lgActor = addLayers(lgActor, inputLayer);
% % lgActor = addLayers(lgActor, commonPath);
% % lgActor = addLayers(lgActor, meanPath);
% % lgActor = addLayers(lgActor, stdPath);
% % 
% % lgActor = connectLayers(lgActor, 'state', 'fc1');
% % lgActor = connectLayers(lgActor, 'relu3', 'mean');
% % lgActor = connectLayers(lgActor, 'relu3', 'std');
% % 
% % actorNet = dlnetwork(lgActor);
% % actor = rlStochasticActorRepresentation(actorNet, obsInfo, actInfo);
% % 
% % %% === RED DEL CRITIC ===
% % criticLayers = [
% %     featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
% %     fullyConnectedLayer(64, 'Name', 'fc1')
% %     reluLayer('Name', 'relu1')
% %     fullyConnectedLayer(64, 'Name', 'fc2')
% %     reluLayer('Name', 'relu2')
% %     fullyConnectedLayer(1, 'Name', 'value')
% % ];
% % 
% % criticNet = dlnetwork(layerGraph(criticLayers));
% % critic = rlValueFunction(criticNet, obsInfo);
% % 
% % %% === CONFIGURACIÓN DEL AGENTE PPO ===
% % agentOptions = rlPPOAgentOptions(...
% %     'SampleTime', 1, ...
% %     'ExperienceHorizon', 256, ...
% %     'ClipFactor', 0.2, ...
% %     'EntropyLossWeight', 0.05, ...
% %     'MiniBatchSize', 128, ...
% %     'NumEpoch', 5, ...
% %     'AdvantageEstimateMethod', 'gae', ...
% %     'GAEFactor', 0.95, ...
% %     'DiscountFactor', 0.99, ...
% %     'ActorOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-4), ...
% %     'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-3));
% % 
% % agent = rlPPOAgent(actor, critic, agentOptions);
% % 
% % %% === OPCIONES DE ENTRENAMIENTO ===
% % trainOpts = rlTrainingOptions(...
% %     'MaxEpisodes', 400, ...
% %     'MaxStepsPerEpisode', env.MaxSteps, ...
% %     'ScoreAveragingWindowLength', 50, ...
% %     'Verbose', false, ...
% %     'Plots', "training-progress", ...
% %     'StopTrainingCriteria','EpisodeCount', ...
% %     'StopTrainingValue', 1000);
% % 
% % %% === ENTRENAMIENTO ===
% % fprintf("Iniciando entrenamiento del agente PPO mejorado...\n");
% % trainingStats = train(agent, env, trainOpts);
% % 
% % %% === GUARDAR RESULTADO ===
% % save('trainedAgent.mat', 'agent');
% % fprintf("Entrenamiento finalizado y agente guardado como 'trainedAgent.mat'.\n");