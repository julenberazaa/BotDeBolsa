clc;
clear;
close all;

% === Cargar entorno con penalización SPO ===
env = PortfolioEnvSPOGuided();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% === Red del actor PPO (estocástica) ===
inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'state');

commonPath = [
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
];

meanPath = fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean');
stdPath = [
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'std')
    softplusLayer('Name', 'std_out')  % asegura que la desviación estándar sea positiva
];

lg = layerGraph();
lg = addLayers(lg, inputLayer);
lg = addLayers(lg, commonPath);
lg = addLayers(lg, meanPath);
lg = addLayers(lg, stdPath);

lg = connectLayers(lg, 'state', 'fc1');
lg = connectLayers(lg, 'relu2', 'mean');
lg = connectLayers(lg, 'relu2', 'std');

actorNet = dlnetwork(lg);

actor = rlStochasticActorRepresentation(actorNet, obsInfo, actInfo);

criticLayers = [
    featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
];

criticNet = dlnetwork(layerGraph(criticLayers));
critic = rlValueFunction(criticNet, obsInfo);

% === Configuración del agente PPO ===
agentOpts = rlPPOAgentOptions( ...
    'SampleTime', 1, ...
    'ExperienceHorizon', 128, ...
    'ClipFactor', 0.2, ...
    'EntropyLossWeight', 0.01, ...
    'MiniBatchSize', 64, ...
    'NumEpoch', 3, ...
    'AdvantageEstimateMethod', 'gae', ...
    'GAEFactor', 0.95, ...
    'DiscountFactor', 0.99, ...
    'ActorOptimizerOptions', rlOptimizerOptions('LearnRate', 1e-4), ...
    'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 5e-4));

agent = rlPPOAgent(actor, critic, agentOpts);

% === Opciones de entrenamiento ===
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', env.MaxSteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 500, ...
    'Plots', 'training-progress');


trainingStats = train(agent, env, trainOpts);

% === Guardar agente entrenado ===
save('ppo_agent_spoguided.mat', 'agent');

