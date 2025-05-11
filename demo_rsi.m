% DEMO_RSI.M - Simple demonstration of the RSI trading strategy
% This script demonstrates the RSI trading signals on a generated price series

% Add paths
addpath('src/strategies');
addpath('src/agents');
addpath('src/utils');

% Generate a synthetic price series with some trend and volatility
days = 252; % One year of trading
initialPrice = 100;
rng(42); % For reproducibility
dailyReturns = 0.0005 + 0.015 * randn(days, 1); % Mean: 0.05% daily, Std: 1.5%
prices = cumprod([initialPrice; 1 + dailyReturns]);

% Calculate RSI
window = 14;
overbought = 70;
oversold = 30;
rsiValues = zeros(size(prices));
signals = zeros(size(prices));

% Compute RSI using our strategy implementation
[signals] = rsi_strategy(prices, window, overbought, oversold);

% Create RSI agent for the same price series
rsiAgent = rsi_agent(prices, window, overbought, oversold);

% Simulate trading based on RSI signals
equity = zeros(size(prices));
equity(1) = 100; % Initial investment

position = 0; % 0 = no position, 1 = long
for i = 2:length(prices)
    % Get signal for current day
    signal = signals(i);
    
    % Trading logic
    if signal == 1 && position == 0 % Buy signal and no position
        position = 1;
        equity(i) = equity(i-1) * (prices(i) / prices(i-1));
    elseif signal == -1 && position == 1 % Sell signal and long position
        position = 0;
        equity(i) = equity(i-1);
    else % Hold
        if position == 1
            equity(i) = equity(i-1) * (prices(i) / prices(i-1));
        else
            equity(i) = equity(i-1);
        end
    end
end

% Calculate buy-and-hold strategy for comparison
buyHoldEquity = 100 * prices / prices(1);

% Calculate performance metrics
rsiReturn = (equity(end) - equity(1)) / equity(1);
rsiDailyReturns = diff(equity) ./ equity(1:end-1);
rsiSharpe = mean(rsiDailyReturns) / std(rsiDailyReturns) * sqrt(252);
rsiMaxDrawdown = max(cummax(equity) - equity) / max(equity);

bhReturn = (buyHoldEquity(end) - buyHoldEquity(1)) / buyHoldEquity(1);
bhDailyReturns = diff(buyHoldEquity) ./ buyHoldEquity(1:end-1);
bhSharpe = mean(bhDailyReturns) / std(bhDailyReturns) * sqrt(252);
bhMaxDrawdown = max(cummax(buyHoldEquity) - buyHoldEquity) / max(buyHoldEquity);

% Print results
fprintf('RSI Strategy - Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    rsiReturn*100, rsiSharpe, rsiMaxDrawdown*100);
fprintf('Buy & Hold - Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
    bhReturn*100, bhSharpe, bhMaxDrawdown*100);

% Plot results
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Price and RSI Signals
subplot(3, 1, 1);
hold on;
plot(prices, 'b-', 'LineWidth', 1.5);
buyPoints = find(signals == 1);
sellPoints = find(signals == -1);
plot(buyPoints, prices(buyPoints), 'g^', 'MarkerSize', 8);
plot(sellPoints, prices(sellPoints), 'rv', 'MarkerSize', 8);
title('Price Chart with RSI Signals', 'FontSize', 14);
legend('Price', 'Buy Signal', 'Sell Signal');
grid on;

% Plot 2: Equity Curves
subplot(3, 1, 2);
hold on;
plot(equity, 'g-', 'LineWidth', 1.5);
plot(buyHoldEquity, 'b-', 'LineWidth', 1.5);
title('Equity Curves', 'FontSize', 14);
legend('RSI Strategy', 'Buy & Hold');
grid on;

% Plot 3: RSI Values
subplot(3, 1, 3);
hold on;

% Calculate RSI values for visualization
changes = diff(prices);
n = length(changes);
rsiVals = zeros(length(prices), 1);

% First window
gains = zeros(window, 1);
losses = zeros(window, 1);
for i = 1:window
    if changes(i) > 0
        gains(i) = changes(i);
    elseif changes(i) < 0
        losses(i) = abs(changes(i));
    end
end
avgGain = mean(gains);
avgLoss = mean(losses);
if avgLoss == 0
    rsiVals(window+1) = 100;
else
    rs = avgGain / avgLoss;
    rsiVals(window+1) = 100 - (100 / (1 + rs));
end

% Rest of the values
for i = window+1:n
    if changes(i) > 0
        avgGain = (avgGain * (window-1) + changes(i)) / window;
        avgLoss = (avgLoss * (window-1)) / window;
    elseif changes(i) < 0
        avgGain = (avgGain * (window-1)) / window;
        avgLoss = (avgLoss * (window-1) + abs(changes(i))) / window;
    else
        avgGain = (avgGain * (window-1)) / window;
        avgLoss = (avgLoss * (window-1)) / window;
    end
    
    if avgLoss == 0
        rsiVals(i+1) = 100;
    else
        rs = avgGain / avgLoss;
        rsiVals(i+1) = 100 - (100 / (1 + rs));
    end
end

plot(rsiVals, 'k-', 'LineWidth', 1.5);
yline(overbought, 'r--', 'LineWidth', 1.5);
yline(oversold, 'g--', 'LineWidth', 1.5);
title('RSI Indicator', 'FontSize', 14);
ylabel('RSI Value');
ylim([0, 100]);
legend('RSI', sprintf('Overbought (%d)', overbought), sprintf('Oversold (%d)', oversold));
grid on;

% Save the figure
if ~exist('results/figures', 'dir')
    mkdir('results/figures');
end
saveas(gcf, 'results/figures/rsi_demo.png');
saveas(gcf, 'results/figures/rsi_demo.fig');

fprintf('\nDemo completed. Visualizations saved to results/figures directory.\n'); 
 
 
 
 