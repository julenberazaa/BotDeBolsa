% DEMO_MACD.M - Simple demonstration of the MACD trading strategy
% This script demonstrates the MACD trading signals on a generated price series

% Run startup script to initialize all paths
startup;

try
    % Generate a synthetic price series with some trend and volatility
    days = 252; % One year of trading
    initialPrice = 100;
    rng(42); % For reproducibility
    dailyReturns = 0.0005 + 0.015 * randn(days, 1); % Mean: 0.05% daily, Std: 1.5%
    
    % Add a trend to make it more interesting for MACD
    trend = linspace(0, 0.01, days/3)';
    trend = [trend; -trend*1.5; trend*0.5];
    dailyReturns = dailyReturns + trend;
    
    prices = cumprod([initialPrice; 1 + dailyReturns]);
    
    % Calculate MACD
    fastPeriod = 12;
    slowPeriod = 26;
    signalPeriod = 9;
    signals = zeros(size(prices));
    
    fprintf('Computing MACD signals...\n');
    % Compute MACD using our strategy implementation
    [signals] = macd_strategy(prices, fastPeriod, slowPeriod, signalPeriod);
    
    % Create MACD agent for the same price series
    macdAgent = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod);
    
    % Simulate trading based on MACD signals
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
    macdReturn = (equity(end) - equity(1)) / equity(1);
    macdDailyReturns = diff(equity) ./ equity(1:end-1);
    macdSharpe = mean(macdDailyReturns) / std(macdDailyReturns) * sqrt(252);
    macdMaxDrawdown = max(cummax(equity) - equity) / max(equity);
    
    bhReturn = (buyHoldEquity(end) - buyHoldEquity(1)) / buyHoldEquity(1);
    bhDailyReturns = diff(buyHoldEquity) ./ buyHoldEquity(1:end-1);
    bhSharpe = mean(bhDailyReturns) / std(bhDailyReturns) * sqrt(252);
    bhMaxDrawdown = max(cummax(buyHoldEquity) - buyHoldEquity) / max(buyHoldEquity);
    
    % Print results
    fprintf('\n--- Performance Metrics ---\n');
    fprintf('MACD Strategy - Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
        macdReturn*100, macdSharpe, macdMaxDrawdown*100);
    fprintf('Buy & Hold - Return: %.2f%%, Sharpe: %.2f, Max Drawdown: %.2f%%\n', ...
        bhReturn*100, bhSharpe, bhMaxDrawdown*100);
    
    % Create visualizations
    fprintf('\nGenerating plots...\n');
    
    % Plot results
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Price and MACD Signals
    subplot(3, 1, 1);
    hold on;
    plot(prices, 'b-', 'LineWidth', 1.5);
    buyPoints = find(signals == 1);
    sellPoints = find(signals == -1);
    plot(buyPoints, prices(buyPoints), 'g^', 'MarkerSize', 8);
    plot(sellPoints, prices(sellPoints), 'rv', 'MarkerSize', 8);
    title('Price Chart with MACD Signals', 'FontSize', 14);
    legend('Price', 'Buy Signal', 'Sell Signal');
    grid on;
    
    % Plot 2: Equity Curves
    subplot(3, 1, 2);
    hold on;
    plot(equity, 'g-', 'LineWidth', 1.5);
    plot(buyHoldEquity, 'b-', 'LineWidth', 1.5);
    title('Equity Curves', 'FontSize', 14);
    legend('MACD Strategy', 'Buy & Hold');
    grid on;
    
    % Plot 3: MACD Components
    subplot(3, 1, 3);
    hold on;
    
    % Calculate MACD components for visualization
    n = length(prices);
    fastEMA = zeros(n, 1);
    slowEMA = zeros(n, 1);
    macdLine = zeros(n, 1);
    signalLine = zeros(n, 1);
    histogram = zeros(n, 1);
    
    % Calculate fast EMA
    alpha_fast = 2 / (fastPeriod + 1);
    fastEMA(1) = prices(1);
    for t = 2:n
        fastEMA(t) = alpha_fast * prices(t) + (1 - alpha_fast) * fastEMA(t-1);
    end
    
    % Calculate slow EMA
    alpha_slow = 2 / (slowPeriod + 1);
    slowEMA(1) = prices(1);
    for t = 2:n
        slowEMA(t) = alpha_slow * prices(t) + (1 - alpha_slow) * slowEMA(t-1);
    end
    
    % Calculate MACD line (fast EMA - slow EMA)
    macdLine = fastEMA - slowEMA;
    
    % Calculate signal line (EMA of MACD line)
    alpha_signal = 2 / (signalPeriod + 1);
    signalLine(1) = macdLine(1);
    for t = 2:n
        signalLine(t) = alpha_signal * macdLine(t) + (1 - alpha_signal) * signalLine(t-1);
    end
    
    % Calculate histogram
    histogram = macdLine - signalLine;
    
    % Plot MACD components
    plot(macdLine, 'b-', 'LineWidth', 1.5);
    plot(signalLine, 'r-', 'LineWidth', 1.5);
    bar(histogram, 0.6, 'FaceColor', [0.7 0.7 0.7]);
    yline(0, 'k--');
    title('MACD Indicator', 'FontSize', 14);
    legend('MACD Line', 'Signal Line', 'Histogram');
    grid on;
    
    % Save the figure
    fprintf('Saving visualizations...\n');
    if ~exist('results/figures', 'dir')
        mkdir('results/figures');
    end
    saveas(gcf, 'results/figures/macd_demo.png');
    saveas(gcf, 'results/figures/macd_demo.fig');
    
    fprintf('\nDemo completed successfully! Visualizations saved to results/figures directory.\n');
catch ME
    fprintf('\nError running MACD demo: %s\n', ME.message);
    fprintf('In file: %s, line %d\n', ME.stack(1).file, ME.stack(1).line);
    
    % Ensure proper path initialization if that's the issue
    fprintf('\nTrying to add paths directly...\n');
    addpath('src/strategies');
    addpath('src/agents');
    addpath('src/utils');
    fprintf('Please try running the demo again.\n');
end 
 
 
 
 