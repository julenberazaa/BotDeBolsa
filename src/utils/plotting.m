function plot_comparison(results, outputPath)
% PLOT_COMPARISON - Plot comparison between different trading strategies
%
% Syntax:  plot_comparison(results, outputPath)
%
% Inputs:
%    results - Structure containing strategy results
%              Each strategy should have equity curve and metrics
%    outputPath - Path to save generated plots
%
% Example:
%    results.RSI.equity = [100, 102, 105, ...];
%    results.RSI.returns = [0, 0.02, 0.03, ...];
%    results.RSI.sharpe = 1.2;
%    results.RSI.maxDrawdown = 0.05;
%    plot_comparison(results, 'results/figures/')

% Input validation
if nargin < 2
    outputPath = 'results/figures/';
end

if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

% Create figure for equity curves
fig1 = figure('Position', [100, 100, 1200, 600]);
hold on;

% Plot equity curves
strategies = fieldnames(results);
colors = lines(length(strategies));
legendEntries = cell(length(strategies), 1);

for i = 1:length(strategies)
    strat = strategies{i};
    if isfield(results.(strat), 'equity')
        plot(results.(strat).equity, 'LineWidth', 2, 'Color', colors(i,:));
        legendEntries{i} = strat;
    end
end

title('Strategy Equity Curves Comparison', 'FontSize', 14);
xlabel('Trading Days', 'FontSize', 12);
ylabel('Portfolio Value', 'FontSize', 12);
legend(legendEntries, 'Location', 'northwest');
grid on;
saveas(fig1, fullfile(outputPath, 'equity_curves_comparison.png'));
saveas(fig1, fullfile(outputPath, 'equity_curves_comparison.fig'));

% Create figure for performance metrics
fig2 = figure('Position', [100, 100, 1200, 600]);

% Collect metrics
sharpeRatios = zeros(length(strategies), 1);
drawdowns = zeros(length(strategies), 1);
returns = zeros(length(strategies), 1);
volatilities = zeros(length(strategies), 1);

for i = 1:length(strategies)
    strat = strategies{i};
    if isfield(results.(strat), 'sharpe')
        sharpeRatios(i) = results.(strat).sharpe;
    end
    if isfield(results.(strat), 'maxDrawdown')
        drawdowns(i) = results.(strat).maxDrawdown;
    end
    if isfield(results.(strat), 'totalReturn')
        returns(i) = results.(strat).totalReturn;
    end
    if isfield(results.(strat), 'volatility')
        volatilities(i) = results.(strat).volatility;
    end
end

% Create subplots for each metric
subplot(2, 2, 1);
bar(sharpeRatios);
set(gca, 'XTick', 1:length(strategies), 'XTickLabel', strategies);
title('Sharpe Ratio', 'FontSize', 12);
grid on;

subplot(2, 2, 2);
bar(drawdowns);
set(gca, 'XTick', 1:length(strategies), 'XTickLabel', strategies);
title('Maximum Drawdown', 'FontSize', 12);
grid on;

subplot(2, 2, 3);
bar(returns);
set(gca, 'XTick', 1:length(strategies), 'XTickLabel', strategies);
title('Total Return', 'FontSize', 12);
grid on;

subplot(2, 2, 4);
bar(volatilities);
set(gca, 'XTick', 1:length(strategies), 'XTickLabel', strategies);
title('Volatility', 'FontSize', 12);
grid on;

suptitle('Performance Metrics Comparison');
saveas(fig2, fullfile(outputPath, 'performance_metrics.png'));
saveas(fig2, fullfile(outputPath, 'performance_metrics.fig'));

% Create additional visualization if return data is available
try
    fig3 = figure('Position', [100, 100, 1200, 600]);
    hold on;
    
    % Plot cumulative returns for strategies with return data
    for i = 1:length(strategies)
        strat = strategies{i};
        if isfield(results.(strat), 'returns') && ~isempty(results.(strat).returns)
            cumReturns = cumprod(1 + results.(strat).returns) - 1;
            plot(cumReturns, 'LineWidth', 2, 'Color', colors(i,:));
        end
    end
    
    title('Cumulative Returns', 'FontSize', 14);
    xlabel('Trading Days', 'FontSize', 12);
    ylabel('Cumulative Return', 'FontSize', 12);
    legend(strategies, 'Location', 'northwest');
    grid on;
    saveas(fig3, fullfile(outputPath, 'cumulative_returns.png'));
    saveas(fig3, fullfile(outputPath, 'cumulative_returns.fig'));
catch
    % Skip if return data is not available
end

% Close all figures
close all;

end 
 
 
 