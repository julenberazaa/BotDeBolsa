# MACD Strategy Documentation

## Overview

The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of an asset's price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. A 9-period EMA of the MACD, called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.

## Implementation

The MACD strategy in BotDeBolsa is implemented in the following components:

1. **Strategy Algorithm** (`src/strategies/macd_strategy.m`): Calculates MACD values and generates trading signals
2. **Agent Wrapper** (`src/agents/macd_agent.m`): Provides a consistent interface for the PortfolioEnv
3. **Experiments** (`experiments/run_MACD_vs_baseline.m`): Compares MACD with Buy-and-Hold
4. **Parameter Optimization** (`experiments/optimize_macd_parameters.m`): Finds optimal MACD parameters

## MACD Components

The MACD consists of three components:

1. **MACD Line**: The difference between the fast EMA and slow EMA (typically 12-period EMA minus 26-period EMA)
2. **Signal Line**: An EMA of the MACD Line (typically 9-period)
3. **Histogram**: The difference between the MACD Line and Signal Line

## Trading Signals

The MACD strategy generates trading signals based on the following conditions:

- **Buy Signal (1)**: When the MACD Line crosses above the Signal Line (bullish crossover)
- **Sell Signal (-1)**: When the MACD Line crosses below the Signal Line (bearish crossover)
- **Hold Signal (0)**: No crossover detected

## Parameters

The MACD strategy uses the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Fast Period | 12 | The period for the fast EMA |
| Slow Period | 26 | The period for the slow EMA |
| Signal Period | 9 | The period for the signal line EMA |

## Running MACD Experiments

To run the MACD experiment:

```matlab
% In MATLAB command window
run run_experiment_macd
```

This will execute the MACD strategy versus Buy-and-Hold baseline comparison and save the results in the `results/` directory.

## Optimization

To find optimal MACD parameters for your specific dataset, run:

```matlab
% In MATLAB command window
cd experiments
optimize_macd_parameters
```

This script performs a grid search over various combinations of MACD parameters to find the set that maximizes the Sharpe ratio.

## Demo

A simplified demonstration of the MACD strategy is available in the `demo_macd.m` script, which:

1. Generates a synthetic price series
2. Calculates MACD signals
3. Simulates trading based on those signals
4. Compares performance with a Buy-and-Hold strategy

```matlab
% In MATLAB command window
run demo_macd
```

## Performance Metrics

The MACD strategy performance is evaluated using the following metrics:

- **Total Return**: Percentage return over the simulation period
- **Sharpe Ratio**: Risk-adjusted return (using daily returns annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns (annualized)

## Visualizations

The `compare_plots` utility function creates the following visualizations:

1. Equity curves for MACD vs. Buy-and-Hold
2. Performance metrics comparison
3. Cumulative returns chart

## Integration with PortfolioEnv

The MACD agent can be used with the PortfolioEnv environment:

```matlab
% Create MACD agent
macdAgent = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod);

% Initialize environment
env = PortfolioEnv(macdAgent);

% Run simulation
[observation, reward, isDone] = step(env);
```

## Limitations and Future Improvements

- MACD is primarily effective in trending markets and may generate false signals in ranging markets
- The strategy could be improved by adding filters to reduce false signals, such as volume confirmation
- Combining MACD with other indicators (RSI, Bollinger Bands, etc.) could create a more robust strategy
- Dynamic parameter adjustment based on market regimes could enhance performance 
 
 
 
 