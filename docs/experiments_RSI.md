# RSI Strategy Experiments

This document explains the Relative Strength Index (RSI) strategy implementation in BotDeBolsa, how it works, its parameters, and how to run experiments with it.

## What is RSI?

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in a traded asset.

The formula for RSI is:
```
RSI = 100 - (100 / (1 + RS))
```
where RS (Relative Strength) is the average of X periods' up closes divided by the average of X periods' down closes.

## Implementation in BotDeBolsa

### RSI Strategy

The RSI strategy (`src/strategies/rsi_strategy.m`) calculates the RSI indicator for a given price series and generates trading signals based on overbought and oversold levels.

**Parameters:**
- `window`: The lookback period for RSI calculation (default: 14)
- `overbought`: The threshold above which an asset is considered overbought (default: 70)
- `oversold`: The threshold below which an asset is considered oversold (default: 30)

**Signals:**
- When RSI falls below the oversold level: Buy signal (+1)
- When RSI rises above the overbought level: Sell signal (-1)
- Otherwise: Hold signal (0)

### RSI Agent

The RSI agent (`src/agents/rsi_agent.m`) is a wrapper that provides a consistent interface between the RSI strategy and the portfolio environment.

The agent precomputes all RSI signals and provides them on demand via the `getSignal(t)` method, which returns the appropriate signal for time step `t`.

## Running RSI Experiments

### RSI vs Buy-and-Hold Comparison

To run a comparison between the RSI strategy and a simple Buy-and-Hold baseline:

1. Navigate to the project root directory
2. Run the experiment script:
   ```matlab
   cd experiments
   run_RSI_vs_baseline
   ```

This script will:
1. Load price data from `data/processed`
2. Create an RSI agent with a 14-period window
3. Create a Buy-and-Hold agent that always generates buy signals
4. Run simulations for both strategies
5. Calculate performance metrics (returns, Sharpe ratio, maximum drawdown)
6. Save results to `results/logs/RSI_vs_BH.mat`
7. Generate comparison plots in `results/figures/`

### Comprehensive Strategy Comparison

To compare RSI with all available strategies (PPO, SPO, Buy-and-Hold):

1. Navigate to the project root directory
2. Run the comparison script:
   ```matlab
   cd experiments
   run_comparison_all
   ```

This script will:
1. Load price data
2. Create all available agents (RSI, PPO, SPO, Buy-and-Hold)
3. Run simulations for each strategy
4. Calculate performance metrics
5. Save combined results to `results/logs/all_comparison.mat`
6. Generate comparison plots in `results/figures/`

## Customizing RSI Parameters

You can modify the RSI parameters in the experiment scripts:

```matlab
% In run_RSI_vs_baseline.m or run_comparison_all.m
rsiWindow = 14;     % Change this to adjust RSI period
overbought = 70;    % Change this to adjust overbought threshold
oversold = 30;      % Change this to adjust oversold threshold

% Then create the agent with custom parameters
rsiAgent = rsi_agent(prices, rsiWindow, overbought, oversold);
```

Common variations include:
- Shorter windows (6-10) for more sensitive signals
- Longer windows (20-25) for less sensitive signals
- Adjusted thresholds (e.g., 80/20) for more selective signals

## Interpreting Results

After running the experiments, you can analyze the results in several ways:

1. Check the console output for summary metrics
2. Examine the generated plots in `results/figures/`
3. Load the saved results for custom analysis:
   ```matlab
   % Load the saved results
   data = load('results/logs/RSI_vs_BH.mat');
   
   % Access the results structure
   results = data.results;
   
   % Look at specific metrics
   rsiSharpe = results.RSI.sharpe;
   bhReturn = results.BuyHold.totalReturn;
   
   % Compare strategy performance
   if results.RSI.sharpe > results.BuyHold.sharpe
       disp('RSI strategy has better risk-adjusted returns');
   end
   ``` 
 
 
 
 