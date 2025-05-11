# BotDeBolsa - Hybrid MACD-IA Trading Strategy

This repository contains a hybrid trading strategy that combines technical analysis (MACD) with artificial intelligence to optimize portfolio allocation across multiple assets.

## Overview

The hybrid MACD-IA strategy leverages both technical signals and AI-based portfolio optimization to create more robust investment decisions across different market regimes.

## Key Components

### Enhanced MACD Agent
- Adaptive parameter selection based on market conditions
- Signal filtering based on volume, trend confirmation, and signal strength
- Position sizing based on signal strength

### IA Models
- **Original IA Model**: Neural network trained to complement MACD signals
- **Regularized IA Model**: Enhanced neural network with L2 regularization, dropout, and early stopping to prevent overfitting and improve robustness

### SPO (Stochastic Portfolio Optimization)
- Optimizes portfolio weights based on expected returns and variance
- Provides target weights for IA model training
- Handles edge cases with fallback mechanisms

## Recent Improvements

### Regularized IA Model
- Added L2 regularization to prevent overfitting
- Enhanced dropout layers (0.4 and 0.3)
- Added batch normalization
- Implemented early stopping with validation patience
- Applied data augmentation techniques
- Added adaptive thresholds for market regime detection
- Implemented sample balancing across market regimes
- Added robust performance metrics

### SPO Function Enhancements
- Improved error handling to deal with NaN/Inf values
- Added fallback mechanisms for optimization failures
- Better handling of asset constraints
- Added validation to ensure the function is available across different contexts
- Created `ensure_spo_available()` utility to validate and locate the function

### Validation Testing
- Created comparison infrastructure to test different model performances
- Added metrics for portfolio turnover and concentration
- Implemented extended validation across multiple stock datasets
- Added comprehensive visualization tools

## Usage

The main scripts for running the strategy are:

- `simulate_hybrid_strategy_comparison.m`: Compares the original and regularized hybrid strategies
- `run_extended_validation.m`: Tests the strategies across multiple stock datasets
- `run_model_comparison.m`: Provides detailed metrics comparing different models

## Getting Started

1. Add required paths:
```matlab
addpath('src/utils');
addpath('src/strategies');
addpath('src/agents');
addpath('src/data');
addpath('proyecto');
```

2. Ensure SPO function is available:
```matlab
ensure_spo_available();
```

3. Run the hybrid strategy simulation:
```matlab
run simulate_hybrid_strategy_comparison.m
```

## Results

The regularized model generally produces more stable portfolios with:
- Lower turnover (reduced transaction costs)
- More balanced position sizing
- Reduced drawdowns
- Comparable or better returns

## Dependencies

- MATLAB (R2019b or later recommended)
- Deep Learning Toolbox for neural network training
- Statistics and Machine Learning Toolbox

## Project Structure

```
BotDeBolsa/
├── src/
│   ├── agents/           # Agent implementations
│   ├── strategies/       # Trading strategies
│   ├── utils/            # Utility functions
│   └── data/             # Data processing
├── proyecto/             # Original project files
├── results/              # Results storage
│   ├── logs/             # Performance logs
│   ├── figures/          # Generated charts
│   └── models/           # Trained models
└── run_*.m               # Simulation scripts
```

## Key Files

- `enhanced_macd_strategy.m`: Improved MACD with signal filtering
- `market_regime_detector.m`: Detects market conditions
- `enhanced_macd_agent.m`: Agent wrapper for MACD strategy
- `train_ia_complementary.m`: Trains AI to complement MACD
- `hybrid_macd_ia_strategy.m`: Combines MACD and AI signals
- `run_enhanced_hybrid_strategy.m`: Main simulation script

## Performance Metrics

The system tracks various performance metrics:

- Total returns
- Sharpe ratio
- Maximum drawdown
- Volatility
- Regime-specific performance

## Visualizations

The simulation generates several visualizations:

- Equity curves comparing strategies
- Market regime classifications over time
- Strategy weight allocations
- Return contribution analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds on traditional MACD strategies with modern enhancements
- The hybrid approach is inspired by adaptive portfolio management techniques 