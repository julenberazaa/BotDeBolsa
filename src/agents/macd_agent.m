classdef macd_agent < handle
    % MACD_AGENT - Agent that generates trading signals based on MACD strategy
    %
    % This agent wraps the MACD strategy and provides a consistent interface
    % with the getSignal method for use in the PortfolioEnv.
    
    properties
        prices       % Historical price data
        fastPeriod   % Fast EMA period
        slowPeriod   % Slow EMA period
        signalPeriod % Signal line EMA period
        signals      % Precomputed signals
    end
    
    methods
        function obj = macd_agent(prices, fastPeriod, slowPeriod, signalPeriod)
            % Constructor for MACD agent
            %
            % Inputs:
            %    prices - Vector of asset prices
            %    fastPeriod - Fast EMA period (default: 12)
            %    slowPeriod - Slow EMA period (default: 26)
            %    signalPeriod - Signal line EMA period (default: 9)
            
            if nargin < 2
                fastPeriod = 12;
            end
            
            if nargin < 3
                slowPeriod = 26;
            end
            
            if nargin < 4
                signalPeriod = 9;
            end
            
            % Store parameters
            obj.prices = prices;
            obj.fastPeriod = fastPeriod;
            obj.slowPeriod = slowPeriod;
            obj.signalPeriod = signalPeriod;
            
            % Precompute signals
            obj.signals = macd_strategy(prices, fastPeriod, slowPeriod, signalPeriod);
        end
        
        function signal = getSignal(obj, t)
            % Get the trading signal for time step t
            %
            % Inputs:
            %    t - Time step (index)
            %
            % Outputs:
            %    signal - Trading signal: 1 (buy), -1 (sell), 0 (hold)
            
            % Return the precomputed signal for this time step
            if t > 0 && t <= length(obj.signals)
                signal = obj.signals(t);
            else
                warning('Time step out of range, returning 0');
                signal = 0;
            end
        end
        
        function setData(obj, prices)
            % Update the price data and recompute signals
            %
            % Inputs:
            %    prices - New vector of asset prices
            
            obj.prices = prices;
            obj.signals = macd_strategy(prices, obj.fastPeriod, obj.slowPeriod, obj.signalPeriod);
        end
    end
end 
 
 
 
 