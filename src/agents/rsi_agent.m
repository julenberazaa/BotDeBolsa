classdef rsi_agent < handle
    % RSI_AGENT - Agent that generates trading signals based on RSI strategy
    %
    % This agent wraps the RSI strategy and provides a consistent interface
    % with the getSignal method for use in the PortfolioEnv.
    
    properties
        prices      % Historical price data
        window      % RSI window size
        overbought  % Overbought threshold
        oversold    % Oversold threshold
        signals     % Precomputed signals
    end
    
    methods
        function obj = rsi_agent(prices, window, overbought, oversold)
            % Constructor for RSI agent
            %
            % Inputs:
            %    prices - Vector of asset prices
            %    window - RSI calculation window (default: 14)
            %    overbought - Overbought threshold (default: 70)
            %    oversold - Oversold threshold (default: 30)
            
            if nargin < 2
                window = 14;
            end
            
            if nargin < 3
                overbought = 70;
            end
            
            if nargin < 4
                oversold = 30;
            end
            
            % Store parameters
            obj.prices = prices;
            obj.window = window;
            obj.overbought = overbought;
            obj.oversold = oversold;
            
            % Precompute signals
            obj.signals = rsi_strategy(prices, window, overbought, oversold);
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
            obj.signals = rsi_strategy(prices, obj.window, obj.overbought, obj.oversold);
        end
    end
end 
 
 
 