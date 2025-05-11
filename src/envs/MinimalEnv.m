classdef MinimalEnv < rl.env.MATLABEnvironment
    methods
        function this = MinimalEnv()
            ObservationInfo = rlNumericSpec([1 1], 'LowerLimit', -inf, 'UpperLimit', inf);
            ActionInfo = rlFiniteSetSpec([-1 0 1]);
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end

        function [obs, reward, isDone, log] = step(this, action)
            obs = randn();
            reward = 1;
            isDone = false;
            log = [];
        end

        function obs = reset(this)
            obs = 0;
        end
    end
end
