classdef (Abstract) double_iter_tl < iter_tl
    % cz 230804
    %

    

    properties (Abstract, SetAccess = immutable)
        %n_intervals
        intervals
    end

    properties (Abstract, SetAccess = protected)
        idx % current interval index
    end

    properties (Abstract)
        % function handles controlling if left and right borders are in
        % range
        lf
        rf
    end

    methods (Abstract)
        % get borders of nth interval
        % get_left(this, n)
        % get_right(this, n)
    end
end