classdef (Abstract) intervals_tl < handle
    properties (Abstract, SetAccess = immutable)
        n_interval
    end
    methods (Abstract)
        get_left(this, n)
        get_right(this, n)
    end
end