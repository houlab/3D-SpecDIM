classdef (Abstract) iter_tl < handle
    methods (Abstract)
        next(this)
        has_next(this)
        advance2next(this)
        reset_iterator(this)
    end

    properties (Abstract, SetAccess = immutable)
        data
        
    end
end