classdef simple_intervals < intervals_tl
    properties (SetAccess = immutable)
        intervals
        n_interval
    end
    methods 
        function this = simple_intervals(in)
            % input: array or matrix of two columns
            % input must be ordered
            if isvector(in)
                this.intervals = simple_intervals.array2interval(in);
                this.n_interval = length(in) - 1;
                return;
            end
            this.intervals = in;
            this.n_interval = length(in);
        end

        function val = get_left(this, n) 
            val = this.intervals(n, 1);
        end
        
        function val = get_right(this, n)
            val = this.intervals(n, 2);
        end
    end

    methods (Static)
        function out = array2interval(in)
            if ~lin_iter.is_numeric_array(in)
                error('input is not an array');
            end
                if ~lin_iter.is_strictly_ordered(in)
                    error('input is not strictly ordered!');
                end
          
            in = in(:);
            out = zeros(length(in) - 1, 2);
            out(:, 1) = in(1 : end - 1);
            out(:, 2) = in(2 : end);
        end
    end
end