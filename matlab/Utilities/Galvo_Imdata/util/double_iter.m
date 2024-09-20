classdef double_iter < double_iter_tl
    properties (SetAccess = immutable)
        intervals
        data
        len
    end
    properties (SetAccess = protected)
        slow
        fast
        idx
    end
    properties
        % function handles for left and right borders
        lf = @(data_val, interval_val) data_val > interval_val;
        rf = @(data_val, interval_val) data_val <= interval_val;
    end
    methods
        function this = double_iter(data, in)

            if ~isa(in, 'intervals_tl')
                error('input is not a valid interval object!')
            end
            if ~lin_iter.is_ordered(data)
                error('input is not an ordered array!')
            end
            this.data = data;
            this.len = length(data);
            this.intervals = in;
        end

        function [is_valid, s, f, lb, rb, id] = next(this)
            s = this.slow;
            f = this.fast;
            
            lb = this.intervals.get_left(this.idx);
            rb = this.intervals.get_right(this.idx);
            id = this.idx;
            
            

            
            this.advance2next();
            is_valid = s <= f;

            % check correctness
            if is_valid
                ind = intersect(find(this.data >= lb), find(this.data <= rb));
                %assert(ind(1) == s && ind(2) == f);
            end
        end

        function tf = has_next(this)
            tf = this.idx <= this.intervals.n_interval;
        end

        function advance2next(this)
            disp('calling advance2next()')
            left_border = this.intervals.get_left(this.idx);
            right_border = this.intervals.get_right(this.idx);
            disp(['lb = ' num2str(left_border) ', rb = ' num2str(right_border) ' curr s/f: ' num2str(this.slow) ', ' num2str(this.fast)]);
            old_slow = this.slow;
            this.slow = this.len + 1;

            for i = old_slow : this.len
                if this.lf(this.data(i), left_border)
                    this.slow = i;
                    break;
                end                
            end

            old_fast = max([this.fast, 1]);
            this.fast = this.len;

            for i = old_fast : this.len
                if ~this.rf(this.data(i), right_border)
                    this.fast = i - 1;
                    break;
                end
            end


%             for i = old_slow : this.len
%                 if this.lf(this.data(i), left_border)
%                     this.slow = i;
%                     break;
%                 end
%             end
% 
%             for i = max([this.fast, 1]) : this.len
%                 if ~this.rf(this.data(i), right_border)
%                     this.fast = i - 1;
%                     break;
%                 end
%                 if i == this.len && this.rf(this.data(i), right_border)
%                     this.fast = this.len;
%                 end
%             end
            this.idx = this.idx + 1;
        end

        function reset_iterator(this)
            this.slow = 1;
            this.fast = 1;
            this.idx = 1;
            this.advance2next();

        end

        function s = to_string(this)
            s = ['data range: ' num2str(min(this.data)) ' - ' num2str(max(this.data)) '; avg. step: ' num2str(mean(diff(this.data))) '\n intervals: ' num2str(this.intervals.get_left(1)) ' - ' num2str(this.intervals.get_right(this.intervals.n_interval))];
        end
    end
end