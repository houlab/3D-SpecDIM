classdef double_iter_bs < double_iter
    properties
        lb_incl = false;
        rb_incl = true;
    end
    % double iterator, but using binary search
    methods
        function [is_valid, s, f, lb, rb, id] = next(this)
            id = this.idx;
            lb = this.intervals.get_left(id);
            rb = this.intervals.get_right(id);
            [s, f] = double_iter_bs.first_and_last(this.data, lb, rb, this.lb_incl, this.rb_incl);
            is_valid = s ~= 0;
            this.advance2next();
        end

        function tf = has_next(this)
            % modified 230815: does not iterate if there are additional
            % intervals that are impossible to overlap with data
            % this assumes the input interval is ordered
            tf = this.idx <= this.intervals.n_interval && (this.intervals.get_left(this.idx) <= this.data(end)) ;
        end

        function advance2next(this)
            this.idx = this.idx + 1;
        end

        function reset_iterator(this)
            this.idx = 1;
            this.lf = @(val) error('no need of this!');
            this.rf = @(val) error('no need of this!');
        end

        function set.lb_incl(this, val)
            if isequal(val, true) || isequal(val, false)
                this.lb_incl = val;
                return;
            end
            error('invalud argument!');
        end
        function set.rb_incl(this, val)
            if isequal(val, true) || isequal(val, false)
                this.rb_incl = val;
                return;
            end
            error('invalud argument!');
        end
    end

    methods (Static)
        function idx = find_first(data, fun)
            % find first true value in [f...ft...t]
            len = length(data);
            tmp_idx = double_iter_bs.find_any(data, fun, 1, len);
            if (tmp_idx == 0)
                idx = 0;
                return;
            end
            curr_idx = tmp_idx;
            while (curr_idx ~= 0)
                prev_idx = curr_idx;
                curr_idx = double_iter_bs.find_any(data, fun, 1, curr_idx - 1);

            end
            idx = prev_idx;
        end

        function idx = find_last(data, fun)
            % find last true value in [t...tf...f]
            if fun(data(end))
                idx = length(data);
                return;
            end
            idx = double_iter_bs.find_first(data, @(val) not(fun(val))) - 1;
            if (idx == -1)
                idx = 0;
            end

        end

        function idx = find_any(data, fun, i1, i2)
            % disp(['lo = ' num2str(i1) ', hi = ' num2str(i2)])
            % find any true value
            if i1 > i2 || i1 <= 0 || i2 > length(data)
                idx = 0;
                return;
            end
            lo = int32(i1);
            hi = int32(i2);
            while hi - lo > 1
                mid = idivide(hi + lo, 2);
                %disp(['hi = ' num2str(hi) ' lo = ' num2str(lo) ' mid = ' num2str(mid)]);
                if fun(data(mid))
                    idx = mid;
                    return;
                end
                lo = mid;
            end
            if fun(data(lo))
                idx = lo;
                return;
            end
            if (fun(data(hi)))
                idx = hi;
                return;
            end
            idx = 0;
        end

        function [l, r] = first_and_last(data, left_border, right_border, lb_incl, rb_incl)

            if lb_incl
                lf = @(val) val >= left_border;
            else
                lf = @(val) val > left_border;
            end

            if rb_incl
                rf = @(val) val <= right_border;
            else
                rf = @(val) val < right_border;
            end

            l = double_iter_bs.find_first(data, lf);
            r = double_iter_bs.find_last(data, rf);

            % disp(['l, r =  ' num2str(l) ', ' num2str(r)])

            if l == 0 || r == 0 || l > r
                l = 0;
                r = 0;
            end

            l = double(l);
            r = double(r);

        end

        function [l, r] = first_and_last_bf(data, left_border, right_border, lb_incl, rb_incl)
            % index_of_first = find(YourVector == TheValue, 1, 'first');
            if lb_incl
                l = find(data >= left_border, 1, 'first');
            else
                l = find(data >  left_border, 1, 'first');
            end

            if rb_incl
                r = find(data <= right_border, 1, 'last');
            else
                r = find(data <  right_border, 1, 'last');
            end

            if isempty(l) || isempty(r) || l > r
                l = 0;
                r = 0;
            end
        end
    end
end