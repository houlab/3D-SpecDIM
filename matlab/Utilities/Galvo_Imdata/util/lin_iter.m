classdef lin_iter < iter_tl
    properties

        edge_thres = .5;
        window_size = []; 
        
    end
    properties (SetAccess = protected)
        slow
        fast 
        left_border
        right_border

        proj_len % projected length of output
    end
    properties (SetAccess = immutable)
        data
        len
        first_val
    end
    methods
        function this = lin_iter(in)
                          assert(lin_iter.is_numeric_array(in));
                          assert(lin_iter.is_ordered(in));
                this.data = in;
                this.len = length(in);
                this.first_val = in(1);

        end
        function reset_iterator(this)
            if isempty(this.window_size)
               error('window_size not initialized!'); 
            end
            this.fast = 0;
            this.left_border = this.first_val;
            this.right_border = this.left_border + this.window_size;
            this.advance2next();
            this.proj_len = ceil((this.data(this.len) - this.first_val - ...
                this.edge_thres*this.window_size)/this.window_size);
        end
        function [s, f] = next(this)
            if (~this.has_next())
                error('calling next() when there is no next value!')
            end
            s = this.slow;
            f = this.fast;

            assert(s <= f);
            % update left, right borders
%             if this.fast < this.len
%                 k = ceil((this.data(this.fast + 1) - this.right_border)/this.window_size);
%                 this.left_border = this.left_border + k*this.window_size;
%                 this.right_border = this.right_border + k*this.window_size;
%             end            

            this.advance2next();
%             if this.fast == this.len
%                 disp('looking good')
%                 % check if right border satisfies edge_thres
%                 if (this.data(this.fast) - this.data(this.left_border) < this.edge_thres*this.window_size)
%                     this.advance2next(); % dump current value
%                 end
%                 return;
%             end

            
        end
        function tf = has_next(this)
                tf = this.slow < this.len;
        end
        function advance2next(this) 
            % left_border and right_border should be updated and valid
            if (this.slow > this.len)
                return;
            end
            if (this.fast == this.len)
                this.slow = this.len + 1;
                return;
            end

            this.slow = this.fast + 1;
            idx = this.slow;
            while (idx <= this.len && this.data(idx) <= this.right_border)
                idx = idx + 1;
            end
            this.fast = idx - 1;

            if this.fast == this.len
                % check if right border satisfies edge_thres
                disp('checking...')
                %error('@@')
                if (this.data(this.fast) - this.left_border < this.edge_thres*this.window_size)
                    disp('invalid edge segment')
                    this.advance2next(); % dump current value
                end
                return;
            end
                disp('updated borders')
                k = ceil((this.data(this.fast + 1) - this.right_border)/this.window_size);
                disp(['k = ' num2str(k)]);
                this.left_border = this.left_border + k*this.window_size;
                this.right_border = this.right_border + k*this.window_size;
                disp(['borders: ' num2str(this.left_border) ' + ' num2str(this.right_border) ])
        end
    end

    methods (Static)
        function tf = is_numeric_array(in)
            tf = isnumeric(in) && isreal(in) && isvector(in);
        end
        function tf = is_ordered(in)
            
            % allows dupe values
            tf = true;
            for i = 1: length(in) - 1
                if (in(i) > in(i + 1))
                    tf = false;
                    return;
                end
            end
        end
        function tf = is_strictly_ordered(in)
            tf = true;
            for i = 1 : length(in) - 1
                if in(i) >= in(i + 1)
                    tf = false;
                    return;
                end
            end
        end
    end
end