classdef adv_im_tl < handle
    % advanced imaging buffer top level
    % just add import of is_photon and kt_index, realize the iterator
    
    properties (SetAccess = immutable)
        is_photon
        kt_index
        len
        
    end
    
    properties
        verbose = true;
    end
    
    properties (SetAccess = protected)
        % two indices of each eod interval
        slow
        fast
        
        total_photons_cache = -1;
        total_cycles_cache = -1;
        cycles_with_photons_cache = -1;
        max_photon_per_cycle_cache = -1;

        

    end
    
    properties (Dependent)
        total_photons
        total_cycles
        cycles_with_photons
        max_photon_per_cycle
    end
    
    methods
        function self = adv_im_tl(varargin)
            self.is_photon = logical(varargin{1});
            self.kt_index = uint8(varargin{2});
            self.len = length(self.is_photon);
            assert(length(self.kt_index) == self.len);
            self.reset_iterator();
        end
        
        function reset_iterator(self)
            self.slow = 1;
            self.fast = self.len;
            for i = 1:self.len
                if self.kt_index(i) ~= self.kt_index(1)
                    break;
                    
                end
                self.fast = i;
            end
        end
        
        function advance2next(self)
            
            if self.fast > self.len
                
                return;
            end
            if self.fast == self.len
                self.fast = self.fast + 1;
                return;
            end
            
            self.slow = self.fast + 1;
            self.fast = self.len;
            for i = self.slow:self.len
                if self.kt_index(i) ~= self.kt_index(self.slow)
                    break;
                    
                end
                self.fast = i;
            end
            
            
        end
        
        function [s, f] = next(self)
            if ~self.has_next
                error('tried to advance to next when there is no next');
            end
            s = self.slow;
            f = self.fast;
            self.advance2next();
        end
        
        function tf = has_next(self)
            tf = self.fast <= self.len;
        end
        
        function tmp = to_mat(self)
            % return a matrix view of data, all entris converted to double
            % override this method in child classes
            tmp = [double(self.is_photon(:)), double(self.kt_index(:))];
            
        end
        
        function p = calc(self)
            self.reset_iterator();
            bad_ct = 0;
            good_ct = 0;
            while self.has_next()
                [s, f] = self.next();
                fake_ct = 0;
                for i = s:f
                    if ~self.is_photon(i)
                        fake_ct = fake_ct + 1;
                    end
                end
                if fake_ct == 2
                    good_ct = good_ct + 1;
                else
                    bad_ct = bad_ct + 1;
                end
            end
            self.reset_iterator();
            p = 1;
            disp(['found ' num2str(good_ct) ' good intervals out of ' num2str(good_ct + bad_ct) ' intervals'])
        end
        
        function update_cache(self)
            % update these immutable values
            %         total_photons_cache = -1;
            %         total_cycles_cache = -1;
            %         cycles_with_photons_cache = -1;
            %         max_photon_per_cycle_cache = -1;
            
            self.reset_iterator();
            self.total_photons_cache = length(find(self.is_photon));
            
            max_ct = 0;
            cycle_ct = 0;
            cycle_wp_ct = 0; %cycle with photon
            alt_photon_ct = 0; % alternative way to count photons
            
            while (self.has_next())
                [s, f] = self.next();
                cycle_ct = cycle_ct + 1;
                if f - s > 1
                    alt_photon_ct = alt_photon_ct + f - s - 1;
                   cycle_wp_ct = cycle_wp_ct + 1; 
                   max_ct = max(max_ct, f - s - 1);
                end
                
                               
            end
            
            %assert(alt_photon_ct == self.total_photons);
            if self.verbose && alt_photon_ct ~= self.total_photons
               warning(['indiscrepancy in how many real photons collected: actual ' ...
                   num2str(self.total_photons) '; tabulated ' num2str(alt_photon_ct)]); 
            end
            
            self.max_photon_per_cycle_cache = max_ct;
            self.cycles_with_photons_cache = cycle_wp_ct;
            self.total_cycles_cache = cycle_ct;
        end
        
        function val = get.total_photons(self)
            if self.total_photons_cache ~= -1
                val = self.total_photons_cache;
                return;
            end
            self.update_cache();
            val = self.total_photons_cache;
        end
        function val = get.total_cycles(self)
            if self.total_cycles_cache ~= -1
                val = self.total_cycles_cache;
                return;
            end
            self.update_cache();
            val = self.total_cycles_cache;
        end
        function val = get.cycles_with_photons(self)
            if self.cycles_with_photons_cache ~= -1
                val = self.cycles_with_photons_cache;
                return;
            end  
            self.update_cache();
             val = self.cycles_with_photons_cache;
        end
        function val = get.max_photon_per_cycle(self)
            if self.max_photon_per_cycle_cache ~= -1
                val = self.max_photon_per_cycle_cache;
                return;
            end  
            self.update_cache();
             val = self.max_photon_per_cycle_cache;
        end        
    end
    
    methods(Static)
        function self = load_from_tdms(full_file_name)
            %function self = import_tdms(full_file_name)

            warning(' calling top level constructor');
            tdms_struct = TDMS_getStruct(full_file_name);
            disp('Data Loaded')
            
            fn = fieldnames(tdms_struct);
            data=tdms_struct.(fn{2});
            
            self = adv_im_tl.import_tdms_struct(data);
            
        end
        function self = import_tdms_struct(data)
            self = adv_im_tl(data.is_photon.data, data.KT_Index.data);
        end
    end
end