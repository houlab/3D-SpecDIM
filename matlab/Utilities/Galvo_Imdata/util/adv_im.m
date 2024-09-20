classdef adv_im < adv_im_tl
    % advanced imaging data: can be used for both legacy or bayesian style
    % imaging data
    % developed for ES v12
    properties(Constant)
        NON_VALUE_I16 = int16(-32768);
        N_FIELDS = 8; % native fields
        TICK = 12.5e-9;
    end
    properties (SetAccess = immutable)
        %         is_photon % boolean
        %         is_dio0 % boolean

        % inherited:
        % is_photon
        % kt_index
        % len
        % boolean
        is_dio0
        is_dio9
        is_galvo

        % i16, nothing here need to be stored as u16
        field1
        field2
        field3

        % expanded fields
        % these are not needed

        %         % i16, non value is -32768
        %         x_galvo
        %         y_galvo
        %
        %         x_stage
        %         y_stage
        %         z_stage
        %
        %         % i16, non value -1;
        %         tag_phase
        %         ticks



    end



    properties
        x_conv = (1/32767*76.249);
        y_conv = (1/32767*76.403);
        z_conv = (1/32767*66.342);

        tag_cycle_len = 14.3e-6; % added 230815

        x_conv_g = .02;
        y_conv_g = .02;

        bin_time = 20; % bin time in us (eod cycle time)

        ti_time = 10; % tracking index time (legacy: 1 index = 10 us)

        bayes_time = 1; % bayes calculation time

        max_cycle_len_tick = 3200; % the program throws an error if trying to add a cycle length longer than this

        bayes_bin_time = 1e-6; % bin time for bayes calculation, in s

    end

    properties (SetAccess = protected)
        % used for iterator
        prev_galvo_x = 0;
        prev_galvo_y = 0;
        prev_galvo_z = 0;
        prev_stage_x = 0;
        prev_stage_y = 0;
        prev_stage_z = 0;
    end

    properties (Dependent)
        em_rate;
        duration;

    end

    properties (SetAccess = protected)
        em_rate_cache = -1;
        duration_cache = -1;
    end

    methods
        function im = crop_data(self, n)
            % chop the first several entries of data
            im = self.index_data(1 : n);
        end
        function im = index_data(self, idx)
            im = adv_im(self.is_photon(idx), self.kt_index(idx), self.is_dio0(idx), self.is_dio9(idx), ...
                self.is_galvo(idx), ... 
                adv_im.rev_convert_custom_field(self.is_photon(idx),self.field1(idx)), ... 
                adv_im.rev_convert_custom_field(self.is_photon(idx),self.field2(idx)), ...
                adv_im.rev_convert_custom_field(self.is_photon(idx),self.field3(idx)));
        end
        function self = adv_im(varargin)
            % order of arguments:
            % is_photon
            % kt_index
            % is_dio0
            % is_dio9
            % is_galvo

            % field_1
            % field_2
            % field_3
            self@adv_im_tl(varargin{1}, varargin{2});
            self.is_dio0 = logical(varargin{3});
            self.is_dio9 = logical(varargin{4});
            self.is_galvo = logical(varargin{5});
            self.field1 = adv_im.convert_custom_field(self.is_photon, varargin{6});
            self.field2 = adv_im.convert_custom_field(self.is_photon, varargin{7});
            self.field3 = adv_im.convert_custom_field(self.is_photon, varargin{8});

            % build other fields

            %self.x_stage = int16(len, 1);



        end

        function data = legacy_ti(self)
            % returns legacy imaging data view with tracking index
            data = self.gather_legacy();
            data(:,3) = round(data(:,3)./self.ti_time.*1e6);
        end

        function data = legacy_ticks(self)
            % returns legacy imaging data view with ticks
            data = self.gather_legacy();
            data(:,3) = round(data(:,3)./self.TICK);
        end

        function data = gather_legacy(self)
            % 4 columns: kt index, phase(0~628), channel, time, in second
            data = zeros(self.total_photons, 4);
            idx = 1;
            self.reset_iterator();
            curr_t = 0;
            self.reset_iterator();
            while self.has_next()
                [s, f] = self.next();
                curr_t = curr_t + self.bin_time*1e-6;
                for i = s:f
                    if ~self.is_photon(i)
                        continue;
                    end
                    chan = -1;
                    if (self.is_dio0(i) && ~self.is_dio9(i))
                        chan = 1; % dio0
                    elseif (~self.is_dio0(i) && self.is_dio9(i))
                        chan = 2; % dio9
                    elseif (self.is_dio0(i) && self.is_dio9(i))
                        chan = 3; % simutaneous
                    end
                    t = curr_t + double(self.field2(i))*self.TICK;
                    data(idx, :) = [double(self.kt_index(i)), double(self.field1(i)), t, chan];
                    idx = idx + 1;
                end
            end
        end

        function data = gather_adv(self)
            % 9 columns: kt index, phase(0~628), channel, time (in second)
            % plus: x_stage, y_stage, z_stage, x_galvo, y_galvo
            % 230306: added z_galvo (to column 10)
            data = zeros(self.total_photons, 9);
            idx = 1;
            self.reset_iterator();
            curr_t = 0;
            curr_xs = 0;
            curr_xg = 0;
            curr_zg = 0;
            curr_ys = 0;
            curr_yg = 0;
            curr_zs = 0;
            % initialize
            for i = 1:self.len
                if ~self.is_photon(i) && self.is_galvo(i)
                    curr_xs = self.field1(i);
                    curr_ys = self.field2(i);
                    curr_zs = self.field3(i);
                    break;
                end
            end
            for i = 1:self.len
                if ~self.is_photon(i) && ~self.is_galvo(i)
                    curr_xg = self.field1(i);
                    curr_yg = self.field2(i);
                    curr_zg = self.field3(i);

                    break;
                end
            end
            self.reset_iterator();
            while self.has_next()
                [s, f] = self.next();
                curr_t = curr_t + self.bin_time*1e-6;

                % gather stage and galvo info
                found_stage = false;
                found_galvo = false;
                for i = s:f
                    if found_stage && found_galvo
                        break;
                    end
                    if ~self.is_photon(i) && self.is_galvo(i)
                        curr_xs = double(self.field1(i));
                        curr_ys = double(self.field2(i));
                        curr_zs = double(self.field3(i));
                        found_stage = true;
                    end
                    if ~self.is_photon(i) && ~self.is_galvo(i)
                        curr_xg = double(self.field1(i));
                        curr_yg = double(self.field2(i));

                        found_galvo = true;
                    end
                end

                for i = s:f
                    if ~self.is_photon(i)
                        continue;
                    end
                    chan = -1;
                    if (self.is_dio0(i) && ~self.is_dio9(i))
                        chan = 1; % dio0
                    elseif (~self.is_dio0(i) && self.is_dio9(i))
                        chan = 2; % dio9
                    elseif (self.is_dio0(i) && self.is_dio9(i))
                        chan = 3; % simutaneous
                    end
                    %t = curr_t + double(self.field2(i))*self.TICK;
                    t = curr_t + double(curr_zg)*self.TICK;
                    data(idx, :) = [double(self.kt_index(i)), double(self.field1(i)), t, chan, curr_xs, curr_ys, curr_zs, curr_xg, curr_yg];
                    idx = idx + 1;
                end
                %error('SS')
            end
        end

        function data = gather_adv2(self)
            % 230306: rephrased gather adv im function utilizing overriden
            % iterator function
            % 9 columns: kt index, phase(0~628), channel, time (in second)
            % plus: x_stage, y_stage, z_stage, x_galvo, y_galvo
            % 230306: added z_galvo (to column 10)
            data = zeros(self.total_photons, 10);
            idx = 1;
            self.reset_iterator();
            curr_t = 0;
            while self.has_next()
                [s, f, kt_idx, galvo_x, galvo_y, stage_x, stage_y, stage_z, galvo_z] = self.next();
                for i = s:f
                    if ~self.is_photon(i)
                        continue;
                    end
                    chan = -1;
                    if (self.is_dio0(i) && ~self.is_dio9(i))
                        chan = 1; % dio0
                    elseif (~self.is_dio0(i) && self.is_dio9(i))
                        chan = 2; % dio9
                    elseif (self.is_dio0(i) && self.is_dio9(i))
                        chan = 3; % simutaneous
                    end
                    t = curr_t + double(self.field2(i))*self.TICK;
                    data(idx, :) = [kt_idx, double(self.field1(i)), t, chan, stage_x, stage_y, stage_z, galvo_x, galvo_y, galvo_z];
                    idx = idx + 1;
                end
                curr_t = curr_t + self.bin_time*1e-6;
            end

        end

        function data = gather_bayes_abandoned(self)
            % 8 columns
            % n, kt_idx, tag_pos, x_stage, y_stage, z_stage, x_galvo,
            % y_galvo

            %                     total_photons
            %         total_cycles
            %         cycles_with_photons
            %         max_photon_per_cycle

            % first, assign matrix

            cycle_len = round( self.bin_time/self.bayes_time);

            data = -1.*ones( self.total_cycles*cycle_len + 100,8);

            data_len = length(data(:,1 ));


        end

        function [nk0, nk9, kt_data, tag_phase, stage_and_galvo, t] = gather_bayes(self)
            % t = self.get_t();
            max_t = self.duration + .1;
            dt = self.bayes_bin_time;

            data_len = round(max_t/dt); % add 100ms

            % initialize data
            nk0 = zeros(data_len, 1);
            nk9 = zeros(data_len, 1);
            kt_data = nan(data_len, 1) - 1;
            tag_phase = nan(data_len, 1);
            stage_and_galvo = nan(data_len, 6);
            t = nan(data_len, 1);

            curr_sg = self.get_initial_sg();
            cache0 = 0;
            cache9 = 0;
            % used to calc tag phase
            prev_t = 0;
            prev_tag_phase = 0;
            % get data stream
            [is_p, ds_kt, is_0, is_9, sg, ds_tag, ds_t] = self.get_data_stream();
            curr_kt = ds_kt(1);

            % create iterator

            %iter = double_iter_bs(ds_t, simple_intervals(linspace(0, max_t, dt)));
            iter = double_iter_bs(ds_t, simple_intervals(0 : dt: max_t) );
            disp(iter.to_string());
            iter.reset_iterator();
            idx = 1;
            
            call_ct = 0;
            while iter.has_next()
                call_ct = call_ct + 1;
                if mod(call_ct, 1e4) == 0
                    disp([num2str(call_ct) ' of calls complete'])
                end
                [is_valid, s, f, lb, ~, ~] = iter.next();
                curr_t = lb + .5*dt;
                t(idx) = curr_t;
                %disp('entering cycle')
                if ~is_valid % nothing in the interval, simple case
                    %disp('1st case')
                    kt_data(idx) = curr_kt;
                    tag_phase(idx) = (curr_t - prev_t)/self.tag_cycle_len*100 + prev_tag_phase; % in 100
                    stage_and_galvo(idx, :) = curr_sg;
                    nk0(idx) = cache0;
                    nk9(idx) = cache9;
                    cache0 = 0;
                    cache9 = 0;
                    idx = idx + 1;

                    continue;
                end
                % if sum(self.is_photon(s : f)) == f - s + 1 % no_fake_photon 
                if sum(is_p(s : f)) == f - s + 1 % no_fake_photon
                    %disp('2nd case')
                    kt_data(idx) = curr_kt;
                    stage_and_galvo(idx, :) = curr_sg;
                    % tag phase
                    prev_t = ds_t(f);
                    prev_tag_phase = ds_tag(f);
                    % is this correct??
                    tag_phase(idx) = mean(ds_t(s : f) - curr_t)/self.tag_cycle_len*100 + mean(ds_tag(s : f));
                    nk0(idx) = sum(and(is_0(s : f), is_p(s : f))) + cache0;
                    nk9(idx) = sum(and(is_9(s : f), is_p(s : f))) + cache9;
                    idx = idx + 1;
                    cache0 = 0;
                    cache9 = 0;
                    continue;
                end

                % first_idx = find(not(is_p(s : f)), 1, 'first');
                first_idx = find(not(is_p(s : f)), 1);

                curr_kt = ds_kt(first_idx);
                %disp('3rd case')
                if isempty(first_idx)
                    error('unexpected error!')
                end
                if length(first_idx) > 1
                    warning(['found multiple (N = ' num2str(length(first_idx)) ') fake photons at i = ' num2str(idx)])
                    first_idx = first_idx(1);
                end
                % first register current entry, then update cache0 and cache9
                nk0(idx) = sum(and(is_0(s : first_idx - 1), is_p(s : first_idx - 1))) + cache0;
                nk9(idx) = sum(and(is_9(s : first_idx - 1), is_p(s : first_idx - 1))) + cache9;

                kt_data(idx) = curr_kt;
                stage_and_galvo(idx, :) = curr_sg;

                cache0 = sum(and(is_0(first_idx + 1 : f), is_p(first_idx + 1 : f)));
                cache9 = sum(and(is_9(first_idx + 1 : f), is_p(first_idx + 1 : f)));

                % next, update curr_kt and curr_sg
                curr_kt = ds_kt(first_idx);
                curr_sg = sg(first_idx, :);

                % finally, update tag phase
                if any(is_p(s : f))
                    tag_phase(idx) = mean(ds_t(is_p(s : f)) - curr_t)/self.tag_cycle_len*100 + mean(ds_tag(is_p(s : f)));
                    last_idx = find(is_p(s : f), 1, 'last');
                    prev_t = ds_t(last_idx);
                    prev_tag_phase = ds_tag(last_idx);                    
                else % no photon in cycle
                    tag_phase(idx) = (curr_t - prev_t)/self.tag_cycle_len*100 + prev_tag_phase; % in 100
                end

                idx = idx + 1;
            end

            fprintf('called next() %.0f times \n', call_ct)

        end

        function [s, f, kt_idx, galvo_x, galvo_y, stage_x, stage_y, stage_z, galvo_z] = next(self)
            % @Override

            % 230306: added an output field 'galvo_z' which denotes thrid
            % field while writing galvo data

            if ~self.has_next()
                error('tried to advance to next when there is no next');
            end

            s = self.slow;
            f = self.fast;

            kt_idx = double(self.kt_index(s));

            found_stage = false;
            found_galvo = false;
            curr_xs = self.prev_stage_x;
            curr_ys = self.prev_stage_y;
            curr_zs = self.prev_stage_z;
            curr_xg = self.prev_galvo_x;
            curr_yg = self.prev_galvo_y;
            curr_zg = self.prev_galvo_z;
            for i = s:f
                if found_stage && found_galvo
                    break;
                end
                if ~self.is_photon(i) && self.is_galvo(i)
                    curr_xs = double(self.field1(i));
                    curr_ys = double(self.field2(i));
                    curr_zs = double(self.field3(i));
                    found_stage = true;
                end
                if ~self.is_photon(i) && ~self.is_galvo(i)
                    curr_xg = double(self.field1(i));
                    curr_yg = double(self.field2(i));
                    curr_zg = double(self.field3(i));

                    found_galvo = true;
                end
            end

            galvo_x = curr_xg;
            galvo_y = curr_yg;
            galvo_z = curr_zg;
            stage_x = curr_xs;
            stage_y = curr_ys;
            stage_z = curr_zs;

            self.prev_stage_x = curr_xs;
            self.prev_stage_y = curr_ys;
            self.prev_stage_z = curr_zs;
            self.prev_galvo_x = curr_xg;
            self.prev_galvo_y = curr_yg;
            self.prev_galvo_z = curr_zg;

            self.advance2next();



        end


        function tmp = to_mat(self)
            % Override
            % returns a matrix representation of raw data with following order:
            % Column No | description
            % 1 : is_photon
            % 2 : kt_index
            % 3 : is dio0
            % 4 : is dio9
            % 5 : is galvo (actually it's is stage)
            % 6 : field_1
            % 7 : field_2
            % 8 : field_3
            tmp = zeros(self.len, self.N_FIELDS);
            tmp(:,1) = double(self.is_photon(:));
            tmp(:,2) = double(self.kt_index(:));
            tmp(:,3) = double(self.is_dio0(:));
            tmp(:,4) = double(self.is_dio9(:));
            tmp(:,5) = double(self.is_galvo(:));
            tmp(:,6) = double(self.field1(:));
            tmp(:,7) = double(self.field2(:));
            tmp(:,8) = double(self.field3(:));

            % 230804: added 9th row reflecting photon arrival time (in s)
            % of both real and fake photon

        end

        function data = to_mat_v2(self)
            [is_p, kt_data, is_0, is_9, stage_and_galvo, tag_phase, t] = self.get_data_stream();
            fprintf('got %.0f entries from original %.0f data points', length(is_p), self.len);
            data = zeros(length(is_p), 12);
            data(:, 1) = is_p;
            data(:, 2) = kt_data;
            data(:, 3) = is_0;
            data(:, 4) = is_9;
            data(:, 5) = tag_phase;
            data(:, 6:11) = stage_and_galvo;
            data(:, 12) = t;
        end

        function t = get_t(self)
            % get time stamps for all photons
            % output: double column array, length = self.len
            t = zeros(self.len, 1);
            curr_t = 0; % last registered time stamp
            curr_t0 = 0; % start time of current cycle
            for i = 1 : self.len
                if (self.is_photon(i))
                    pre_curr_t = curr_t0 + double(self.field2(i))*self.TICK;
                    if pre_curr_t <= curr_t
                        if self.verbose
                            disp(['i = ' num2str(i) ': potentially missing fake photon(s)'])
                        end
                        curr_t0 = t(i - 1) + double(self.field2(i))*self.TICK;
                        curr_t = curr_t0;
                        t(i) = curr_t;
                        continue;
                    end
                    curr_t = pre_curr_t;
                    t(i) = curr_t;
                    continue;
                end

                if (self.is_galvo(i))
                    if (i ~= self.len && ~self.is_galvo(i + 1))
                        curr_t0 = curr_t0 + double(self.field3(i + 1) + 1)*self.TICK;
                        if self.field3(i + 1) > self.max_cycle_len_tick
                            % if there are no cycles longer than max_cycle_len_tick, it's
                            % likely a bug
                            error(['i = ' num2str(i + 1) ': trying to add a cycle longer than ' num2str(self.max_cycle_len_tick) ' ticks!'])
                        end
                        if curr_t0 <= curr_t % this is from previous cycles
                            if self.verbose
                                warning(['time stamp rollback at i = ' num2str(i)])
                            end
                            curr_t0 = curr_t + self.TICK*5;
                        end
                        curr_t = curr_t0;
                        t(i) = curr_t;
                        continue;
                    end
                    if self.verbose
                        warning(['missing 2nd fake photon at i = ' num2str(i)])
                        if i ~= 1
                            curr_t0 = t(i - 1) + 5*self.TICK;
                            curr_t = curr_t0;
                            t(i) = curr_t;
                        end
                    end
                    continue;
                end

                % this is 2nd entry (and fake photon) in a cycle
                if ~self.is_galvo(i)
                    if i ~= 1 && self.is_photon(i - 1)
                        if self.verbose
                            warning(['missing 1st fake photon at i = ' num2str(i)]);
                        end
                        curr_t0 = t(i - 1) + self.TICK*5;
                        curr_t = curr_t0;
                        t(i) = curr_t;
                        continue;
                    end

                    % what's this line??? 230814
                    % curr_t0 = curr_t0 + double(self.field3(i + 1))*self.TICK;

                    curr_t = curr_t0 + self.TICK*2;
                    %curr_t = curr_t0;
                    t(i) = curr_t;
                    continue;
                end

                if self.verbose
                    warning('something went wrong!');
                end

            end

        end

        function [is_p, kt_data, is_0, is_9, stage_and_galvo, tag_phase, t] = get_data_stream(self)
            % assign data
            is_p = false(self.len, 1);
            is_0 = false(self.len, 1);
            is_9 = false(self.len, 1);
            kt_data = zeros(self.len, 1);
            stage_and_galvo = nan(self.len, 6);
            tag_phase = nan(self.len, 1);
            t = zeros(self.len, 1);

            idx = 1;
            time_stamps = self.get_t();

            curr_sg = self.get_initial_sg();

            for i = 1 : self.len
                if ~self.is_photon(i) && ~self.is_galvo(i)
                    % do not write anything here
                    curr_sg(4:6) = [double(self.field1(i)), double(self.field2(i)), double(self.field3(i))];
                    continue;
                end
                if ~self.is_photon(i) && self.is_galvo(i)
                    if i ~= self.len && ~self.is_photon(i + 1)
                        curr_sg(4:6) = [double(self.field1(i + 1)), double(self.field2(i + 1)), double(self.field3(i + 1))];
                    end
                    curr_sg(1:3) = [double(self.field1(i)), double(self.field2(i)), double(self.field3(i))];
                    is_p(idx) = false;
                    kt_data(idx) = self.kt_index(i);
                    stage_and_galvo(idx, :) = curr_sg;
                    t(idx) = time_stamps(i);
                    idx = idx + 1;
                    continue;
                end
                % if self.is_photon(i)
                is_p(idx) = true;
                is_0(idx) = self.is_dio0(i);
                is_9(idx) = self.is_dio9(i);
                kt_data(idx) = self.kt_index(i);
                stage_and_galvo(idx, :) = curr_sg;
                tag_phase(idx) = double(self.field1(i));
                t(idx) = time_stamps(i);
                idx = idx + 1;
            end

            % trim data
            is_p(idx : end) = [];
            tag_phase(idx : end) = [];
            is_0(idx : end) = [];
            is_9(idx : end) = [];
            kt_data(idx : end) = [];
            stage_and_galvo(idx : end, :) = [];
            t(idx : end) = [];

            %error('@@')
        end

        function curr_sg = get_initial_sg(self)
            % get initial stage and galvo position
            found_stage = false;
            found_galvo = false;
            for i = 1 : self.len
                if found_stage && found_galvo
                    break;
                end
                if ~self.is_photon(i) && self.is_galvo(i)
                    found_stage = true;
                    curr_sg(1:3) = [double(self.field1(i)), double(self.field2(i)), double(self.field3(i))];

                end
                if ~self.is_photon(i) && ~self.is_galvo(i)
                    found_galvo = true;
                    curr_sg(4:6) = [double(self.field1(i)), double(self.field2(i)), double(self.field3(i))];
                end
            end
            if ~found_stage || ~found_galvo
                error('unexpected error!')
            end
        end

        function [nk, stage_and_galvo] = get_nk_sg(self)
            raw_nk = diff( find( diff(double(self.kt_index)) ~= 0)) - 2;
            stage_idx = and(not(self.is_photon), self.is_galvo);
            galvo_idx = and(not(self.is_photon), not(self.is_galvo));

            pre_stage_data = [double(self.field1(stage_idx))'.*self.x_conv, ...
                double(self.field2(stage_idx))'.*self.y_conv, ... 
                double(self.field3(stage_idx))'.*self.z_conv ];

            pre_galvo_data = [double(self.field1(galvo_idx))'.*self.x_conv_g, ...
                double(self.field2(galvo_idx))'.*self.y_conv_g, ... 
                double(self.field3(galvo_idx))'.*1 ];
            data_len = max([length(raw_nk), length(pre_stage_data(:, 1)), length(pre_galvo_data(:, 1))]);
            
            nk = zeros(data_len, 1);
            nk(1 : length(raw_nk)) = raw_nk;

            stage_and_galvo = zeros(data_len, 6);
            stage_and_galvo(1:length(pre_stage_data(:, 1)), 1:3) = pre_stage_data;
            stage_and_galvo(1:length(pre_galvo_data(:, 1)), 4:6) = pre_galvo_data;
        end

        function [t, int_data, stage_and_galvo] = get_khz_data(self)
            [nk, pre_stage_and_galvo] = self.get_nk_sg();
            avg_cycle_len = mean(pre_stage_and_galvo(:, 6))*self.TICK;
            dec_factor = round(1e-3/avg_cycle_len);
            if self.verbose
                disp(['dec factor: ' num2str(dec_factor)])
            end
            
            int_data = dectrackdata(nk(:)./avg_cycle_len, dec_factor);
            stage_and_galvo = dectrackdata(pre_stage_and_galvo(:, 1:5), dec_factor);
            t = [1:length(int_data)]./1e3;
        end
 
        function val = get.em_rate(self)
            if (self.em_rate_cache ~= -1)
                val = self.em_rate_cache;
                return;
            end
            self.em_rate_cache = self.total_photons/self.duration;
            val = self.em_rate_cache;
        end

        function val = get.duration(self)
            if (self.duration_cache ~= -1)
                val = self.duration_cache;
                return;
            end
            self.duration_cache = self.TICK*sum(double(self.field3(and(~self.is_photon, ~self.is_galvo))));
            val = self.duration_cache;
        end
    end

    methods (Static)
        function self = import_tdms_struct(data)
            self = adv_im(data.is_photon.data,  data.KT_Index.data, data.is_dio0.data, ...
                data.is_dio9.data, data.is_galvo.data , data.field_1.data, data.field_2.data, data.field_3.data);
        end
        function self = import_tdms(full_file_name)
            tdms_struct = TDMS_getStruct(full_file_name);
            disp('Data Loaded')

            fn = fieldnames(tdms_struct);
            data=tdms_struct.(fn{2});

            self = adv_im.import_tdms_struct(data);
        end
        function self = import_manual()
            [fname, pname] = uigetfile('*.tdms');
            self = adv_im.import_tdms(fullfile(pname, fname));
        end

        function out = convert_custom_field(is_photon, custom_data)
            % noted 230301: below codes are problematic, resulting in <0
            % galvo values coerced to 0
            out = int16(zeros(length(is_photon), 1));
            is_photon = logical(is_photon);
            out(is_photon) = custom_data(is_photon);
            % following line is fixed
            out(~is_photon) = int16(double(custom_data(~is_photon)) - 32768);

            out = out(:)';
        end

        function out = rev_convert_custom_field(is_photon, custom_data)
            % 230821
            out = uint16(zeros(length(is_photon), 1));
            is_photon = logical(is_photon);
            out(is_photon) = custom_data(is_photon);
            out(~is_photon) = uint16(double(custom_data(~is_photon)) + 32768);
            out = out(:)';
        end

    end
end