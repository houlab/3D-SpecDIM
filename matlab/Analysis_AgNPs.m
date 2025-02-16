pname = 'E:\data\spms_track\AgNPs\';
fileList = ["bleb/on_membrane", "bleb/inside_membrane","no_bleb/on_membrane", "no_bleb/inside_membrane"];
dataStruct = struct();

for i = 1:length(fileList)
    dataStruct.(strrep(fileList(i),'/','_')) = readDataFolders(strcat(pname,fileList(i)));
end


%% 速度变化 & 光谱变化 Fig3k
bleb_timepoint = struct();
bleb_timepoint.key_231111_TR020=[186.79,3724.73];
bleb_timepoint.key_231116_TR031=[323.85,	3538.87];
bleb_timepoint.key_231116_TR040=[968.18,	1219.12];
bleb_timepoint.key_231121_TR018=[1473.43,	1879.5];
bleb_timepoint.key_231128_TR004=[88.2,	320.2];
bleb_timepoint.key_231128_TR013=[268.17,	620.85;];
bleb_timepoint.key_231128_TR016=[229.76,	606.31];
bleb_timepoint.key_231128_TR017=[779.57,	1087.58];

%%
close all
fields = fieldnames(dataStruct);

interval = 60000; %The statistics rate is collected at an interval of 60s
xy_velocity = [];
z_velocity = [];
spec_velocity = [];
dis_changes = zeros(8,3);

vel_nums = [];
for i=1:numel(fields)
    fieldName = fields{i}; 
    data = dataStruct.(fieldName);
    fields_ = fieldnames(data);

    if contains(fieldName,"no_bleb")
        continue
    end

    for j = 1:numel(fields_)
        data_ = data.(fields_{j});
        traj_time = length(data_.traj_data)/1000;
        z = data_.traj_data(:,6);
        y = data_.traj_data(:,5);
        x = data_.traj_data(:,4);
        raw_centroids = data_.spec_data.raw_centroid;
        scale_factor = 1;

        if contains(fields_{j}, "key_231111_TR020")
            scale_factor = 10;
        elseif contains(fields_{j}, "key_231116_TR031")
            scale_factor = 3;
        end

        exposure_time = (traj_time * scale_factor)/length(raw_centroids);
        
        bleb_time = bleb_timepoint.(fields_{j});        

        t1 = floor(bleb_time(1)*1000);
        t2 = floor(bleb_time(2)*1000);
        vel_num = 0;

        for k = t1:interval:t2
            t1_ = floor(k / scale_factor);
            t2_ = min(floor((k+interval)/scale_factor), floor(t2/scale_factor));

            vel_xy = (sqrt((x(t2_)-x(t1_))^2 + (y(t2_)-y(t1_))^2))/interval * 60000;
            vel_z = abs(z(t2_)-z(t1_))/interval * 60000;

            % vel_x = (mean(x(t2_-300:t2_-100)) - mean(x(t1_+100:t1_+300)));
            % vel_y = (mean(y(t2_-300:t2_-100)) - mean(y(t1_+100:t1_+300)));
            % vel_xy = (vel_x + vel_y) / 2;
            % vel_z = (mean(z(t2_-300:t2_-100)) - mean(z(t1_+100:t1_+300)));

            startFrame = floor(t1_*scale_factor/exposure_time/1000);
            endFrame = floor(t2_*scale_factor/exposure_time/1000);

            vel_spec = (raw_centroids(endFrame) - raw_centroids(startFrame))/interval * 60000;
    
            xy_velocity = [xy_velocity;vel_xy];
            z_velocity = [z_velocity;vel_z];
            spec_velocity = [spec_velocity;vel_spec];
            vel_num = vel_num + 1;
        end
        vel_nums = [vel_nums;vel_num];
    end
end


xy_mean = mean(xy_velocity);
z_mean = mean(z_velocity);
spec_mean = -mean(spec_velocity(spec_velocity~=0));

xy_err = std(xy_velocity) / sqrt(length(xy_velocity));
z_err = std(z_velocity) / sqrt(length(z_velocity));
spec_err = -std(spec_velocity(spec_velocity~=0)) / sqrt(length(spec_velocity(spec_velocity~=0)));


figure;

yyaxis left;
bar(1, xy_mean, 'FaceColor', [0,0,0]); 
hold on;
bar(2, z_mean, 'FaceColor', [0.5,0.5,0.5]); 
errorbar([1, 2], [xy_mean, z_mean], [xy_err, z_err], 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
ylabel('Coordinate Change');

yyaxis right;
bar(3, spec_mean, 'FaceColor', [0.8,0.8,0.9]);
errorbar(3, spec_mean, spec_err, 'k', 'LineStyle', 'none', 'LineWidth', 1.5); 
ylabel('Spectral Change');

xticks([1 2 3]);
xticklabels({'xy\_velocity', 'z\_velocity', 'spec\_velocity'});

legend({'xy\_velocity', 'z\_velocity', 'spec\_velocity'}, 'Location', 'best');
title('Mean Velocity with Error Bars');
grid on;
hold off;

%%

data = [xy_velocity', z_velocity']'; % 合并为列向量
group = [repmat({'xy'}, 148, 1); repmat({'z'}, 148, 1)]; % 创建分组标签

figure;
violinplot(data, group,'ShowMean', true, 'ShowBox', true, 'ShowData', false);
% ylim([0,2])

ylabel('Moving speed [um/min]');
title('Comparison of D');

data = spec_velocity(spec_velocity~=0); % 合并为列向量
group = [repmat({'spec'}, 148, 1)]; % 创建分组标签

figure;
violinplot(data, group,'ShowMean', true, 'ShowBox', true, 'ShowData', false);

ylabel('Moving speed [um/min]');
title('Comparison of D');


