pname = '/Volumes/shah_ssd/data/spms_track/manuscript/Fig3/AgNPs/';
fileList = ["bleb/on_membrane", "bleb/inside_membrane","no_bleb/on_membrane", "no_bleb/inside_membrane"];
dataStruct = struct();

for i = 1:length(fileList)
    dataStruct.(strrep(fileList(i),'/','_')) = readDataFolders(strcat(pname,fileList(i)));
end

%%  Fig3j
% 
% bleb_time = 0:50:1600;
% bleb_num = [0,1,2,3,4,5,6,7,7,7,8]/8;
% bleb_num = [0,0,1,1,2,3,4,5,6,7,7,7,8];
% cumsum([88.2,186.79,229.76,268.17,323.85,779.57,968.18,1473.43]);

y = [0,1,2,3,4,5,6,7,8] / 8;
x = [0, 88.2,186.79,229.76,268.17,323.85,779.57,968.18,1473.43];

% 定义拟合模型 y = a*(1-exp(-b*x))
fitModel = fittype('a*(1-exp(-b*x))', 'independent', 'x', 'coefficients', {'a', 'b'});

% 初始参数估计
startPoints = [1, 0.05];  % 对 a 和 b 进行初始估计

% 进行拟合
[fitResult, gof] = fit(x', y', fitModel, 'StartPoint', startPoints);

% 显示拟合结果
disp(fitResult);
disp(gof);

% 绘制拟合结果
plot(fitResult, x, y);
xlabel('x');
ylabel('y');
title('Nonlinear Fit: y = a*(1 - exp(-b*x))');
grid on

% 定义方程 y = 0.5，对应的 x 值
target_y = 0.5;
% 定义一个匿名函数，用于求解 y = 0.5 时的 x
f = @(x) fitResult.a * (1 - exp(-fitResult.b * x)) - target_y;
% 使用 fzero 求解 x
x_value_at_y_0_5 = fzero(f, 1); % 1 是初始猜测的 x 值
% 显示结果
disp(['When y = 0.5, the corresponding x value is: ', num2str(x_value_at_y_0_5)]);
%%
figure;
% bar(bleb_time, bleb_num);
% h = histogram('BinEdges', bleb_time, 'BinCounts', bleb_num);
plot(bleb_time,bleb_num, '-o')

xlabel('Time (s)');
ylabel('Probability (%)');

% 设置轴标签
xlabel('Time (s)');
ylabel('Probability (%)');

% 显示网格线
grid on;


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
mean([186.79,323.85,968.18,1473.43,88.2,268.17,229.76,779.57])

cumsum([88.2,186.79,229.76,268.17,323.85,779.57,968.18,1473.43]);
%%

close all
fields = fieldnames(dataStruct);

interval = 60000; %以60s为间隔统计速度
xy_velocity = [];
z_velocity = [];
spec_velocity = [];

for i=1:numel(fields)
    fieldName = fields{i}; % 当前字段名
    data = dataStruct.(fieldName); % 当前字段的值
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

        for k = t1:interval:t2
            t1_ = floor(k / scale_factor);
            t2_ = min(floor((k+interval)/scale_factor), floor(t2/scale_factor));

            vel_xy = (sqrt((x(t2_)-x(t1_))^2 + (y(t2_)-y(t1_))^2))/interval * 60000;
            vel_z = abs(z(t2_)-z(t1_))/interval * 60000;

            startFrame = floor(t1_*scale_factor/exposure_time/1000);
            endFrame = floor(t2_*scale_factor/exposure_time/1000);

            vel_spec = (raw_centroids(endFrame) - raw_centroids(startFrame))/interval * 60000;
    
            xy_velocity = [xy_velocity;vel_xy];
            z_velocity = [z_velocity;vel_z];
            spec_velocity = [spec_velocity;vel_spec];
        end
    end
end
spec_velocity(spec_velocity==0) = [];

% 计算平均值
xy_mean = mean(xy_velocity);
z_mean = mean(z_velocity);
spec_mean = -mean(spec_velocity);

% 计算标准误差
xy_err = std(xy_velocity) / sqrt(length(xy_velocity));
z_err = std(z_velocity) / sqrt(length(z_velocity));
spec_err = -std(spec_velocity) / sqrt(length(spec_velocity));

% 创建一个带有双纵坐标的图形
figure;

% 左侧纵坐标（xy_velocity 和 z_velocity）
yyaxis left;
bar(1, xy_mean, 'FaceColor', [0,0,0]); % 绘制 xy_velocity 的 bar
hold on;
bar(2, z_mean, 'FaceColor', [0.5,0.5,0.5]); % 绘制 z_velocity 的 bar
errorbar([1, 2], [xy_mean, z_mean], [xy_err, z_err], 'k', 'LineStyle', 'none', 'LineWidth', 1.5); % 添加误差条
ylabel('Coordinate Change');

% 右侧纵坐标（spec_velocity）
yyaxis right;
bar(3, spec_mean, 'FaceColor', [0.8,0.8,0.9]); % 绘制 spec_velocity 的 bar
errorbar(3, spec_mean, spec_err, 'k', 'LineStyle', 'none', 'LineWidth', 1.5); % 添加误差条
ylabel('Spectral Change');

% 设置X轴标签和刻度
xticks([1 2 3]);
xticklabels({'xy\_velocity', 'z\_velocity', 'spec\_velocity'});

% 添加图例
legend({'xy\_velocity', 'z\_velocity', 'spec\_velocity'}, 'Location', 'best');

% 设置标题
title('Mean Velocity with Error Bars');

% 显示网格线
grid on;
hold off;



%%  Supplementary Tables 4 and eFig9d
close all
Traj_nums = zeros(2,2);
total_times_bleb = [];
total_times_nobleb = [];
x_changes_bleb = [];
x_changes_nobleb = [];
y_changes_bleb = [];
y_changes_nobleb = [];
z_changes_bleb = [];
z_changes_nobleb = [];

cent_changes = zeros(2,2);
cent_var = zeros(2,2);

fields = fieldnames(dataStruct);

for i=1:numel(fields)
    fieldName = fields{i}; % 当前字段名
    data = dataStruct.(fieldName); % 当前字段的值

    fields_ = fieldnames(data);
    Traj_nums(ceil(i/2),2-mod(i,2)) = numel(fields_);

    tmp_cent = [];

    for j = 1:numel(fields_)
        data_ = data.(fields_{j});
        traj_time = length(data_.traj_data)/1000;
        raw_centroids = data_.spec_data.raw_centroid;

        scale_factor = 1;

        if contains(fields_{j}, "key_231111_TR020")
            scale_factor = 10;
        elseif contains(fields_{j}, "key_231116_TR031")
            scale_factor = 3;
        end

        exposure_time = (traj_time * scale_factor)/length(raw_centroids);
        

        if(~contains(fieldName,'no_bleb'))
            bleb_time = bleb_timepoint.(fields_{j});     
            t1 = floor(bleb_time(1)*1000);
            t2 = floor(bleb_time(2)*1000);
            
            startFrame = floor(t1/exposure_time/1000);
            endFrame = floor(t2/exposure_time/1000);
    
            raw_centroids = raw_centroids(startFrame:endFrame);
        else
            raw_centroids = raw_centroids(1:end);
        end



        idx = find(diff(raw_centroids) ~= 0);
        filteredIndices = [1, (idx + 1)', length(raw_centroids)];
    
        raw_centroids = raw_centroids(filteredIndices);

        if length(raw_centroids) > 100
            cent_changes_ = mean(raw_centroids(end-50:end-30)) - mean(raw_centroids(30:50));
            tmp_cent = [tmp_cent cent_changes_];
        end


    end

    cent_changes(ceil(i/2),2-mod(i,2)) = mean(tmp_cent);
    cent_var(ceil(i/2),2-mod(i,2)) = std(tmp_cent)/sqrt(length(tmp_cent));

end

%%%%%%%%%%%%%% eFig9d
hNum = figure;
groupLabels = {'Blebbing', 'No Blebbing'};
% 绘制分组条形图
hb = bar(Traj_nums, 'FaceColor', 'flat');

% 设置灰度颜色
colors = [0.53 0.71 0.82; 0.95 0.79 0.53]; % 灰度颜色数组，每行一个颜色
for k = 1:size(Traj_nums, 2)
    hb(k).CData = colors(k,:);
end

% 设置图形的其他属性以匹配 Origin 风格
set(gca, 'Box', 'off', 'FontSize', 10, 'FontName', 'Arial'); % 关闭图框，设置字体大小和类型
xlabel('Number of tracks', 'FontSize', 12);
ylabel('Counts', 'FontSize', 12);
set(gca, 'XTick', 1:size(Traj_nums, 2), 'XTickLabel', groupLabels);
% 设置背景色为白色
set(hNum, 'Color', 'w');
grid on; % 添加网格线，根据个人偏好选择是否启用

% 调整坐标轴范围以优化显示
ylim([0, floor(max(max(Traj_nums)) * 1.4)]);
figureandaxiscolors('w','k','')
legend({'On membrane', 'in mebrane'}, 'Location', 'NorthEast', 'Box', 'off','Color','k'); % 添加图例，关闭图例边框


%% 231116 TR031 光谱随距离的变化
bleb_time = bleb_timepoint.key_231128_TR013;       
scale_factor = 1;
t1 = floor(bleb_time(1)*1000/scale_factor);
t2 = floor(bleb_time(2)*1000/scale_factor);

data = dataStruct.bleb_on_membrane;
data_ = data.key_231128_TR013;
traj_time = length(data_.traj_data)/1000;
z = data_.traj_data(:,6);
y = data_.traj_data(:,5);
x = data_.traj_data(:,4);
raw_centroids = data_.spec_data.raw_centroid;

dis = [x(t1:t2),y(t1:t2),z(t1:t2)];
dis = sqrt(sum((dis - dis(1,:)).^2, 2)); %计算三维距离

exposure_time = (traj_time * scale_factor)/length(raw_centroids);
startFrame = floor(t1 * scale_factor/exposure_time/1000);
endFrame = floor(t2 * scale_factor/exposure_time/1000);

slice_inter = endFrame-startFrame+1;
nGroups = floor(length(dis) / slice_inter); 
fullDataLength = nGroups * 1000;
dis = reshape(dis(1:floor(length(dis) / slice_inter)*slice_inter), nGroups, []);
dis = mean(dis, 1);

[raw_centroids_unique, idx] = unique(raw_centroids(startFrame:endFrame), 'stable');

figure
scatter(dis(idx),raw_centroids_unique);
xlabel("distance [um]")
ylabel("centroids changes[nm]")
grid on

%% 起泡随时间变化图
traj_acc = [];
traj_bleb_num = [];

for i=1:numel(fields)
    fieldName = fields{i}; % 当前字段名
    data = dataStruct.(fieldName); % 当前字段的值
    fields_ = fieldnames(data);

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
        traj_time = traj_time * scale_factor;

        if ~contains(fieldName,"no_bleb")
            bleb_time = bleb_timepoint.(fields_{j});     
            traj_bleb_num = [traj_bleb_num; 1];
            t1 = floor(bleb_time(1)*1000);
            t2 = floor(bleb_time(2)*1000); 
        else
            traj_bleb_num = [traj_bleb_num; 0];
        end
        traj_acc = [traj_acc; traj_time];


        
    end
end


%%
bleb_time = 500:500:4500;
bleb_num = [1,3,6,7,7,7,7,7,9]./24;
figure;
bar(bleb_time, bleb_num);
xlabel('Time (s)');
ylabel('Probability (%)');



