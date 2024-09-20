%%
addpath(genpath('/Users/shahao/projects/spms_track/manuscript/code'))

%% x,y,z精度拟合 FigS1

z=trackDataFilt(10000:20000,6);
y=trackDataFilt(10000:20000,5);
x=trackDataFilt(10000:20000,4);
time=(1:length(x(:,1)))/1000;

% 设置滑动窗口的大小和步长
window_size = 1000;
step_size = 10;

% 计算滑动窗口的数量
num_windows = floor((length(x) - window_size) / step_size) + 1;

% 初始化存储标准差的数组
std_errors_x = zeros(num_windows, 1);
std_errors_y = zeros(num_windows, 1);
std_errors_z = zeros(num_windows, 1);

% 进行滑动窗口标准差计算
for i = 1:num_windows
    % 定位当前窗口的起始和结束索引
    start_idx = (i-1) * step_size + 1;
    end_idx = start_idx + window_size - 1;
    
    % 计算标准差并存储
    std_errors_x(i) = std(x(start_idx:end_idx))*1000;
    std_errors_y(i) = std(y(start_idx:end_idx))*1000;
    std_errors_z(i) = std(z(start_idx:end_idx))*1000;
end


hfig = figure;
pos1 = [0.1, 0.7, 0.74, 0.25]; % [left bottom width height]
pos3 = [0.1, 0.4, 0.74, 0.25]; % [left bottom width height]
pos5 = [0.1, 0.1, 0.74, 0.25]; % [left bottom width height]
%%%%%%%%%%%%%%%
ax1 = axes('Position', pos1);
plot(ax1, time(window_size:step_size:end), std_errors_x, 'Color',"#0072BD", 'LineWidth',1); % 假设数据在 dataVector 变量中
ylabel('X Precision [nm]');

grid on
%%%%%%%%%%%%
ax3 = axes('Position', pos3);
plot(ax3, time(window_size:step_size:end), std_errors_y, 'Color',"#0072BD", 'LineWidth',1); % 假设数据在 dataVector 变量中
ylabel('Y Precision [nm]');

grid on
%%%%%%%%%%%%%%%%%%%%%%
ax5 = axes('Position', pos5);

plot(ax5, time(window_size:step_size:end), std_errors_z, 'Color',"#0072BD", 'LineWidth',1); % 假设数据在 dataVector 变量中
ylabel('Z Precision [nm]');
xlabel('Time [sec]')
grid on

%% 光谱精度 Fig2d
pname = '/Volumes/shah_ssd/data/spms_track/manuscript/Fig2/spec_resolution/';
fileList = ["100ms"];
dataStruct = struct();

for i = 1:length(fileList)
    dataStruct.(strcat("exposure_",fileList(i))) = readDataFolders(strcat(pname,fileList(i)));
end

fields = fieldnames(dataStruct);

figure;
hold on;

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"];
k=1;
for i=1:numel(fields)
    cd (pname);
    fieldName = fields{i}; % 当前字段名
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName); % 当前字段的值
    cd (fileList(k))
    k=k+1;
    vit_data = readtable("MDD_ViT-learned_outputs.csv");

    fields_ = fieldnames(data);
    intensity = [];
    spec_res = [];
    vit_res = [];

    for j = 1:numel(fields_)
        data_ = data.(fields_{j});
        intensity = [intensity; mean(data_.traj_data(:,11))];
        spec_res = [spec_res; std(data_.spec_data.raw_centroid)];
        
        tiff_name = fields_{j};
        vit_preds = vit_data.preds(vit_data.t_labels==strcat(tiff_name(5:end),".tiff"));
        vit_res = [vit_res; std(vit_preds)];
    end
    f = fit(intensity, spec_res, 'power2');
    scatter(intensity/1000, spec_res, 100, 'o', 'MarkerFaceColor', '#4B7BB3', ...
        'MarkerEdgeColor', '#4B7BB3', 'HandleVisibility', 'off'); % 使用蓝色圆形标记mVenus
    plot(50:1:800, f(50000:1000:800000), 'Color', '#4B7BB3', 'LineWidth', 3.5, 'DisplayName', 'Norm');
    
    scatter(intensity/1000, vit_res, 100, '^', 'MarkerFaceColor', '#F4BA1D', ...
        'MarkerEdgeColor', '#F4BA1D', 'HandleVisibility', 'off'); % 使用金色三角形标记mGold
    f = fit(intensity, vit_res, 'power2');
    plot(50:1:800, f(50000:1000:800000), 'Color', '#F4BA1D', 'LineWidth', 3.5, 'DisplayName', 'ViT');
end


legend
grid on

%% 光谱时间分辨率 Fig2e
pname = '/Volumes/shah_ssd/data/spms_track/manuscript/Fig2/time_resolution/';
fileList = ["100us", "1ms", "5ms", "10ms"];

dataStruct = struct();

for i = 1:length(fileList)
    dataStruct.(strcat("exposure_",fileList(i))) = readDataFolders(strcat(pname,fileList(i)));
end


fields = fieldnames(dataStruct);
figure;
hold on;

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7E2F8E"];

exposure_time = [0.1, 1, 5, 10]';
spec_res = [];
vit_res = [];
intensity = [];

for i=1:numel(fields)
    cd (pname);
    fieldName = fields{i}; % 当前字段名
    
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName); % 当前字段的值
    fields_ = fieldnames(data);
    data_ = data.(fields_{1});

    intensity = [intensity; mean(data_.traj_data(:,11))];
    spec_res = [spec_res; std(data_.spec_data.raw_centroid)];
end

f = fit(exposure_time, spec_res, 'power2');
scatter(exposure_time, spec_res, 100, 'o', 'MarkerFaceColor', colors(i), ...
    'MarkerEdgeColor', colors(i), 'HandleVisibility', 'off'); % 使用蓝色圆形标记mVenus
plot(0.1:0.1:10, f(0.1:0.1:10), 'Color', colors(i), 'LineWidth', 1.5, 'DisplayName', fileList(i));
%     scatter(intensity/1000, vit_res, 50, '^', 'MarkerFaceColor', colors(i), ...
%         'MarkerEdgeColor', colors(i), 'HandleVisibility', 'off'); % 使用金色三角形标记mGold
%     f = fit(intensity, vit_res, 'power2');
%     plot(5:0.1:35, f(5000:100:35000), 'Color', '#F4BA1D', 'LineWidth', 1.5, 'DisplayName', 'ViT');
legend
grid on
axis tight
xlim([0,10])
ylim([0,60])
truncAxis('Y', [5 45]);


%% 光谱灵敏度 Fig2f
pname = '/Volumes/shah_ssd/data/spms_track/manuscript/Fig2/sensitive/';
fileList = ["500ms", "200ms", "100ms", "50ms"];

dataStruct = struct();

for i = 1:length(fileList)
    dataStruct.(strcat("exposure_",fileList(i))) = readDataFolders(strcat(pname,fileList(i)));
end

fields = fieldnames(dataStruct);
figure;
hold on;

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7E2F8E"];

k=1;
for i=1:numel(fields)
    cd (pname);
    fieldName = fields{i}; % 当前字段名
    
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName); % 当前字段的值
    cd (fileList(k))
    k=k+1;
    vit_data = readtable("MDD_ViT-learned_outputs.csv");

    fields_ = fieldnames(data);
    intensity = [];
    spec_res = [];
    vit_res = [];

    for j = 1:numel(fields_)
        data_ = data.(fields_{j});
        intensity = [intensity; mean(data_.traj_data(:,11))];
        spec_res = [spec_res; std(data_.spec_data.raw_centroid)];
        
        tiff_name = fields_{j};
        vit_preds = vit_data.preds(vit_data.t_labels==strcat(tiff_name(5:end),".tiff"));
        vit_res = [vit_res; std(vit_preds)];
    end
    f = fit(intensity, spec_res, 'power2');
    scatter(intensity/1000*9/6, spec_res, 100, 'o', 'MarkerFaceColor', colors(i), ...
        'MarkerEdgeColor', colors(i), 'HandleVisibility', 'off'); % 使用蓝色圆形标记mVenus
    plot([5:0.1:35]*9/6, f(5000:100:35000), 'Color', colors(i), 'LineWidth', 1.5, 'DisplayName', fileList(i));
    
%     scatter(intensity/1000, vit_res, 50, '^', 'MarkerFaceColor', colors(i), ...
%         'MarkerEdgeColor', colors(i), 'HandleVisibility', 'off'); % 使用金色三角形标记mGold
%     f = fit(intensity, vit_res, 'power2');
%     plot(5:0.1:35, f(5000:100:35000), 'Color', '#F4BA1D', 'LineWidth', 1.5, 'DisplayName', 'ViT');
end


legend
grid on
axis tight

%% 计算所有tr-Halo-mGold曲线的酸化时间及扩散系数 Fig3c
pname = '/Volumes/shah_ssd/data/spms_track/manuscript/Fig3/mito/';
fileList = ["tr_Halo_mGold", "tr_Halo_mGold_2"];

fnames = {};
for j = 1:length(fileList)
    
    folderPath = strcat(pname,fileList(j));
    files = dir(folderPath);
    files = files(~[files.isdir]);
    for i = 1:length(files)
        fileName = files(i).name;
        if contains(fileName, '.tdms') && ~contains(fileName, 'SM') &&  ~contains(fileName, 'IM') 
            % 将符合条件的文件名添加到数组中
            fnames{end+1} = strcat(folderPath,'/',fileName); % 将符合条件的文件名添加到selectedFiles
        end
    end
end

total_autophagy_time = [];
autophagy_names = [];
Diff = [];
Diff_norm = [];

for j = 1:length(fnames)
    [pname, fname, ~] = fileparts(fnames{j});
    fname = strcat(fname,'.tdms');
    sname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 SM$2 TR$2$3');  

    [~,dirname,~]=fileparts(fname);
    cd(fullfile(pname,dirname));  
    load(strcat(dirname,' ph.mat'));
    load(strcat(dirname,'.mat'));
    exposureTime = extrack_exposureTime('./');

    z=trackDataFilt(:,6);
    y=trackDataFilt(:,5);
    x=trackDataFilt(:,4);

    total_autophagy_time = [total_autophagy_time, autophagy_time];
    if autophagy_time ~= 0
        [selectedStarts, selectedEnds] = selectNormPoints(floor(startPoints*exposureTime), ...
            floor(endPoints*exposureTime),length(x),length(autophagy_time));
    end

    for k = 1:length(autophagy_time)
        autophagy_names = [autophagy_names,fname];
        if autophagy_time(k) == 0
            continue
        else
            if startPoints(k) == 1
                total_autophagy_time(end) = 0;
                continue
            end
            t1 = floor(startPoints(k)*exposureTime);
            t2 = floor(endPoints(k)*exposureTime);
            t1_norm = selectedStarts(k);
            t2_norm = selectedEnds(k);

            [~,msd,D]=msdcalc(x(t1:t2),y(t1:t2),z(t1:t2),1000);
            [~,msd_norm,D_norm]=msdcalc(x(t1_norm:t2_norm),y(t1_norm:t2_norm),z(t1_norm:t2_norm),1000);
            Diff = [Diff, D];
            Diff_norm = [Diff_norm, D_norm];
        end
    end
        disp(fname + ' completed!')
end
autophagy_names(total_autophagy_time==0) = [];
total_autophagy_time(total_autophagy_time==0)=[];

data =  total_autophagy_time;

mu = mean(total_autophagy_time);
sigma = std(total_autophagy_time);

figure;
histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.5]); % 使用灰度色彩
hold on;

x = linspace(0, max(data), 100);
pdf = normpdf(x, mu, sigma);
plot(x, pdf, 'LineWidth', 2, 'Color', 'k'); % 黑色线条表示正态分布拟合

% 标注均值和方差
text(mu*1.5, max(pdf), sprintf('Mean = %.2f ± %.2f s', mu, sigma), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'k');

% 设置图形的其它属性
xlabel('Data');
ylabel('Probability Density');
title(sprintf('Autophagy Time (N=%d)', length(data)));
grid on;
hold off;

%%%%%%%%%%%
% 设置背景色为白色
set(gcf, 'Color', 'w');

barData = [mean(Diff);mean(Diff_norm)];
errData = [std(Diff)/sqrt(length(Diff)); std(Diff_norm)/sqrt(length(Diff_norm))];
fig = figure;
barHandle = bar(barData, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);

% 设置条形图的颜色为灰度
barHandle.CData(1,:) = [0.5 0.5 0.5]; % 第一个条形的颜色
barHandle.CData(2,:) = [0 0 0]; % 第二个条形的颜色

hold on;

% 添加误差线
% 计算条形图的中心位置用于定位误差线
numBars = numel(barData);
for i = 1:numBars
    errorbar(i, barData(i),0, errData(i), 'k', 'LineWidth', 1.5);
    x = i;
    y = barData(i) + max(barData) * 0.1; % 确保文本显示在条形上方
    text(x, y, sprintf('%.2f', barData(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

hold off;

% 设置图表的其他属性
set(gca, 'XTick', 1:length(barData), 'XTickLabel', {'Autophagy', 'Norm'},'Box', 'off', 'Color', 'none');
ylabel('D [um^2/s]');
title('Comparison of D');

% 设置背景色为白色
set(fig, 'Color', 'w');
grid on; % 添加网格线，根据个人偏好选择是否启用

% 调整坐标轴范围以优化显示
ylim([0, max(barData) * 1.2]);
figureandaxiscolors('w','k','')




%% eFig5
% 定义时间点和波长范围
t = linspace(1, 100, 100);
% lambda = linspace(500, 700, 1000);

% 定义两个高斯峰的中心和宽度
peak1_center = 531;
peak2_center = 571;
width1 = 17;   % 峰的宽度
width2 = 23;   % 峰的宽度

% peak1_center = 664;
% peak2_center = 680;
% width1 = 6;   % 峰的宽度
% width2 = 10;   % 峰的宽度

% 高斯函数定义
gauss = @(x, mu, sigma, amplitude) amplitude * exp(-((x - mu).^2) / (2 * sigma^2));

mGold_spec = normalize(mGoldHaloSpec.mGold,'range')';
JF549_spec = normalize(movmean(mGoldHaloSpec.JF549', 40),'range')';
lambda = mGoldHaloSpec.spec';

% 初始峰值强度
initial_amplitude1 = 10;
initial_amplitude2 = 1;

% 指数衰减函数，随时间 t 减小幅度
decay = @(t, tau, initial) initial * exp(-t / tau);

% 存储两个峰的数据
spectrum_over_time = zeros(length(t), length(lambda));

gt_int = zeros(length(t),1);
apd_int = zeros(length(t),1);
apd2_int = zeros(length(t),1);
spec_int = zeros(length(t),1);

% filter1_center = 525; % 653/47 nm
% filter1_width = 50;
% filter2_center = 600; % 684/40 nm
% filter2_width = 52;

filter1_center = 662; % 653/47 nm
filter1_width = 11;
filter2_center = 680; % 684/40 nm
filter2_width = 22;

% 滤光片函数
filter_func = @(x, center, width) exp(-((x-center).^2)/(2*(width/2.355)^2));

donor_int = zeros(length(t),1);
acceptor_int = zeros(length(t),1);
% 模拟随时间变化的光谱
for i = 1:length(t) 
    % 计算当前时间点的幅度
    current_amplitude1 = decay(t(i), 20, initial_amplitude1);  % 较快衰减
    current_amplitude2 = decay(t(i), 1e5, initial_amplitude2);  % 较慢衰减
    gt_int(i) = current_amplitude1 / current_amplitude2;

    noise1 = 0.1 * current_amplitude1 * randn(size(lambda));
    noise2 = 0.1 * current_amplitude2 * randn(size(lambda));
    % noise1 = 0;
    % noise2 = 0;
    
    % 合成光谱
%     spectrum_over_time(i, :) = gauss(lambda, peak1_center, width1, current_amplitude1) + ...
%                                gauss(lambda, peak2_center, width2, current_amplitude2) + ...
%                                noise1 + noise2;

    spectrum_over_time(i, :) = mGold_spec .* current_amplitude1 + JF549_spec .* current_amplitude2 + ...
                           noise1 + noise2;
                           
    % 通过滤光片的光谱
    filtered1 = spectrum_over_time(i, :) .* filter_func(lambda, filter1_center, filter1_width);
    filtered2 = spectrum_over_time(i, :) .* filter_func(lambda, filter2_center, filter2_width);

    % 累计光强
    sum_filtered1 = sum(filtered1);
    sum_filtered2 = sum(filtered2);
    apd_int(i) = sum_filtered1 / sum_filtered2;
    
    
    % 光谱解混方法
    % 构造光谱矩阵和混叠光谱向量
%     a1 = gauss(lambda, peak1_center, width1, 1);
%     a2 = gauss(lambda, peak2_center, width2, 1);
    a1 = mGold_spec;
    a2 = JF549_spec;
    M = [a1; a2]';
    b_vec = spectrum_over_time(i, :)';

    % 使用线性代数方法解丰度系数
    x = M \ b_vec;  % 解线性方程 Mx = b
    spec_int(i) = x(1) / x(2);
    
    donor_int(i) = current_amplitude1/current_amplitude2;
    % acceptor_int(i) = sum_filtered2;
                      
end

figure;


% spectrum_merge = gauss(lambda, peak1_center, width1, 1) + ...
% gauss(lambda, peak2_center, width2, 1);
spectrum_merge = mGold_spec + JF549_spec;
plot(lambda, spectrum_merge, 'r-');  % 初始光谱
hold on
% area(lambda, gauss(lambda, peak1_center, width1, 1), 'FaceColor', 'r', 'FaceAlpha', 0.5);
% area(lambda, gauss(lambda, peak2_center, width2, 1), 'FaceColor', 'b', 'FaceAlpha', 0.5);

area(lambda, mGold_spec, 'FaceColor', 'r', 'FaceAlpha', 0.5);
area(lambda, JF549_spec, 'FaceColor', 'b', 'FaceAlpha', 0.5);
xlabel('Wavelength (nm)');
ylabel('Intensity');
title('Synthetic Spectrum with Emission Peaks');
legend('Spectrum', '664 nm Peak', '680 nm Peak');
grid on

% 绘制特定时间点的光谱
figure;
plot(lambda, spectrum_over_time(1, :), 'r-', 'DisplayName', 't = 1');  % 初始光谱
hold on;
plot(lambda, spectrum_over_time(50, :), 'g-', 'DisplayName', 't = 50'); % 中间时间点光谱
plot(lambda, spectrum_over_time(100, :), 'b-', 'DisplayName', 't = 100'); % 最后时间点光谱

% 计算带通滤光片的起始和结束波长
start1 = filter1_center - filter1_width/2;
end1 = filter1_center + filter1_width/2;
start2 = filter2_center - filter2_width/2;
end2 = filter2_center + filter2_width/2;

% 计算两个滤光片之间的区域
x_fill = [start1, end1, end1, start1];
y_fill = [0, 0, max(max(spectrum_over_time)), max(max(spectrum_over_time))]; % Y轴范围从0到最大强度
% 添加矩形区域
fill(x_fill, y_fill, 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % 黑色填充，50%透明度

x_fill = [start2, end2, end2, start2];
y_fill = [0, 0, max(max(spectrum_over_time)), max(max(spectrum_over_time))]; % Y轴范围从0到最大强度
fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % 黑色填充，50%透明度

xlabel('Wavelength (nm)');
ylabel('Intensity');
title('Spectral Evolution Over Time');
legend show;
grid on;

% 可视化所有时间点的光谱
figure;
imagesc(t, lambda, spectrum_over_time');
xlabel('Time');
ylabel('Wavelength (nm)'); 
title('Spectral Intensity Over Time');
colorbar;
axis xy;

figure
plot(donor_int, gt_int, 'r', 'LineWidth',1);
hold on 
plot(donor_int, apd_int, 'g');
plot(donor_int, spec_int, 'b');
grid on
set(gca, 'XScale', 'log'); % 设置 Y 轴为对数刻度
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度
legend("gt", "filter", "unmix","3nm-filter")




%% 极性表征 eFig8
%cyclohexane toluene 1,4-dioxane (ethyl acetate) acetone ethanol methanol 
% acetonitrile DMF (DMSO buﬀer) 
lambda_ex = [528 571 581 593 615 636 640 621 618 629];
ET_30 = [30.9 33.9 36.0 38.1 42.2 51.9 55.4 45.6 43.2 45.1];

% DOPC DOPC/Chol SM/Chol
SLBs = [667 641 635];

p = polyfit(ET_30, lambda_ex, 1); % 一次多项式拟合
y_fit = polyval(p, ET_30);

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"];

labels = {'DOPC', 'SM/Chol', 'DOPC/Chol'};
x_coords = (SLBs - p(2)) / p(1);

figure;
hold on;
scatter(ET_30, lambda_ex, 50, 'square', 'filled', 'MarkerFaceColor','k', ...
    'HandleVisibility', 'off');
plot(ET_30, y_fit, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');

% 在拟合曲线上显示指定的坐标并添
for i = 1:length(x_coords)
    scatter(x_coords(i), SLBs(i), 70, 'diamond', ...
        'filled', 'MarkerFaceColor', colors(i), 'DisplayName', labels{i});
end

xlabel('E_T(30)');
ylabel('NR Centriods [nm]');

figureandaxiscolors('w','k','')
legend('show');
grid on;

%% mGold 光谱分析 FigS3
x = mGoldHaloSpec.spec;
y_data = {mean(mGoldHaloSpec.pH3,2), mean(mGoldHaloSpec.pH4,2), mean(mGoldHaloSpec.pH5,2), mean(mGoldHaloSpec.pH6,2), ...
    mean(mGoldHaloSpec.pH7,2), mean(mGoldHaloSpec.pH8,2), mean(mGoldHaloSpec.pH9,2), mean(mGoldHaloSpec.pH10,2)};

x_561 = mGoldHaloSpec561.spec;
y_data_561 = {mean(mGoldHaloSpec561.pH3,2), mean(mGoldHaloSpec561.pH4,2), mean(mGoldHaloSpec561.pH5,2), mean(mGoldHaloSpec561.pH6,2), ...
    mean(mGoldHaloSpec561.pH7,2), mean(mGoldHaloSpec561.pH8,2), mean(mGoldHaloSpec561.pH9,2), mean(mGoldHaloSpec561.pH10,2)};

% pH_Range = ["pH 3", "pH 4", "pH 5", "pH 6", "pH 7", "pH 8", "pH 9", "pH 10"];

% 归一化函数
maxValue = max([y_data{:}], [], 'all');
minValue = min([y_data{:}], [], 'all');
normalize = @(y) (y - minValue) / (maxValue - minValue);

maxValue = max([y_data_561{:}], [], 'all');
minValue = min([y_data_561{:}], [], 'all');
normalize_561 = @(y) (y - minValue) / (maxValue - minValue);

fit_curve_combined = zeros(2401,8);

% 对每组数据进行高斯拟合，并绘制叠加曲线
spec = 500:0.1:740;

for i = 1:8
    % 高斯拟合
    y_norm = normalize(y_data{i});
    y_norm_561 = normalize_561(y_data_561{i});
    fit1 = fit(x(1:13,1), y_norm(1:13,1), 'gauss1'); % 双高斯拟合
    fit2 = fit(x_561, y_norm_561, 'gauss1'); % 双高斯拟合

    % 生成拟合曲线
    fit_curve1 = feval(fit1, spec);
    fit_curve2 = feval(fit2, spec);

    % 叠加两组数据
    fit_curve_combined(:,i) = fit_curve1 + fit_curve2;
    % fit_curve_combined(:,i) = fit_curve2;
end

% 创建figure
figure
hold on;
maxValue = max(fit_curve_combined, [], 'all');
minValue = min(fit_curve_combined, [], 'all');

% 生成渐变颜色
colors = cool(8); % 使用jet渐变色
int_mGold = [];
int_Halo = [];
pH_Range = ["pH 3", "pH 4", "pH 5", "pH 6", "pH 7", "pH 8", "pH 9", "pH 10"];

for i = 1:8
    % 绘制叠加曲线
    fit_curve_norm = (fit_curve_combined(:,i) - minValue) / (maxValue - minValue);
    plot(spec, fit_curve_norm, 'Color', colors(i, :), ...
        'DisplayName', pH_Range(i),'LineWidth',2);

    int_mGold = [int_mGold;max(fit_curve_norm(340:360))];
    int_Halo = [int_Halo;max(fit_curve_norm(750:850))];
end

hold off;
figureandaxiscolors('w','k','')
legend('show');
xlabel('Wavelength [nm]');
ylabel('Intensity');
axis tight
grid on;

% 绘制图像
figure;
hold on;

pH_values = 3:1:10;
scatter(pH_values, int_Halo, 100, 'o', 'MarkerFaceColor', '#4B7BB3', ...
        'DisplayName', 'HaloTag'); % 使用蓝色圆形标记mVenus
scatter(pH_values, int_mGold, 100, '^', 'MarkerFaceColor', '#F4BA1D', ...
        'DisplayName', 'mGold'); % 使用金色三角形标记mGold

% 拟合曲线

% 自定义sigmoid函数
sigmoid = @(b,x) b(1) ./ (1 + exp(-b(2)*(x-b(3))));
% 初始参数猜测
initialGuess = [max(int_mGold), 1, mean(pH_values)]; % 最大值，增长率，中点
% 进行拟合
opts = optimset('Display', 'off'); % 关闭拟合过程中的输出
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(pH_values', int_mGold, sigmoid, initialGuess, opts);
% 生成拟合数据
fitValuesGold = sigmoid(beta, 3:0.1:10);

% 初始参数猜测
initialGuess = [max(int_Halo), 1, mean(pH_values)]; % 最大值，增长率，中点
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(pH_values', int_Halo, sigmoid, initialGuess, opts);
fitValuesHalo = sigmoid(beta, 3:0.1:10);

% pHalo = polyfit(pH_values, int_Halo, 3); % 使用3次多项式拟合mVenus数据
% pGold = polyfit(pH_values, int_mGold, 3); % 使用3次多项式拟合mGold数据
% fitValuesVenus = polyval(pHalo, 3:0.1:10);
% fitValuesGold = polyval(pGold, 3:0.1:10);
plot(3:0.1:10, fitValuesHalo, 'Color', '#4B7BB3', 'LineWidth', 3.5, 'HandleVisibility', 'off'); % 绘制mVenus拟合曲线
plot(3:0.1:10, fitValuesGold, 'Color', '#F4BA1D', 'LineWidth', 3.5, 'HandleVisibility', 'off'); % 绘制mGold拟合曲线

% 设置图例、轴标签和标题
xlabel('pH');
ylabel('Normalized Fluorescence');
hold off;
figureandaxiscolors('w','k','')
legend('show');
axis tight
grid on;

%% 绘制profile line FigS6 a-d
pname = "/Users/shahao/projects/spms_track/manuscript/初稿/figures/efig3/";
fnames = ["mGold1.tif","Halo.tif", "mGold-Halo-merge.tif"];

hFig = figure;

pos1 = [0.1, 0.55, 0.4, 0.4]; % [left bottom width height]
pos2 = [0.55, 0.55, 0.4, 0.4];
pos3 = [0.1, 0.1, 0.4, 0.4]; % [left bottom width height]
pos4 = [0.55, 0.1, 0.4, 0.4];

ax1 = axes('Position', pos1);
img_1 = imread(strcat(pname,fnames(1)));
img_1 = img_1(:,:,1:3);
imshow(img_1);
img_1 = rgb2gray(img_1);
title('Select two points for the profile line');
[x, y] = ginput(2);
x = round(x);
y = round(y);

hold on;
plot(x, y, 'w-', 'LineWidth', 2);
int_profile_1 = improfile(img_1, x, y);

ax2 = axes('Position', pos2);
img_2 = imread(strcat(pname,fnames(2)));
img_2 = img_2(:,:,1:3);
imshow(img_2);
hold on
img_2 = rgb2gray(img_2);
plot(x, y, 'w-', 'LineWidth', 2);
int_profile_2 = improfile(img_2, x, y);

ax3 = axes('Position', pos3);
img_3 = imread(strcat(pname,fnames(3)));
img_3 = img_3(:,:,1:3);
imshow(img_3);
hold on
plot(x, y, 'w-', 'LineWidth', 2);

ax4 = axes('Position', pos4);
plot(ax4,normalize(int_profile_1, 'range'),'r','LineWidth',1)
hold on
plot(ax4,normalize(int_profile_2, 'range'),'Color',"#0072BD",'LineWidth',1)
ylabel('Normalized Intensity')
xlabel('Distance (pixels)')
figureandaxiscolors('w','k','')


%% 联合标定明场和轨迹数据 FigS7
close all
plot_traj(trackDataFilt, '');

z=trackDataFilt(:,6);
y=trackDataFilt(:,5);
x=trackDataFilt(:,4);

centerEMCCD = [63,161];
posImg = TR001_yp_xp(300:506,115:321);
plot_traj_emccd(y,x,z,' ', posImg,centerEMCCD)

%% 显示光谱曲线 FigS8
close all
hFig = figure;

pos1 = [0.2, 0.8, 0.65, 0.1]; % [left bottom width height]
pos2 = [0.2, 0.1, 0.65, 0.65];

ax1 = axes('Position', pos1);
img = spec_data.trackImg(:,:,1755);
imagesc(img);
colormap hot;
set(ax1, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);

int_profile_1 = spec_data.trackCurve(1755,:);

ax2 = axes('Position', pos2);
plot(int_profile_1,'r','LineWidth',1)
hold on
ylabel('Normalized Intensity')
xlabel('Distance (pixels)')
figureandaxiscolors('w','k','')


%% 光谱分段显示 lambda-time-intensity
% 假设你有波长、时间和强度数据的矩阵

scale_factor=3;
traj_time = length(trackDataFilt)/1000;
exposure_time = (traj_time * scale_factor)/length(spec_data.raw_centroid);

t1_ = 500 * 1000;
t2_ = 2000 * 1000;

startFrame = floor(t1_/exposure_time/1000);
endFrame = floor(t2_/exposure_time/1000);

spec_data_ = crop_spec_data(spec_data,startFrame, endFrame);

cjet=colormap(parula(length(spec_data_.trackCurve)));

time = linspace(0, 10, 5);  % 时间数据，假设100个点
wavelengths = 500:3.75:739;  % 光谱波长数据，假设50个波长点
% intensity = squeeze(mean(spec_data_.trackImg(:, 17:80,1:69:345),1))';  % 强度数据，100组时间，每组50个波长点
intensity = spec_data.trackCurve(1:69:345,17:80);


for i = 1:length(cjet)
    % plot3(time(i)*ones(size(wavelengths)), wavelengths, intensity(i, :)/max(intensity(i, 23:43)), 'LineWidth', 2);
    intensity = spec_data_.trackCurve(i,17:80);
    plot(wavelengths, intensity, 'LineWidth', 1, 'Color', [cjet(i,:),1]);
    hold on
end

% 设置坐标轴标签
xlabel('Time (s)');
ylabel('Wavelength (nm)');
zlabel('Intensity (kHz)');

% 添加网格和视角
grid on;

% % 调整轴的范围
% xlim([0 10]);
% ylim([500 740]);
% zlim([0 4]);

hold off;
%%

% 假设有10组光谱数据
wavelengths = 500:3.75:739;
intensity = spec_data_.trackCurve(1:35:345,17:80);

% 创建图形
figure;
hold on;

% 循环遍历每一组数据，并按步长 0.2 偏移横坐标
for i = 1:size(intensity,1)
    plot(intensity(i,:)/max(intensity(i, 23:43)) + (i-1)*1, wavelengths, 'LineWidth', 2);  % 按步长 0.2 偏移横坐标
end

% 设置图形标签
xlabel('Intensity (with offset)');
ylabel('Spectra');
grid on;

hold off;

%% 保存视频

video_name = strcat('spec_image','.mp4');
outputVideo = VideoWriter(video_name, 'MPEG-4');
open(outputVideo);

close all
hFig = figure;

pos1 = [0.2, 0.8, 0.65, 0.1]; % [left bottom width height]
pos2 = [0.2, 0.1, 0.65, 0.65];

for i = 1:length(spec_data_.trackCurve)
    ax1 = axes('Position', pos1);
    img = spec_data_.trackImg(:,16:end,i);
    imagesc(img);
    colormap hot;
    set(ax1, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
    
    int_profile_1 = normalize(spec_data_.trackCurve(i,16:end),'range');
    
    ax2 = axes('Position', pos2);
    plot(int_profile_1,'r','LineWidth',1)

    hold on
    % ylabel('Normalized Intensity')
    % xlabel('Distance (pixels)')
    figureandaxiscolors('w','k','')
    set(gca, 'XTick', [], 'YTick', []);
    % axis off

    %     %%%%%
    drawnow;
    pause(0.01);
    currFrame = getframe(hFig);
    writeVideo(outputVideo, currFrame);
    %     %%%%%%%%
end
close(outputVideo);


%% 将所有荧光小球的光谱数据存储为16*80大小，用于深度学习重建
pname = '/Volumes/shah/data/spms_track/Ps_beads/';
fileList = ["488_beads", "561_beads"];

fnames = {};
for j = 1:length(fileList)
    
    folderPath = strcat(pname,fileList(j));
    files = dir(folderPath);
    files = files(~[files.isdir]);
    for i = 1:length(files)
        fileName = files(i).name;
        if contains(fileName, '.tdms') && ~contains(fileName, 'SM')
            % 将符合条件的文件名添加到数组中
            fnames{end+1} = strcat(folderPath,'/',fileName); % 将符合条件的文件名添加到selectedFiles
        end
    end
end

for j = 1:length(fnames)
    close all
    [pname, fname, ~] = fileparts(fnames{j});
    disp(fname)
    fname = strcat(fname,'.tdms');
    sname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 SM$2 TR$2$3');

    [~,dirname,~]=fileparts(fname);
    cd(fullfile(pname,dirname));
    
    load('spec_data.mat');
    ind = find(spec_data.pos(:,1)==0);
    spec_data.trackImg(:,:,ind) = [];
    
    tiff_fileName = strrep(dirname, ' ', '_')+".tiff";    
    t = Tiff(strcat('/Users/shahao/Desktop/new_folder/',tiff_fileName),'w');
    tagstruct.ImageLength = size(spec_data.trackImg, 1);
    tagstruct.ImageWidth = size(spec_data.trackImg, 2);
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 64;
    tagstruct.RowsPerStrip = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    t.setTag(tagstruct);
    
    % 循环遍历图像序列，并将每个图像写入TIFF文件
    for i = 1:size(spec_data.trackImg, 3)
        img = spec_data.trackImg(:, :, i);
        t.setTag(tagstruct);
        t.write(img);
        t.writeDirectory();
    end  

end






%% 纳米银数据分析，同时展示EMCCD图片和光谱轨迹
z=trackDataFilt(1:end,6);
y=trackDataFilt(1:end,5);
x=trackDataFilt(1:end,4);
traj_int=trackDataFilt(1:end,1);
spec_data_crop = crop_spec_data(spec_data, 1, 6225);

time=(1:length(x(:,1)))/1000;

centerEMCCD = [154,149];
posImg = image_ccd(:,1:251,1:6225);
showFrame = 1145;
plot_spec_emccd(traj_int,y,x,z,spec_data_crop,'231116 TR031', posImg,showFrame,centerEMCCD)



%%

fname_ = "./TR109-three-5mw-1.tif";
info = imfinfo(fname_);  % 替换'yourfile.tif'为你的文件名
num_frames = numel(info);
image_ccd = zeros(65,233,num_frames);  % 创建一个单元数组来存储所有帧
for k = 1:num_frames
    image_ccd(:,:,k) = imread(fname_, k);  % 读取第k帧
end
save('image_ccd.mat', 'image_ccd','-v7.3')

%% 批处理打开.tdms文件
pname = '/Volumes/shah_ssd/data/spms_track/单分子FRET/20240902/';
fileList = ["240902 TR081/"];

fnames = {};
for j = 1:length(fileList)
    
    folderPath = strcat(pname,fileList(j));
    files = dir(folderPath);
    files = files(~[files.isdir]);
    for i = 1:length(files)
        fileName = files(i).name;
        if contains(fileName, '.tdms') && ~contains(fileName, 'SM')
            % 将符合条件的文件名添加到数组中
            fnames{end+1} = strcat(folderPath,'/',fileName); % 将符合条件的文件名添加到selectedFiles
        end
    end
end

for j = 1:length(fnames)
    close all
    
    [pname, fname, ~] = fileparts(fnames{j}); 
    sname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 SM$2 TR$2$3');  
    iname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 IM$2 TR$2$3');

    fname_ = cell2mat(strcat(fname, '.tdms'));
    [trackData,trackDataFilt] = trajLoadTDMSCalibrated_galvo_v2(1,fname_,pname);
    disp(fname_)
    image_ccd = showEMCCDImg(length(trackDataFilt),1,sname,pname);
% 
%     cd(pname)
%     cd(fname)
%     disp(fname)
%     load("image_ccd.mat");
    spec_data = AnalysisSpecImg_centroid(image_ccd, fname);
    save("spec_data","spec_data");
end





%%
function exposureTime = extrack_exposureTime(filePath)
% 获取路径下所有 TIFF 文件的信息
tiffFiles = dir(fullfile(filePath, '*.tiff'));

% 遍历文件，提取曝光时间
for i = 1:length(tiffFiles)
    % 获取当前文件名
    fileName = tiffFiles(i).name;
    
    % 正则表达式匹配曝光时间
    % 假设曝光时间总是在最后一个下划线和 ".tiff" 之间
    token = regexp(fileName, '_(\d+\.\d+)\.tiff$', 'tokens');
    
    if ~isempty(token)
        % 将提取的曝光时间转换为数值并存储
        exposureTime = str2double(token{1}{1});
    end
end

end

function [selectedStarts, selectedEnds] = selectNormPoints(startPoints,endPoints,totalTime,nums)
    available = true(1, totalTime);
    % 标记不可选的时间段为 false
    for i = 1:length(startPoints)
        available(startPoints(i):endPoints(i)) = false;
    end
    
    % 生成所有可能的起始点，排除那些不满足条件的点
    possibleStarts = find(available(1:end-6000)); % 确保选择的起始点和终止点都在可选范围内
    
    % 从过滤后的数组中随机选择一个起始点
    % 由于要随机选择 6 组这样的点，我们需要检查并确保选取的起始点合法
    selectedStarts = zeros(1, nums); % 存储选中的起始点
    selectedEnds = zeros(1, nums); % 存储对应的终止点
    
    for n = 1:nums
        isValid = false;
        while ~isValid
            idx = randi(length(possibleStarts));
            startPoint = possibleStarts(idx);
            endPoint = startPoint + 6000;
            % 检查选中的起始点和终止点是否合法（即不位于不可选的时间段内）
            if all(available(startPoint:endPoint))
                isValid = true;
                selectedStarts(n) = startPoint;
                selectedEnds(n) = endPoint;
                % 将选中的时间段标记为不可选，避免重复选择
                available(startPoint:endPoint) = false;
            end
        end
    end
end

function plot_spec_emccd(traj_int,x,y,z,spec_data, dirname, posImg,showFrame,centerEMCCD)
    close all
    time=(1:length(traj_int(:,1)))/1000;
    len=length(x);
    
    pixSize = 0.237; % um
    
    scan_y_start = y(1) - centerEMCCD(1) * pixSize;
    scan_y_end = y(1) + (size(posImg,1) - centerEMCCD(1)) * pixSize;
    scan_x_start = x(1) - centerEMCCD(2) * pixSize;
    scan_x_end = x(1) + (size(posImg,2) - centerEMCCD(2)) * pixSize;
    

    exposure_time = length(x) / size(posImg,3);
    time_point = floor(exposure_time * showFrame);
    center = [x(time_point) y(time_point) z(time_point)]; % x y z
    
    % 创建 x 轴和 y 轴的坐标向量
    start_x = linspace(scan_x_start, scan_x_end, size(posImg(:,:,showFrame),2));  % x 轴从 5um 到 45um，步进为 0.2um
    start_y = linspace(scan_y_start, scan_y_end, size(posImg(:,:,showFrame),1));  % y 轴从 5um 到 45um，步进为 0.2um
    
    % 创建一个 z 轴位置的常数值，例如 z = 10um
    start_z = center(1,3) * ones(size(posImg(:,:,showFrame)));
    
    adjustedImg = adapthisteq(double(posImg(:,:,showFrame)) / double(max(posImg(:))));
    % 
    
    hTraj = figure;
    surf(start_x, start_y, start_z, adjustedImg, 'EdgeColor', 'none', 'FaceColor', 'interp')
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    hold on

    daspect([1 1 0.1]);

    view([0 0 -1]);
%     xlim([min(x),max(x)])
%     ylim([min(y),max(y)])
%     zlim([min(z),max(z)])
    


    figureandaxiscolors('w','k',strcat(dirname,' Spec'))
    hit_centroid = floor(min(spec_data.raw_centroid)):0.01: ...
        ceil(max(spec_data.raw_centroid));
    cjet=colormap(parula(length(hit_centroid)));
    
    colormap('bone')
%     hOriginalColorbar = colorbar; 
%     
%     % 创建一个隐藏的轴
%     hHiddenAxes = axes('Visible', 'off');
%     caxis([min(hit_centroid) max(hit_centroid)]);
%        
%     h = colorbar('peer', hHiddenAxes, 'Location', 'eastoutside');
%     % 添加颜色条，并设置字体颜色和范围
%     set( h,'fontsize',10, 'Color', 'k');
%     h.Label.String = 'centroid';%添加单位
%     set(h,'fontsize',10,'Color', 'k');
%     h_text = h.Label;%将“cm”的句柄赋值给h_text
%     set(h_text,'Position',[ 0.5 min(spec_data.raw_centroid)-0.25 ],'Rotation',360);
%     
%     % 隐藏原始的 colorbar
%     set(hOriginalColorbar, 'Visible', 'off');
    
    hold on

    %%%%%%%%%%%%%%%%%%

    hcentroid = figure;
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))+5])
    ylim([min(hit_centroid) max(hit_centroid)])
    xlabel('Time(s)')
    ylabel('Centroids')
    hold on
    % figTrajname=[dirname ' pH ratio' '.fig'];
    % saveas(hSpec,figTrajname,'fig');
    
    hx = figure;
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))+5])
    ylim([min(x) max(x)])
    xlabel('Time(s)')
    ylabel('x (um)')
    hold on
    
    hy = figure;
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))+5])
    ylim([min(y) max(y)])
    xlabel('Time(s)')
    ylabel('y (um)')
    hold on
    
    hz = figure;
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))+5])
    ylim([min(z) max(z)])
    xlabel('Time(s)')
    ylabel('z (um)')
    hold on
    
    disp_rate = 1; % EMCCD Frame
    cjlen = floor(numel(spec_data.raw_centroid)/disp_rate);
    
    cen_len = numel(spec_data.raw_centroid);
    spec_time=time(round(linspace(1, len, cen_len)));

    for i=1:cjlen-1
    
        seg=(floor((i-1)*len/cjlen)+1):floor((i)*len/cjlen);
        p1=plot3(hTraj.CurrentAxes, x(seg),y(seg),z(seg));
        tolerance = 1e-6;  
        cid=find(abs(hit_centroid-roundn(spec_data.raw_centroid(i),-2)) < tolerance);
        set(p1,'Color',cjet(cid,:),'LineWidth',1);
        
        plot(hx.CurrentAxes, time(seg), x(seg),'Color',"#0072BD",'LineWidth',1);
        plot(hy.CurrentAxes, time(seg), y(seg),'Color',"#0072BD",'LineWidth',1);
        plot(hz.CurrentAxes, time(seg), z(seg),'Color',"#0072BD",'LineWidth',1);
    
        seg = ((i-1)*disp_rate+1:(i+1)*disp_rate);
        plot(hcentroid.CurrentAxes, spec_time(seg), spec_data.raw_centroid(seg),'r','LineWidth',1)    
    
    %     %%%%%
    %     drawnow;
    %     pause(0.01);
    %     currFrame = getframe(hScatter);
    %     writeVideo(outputVideo, currFrame);
    %     %%%%%%%%
    
    end
    % set(hScatter, 'Renderer', 'painters');
    % print(hScatter, strcat(dirname,'.eps'), '-epsc', '-painters');

end

function plot_traj_emccd(x,y,z,dirname, posImg,centerEMCCD)
    cjet=colormap(parula(round(length(x)/33)));
    len=length(x);
    
    pixSize = 0.237; % um
    
    scan_y_start = y(1) - centerEMCCD(1) * pixSize;
    scan_y_end = y(1) + (size(posImg,1) - centerEMCCD(1)) * pixSize;
    scan_x_start = x(1) - centerEMCCD(2) * pixSize;
    scan_x_end = x(1) + (size(posImg,2) - centerEMCCD(2)) * pixSize;
    

    center = [x(100) y(100) z(100)]; % x y z
    
    % 创建 x 轴和 y 轴的坐标向量
    start_x = linspace(scan_x_start, scan_x_end, size(posImg,2));  % x 轴从 5um 到 45um，步进为 0.2um
    start_y = linspace(scan_y_start, scan_y_end, size(posImg,1));  % y 轴从 5um 到 45um，步进为 0.2um
    
    % 创建一个 z 轴位置的常数值，例如 z = 10um
    start_z = center(1,3) * ones(size(posImg));
    
    adjustedImg = adapthisteq(double(posImg) / double(max(posImg(:))));
    % 
    
    hTraj = figure;
    surf(start_x, start_y, start_z, adjustedImg, 'EdgeColor', 'none', 'FaceColor', 'interp')
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    hold on

    colormap('bone')
    
    view([0 0 1]);
    figureandaxiscolors('w','k',strcat(dirname,' traj')) 
    hold on

    %%%%%%%%%%%%%%%%%%
    cjlen = size(cjet,1);
    for i=1:cjlen-1
        seg=(floor((i-1)*len/cjlen)+1):floor((i)*len/cjlen);
        p1=plot3(x(seg),y(seg),z(seg));
        set(p1,'Color',cjet(i,:),'LineWidth',1);       
    end

end

function read_save_datfile(pname, fname, w, h)
    fid=fopen(strcat(pname,fname(1)));
    data_1=fread(fid, Inf, 'int32', 0);  % 以uint32格式读取数据，'b'表示大端字节序
    fclose(fid);
    
    row = h;
    col = w;
    img_ccd_1 = reshape(data_1, row, col, []);
    image_ccd = img_ccd_1(:,:,1:end);
    
    save('image_ccd.mat', 'image_ccd','-v7.3')
    
    
    [~,dirname,~]=fileparts(fname(1));
    tiff_fileName = dirname+".tiff";
    
    t = Tiff(tiff_fileName,'w');
    tagstruct.ImageLength = size(image_ccd, 1);
    tagstruct.ImageWidth = size(image_ccd, 2);
    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 16;
    tagstruct.RowsPerStrip = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    t.setTag(tagstruct);
    
    % 循环遍历图像序列，并将每个图像写入TIFF文件
    for i = 1:1:size(image_ccd,3)
        img = uint16(image_ccd(:,:,i));
        t.setTag(tagstruct);
        t.write(img);
        t.writeDirectory();
        
        if mod(i, 500)==0
            disp("processing "+string(i) + " / "+string(size(image_ccd,3)));
        end

    end
end

function plot_fret(trackDataFilt,spec_data,fname)
    z=trackDataFilt(1:end,6);
    y=trackDataFilt(1:end,5);
    x=trackDataFilt(1:end,4);
    traj_int = trackDataFilt(1:end,1);
    len=length(x);    
    
    [~,dirname,~]=fileparts(fname);
    
    turncat = len;
    x = x(1:turncat);
    y = y(1:turncat);
    z = z(1:turncat);
    traj_int = traj_int(1:turncat);
    
    tmp = round(turncat / len * numel(spec_data.frame));
    spec_data.frame = spec_data.frame (1:tmp,:);
    spec_data.trackImg = spec_data.trackImg(:,:,1:tmp);
    spec_data.raw_centroid = spec_data.raw_centroid(1:tmp);
    
    time=(1:length(traj_int(:,1)))/1000;
    
    visibility='on';
    
    hScatter=figure;
    set(hScatter,'Visible',visibility);
    set(hScatter,'Renderer','OpenGL')
    set(0,'CurrentFigure',hScatter)
    
    pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]
    pos2 = [0.6, 0.5, 0.35, 0.35];
    pos3 = [0.1, 0.1, 0.25, 0.25];
    pos4 = [0.4, 0.1, 0.25, 0.25];
    pos5 = [0.7, 0.1, 0.25, 0.25];
    
    ax1 = axes('Position', pos1);
    
    caxis([0 length(x)/1000]);
    h = colorbar;
    set( h,'fontsize',10, 'Color', 'w');
    h.Label.String = 'time(s) ';%添加单位
    set(h,'fontsize',10,'Color', 'w');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 -0.8 ],'Rotation',360);
    
    n = 1; % 设置n的值
    numGroups = floor(numel(spec_data.raw_centroid) / n);  % 计算能整除的最大组数
    centroids = spec_data.raw_centroid(1:numGroups * n);  % 截取能整除的部分数据
    centroids = mean(reshape(centroids,n,[]),1);
    
    view([1 1 1]);
    axis image
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    
    hit_centroid = min(roundn(spec_data.raw_centroid,-2)):0.01: ...
       max(roundn(spec_data.raw_centroid,-2));
    % hit_centroid = 520:0.01:530;
    cjet=colormap(parula(length(hit_centroid)));
    
    caxis([min(spec_data.raw_centroid) max(spec_data.raw_centroid)]);
    h = colorbar;
    % 添加颜色条，并设置字体颜色和范围
    set( h,'fontsize',10, 'Color', 'k');
    h.Label.String = 'centroid';%添加单位
    set(h,'fontsize',10,'Color', 'k');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 min(spec_data.raw_centroid)-0.15 ],'Rotation',360);
    
    % axis tight
    % axis equal
    figureandaxiscolors('w','k',strcat(dirname,' Spec'))
    
    xlim([min(x),max(x)])
    ylim([min(y),max(y)])
    zlim([min(z),max(z)])
    
    hold on
    
    
    %%%%%%%%%%%%%%%%%%
    ax2 = axes('Position', pos2);
    figureandaxiscolors('w','k',dirname)
    xlim([0 ceil(max(time))])
    ylim([-0.1, 1.1])
    xlabel('Time(s)')
    ylabel('Int')
    hold on
    % figTrajname=[dirname ' pH ratio' '.fig'];
    % saveas(hSpec,figTrajname,'fig');
    
    ax3 = axes('Position', pos3);
    figureandaxiscolors('w','k',dirname)
    xlim([0 ceil(max(time))])
    ylim([min(x) max(x)])
    xlabel('Time(s)')
    ylabel('x (um)')
    hold on
    
    ax4 = axes('Position', pos4);
    figureandaxiscolors('w','k',dirname)
    xlim([0 ceil(max(time))])
    ylim([min(y) max(y)])
    xlabel('Time(s)')
    ylabel('y (um)')
    hold on
    
    ax5 = axes('Position', pos5);
    figureandaxiscolors('w','k',dirname)
    xlim([0 ceil(max(time))])
    ylim([min(z) max(z)])
    xlabel('Time(s)')
    ylabel('z (um)')
    hold on
    
    disp_rate = 1; % EMCCD Frame
    cjlen = floor(numel(spec_data.raw_centroid)/disp_rate);
    
    cen_len = numel(spec_data.raw_centroid);
    spec_time=time(round(linspace(1, len, cen_len)));

    spec_488_norm = normalize(spec_data.spec_488, 'range');
    spec_640_norm = normalize(spec_data.spec_640, 'range');

    % video_name = strcat(dirname,'.mp4');
    % outputVideo = VideoWriter(video_name, 'MPEG-4');
    % open(outputVideo);
    
    for i=1:cjlen-1
    
        seg=(floor((i-1)*len/cjlen)+1):floor((i)*len/cjlen);
    
        p1=plot3(ax1, x(seg),y(seg),z(seg));
        tolerance = 1e-6;  
        cid=find(abs(hit_centroid-roundn(spec_data.raw_centroid(i),-2)) < tolerance);
        set(p1,'Color',cjet(cid,:),'LineWidth',1);
    
        plot(ax3, time(seg), x(seg),'Color',"#0072BD",'LineWidth',1);
        plot(ax4, time(seg), y(seg),'Color',"#0072BD",'LineWidth',1);
        plot(ax5, time(seg), z(seg),'Color',"#0072BD",'LineWidth',1);
    
        seg = ((i-1)*disp_rate+1:(i+1)*disp_rate);        
        plot(ax2, spec_time(seg), spec_488_norm(seg),'b','LineWidth',1)  
        plot(ax2, spec_time(seg), spec_640_norm(seg),'r','LineWidth',1)  
    
    %     %%%%%
    %     drawnow;
    %     pause(0.01);
    %     currFrame = getframe(hScatter);
    %     writeVideo(outputVideo, currFrame);
    %     %%%%%%%%
    
    end
    % close(outputVideo);
    saveas(hScatter,strcat(dirname, ' spec'), 'fig');

end

function truncAxis(varargin)

% 获取参数
if isa(varargin{1},'matlab.graphics.axis.Axes')
    ax=varargin{1};varargin(1)=[];
else
    ax=gca;
end
hold(ax,'on');
% box(ax,'off')
ax.XAxisLocation='bottom';
ax.YAxisLocation='left';

axisPos=ax.Position;
axisXLim=ax.XLim;
axisYLim=ax.YLim;

axisXScale=diff(axisXLim);
axisYScale=diff(axisYLim);


truncRatio=1/20;
Xtrunc=[];Ytrunc=[];
for i=1:length(varargin)-1
    switch true
        case strcmpi('X',varargin{i}),Xtrunc=varargin{i+1};
        case strcmpi('Y',varargin{i}),Ytrunc=varargin{i+1};
    end
end


switch true
    case isempty(Xtrunc)
        % 复制坐标区域
        ax2=copyAxes(ax);
        % 修改轴基础属性
        ax2.XTickLabels=[];
        ax2.XColor='none';
        % 修改坐标区域范围
        ax.YLim=[axisYLim(1),Ytrunc(1)];
        ax2.YLim=[Ytrunc(2),axisYLim(2)];
        % 坐标区域重定位
        ax.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(Ytrunc(1)-axisYLim(1));
        ax2.Position(2)=axisPos(2)+ax.Position(4)+axisPos(4)*truncRatio;
        ax2.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(axisYLim(2)-Ytrunc(2));
        % 链接轴范围变动
        linkaxes([ax,ax2],'x')
        % 添加线和标识符
        if strcmp(ax.Box,'on')
        ax.Box='off';ax2.Box='off';
        annotation('line',[1,1].*(ax.Position(1)+ax.Position(3)),[ax.Position(2),ax.Position(2)+ax.Position(4)],'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax.Position(1)+ax.Position(3)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax2.Position(2)+ax2.Position(4)),'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        else
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax.Position(2)+ax.Position(4)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax2.Position(2)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        end
        createSlash([ax.Position(1)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
        createSlash([ax.Position(1)-.2,ax2.Position(2)-.2,.4,.4])
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax2.Position(2)-.2,.4,.4])
    case isempty(Ytrunc) 
        % 复制坐标区域
        ax2=copyAxes(ax);
        % 修改轴基础属性
        ax2.YTickLabels=[];
        ax2.YColor='none';
        % 修改坐标区域范围
        ax.XLim=[axisXLim(1),Xtrunc(1)];
        ax2.XLim=[Xtrunc(2),axisXLim(2)];
        % 坐标区域重定位
        ax.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(Xtrunc(1)-axisXLim(1));
        ax2.Position(1)=axisPos(1)+ax.Position(3)+axisPos(3)*truncRatio;
        ax2.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(axisXLim(2)-Xtrunc(2));
        % 链接轴范围变动
        linkaxes([ax,ax2],'y')
        % 添加线和标识符
        if strcmp(ax.Box,'on')
        ax.Box='off';ax2.Box='off';
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax.Position(2)+ax.Position(4)),'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax2.Position(1),ax2.Position(1)+ax2.Position(3)],[1,1].*(ax.Position(2)+ax.Position(4)),'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax2.Position(1)+ax2.Position(3)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        else
        annotation('line',[1,1].*(ax.Position(1)+ax.Position(3)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax2.Position(1)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        end
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax.Position(2)-.2,.4,.4])
        createSlash([ax2.Position(1)-.2,ax.Position(2)-.2,.4,.4])
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
        createSlash([ax2.Position(1)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
    case (~isempty(Ytrunc))&(~isempty(Ytrunc))
        % 复制坐标区域
        ax2=copyAxes(ax);
        ax3=copyAxes(ax);
        ax4=copyAxes(ax);
        % 修改轴基础属性
        ax2.XTickLabels=[];
        ax2.XColor='none';
        ax3.XTickLabels=[];
        ax3.XColor='none';
        ax3.YTickLabels=[];
        ax3.YColor='none';
        ax4.YTickLabels=[];
        ax4.YColor='none';
        % 修改坐标区域范围
        ax.YLim=[axisYLim(1),Ytrunc(1)];
        ax.XLim=[axisXLim(1),Xtrunc(1)];
        ax2.XLim=[axisXLim(1),Xtrunc(1)];
        ax2.YLim=[Ytrunc(2),axisYLim(2)];
        ax3.XLim=[Xtrunc(2),axisXLim(2)];
        ax3.YLim=[Ytrunc(2),axisYLim(2)];
        ax4.XLim=[Xtrunc(2),axisXLim(2)];
        ax4.YLim=[axisYLim(1),Ytrunc(1)];
        % 坐标区域重定位
        ax.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(Xtrunc(1)-axisXLim(1));
        ax.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(Ytrunc(1)-axisYLim(1));
        ax2.Position(2)=axisPos(2)+ax.Position(4)+axisPos(4)*truncRatio;
        ax2.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(Xtrunc(1)-axisXLim(1));
        ax2.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(axisYLim(2)-Ytrunc(2));
        ax3.Position(1)=axisPos(1)+ax.Position(3)+axisPos(3)*truncRatio;
        ax3.Position(2)=axisPos(2)+ax.Position(4)+axisPos(4)*truncRatio;
        ax3.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(axisXLim(2)-Xtrunc(2));
        ax3.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(axisYLim(2)-Ytrunc(2));
        ax4.Position(1)=axisPos(1)+ax.Position(3)+axisPos(3)*truncRatio;
        ax4.Position(3)=axisPos(3)*(1-truncRatio)/(axisXScale-diff(Xtrunc))*(axisXLim(2)-Xtrunc(2));
        ax4.Position(4)=axisPos(4)*(1-truncRatio)/(axisYScale-diff(Ytrunc))*(Ytrunc(1)-axisYLim(1));
        % 链接轴范围变动
        linkaxes([ax3,ax2],'y')
        linkaxes([ax4,ax3],'x')
        linkaxes([ax,ax2],'x')
        linkaxes([ax,ax4],'y')
        % 添加线和标识符
        if strcmp(ax.Box,'on')
        ax.Box='off';ax2.Box='off';ax3.Box='off';ax4.Box='off';
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax2.Position(2)+ax2.Position(4)),'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax3.Position(1),ax3.Position(1)+ax3.Position(3)],[1,1].*(ax2.Position(2)+ax2.Position(4)),'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax4.Position(1)+ax4.Position(3)),[ax3.Position(2),ax3.Position(2)+ax3.Position(4)],'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax4.Position(1)+ax4.Position(3)),[ax4.Position(2),ax4.Position(2)+ax4.Position(4)],'LineStyle','-','LineWidth',ax.LineWidth,'Color',ax.XColor);
        else
        annotation('line',[1,1].*(ax.Position(1)+ax.Position(3)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax3.Position(1)),[ax2.Position(2),ax2.Position(2)+ax2.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax.Position(1)+ax.Position(3)),[ax.Position(2),ax.Position(2)+ax.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[1,1].*(ax3.Position(1)),[ax.Position(2),ax.Position(2)+ax.Position(4)],'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax.Position(2)+ax.Position(4)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax.Position(1),ax.Position(1)+ax.Position(3)],[1,1].*(ax2.Position(2)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax4.Position(1),ax4.Position(1)+ax4.Position(3)],[1,1].*(ax.Position(2)+ax.Position(4)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        annotation('line',[ax4.Position(1),ax4.Position(1)+ax4.Position(3)],[1,1].*(ax2.Position(2)),'LineStyle',':','LineWidth',ax.LineWidth,'Color',ax.XColor);
        end
        createSlash([ax.Position(1)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
        createSlash([ax.Position(1)-.2,ax2.Position(2)-.2,.4,.4])
        createSlash([ax4.Position(1)+ax4.Position(3)-.2,ax.Position(2)+ax.Position(4)-.2,.4,.4])
        createSlash([ax4.Position(1)+ax4.Position(3)-.2,ax2.Position(2)-.2,.4,.4])
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax.Position(2)-.2,.4,.4])
        createSlash([ax.Position(1)+ax.Position(3)-.2,ax2.Position(2)+ax2.Position(4)-.2,.4,.4])
        createSlash([ax4.Position(1)-.2,ax.Position(2)-.2,.4,.4])
        createSlash([ax4.Position(1)-.2,ax2.Position(2)+ax2.Position(4)-.2,.4,.4])
        % 修改当前坐标区域，方便legend添加
        set(gcf,'currentAxes',ax3)
end
% 复制原坐标区域全部可复制属性
    function newAX=copyAxes(ax)
        axStruct=get(ax);
        fNames=fieldnames(axStruct);
        newAX=axes('Parent',ax.Parent);

        coeList={'CurrentPoint','XAxis','YAxis','ZAxis','BeingDeleted',...
            'TightInset','NextSeriesIndex','Children','Type','Legend'};
        for n=1:length(coeList)
            coePos=strcmp(fNames,coeList{n});
            fNames(coePos)=[];
        end
        
        for n=1:length(fNames)
            newAX.(fNames{n})=ax.(fNames{n});
        end

        copyobj(ax.Children,newAX)
    end
% 添加截断标识符函数
    function createSlash(pos)
        anno=annotation('textbox');
        anno.String='/';
        anno.LineStyle='none';
        anno.FontSize=15;
        anno.Position=pos;
        anno.FitBoxToText='on';
        anno.VerticalAlignment='middle';
        anno.HorizontalAlignment='center';
    end
end

