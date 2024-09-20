%% Fig2c
%%%%%%%%%

pname = "/Volumes/shah_ssd/data/spms_track/manuscript/Fig2/230828 TR012/";
cd(pname)
fnames = "488beads_vit.csv";

load("spec_data.mat");
load("230828 TR012.mat");
dirname = "230828 TR012";

data = readtable(strcat(pname,fnames));
data = data(data.preds ~= 0, :);
vit_preds = data.preds;

fitted_struct.pred_centroid = vit_preds;

z=trackDataFilt(1:end,6);
y=trackDataFilt(1:end,5);
x=trackDataFilt(1:end,4);
time=(1:length(x(:,1)))/1000;
cen_len = numel(fitted_struct.pred_centroid);
spec_time=time(round(linspace(1, length(x), cen_len)));

hfig = figure;
pos1 = [0.1, 0.7, 0.74, 0.25]; % [left bottom width height]
pos2 = [0.85, 0.7, 0.1, 0.25];
%%%%%%%%%%%%%%%
ax1 = axes('Position', pos1);
dataVector = fitted_struct.pred_centroid;

plot(ax1, spec_time, dataVector, 'Color',"r", 'LineWidth',1); % 假设数据在 dataVector 变量中
ylabel('Spec Cent. [nm]');
ylim([505 525])
% 计算数据的均值和标准差
mu = mean(dataVector);
sigma = std(dataVector);
grid on

% 绘制旋转的直方图和正态分布拟合
ax2 = axes('Position', pos2);
histogram(ax2, dataVector, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','r','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;
ylim([505 525])
% 计算正态分布数据
x_values = linspace(min(dataVector), max(dataVector), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax2, pdf_values, x_values, 'Color','r', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));

hold off;
axis off

pos3 = [0.1, 0.4, 0.74, 0.25]; % [left bottom width height]
pos4 = [0.85, 0.4, 0.1, 0.25];
%%%%%%%%%%%%%%%
ax3 = axes('Position', pos3);
ind = find(spec_data.pos(:,1)~=0);
raw_centroid = spec_data.raw_centroid(ind);
dataVector = raw_centroid;

cen_len = size(dataVector,1);
spec_time=time(round(linspace(1, length(x), cen_len)));

plot(ax3, spec_time, dataVector, 'Color',"r", 'LineWidth',1); % 假设数据在 dataVector 变量中
ylabel('Spec Cent. [nm]');
ylim([505 525])

% 计算数据的均值和标准差
mu = mean(dataVector);
sigma = std(dataVector);
grid on

% 绘制旋转的直方图和正态分布拟合
ax4 = axes('Position', pos4);
histogram(ax4, dataVector, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','r','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;
ylim([505 525])

% 计算正态分布数据
x_values = linspace(min(dataVector), max(dataVector), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax4, pdf_values, x_values, 'Color','r', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));

hold off;
axis off

%% 画光谱精度
z=trackDataFilt(1:end,6);
y=trackDataFilt(1:end,5);
x=trackDataFilt(1:end,4);
time=(1:length(x(:,1)))/1000;

% ind = find(spec_data.pos(:,1)==0);
% spec_data.raw_centroid(ind) = [];
cen_len = numel(raw_centroid);
spec_time=time(round(linspace(1, length(x), cen_len)));


hfig = figure;
pos1 = [0.1, 0.7, 0.72, 0.25]; % [left bottom width height]
pos2 = [0.85, 0.7, 0.1, 0.25];
%%%%%%%%%%%%%%%
ax1 = axes('Position', pos1);
yyaxis left;
dataVector = raw_centroid;

plot(ax1, spec_time, dataVector, 'Color',"r", 'LineWidth',1); % 假设数据在 dataVector 变量中

leftAxis = gca; % 获取当前坐标轴句柄
leftAxis.YColor = 'r'; % 设置 y 轴及刻度颜色为红色

% 设置左侧 Y 轴标签为黑色
ylabelHandleLeft = ylabel('Spec Cent. [nm]');
ylabelHandleLeft.Color = 'k'; % 设置标签颜色为黑色
ylim([505 525])

mu = mean(dataVector);
sigma = std(dataVector);

ax2 = axes('Position', pos2);

histogram(ax2, dataVector, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','r','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;

% 计算正态分布数据
x_values = linspace(min(dataVector), max(dataVector), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax2, pdf_values, x_values, 'Color','r', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));

%%%%%%%%%%%%%%%%%
axes(ax1)
yyaxis right;
dataVector = fitted_struct.pred_centroid;
cen_len = numel(fitted_struct.pred_centroid);
spec_time=time(round(linspace(1, length(x), cen_len)));
plot(ax1, spec_time, dataVector, 'Color',"#0072BD", 'LineWidth',1); % 假设数据在 dataVector 变量中
set(gca, 'ycolor', '#0072BD');
ylim([505 525])


grid on
set(gca, 'GridColor', [0.8 0.8 0.8]);
% 计算数据的均值和标准差
mu = mean(dataVector);
sigma = std(dataVector);


% 绘制旋转的直方图和正态分布拟合
axes(ax2)
histogram(ax2, dataVector, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','#0072BD','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;

% 计算正态分布数据
x_values = linspace(min(dataVector), max(dataVector), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax2, pdf_values, x_values, 'Color','#0072BD', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));



axis off

%% Fig2a 使用ViT输出结果代替Norm Fit 
pname = "/Volumes/shah_ssd/data/spms_track/manuscript/Fig2/230828 TR012/";
cd(pname)
fnames = "488beads_vit.csv";
load("spec_data.mat");
load("230828 TR012.mat");
dirname = "230828 TR012";

% fitted_struct = struct();
data = readtable(strcat(pname,fnames));
data = data(data.preds ~= 0, :);
vit_preds = data.preds;

spec_data_ = spec_data;

ind = sum(spec_data.pos, 2);
ind(spec_data.snr<0.5) = 0;
spec_data_.raw_centroid = replaceZeros(ind, vit_preds);

z=trackDataFilt(1:end,6);
y=trackDataFilt(1:end,5);
x=trackDataFilt(1:end,4);
traj_int=trackDataFilt(1:end,1);

plot_spec(traj_int,x,y,z,spec_data_,dirname) %Please set the colormap from 505 to 525!


%% eFig4 在仿真数据集上对比CNN/ViT/ViT_domain 
pname = "/Volumes/shah_ssd/data/spms_track/manuscript/eFig4/";
fnames = ["norm.csv", "ViT.csv", "MDD_ViT.csv"];

fitted_struct = struct();
for i = 1:length(fnames)
    data = readtable(strcat(pname,fnames(i)));
    data = data(data.preds ~= 0, :);
    [~,s,~] = fileparts(fnames(i));
    s = strcat(s,"_preds");
    fitted_struct.(s) = data.preds;
    s = strcat(s,"_labels");
    fitted_struct.(s) = data.gt_centroids;
end

data = struct();
labels = unique(fitted_struct.norm_preds_labels);

for i = 1:length(labels)
    idx = find(fitted_struct.norm_preds_labels == labels(i));
    norm_pred = fitted_struct.norm_preds(idx);
    data.norm_centroid(i) =  mean(norm_pred);
    data.norm_std(i) = std(norm_pred);
    
    idx = find(fitted_struct.ViT_preds_labels == labels(i));
    ViT_pred = fitted_struct.ViT_preds(idx);
    data.ViT_centroid(i) =  mean(ViT_pred);
    data.ViT_std(i) = std(ViT_pred);

    idx = find(fitted_struct.MDD_ViT_preds_labels == labels(i));
    MDD_ViT_pred = fitted_struct.MDD_ViT_preds(idx);
    data.MDD_ViT_centroid(i) =  mean(MDD_ViT_pred);
    data.MDD_ViT_std(i) = std(MDD_ViT_pred);


end

% 创建误差线图 
close all
hFig = figure;

pos1 = [0.1, 0.55, 0.3, 0.3]; % [left bottom width height]
pos2 = [0.57, 0.55, 0.3, 0.3];
pos3 = [0.1 0.1, 0.3, 0.3];
pos4 = [0.57, 0.1, 0.3, 0.3];

ax1 = axes('Position', pos1);
plot_error_fig(1, labels, data.norm_centroid, data.norm_std, 'title', 'Norm Fitting');

ax2 = axes('Position', pos2);
plot_error_fig(1, labels, data.ViT_centroid, data.ViT_std, 'title', 'ViT outputs');

ax3 = axes('Position', pos3);
plot_error_fig(1, labels, data.MDD_ViT_centroid, data.MDD_ViT_std, 'title', 'Domain Adaption');


%% eFig4d 在荧光小球数据集上对比ViT/ViT_domain 
fname = '230828_TR007.tdms';
file_path = '/Volumes/shah_ssd/data/spms_track/manuscript/eFig4/eFig4d/MDD_ViT-learned_outputs.csv';
% [fname,pname]=uigetfile('*.csv');
ViT_table = readtable(file_path);
% ViT_table = ViT_table(fitted_struct.raw_centroid ~= 0, :);
fitted_struct.pred_centroid = ViT_table.preds;
fitted_struct.fileName = ViT_table.t_labels;

ind = [];

for i = 1:numel(fitted_struct.fileName)
    if contains(fitted_struct.fileName{i}, fname(1:12))
        % 如果匹配成功，将索引添加到 matching_indices 数组中
        ind = [ind, i];
    end
end
fitted_struct.pred_centroid = fitted_struct.pred_centroid(ind);

file_path = '/Volumes/shah_ssd/data/spms_track/manuscript/eFig4/eFig4d/ViT_outputs.csv';
ViT_table = readtable(file_path);
fitted_struct.ViT_centroid = ViT_table.preds;
fitted_struct.ViT_fileName = ViT_table.t_labels;
ind = [];
for i = 1:numel(fitted_struct.ViT_fileName)
    if contains(fitted_struct.ViT_fileName{i}, fname(1:12))
        % 如果匹配成功，将索引添加到 matching_indices 数组中
        ind = [ind, i];
    end
end
fitted_struct.ViT_centroid = fitted_struct.ViT_centroid(ind);
fitted_struct.raw_centroid = spec_data.raw_centroid;



hInt=figure;
visibility='on';
set(hInt,'Visible',visibility);
set(hInt,'Renderer','OpenGL')
set(gcf, 'Position', [100 100 600 600]); 
t = tiledlayout(1,2,'TileSpacing','Compact');

norm_var = roundn(sqrt(var(fitted_struct.raw_centroid)),-2);
norm_mu = roundn(mean(fitted_struct.raw_centroid),-2);
vit_var = roundn(sqrt(var(fitted_struct.pred_centroid)),-2);
vit_mu = roundn(mean(fitted_struct.pred_centroid),-2);
domain_var = roundn(sqrt(var(fitted_struct.ViT_centroid)),-2);
domain_mu = roundn(mean(fitted_struct.ViT_centroid),-2);

data1 = fitted_struct.raw_centroid;
data2 = fitted_struct.pred_centroid;
data3 = fitted_struct.ViT_centroid;
% 设置统一的 bin 边缘
binEdges = linspace(min([data1; data2; data3]), max([data1; data2; data3]), 40);  % 10 bins

% 使用 histogram 函数绘制 data1 的直方图
h1 = histogram(data1, binEdges, 'Normalization', 'probability', ...
    'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'w', 'DisplayName', 'Norm');
hold on;

% 拟合 data1 的正态分布并绘制
pd1 = fitdist(data1, 'Normal');
x = linspace(min(data1), max(data1), 100);
y = normpdf(x, pd1.mu, pd1.sigma);
plot(x, y * diff(binEdges(1:2)), 'Color', [0 0.4470 0.7410], 'LineWidth', 3);
text(x(numel(x)/2)-6,max(y)* diff(binEdges(1:2))+0.01,...
    '\sigma_z='+string(norm_mu)+'\pm'+string(norm_var)+' nm',...
    'color','k', 'FontWeight','bold');

% 对 data2 和 data3 重复相同的步骤
h2 = histogram(data2, binEdges, 'Normalization', 'probability', ...
    'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'w', 'DisplayName', 'ViT');
pd2 = fitdist(data2, 'Normal');
x = linspace(min(data2), max(data2), 100);
y = normpdf(x, pd2.mu, pd2.sigma);
plot(x, y * diff(binEdges(1:2)), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 3);
text(x(numel(x)/2)-1,max(y)* diff(binEdges(1:2))+0.01,...
    '\sigma_z='+string(vit_mu)+'\pm'+string(vit_var)+' nm', ...
    'color','k', 'FontWeight','bold');

h3 = histogram(data3, binEdges, 'Normalization', 'probability', ...
    'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor', 'w', 'DisplayName', 'Domain');
pd3 = fitdist(data3, 'Normal');
x = linspace(min(data3), max(data3), 100);
y = normpdf(x, pd3.mu, pd3.sigma);
plot(x, y * diff(binEdges(1:2)), 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 3);
text(x(numel(x)/2)-2,max(y)* diff(binEdges(1:2))+0.01,...
    '\sigma_z='+string(domain_mu)+'\pm'+string(domain_var)+' nm','color','k', 'FontWeight','bold');



hBar = findobj(gca, 'Type', 'Hist');


ylabel("Probability")
xlabel("Fitted centroid")
% title('Fitted Centroid', 'FontSize', 14, 'FontWeight','bold');
% title(fname(1:end-5),'FontSize', 14, 'FontWeight','bold')

figureandaxiscolors('w','k',strrep(fname(1:end-5),'_',' '))
set(gca, 'FontName', 'Arial', 'FontSize', 12);

legend(hBar(end:-1:end-2), {'Norm', 'Domain', 'ViT'}, 'Location','best');



%%
%%%%%%%%%
function B = replaceZeros(A, B)
    % A是可能包含零的数组
    % B是需要修改的数组
    % 确保A和B具有相同的尺寸
    if size(A) ~= size(B)
        error('两个输入数组必须有相同的尺寸。');
    end
    
    % 寻找A中为零的元素的索引
    zeroIndices = find(A == 0);
    
    % 对于每个零元素，寻找最近的非零元素
    for idx = zeroIndices'
        % 计算所有非零元素与当前零元素的距离
        nonZeroIndices = find(A ~= 0);
        [~, nearest] = min(abs(nonZeroIndices - idx));
        nearestIndex = nonZeroIndices(nearest);
        
        % 将B中对应的元素替换为最近非零元素的对应值
        B(idx) = B(nearestIndex);
    end
end

function plot_error_fig(ii,labels,cent, err, varargin)
     if nargin>3
        n=numel(varargin);
        assert(rem(n,2)==0)
        for i=1:2:n
            opt.(varargin{i})=varargin{i+1};
        end
    else
        opt=struct;
    end
    
    figure(ii)
    if isfield(opt,'subplot')
       subplot(opt.subplot), 
    end

    lower_bound = cent - err;
    upper_bound = cent + err;
    
    % 使用 fill 函数创建 fill_between 效果
    for i = 1:length(labels) - 1
        fill([labels(i), labels(i+1), labels(i+1), labels(i)], ...
            [lower_bound(i), lower_bound(i+1), upper_bound(i+1), upper_bound(i)], ...
            [0.2, 0.4, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        hold on;
    end
    scatter(labels, cent , 20, [0.2, 0.4, 0.8], 'filled');
    
    % 添加斜率为1的虚线
    plot([min(labels), max(labels)], [min(labels), max(labels)], '--', 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2]);
    
    % 设置图形属性
    xlabel('Ground Truth', 'FontSize', 10);
    ylabel('Prediction', 'FontSize', 10); 
    figureandaxiscolors('w','k',opt.title)
    grid on;
    
    axis equal;    
end




