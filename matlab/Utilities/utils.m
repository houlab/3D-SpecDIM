%%
addpath(genpath('./'))

%% FigS1

z=trackDataFilt(10000:20000,6);
y=trackDataFilt(10000:20000,5);
x=trackDataFilt(10000:20000,4);
time=(1:length(x(:,1)))/1000;

% Set the size and step of the sliding window
window_size = 1000;
step_size = 10;

num_windows = floor((length(x) - window_size) / step_size) + 1;

std_errors_x = zeros(num_windows, 1);
std_errors_y = zeros(num_windows, 1);
std_errors_z = zeros(num_windows, 1);

for i = 1:num_windows
    start_idx = (i-1) * step_size + 1;
    end_idx = start_idx + window_size - 1;

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

%% Fig2d
pname = 'E:\data\spms_track\manuscript\Fig2\spec_resolution\';
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
    fieldName = fields{i}; 
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName); 
    cd (fileList(k))
    k=k+1;
    vit_data = readtable("MDD_ViT-learned_outputs.csv");
    cnn_data = readtable("cnn-learned_outputs.csv");

    fields_ = fieldnames(data);
    intensity = [];
    spec_res = [];
    vit_res = [];
    cnn_res = [];

    spec_mean = [];
    vit_mean = [];
    cnn_mean = [];


    for j = 1:numel(fields_)
        data_ = data.(fields_{j});
        intensity = [intensity; mean(data_.traj_data(:,11))/0.69*0.9];
        spec_res = [spec_res; std(data_.spec_data.raw_centroid)];
        spec_mean = [spec_mean; mean(data_.spec_data.raw_centroid)-20];
        
        tiff_name = fields_{j};
        vit_preds = vit_data.preds(vit_data.t_labels==strcat(tiff_name(5:end),".tiff"));
        vit_res = [vit_res; std(vit_preds)];
        vit_mean = [vit_mean; mean(vit_preds)-15];

        cnn_preds = cnn_data.preds(cnn_data.t_labels==strcat(tiff_name(5:end),".tiff"));
        cnn_res = [cnn_res; std(cnn_preds)];
        cnn_mean = [cnn_mean; mean(cnn_preds)];
    end
    f = fit(intensity, spec_res, 'power2');
    scatter(intensity/1000, spec_res, 100, 'o', 'MarkerFaceColor', '#4B7BB3', ...
        'MarkerEdgeColor', '#4B7BB3', 'HandleVisibility', 'off'); % 使用蓝色圆形标记mVenus
    plot(50:1:1000, f(50000:1000:1000000), 'Color', '#4B7BB3', 'LineWidth', 3.5, 'DisplayName', 'Norm');
    
    scatter(intensity/1000, vit_res, 100, '^', 'MarkerFaceColor', '#F4BA1D', ...
        'MarkerEdgeColor', '#F4BA1D', 'HandleVisibility', 'off'); % 使用金色三角形标记mGold
    f = fit(intensity, vit_res, 'power2');
    plot(50:1:1000, f(50000:1000:1000000), 'Color', '#F4BA1D', 'LineWidth', 3.5, 'DisplayName', 'ViT');

    scatter(intensity/1000, cnn_res, 100, 's', 'MarkerFaceColor', 'r', ...
        'MarkerEdgeColor', 'r', 'HandleVisibility', 'off'); % 使用金色三角形标记mGold
    f = fit(intensity, cnn_res, 'power2');
    plot(50:1:1000, f(50000:1000:1000000), 'Color', 'r', 'LineWidth', 3.5, 'DisplayName', 'CNN');
end

figure
scatter(intensity/1000, spec_mean, 100, 'o', 'MarkerFaceColor', '#4B7BB3', ...
        'MarkerEdgeColor', '#4B7BB3', 'DisplayName', 'Norm');
hold on
scatter(intensity/1000, vit_mean, 100, '^', 'MarkerFaceColor', '#F4BA1D', ...
        'MarkerEdgeColor', '#F4BA1D', 'DisplayName', 'ViT');
scatter(intensity/1000, cnn_mean, 100, 's', 'MarkerFaceColor', 'r', ...
        'MarkerEdgeColor', 'r', 'DisplayName', 'CNN');


legend
grid on



%% Fig2e
pname = 'E:\data\spms_track\Ps_beads\time_resolution\';
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
    fieldName = fields{i}; 
    
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName); 
    fields_ = fieldnames(data);
    data_ = data.(fields_{1});

    intensity = [intensity; mean(data_.traj_data(:,11))];
    spec_res = [spec_res; std(data_.spec_data.raw_centroid)];
end

f = fit(exposure_time, spec_res, 'power2');
scatter(exposure_time, spec_res, 100, 'o', 'MarkerFaceColor', colors(i), ...
    'MarkerEdgeColor', colors(i), 'HandleVisibility', 'off'); 
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


%% Fig2f
pname = 'E:\data\spms_track\Ps_beads\sensitive\';
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
    fieldName = fields{i}; 
    
    if ~contains(fieldName, fileList)
        continue
    end

    data = dataStruct.(fieldName);
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
    

    target_value = 5; 
    equation = @(x) f(x) - target_value;
    x_guess = 10000; 
    x_solution = fzero(equation, x_guess);
    disp(['f(x) = 3 时，x = ', num2str(x_solution*9/6)]);
end

legend
grid on
axis tight

%% Fig3c
pname = 'E:\data\spms_track\mito\';
fileList = ["tr_Halo_mGold", "tr_Halo_mGold_2"];

fnames = {};
for j = 1:length(fileList)
    
    folderPath = strcat(pname,fileList(j));
    files = dir(folderPath);
    files = files(~[files.isdir]);
    for i = 1:length(files)
        fileName = files(i).name;
        if contains(fileName, '.tdms') && ~contains(fileName, 'SM') &&  ~contains(fileName, 'IM') 
            fnames{end+1} = strcat(folderPath,'/',fileName); 
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
histogram(data, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.5]); 
hold on;

x = linspace(0, max(data), 100);
pdf = normpdf(x, mu, sigma);
plot(x, pdf, 'LineWidth', 2, 'Color', 'k'); 

text(mu*1.5, max(pdf), sprintf('Mean = %.2f ± %.2f s', mu, sigma), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'k');

xlabel('Data');
ylabel('Probability Density');
title(sprintf('Autophagy Time (N=%d)', length(data)));
grid on;
hold off;

%%%%%%%%%%%
set(gcf, 'Color', 'w');

barData = [mean(Diff);mean(Diff_norm)];
errData = [std(Diff)/sqrt(length(Diff)); std(Diff_norm)/sqrt(length(Diff_norm))];
fig = figure;
barHandle = bar(barData, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);

barHandle.CData(1,:) = [0.5 0.5 0.5]; 
barHandle.CData(2,:) = [0 0 0]; 

hold on;
numBars = numel(barData);
for i = 1:numBars
    errorbar(i, barData(i),0, errData(i), 'k', 'LineWidth', 1.5);
    x = i;
    y = barData(i) + max(barData) * 0.1; 
    text(x, y, sprintf('%.2f', barData(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

hold off;

set(gca, 'XTick', 1:length(barData), 'XTickLabel', {'Autophagy', 'Norm'},'Box', 'off', 'Color', 'none');
ylabel('D [um^2/s]');
title('Comparison of D');

set(fig, 'Color', 'w');
grid on; 

ylim([0, max(barData) * 1.2]);
figureandaxiscolors('w','k','')

data = [Diff, Diff_norm]'; 
group = [repmat({'Autophagy'}, 66, 1); repmat({'Norm'}, 66, 1)]; 

figure;
violinplot(data, group);

ylabel('D [\mum^2/s]');
title('Comparison of D');
%% eFig5
t = linspace(1, 100, 100);
% lambda = linspace(500, 700, 1000);

peak1_center = 531;
peak2_center = 571;
width1 = 17;  
width2 = 23;   

% peak1_center = 664;
% peak2_center = 680;
% width1 = 6;   
% width2 = 10;   

gauss = @(x, mu, sigma, amplitude) amplitude * exp(-((x - mu).^2) / (2 * sigma^2));

mGold_spec = normalize(mGoldHaloSpec.mGold,'range')';
JF549_spec = normalize(movmean(mGoldHaloSpec.JF549', 40),'range')';
lambda = mGoldHaloSpec.spec';

initial_amplitude1 = 10;
initial_amplitude2 = 1;

decay = @(t, tau, initial) initial * exp(-t / tau);

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

filter_func = @(x, center, width) exp(-((x-center).^2)/(2*(width/2.355)^2));

donor_int = zeros(length(t),1);
acceptor_int = zeros(length(t),1);

for i = 1:length(t) 
    current_amplitude1 = decay(t(i), 20, initial_amplitude1);  
    current_amplitude2 = decay(t(i), 1e5, initial_amplitude2);  
    gt_int(i) = current_amplitude1 / current_amplitude2;

    noise1 = 0.1 * current_amplitude1 * randn(size(lambda));
    noise2 = 0.1 * current_amplitude2 * randn(size(lambda));
    % noise1 = 0;
    % noise2 = 0;
    
    
%     spectrum_over_time(i, :) = gauss(lambda, peak1_center, width1, current_amplitude1) + ...
%                                gauss(lambda, peak2_center, width2, current_amplitude2) + ...
%                                noise1 + noise2;

    spectrum_over_time(i, :) = mGold_spec .* current_amplitude1 + JF549_spec .* current_amplitude2 + ...
                           noise1 + noise2;
                           
    filtered1 = spectrum_over_time(i, :) .* filter_func(lambda, filter1_center, filter1_width);
    filtered2 = spectrum_over_time(i, :) .* filter_func(lambda, filter2_center, filter2_width);

    sum_filtered1 = sum(filtered1);
    sum_filtered2 = sum(filtered2);
    apd_int(i) = sum_filtered1 / sum_filtered2;
    
    
%     a1 = gauss(lambda, peak1_center, width1, 1);
%     a2 = gauss(lambda, peak2_center, width2, 1);
    a1 = mGold_spec;
    a2 = JF549_spec;
    M = [a1; a2]';
    b_vec = spectrum_over_time(i, :)';

    x = M \ b_vec;  % 解线性方程 Mx = b
    spec_int(i) = x(1) / x(2);
    
    donor_int(i) = current_amplitude1/current_amplitude2;
    % acceptor_int(i) = sum_filtered2;
                      
end

figure;


% spectrum_merge = gauss(lambda, peak1_center, width1, 1) + ...
% gauss(lambda, peak2_center, width2, 1);
spectrum_merge = mGold_spec + JF549_spec;
plot(lambda, spectrum_merge, 'r-');  
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

figure;
plot(lambda, spectrum_over_time(1, :), 'r-', 'DisplayName', 't = 1');  
hold on;
plot(lambda, spectrum_over_time(50, :), 'g-', 'DisplayName', 't = 50'); 
plot(lambda, spectrum_over_time(100, :), 'b-', 'DisplayName', 't = 100'); 

start1 = filter1_center - filter1_width/2;
end1 = filter1_center + filter1_width/2;
start2 = filter2_center - filter2_width/2;
end2 = filter2_center + filter2_width/2;

x_fill = [start1, end1, end1, start1];
y_fill = [0, 0, max(max(spectrum_over_time)), max(max(spectrum_over_time))]; 

fill(x_fill, y_fill, 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); 

x_fill = [start2, end2, end2, start2];
y_fill = [0, 0, max(max(spectrum_over_time)), max(max(spectrum_over_time))]; 
fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); 

xlabel('Wavelength (nm)');
ylabel('Intensity');
title('Spectral Evolution Over Time');
legend show;
grid on;

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
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
legend("gt", "filter", "unmix","3nm-filter")

%% FigS3
x = mGoldHaloSpec.spec;
y_data = {mean(mGoldHaloSpec.pH3,2), mean(mGoldHaloSpec.pH4,2), mean(mGoldHaloSpec.pH5,2), mean(mGoldHaloSpec.pH6,2), ...
    mean(mGoldHaloSpec.pH7,2), mean(mGoldHaloSpec.pH8,2), mean(mGoldHaloSpec.pH9,2), mean(mGoldHaloSpec.pH10,2)};

x_561 = mGoldHaloSpec561.spec;
y_data_561 = {mean(mGoldHaloSpec561.pH3,2), mean(mGoldHaloSpec561.pH4,2), mean(mGoldHaloSpec561.pH5,2), mean(mGoldHaloSpec561.pH6,2), ...
    mean(mGoldHaloSpec561.pH7,2), mean(mGoldHaloSpec561.pH8,2), mean(mGoldHaloSpec561.pH9,2), mean(mGoldHaloSpec561.pH10,2)};

% pH_Range = ["pH 3", "pH 4", "pH 5", "pH 6", "pH 7", "pH 8", "pH 9", "pH 10"];

maxValue = max([y_data{:}], [], 'all');
minValue = min([y_data{:}], [], 'all');
normalize = @(y) (y - minValue) / (maxValue - minValue);

maxValue = max([y_data_561{:}], [], 'all');
minValue = min([y_data_561{:}], [], 'all');
normalize_561 = @(y) (y - minValue) / (maxValue - minValue);

fit_curve_combined = zeros(2401,8);

spec = 500:0.1:740;

for i = 1:8
    y_norm = normalize(y_data{i});
    y_norm_561 = normalize_561(y_data_561{i});
    fit1 = fit(x(1:13,1), y_norm(1:13,1), 'gauss1'); 
    fit2 = fit(x_561, y_norm_561, 'gauss1'); 

    fit_curve1 = feval(fit1, spec);
    fit_curve2 = feval(fit2, spec);

    fit_curve_combined(:,i) = fit_curve1 + fit_curve2;
    % fit_curve_combined(:,i) = fit_curve2;
end

figure
hold on;
maxValue = max(fit_curve_combined, [], 'all');
minValue = min(fit_curve_combined, [], 'all');

colors = cool(8); 
int_mGold = [];
int_Halo = [];
pH_Range = ["pH 3", "pH 4", "pH 5", "pH 6", "pH 7", "pH 8", "pH 9", "pH 10"];

for i = 1:8
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

figure;
hold on;

pH_values = 3:1:10;
scatter(pH_values, int_Halo, 100, 'o', 'MarkerFaceColor', '#4B7BB3', ...
        'DisplayName', 'HaloTag'); 
scatter(pH_values, int_mGold, 100, '^', 'MarkerFaceColor', '#F4BA1D', ...
        'DisplayName', 'mGold'); 

sigmoid = @(b,x) b(1) ./ (1 + exp(-b(2)*(x-b(3))));
initialGuess = [max(int_mGold), 1, mean(pH_values)]; 
opts = optimset('Display', 'off'); 
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(pH_values', int_mGold, sigmoid, initialGuess, opts);
fitValuesGold = sigmoid(beta, 3:0.1:10);

initialGuess = [max(int_Halo), 1, mean(pH_values)]; 
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(pH_values', int_Halo, sigmoid, initialGuess, opts);
fitValuesHalo = sigmoid(beta, 3:0.1:10);

% pHalo = polyfit(pH_values, int_Halo, 3); 
% pGold = polyfit(pH_values, int_mGold, 3); 
% fitValuesVenus = polyval(pHalo, 3:0.1:10);
% fitValuesGold = polyval(pGold, 3:0.1:10);
plot(3:0.1:10, fitValuesHalo, 'Color', '#4B7BB3', 'LineWidth', 3.5, 'HandleVisibility', 'off'); 
plot(3:0.1:10, fitValuesGold, 'Color', '#F4BA1D', 'LineWidth', 3.5, 'HandleVisibility', 'off'); 

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


%% FigS7
close all
plot_traj(trackDataFilt, '');

z=trackDataFilt(:,6);
y=trackDataFilt(:,5);
x=trackDataFilt(:,4);

centerEMCCD = [63,161];
posImg = TR001_yp_xp(300:506,115:321);
plot_traj_emccd(y,x,z,' ', posImg,centerEMCCD)


%% Single molecule trajectory analysis Fig2j-m
warning off
pname = 'E:\data\spms_track\manuscript\Review\data\Fig2\Fig2j-m\';
fileList = ["Atto665N","Setau647N", "Atto565N"];


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

file_names = [];
Diff = [];
Diff_R = [];
traj_Time = [];
traj_Int = [];

Spec_cent = [];
Spec_cent_std = [];


for j = 1:length(fnames)
    close all
    [pname, fname, ~] = fileparts(fnames{j});
    fname = strcat(fname,'.tdms');
    sname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 SM$2 TR$2$3');  
    
    cd(pname)
    [~,dirname,~]=fileparts(fname);
    image_ccd = read_and_rotate_tiff(strcat(regexp(dirname, 'TR\d+', 'match'),'-1.tif'));

    if exist(fullfile(pname,dirname), 'dir')
        cd(fullfile(pname,dirname));  
        load(strcat(dirname,'.mat'));
    else
        [trackData,trackDataFilt] = trajLoadTDMSCalibrated_galvo_v2(1,char(fname),pname);
        cd(fullfile(pname,dirname));  
    end

    z=trackDataFilt(:,6);
    y=trackDataFilt(:,5);
    x=trackDataFilt(:,4);
    traj_i = trackDataFilt(:,11);
    
    if exist('spec_data.mat', 'file')
        load('spec_data.mat');
        % spec_data = AnalysisSpecImg_centroid_v2(image_ccd, fname);
        % plot_spec(traj_i,x,y,z,spec_data,dirname)
        % save("spec_data","spec_data");
    else
        % image_ccd = showEMCCDImg_v2(length(trackDataFilt),1,sname,pname);
        spec_data = AnalysisSpecImg_centroid_v2(image_ccd, fname);
        plot_spec(traj_i,x,y,z,spec_data,dirname)
        save("spec_data","spec_data");
    end
    
    Spec_cent = [Spec_cent, mean(spec_data.raw_centroid)];
    Spec_cent_std = [Spec_cent_std, std(spec_data.raw_centroid)];

    % exposureTime = length(trackDataFilt) / length(spec_data.raw_centroid); % ms 
    % spec_cent = spec_data.raw_centroid;

    track_duration = length(trackDataFilt);
    if track_duration < 500
        continue
    end

    t1 = 100;
    t2 = track_duration - 100;        
    [~,msd,D,r]=msdcalc(x(t1:t2),y(t1:t2),z(t1:t2),1000);
    
    Diff = [Diff, D];
    Diff_R = [Diff_R, r];

    traj_Time = [traj_Time, (track_duration - 200)/1000];
    traj_Int = [traj_Int, mean(traj_i)];
    file_names = [file_names, dirname];

    disp(fname + ' completed!')
end
atto665N_traj_D = Diff(1:32)';
atto665N_traj_d = Diff_R(1:32)'*2;
atto665N_traj_duration = traj_Time(1:32)';
atto665N_traj_Int = traj_Int(1:32)';
atto665N_Spec_cent = Spec_cent(1:32)';
atto665N_Spec_cent_std = Spec_cent_std(1:32)';

setau647_traj_D = Diff(33:67)';
setau647_traj_d = Diff_R(33:67)'*2;
setau647_traj_duration = traj_Time(33:67)';
setau647_traj_Int = traj_Int(33:67)';
setau647_Spec_cent = Spec_cent(33:67)';
setau647_Spec_cent_std = Spec_cent_std(33:67)';


atto565N_traj_D = Diff(68:96)';
atto565N_traj_d = Diff_R(68:96)'*2;
atto565N_traj_duration = traj_Time(68:96)';
atto565N_traj_Int = traj_Int(68:96)';
atto565N_Spec_cent = Spec_cent(68:96)';
atto565N_Spec_cent_std = Spec_cent_std(68:96)';

figure(1)
ax1 = subplot(221);
plot_hist({atto665N_traj_d, setau647_traj_d, atto565N_traj_d}, [0.2,0.2,0.2], ...
    {'atto647N','setau647','atto565N'}, 'count', ax1,'equalInterval')
xlabel('Diameter (nm)')
ylabel('Density')

ax2 = subplot(222);
plot_hist({atto665N_Spec_cent, setau647_Spec_cent, atto565N_Spec_cent}, [1,1,1], ...
    {'atto647N','setau647','atto565N'}, 'count', ax2,'equalInterval')
xlabel('Spec. Cent. (nm)')
ylabel('Density')
truncAxis('X', [600,675])


ax3 = subplot(223);
plot_hist({atto665N_traj_duration, setau647_traj_duration, atto565N_traj_duration}, [1,1,1], ...
    {'atto647N','setau647','atto565N'}, 'count', ax3,'equalInterval')
xlabel('Time (s)')
ylabel('Density')

ax4 = subplot(224);
plot_hist({atto665N_traj_Int/1000, setau647_traj_Int/1000, atto565N_traj_Int/1000}, [0.4,0.4,0.4], ...
    {'atto647N','setau647','atto565N'}, 'count', ax4,'equalInterval')
xlabel('Intensity (kHz)')
ylabel('Density')


%% Positioning accuracy vs spectral accuracy
trackDataFilt = trackDataFilt(500:end-500,:);

exposureTime = length(trackDataFilt) / length(spec_data.raw_centroid); % ms 

z=trackDataFilt(:,6);
y=trackDataFilt(:,5);
x=trackDataFilt(:,4);
spec_cent = spec_data.raw_centroid(2:end-1);

track_duration = length(trackDataFilt);
time_win = 1000;
step_size = 1000;

num_windows = floor((length(x) - time_win) / step_size) + 1;
std_errors_x = zeros(num_windows, 1);
std_errors_y = zeros(num_windows, 1);
std_errors_z = zeros(num_windows, 1);
std_errors_spec = zeros(num_windows, 1);
for i= 3: num_windows-3
    t1 = (i-1) * step_size + 1;
    t2 = t1 + time_win - 1;
    
    % 计算标准差并存储
    std_errors_x(i) = std(x(t1:t2))*1000;
    std_errors_y(i) = std(y(t1:t2))*1000;
    std_errors_z(i) = std(z(t1:t2))*1000;
    
    frame_t1 = round(t1 / exposureTime);
    frame_t2 = min(round(t2 / exposureTime), length(spec_cent));

    std_errors_spec(i) = std(spec_cent(frame_t1:frame_t2));
end

% scatter(std_errors_x(3:end-3), std_errors_spec(3:end-3), 'o','DisplayName', 'std_x');
% hold on
% scatter(std_errors_y(3:end-3), std_errors_spec(3:end-3), 'o','DisplayName', 'std_y');
scatter(std_errors_z(3:end-3), std_errors_spec(3:end-3), 60, 'filled', 'MarkerFaceAlpha',1, 'DisplayName', 'std_z');

xlabel('Z Localizaion Errors (nm)')
ylabel('Spectral Erros (nm)')
grid on


%% FrameRate vs Numbers of Fluorophore
exposureTime = [50, 100, 200 ,500];
% FloNums =[14504*0.05, 10885*0.1, 6625*0.2, 4933*0.5]; % 3nm spec precision
setau647_molecule_Photos = 9.63;
atto647N_molecule_Photos = 8.43;

setau647_Photos = setau647_molecule_Photos * [50, 100, 200 ,500];
atto647N_Photos = atto647N_molecule_Photos * [50, 100, 200 ,500];

int_5nm_spec =  8033*0.1;
% int_5nm_spec =[9614*0.05, 8033*0.1, 4383*0.2, 3444*0.5]; % 5nm spec precision

FloNums_setau =  int_5nm_spec ./ setau647_Photos;

f = fit(exposureTime', FloNums_setau', 'power2');
hold on

x = logspace(0.1, 3, 1000);
plot(x, f(x), 'LineWidth', 1.5);
set(gca, 'XScale', 'log');

scatter(100, 1, 200,'o','filled','DisplayName','setau647N') % 
n = 1.3442e+05/setau647_molecule_Photos/1000;
scatter(10, n, 200,'s','filled','DisplayName','fig2a') % 
n = 5.8732e+04/setau647_molecule_Photos/1000;
scatter(100, n, 200,'d','filled','DisplayName','fig2d') % 
% n = 3.1571e+06/setau647_molecule_Photos/1000;
% scatter(1, n, 80,'d','filled','DisplayName','fig2e') % 
n = 9.0172e+04/setau647_molecule_Photos/1000;
scatter(30, n, 200,'v','filled','DisplayName','fig3a') % 
n = 1.3049e+05/setau647_molecule_Photos/1000;
scatter(30, n, 200,'p','filled','DisplayName','fig4a') % 

grid on
xlabel("Exposure Time (ms)")
ylabel("Numbers of Setau647 molecule")


%% Evaluation of EMCCD positioning accuracy
time = (1:length(spec_data.raw_centroid)) * (length(trackDataFilt)/1000/length(spec_data.raw_centroid));
initialPosition = mean(spec_data.pos,1); 
xyData = spec_data.pos;
displacement = xyData - initialPosition; 

xError = displacement(:, 1) * 0.238; 
yError = displacement(:, 2) * 0.238; 
totalError = sqrt(xError.^2 + yError.^2); 


figure;
subplot(2, 1, 1);
imagesc(time, 1:size(spec_data.trackCurve(:,1:16),2), spec_data.trackCurve(:,1:16)');
colormap('parula'); 
colorbar; 
xlabel('Time (sec)');
ylabel('Pixel shift (px)');

subplot(2, 1, 2);
hold on;
plot(time, xError, 'r-', 'LineWidth', 1, 'DisplayName', 'X Error');
plot(time, yError, 'b-', 'LineWidth', 1, 'DisplayName', 'Y Error');
% plot(time, totalError, 'k-', 'LineWidth', 1, 'DisplayName', 'Total Error');
xlabel('Time (s)');
ylabel('Error (pixels)');
legend('Location', 'Best');
grid on;
ylim([-0.25,0.25])

%%


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


function plot_hist(dataGroups, binCounts, groupLabels, yType, axHandle, binType)
% Input parameters:
% dataGroups - Cellular array containing multiple sets of data (each set of data is a one-dimensional array)
% binCounts - The number or interval of bins in each group of data
% groupLabels - Labels for each set of data (an array of cells of the same length as the dataGroups)
% yType - ordinate type, 'count' or 'pdf' (optional, default is 'count')
% axHandle - Target axis handle (optional, default is the current axis)
% binType - 'equalBinCount' or 'equalInterval' controls the behavior of the bin
%
% Example usage:
%   fig = figure;
%   ax1 = axes(fig);
%   data1 = randn(1, 32) * 10 + 50;  % First set of 32 data
%   data2 = randn(1, 45) * 15 + 60;  The second set of 45 data
%   plot_hist({data1, data2}, [10, 15], {'Group 1', 'Group 2'}, 'count', ax1, 'equalBinCount');

    if nargin < 4 || isempty(yType)
        yType = 'count'; 
    end
    if nargin < 5 || isempty(axHandle)
        axHandle = gca;
    end
    if nargin < 6 || isempty(binType)
        binType = 'equalBinCount'; 
    end

    if nargin < 3 || isempty(groupLabels)
        groupLabels = arrayfun(@(i) sprintf('Group %d', i), 1:length(dataGroups), 'UniformOutput', false);
    end


    colors = lines(length(dataGroups));


    axes(axHandle);
    hold on;

    max_data = 0.0;
    min_data = Inf;

    for i = 1:length(dataGroups)
        if max_data < max(dataGroups{i})
            max_data = max(dataGroups{i});
        end

        if min_data > min(dataGroups{i})
            min_data = min(dataGroups{i});
        end

    end

    for i = 1:length(dataGroups)
        data = dataGroups{i};
        binCount = binCounts(i);

        if strcmp(binType, 'equalBinCount')
            edges = linspace(min(data), max(data), binCount + 1);
        elseif strcmp(binType, 'equalInterval')
            % binCount_ = floor((max(data)-min(data))/binCount);
            % edges = linspace(min(data), max(data), binCount_ + 1);

            binCount_ = floor((max_data-min_data)/binCount);
            edges = linspace(min_data, max_data, binCount_ + 1);
        else
            error('Invalid binType. Use ''equalBinCount'' or ''equalInterval''.');
        end


        if strcmp(yType, 'pdf')
            normalization = 'pdf'; 
            ylabel(axHandle, 'Probability Density');
        elseif strcmp(yType, 'count')
            normalization = 'count';
            ylabel(axHandle, 'Count');
        else
            error('Invalid yType. Use ''count'' or ''pdf''.');
        end


        histogram(data, edges, 'Normalization', normalization, ...
                  'FaceAlpha', 0.5, 'FaceColor', colors(i, :), 'DisplayName', groupLabels{i});


        pd = fitdist(data, 'Normal'); 
        mu = pd.mu; 
        sigma = pd.sigma; 

        x = linspace(min(data), max(data), 1000);
        y = pdf(pd, x); 

 
        if strcmp(yType, 'count')
            y = y * length(data); 
        end

        text(axHandle, mean(x), max(y) * (0.9 - 0.1 * i), ...
             sprintf('\\mu=%.2f, \\sigma=%.2f', mu, sigma), ...
             'Color', colors(i, :), 'FontSize', 10);
    end

    legend(axHandle, 'show');
    grid(axHandle, 'on');
    hold off;
end


function plot_kymograms(imageData, emissionData, time)
    % Kymograms and corresponding Emission curves were drawn

    % kymogram = reshape(mean(imageData, 1), [cols, timeSteps])'; % 按列求平均
    kymogram = imageData;

    figure;

    subplot(2, 1, 1);
    imagesc(time, 1:size(imageData,2), kymogram');
    colormap('parula'); 
    % colorbar;
    xlabel('Time (sec)');
    ylabel('Pixel shift (px)');

    subplot(2, 1, 2);
    plot(time, emissionData, 'b-', 'LineWidth', 1);
    xlabel('Time (sec)');
    ylabel('Emission max (nm)');
    grid on
end

