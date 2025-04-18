
addpath(genpath('/Users/shahao/projects/spms_track/manuscript/code'))
%%
[fname,pname]=uigetfile('*.TDMS');
sname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 SM$2 TR$2$3');  
iname = regexprep(fname, '(\d{6}) TR(\d+)(\.tdms)', '$1 IM$2 TR$2$3');

%%
[trackData,trackDataFilt] = trajLoadTDMSCalibrated_v2_1(1,fname,pname);

%%
image_ccd = showEMCCDImg(length(trackDataFilt),1,sname,pname);

%%
tickTime=10e-6;  %
imData = load_imgdata(fullfile(pname,iname));

apd1_imData = imData(imData(:,1)==1,:);
apd2_imData = imData(imData(:,1)==2,:);

ticksbtw=[1 diff(apd1_imData(:,4))']; % trans ticks to time
satuposition=find(ticksbtw<0);
ticksbtw(satuposition)=2^32+ticksbtw(satuposition)+1; % cycle point
Totalticks=cumsum(ticksbtw); % accumulated times 
apd1_imData(:,4)=Totalticks * tickTime;

ticksbtw=[1 diff(apd2_imData(:,4))']; % trans ticks to time
satuposition=find(ticksbtw<0);
ticksbtw(satuposition)=2^32+ticksbtw(satuposition)+1; % cycle point
Totalticks=cumsum(ticksbtw); % accumulated times 
apd2_imData(:,4)=Totalticks * tickTime;
%%
image_ccd = image_ccd(:,:,1:end);

z=trackDataFilt(1:end,6);
y=trackDataFilt(1:end,5);
x=trackDataFilt(1:end,4);
traj_int = trackDataFilt(1:end,1);

num_frames = length(image_ccd);
len=length(x);

exposure_time = len / num_frames / 1000;
spec_data = AnalysisSpecImg_centroid(image_ccd, fname);
save("spec_data","spec_data");

%%
img_488 = spec_data.trackImg(:,17:38,:);
img_561 = spec_data.trackImg(:,39:59,:);
img_640 = spec_data.trackImg(:,60:80,:);

trackCurve = [spec_data.trackCurve(:,1:16)';...
              squeeze(normalize(mean(img_488, 1),'range')); ...
              squeeze(normalize(mean(img_561, 1),'range')); ...
              squeeze(normalize(mean(img_640, 1),'range'))];

centroids_488 = zeros(size(trackCurve,2),1);
centroids_561 = zeros(size(trackCurve,2),1);
centroids_640 = zeros(size(trackCurve,2),1);

for i = 1:size(trackCurve,2)
    centroids_488(i) = fitCentroid_v3(trackCurve(1:38,i),'488');
    centroids_561(i) = fitCentroid_v3([trackCurve(1:16,i); trackCurve(39:59,i)],'561');
    centroids_640(i) = fitCentroid_v3([trackCurve(1:16,i); trackCurve(60:80,i)],'640');
end

%%
[~,dirname,~]=fileparts(fname);

timePrecision = 0.01;
endTimeApd1 = max(apd1_imData(:,4));
timeBins = 0:timePrecision:endTimeApd1; %ms
[int_apd1, ~] = histcounts(apd1_imData(:,4), timeBins);
int_apd1 = int_apd1 / timePrecision;

[int_apd2, ~] = histcounts(apd2_imData(:,4), timeBins);
int_apd2 = int_apd2 / timePrecision;

time_apd2=(1:length(int_apd2))*timePrecision;
time_apd1=(1:length(int_apd1))*timePrecision;

hALLinOne = figure;
pos1 = [0.1, 0.55, 0.35, 0.35]; % [left bottom width height]
pos2 = [0.6, 0.55, 0.35, 0.35];
pos3 = [0.1, 0.1, 0.35, 0.35]; % [left bottom width height]
pos4 = [0.6, 0.1, 0.35, 0.35];

ax1 = axes('Position', pos1);
plot(ax1, time_apd1,int_apd1,'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Intensity [counts/sec]')
figureandaxiscolors('w','k',strcat(dirname, ' APD1'))

ax2 = axes('Position', pos2);
plot(ax2, time_apd2,int_apd2,'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Intensity [counts/sec]')
figureandaxiscolors('w','k',strcat(dirname, ' APD2'))
%%
time=(1:length(trackDataFilt(:,1)))/1000;
cen_len = size(trackCurve,2);
len = length(trackDataFilt(:,1));
spec_time=time(round(linspace(1, len, cen_len)));
ax3 = axes('Position', pos3);
plot(ax3, spec_time, centroids_561, 'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Centroids [nm]')
figureandaxiscolors('w','k',strcat(dirname, ' 561 channel'))

ax4 = axes('Position', pos4);
plot(ax4, spec_time,centroids_488,'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Centroids [nm]')
figureandaxiscolors('w','k',strcat(dirname, ' 488 channel'))
saveas(hALLinOne,strcat(dirname, ' two APD'), 'fig');

%%
apd_ratio = int_apd1 ./ int_apd2;
spec_ratio = centroids_561 ./ centroids_488;

hALLinOne = figure;

pos1 = [0.1, 0.55, 0.35, 0.35]; % [left bottom width height]
pos2 = [0.6, 0.55, 0.35, 0.35];
pos3 = [0.1, 0.1, 0.35, 0.35]; % [left bottom width height]
pos4 = [0.6, 0.1, 0.35, 0.35];

ax1 = axes('Position', pos1);
plot(ax1, time_apd1,apd_ratio,'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Intensity [counts/sec]')
figureandaxiscolors('w','k',strcat(dirname, ' APD ratio'))

ax3 = axes('Position', pos3);
plot(ax3, spec_time, spec_ratio, 'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Intensity [counts/sec]')
figureandaxiscolors('w','k',strcat(dirname, ' spec ratio'))


mean_apd = mean(apd_ratio);
std_apd = std(apd_ratio);

mean_spec = mean(spec_ratio);
std_spec = std(spec_ratio);

cv_apd = std(apd_ratio)/mean(apd_ratio);
cv_spec = std(spec_ratio)/mean(spec_ratio);
ax2 = axes('Position', pos2);
b = bar(ax2, [cv_apd, cv_spec], 'FaceColor', 'flat', 'EdgeColor','k', 'LineWidth',1.5);
b.FaceColor = 'flat';
b.CData(1,:) = [0 0 0]; % 第一个变异系数的灰度色
b.CData(2,:) = [0.5 0.5 0.5]; % 第二个变异系数的灰度色
b.CData(1,:) = [0 0 0];

text(1, cv_apd*1.05, sprintf('%.4f', cv_apd), 'Color' ,'k', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(2, cv_spec*1.05, sprintf('%.4f', cv_spec),'Color' ,'k', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

set(gca, 'XTickLabel', {'apd\_ratio', 'spec\_ratio'}, 'XTick', 1:2);
ylabel('Coefficient of Variation (CV)');
title('Comparison of Coefficient of Variation');
grid on
figureandaxiscolors('w','k',strcat('Comparison of Coefficient of Variation'))
saveas(hALLinOne,strcat(dirname, ' comparison two APD'), 'fig');


%%
%%%%%%%%%%%%%%
function [imData]=load_imgdata(fname)
    % load trajectory data
    
    tic
    tdms_struct = TDMS_getStruct(fname);
    disp('imgData Loaded')
    toc
    
    fn = fieldnames(tdms_struct);
    imData=tdms_struct.(fn{2});
    % preprocess the imData 
    imData=[double((imData.Channel.data)') double((imData.KT_Index.data)') ...
        double((imData.TAG_Phase.data)') double((imData.Ticks.data)')];
    
end

%Decimates trackdata
function tempDataFilt=dectrackdata(tempData,factor)
    s=size(tempData);
    for j=1:s(2);
        tempDataFilt(:,j)=decimate(tempData(:,j),factor);
    end
end


function centroid=fitCentroid_v3(specCurve, channel)
    % 自定义双峰高斯分布模型
    gaussianModel = @(p, x) p(1) * exp(-((x - p(2)) / p(3)).^2);

    % 初始参数猜测
    initialGuess_psf = [1, 8, 0.5];   
    psf_curve = specCurve(1:16);
    spec_curve = specCurve(17:end);

    [~,init_value]= max(spec_curve);
    initialGuess_spec = [1, init_value, 0.5];

    try
        % 使用 fit 函数进行拟合
        nlModel_psf = fitnlm((1:length(psf_curve))',psf_curve',gaussianModel,initialGuess_psf);
        fitRMSE_psf = nlModel_psf.RMSE;

        nlModel_spec  = fitnlm((1:length(spec_curve))',spec_curve',gaussianModel,initialGuess_spec);
        fitRMSE_spec = nlModel_spec.RMSE;
    catch
        centroid = 0;
        % disp("can not fit the curve!")
        return
    end
    pos = nlModel_psf.Coefficients.Estimate(2);
    spec = nlModel_spec.Coefficients.Estimate(2)+16;

    pix_dis = spec - pos;

    if pos > 16 || pos < 1 || spec > 38 || spec < 16 ||...
            pix_dis > 38 || pix_dis < 0 || fitRMSE_spec > 0.4 || fitRMSE_psf > 0.1
        centroid = 0;
        return
    end

    f_sp = @(x)(0.02215*x.^2+3.077*x+591.6);

    if contains(channel, '488') 
        centroid = f_sp(pix_dis-50);
    elseif contains(channel, '561') 
        centroid = f_sp(pix_dis-20);
    elseif contains(channel, '640') 
        centroid = f_sp(pix_dis+3);
    end

    
end

