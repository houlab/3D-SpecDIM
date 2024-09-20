
addpath(genpath('/Users/shahao/projects/spms_track/manuscript/code'))
%%
tic
im = adv_im.import_manual;
im.bin_time = 28;
% create an advanced image object using import_tdms(full_file_path) function

% useful parameters
disp(num2str(im.em_rate, '%.0f')) % emission rate
disp(num2str(im.duration, '%.2f')) % length of traj
disp('Data Loaded')

disp('数据正在分离')
% 按时间间隔为1us分离数据
im.times_fit_interval = 1e-6;
[data2,data_interval] = im.to_mat_v2();
disp('数据分离完成')
toc
%%
save('Imdata.mat', 'data2','-v7.3')
%%
tickTime=12.5e-9;  %

fname = '240719 TR144';
imData_ = [data2(:,3), data2(:,2), data2(:,5), floor(data2(:,12)/tickTime),data2(:,12),...
    data2(:,6), data2(:,7), data2(:,8)];

%1st column - channel (1 = tracking; 0 = imaging; 3 = both)
%2nd column - KT index (pattern coordinates)
%3rd column - TAG Phaseo
%4th column - Ticks
%5th column - Time
%6th column - x
%7th column - y
%8th column - z
%%
counts1 = 28000000;
counts2 = 32500000;

tmp_data = imData_(counts1:counts2,:);

apd1_imData = tmp_data(tmp_data(:,1)==0,:);
apd2_imData = tmp_data(tmp_data(:,1)==1,:);

[imData, ~] = plot_two_APD(apd1_imData, apd2_imData, tickTime, fname);

trackDataFilt_ = trackDataFilt(floor(imData_(counts1,4)*tickTime*1000):floor(imData_(counts2,4)*tickTime*1000),:);

exposure_time = length(trackDataFilt)/length(spec_data.raw_centroid);
startFrame = floor(imData_(counts1,4)*tickTime*1000/exposure_time);
endFrame = floor(imData_(counts2,4)*tickTime*1000/exposure_time);
spec_data_ = crop_spec_data(spec_data,startFrame, endFrame);

%%
plot_traj(trackDataFilt_,fname)
plot_spec(trackDataFilt_(:,11),trackDataFilt_(:,4),trackDataFilt_(:,5),trackDataFilt_(:,6), spec_data_, '240719 TR144');
%%

dataFinal=[];
% tickTime=10e-6;  %嘀嗒时间和FPGA程序的时钟一致
TAGfreq=70000;
KTBinTime=20e-6;
xkt=[1,2,1,3,5,4,5,3,4,5,3,1,2,4,5,4,2,1,3,5,4,2,3,2,1]-3; % x-scan pattern
ykt=[5,3,1,2,1,3,5,4,2,4,5,4,2,1,3,5,4,2,1,2,4,5,3,1,3]-3; % y-scan pattern
ImData=imData(1:end,:); 
SegPhotons = 1e6; % 每个段处理的光子数量
frame_time = 1; % 1000 ms

% 处理数据
[M, pos, seg_pos, seg_t, int] = processData(ImData, SegPhotons, ...
    xkt, ykt, tickTime, TAGfreq, trackDataFilt_, frame_time);

%%
save("M.mat",'M', 'pos', 'seg_pos', 'seg_t', 'int','counts1','counts2')

%% 从切片数据中生成三维数组
% 重建代码（创建一个三维空间将像素按轨迹放入其中）
bin_size = [5, 5, 10];
num_bins = size(M,3)/10;
MM = zeros([bin_size, num_bins]);

% 将M中的数据放入四维数组
for t = 1:num_bins
    MM(:, :, :, t) = M(:,:,(t-1)*10+1:(t*10));
end

xySize = 0.2;
zSize = 0.18;
cloud_ = combineVoxels(MM, pos, xySize, zSize);

%% 
plot_project(cloud_);

%%
figure
timePrecision = 0.01;

ticksbtw=[1 diff(apd1_imData(:,4))']; % trans ticks to time
satuposition=find(ticksbtw<0);
ticksbtw(satuposition)=2^32+ticksbtw(satuposition)+1; % cycle point
Totalticks=cumsum(ticksbtw); % accumulated times 
apd1_imData(:,4)=Totalticks;
a=ticksbtw==2;
apd1_imData(a,:)=[];

endTimeApd2 = max(apd1_imData(:,4) * tickTime);
timeBins = 0:timePrecision:endTimeApd2; %ms
[int_apd1, ~] = histcounts(apd1_imData(:,4) * tickTime, timeBins);
int_apd1 = int_apd1 / timePrecision;

time_apd1=(1:length(int_apd1))*timePrecision;
int_lyso_emccd = mean(spec_data_.trackCurve(:,60:80),2);
time_cent=(1:length(cent))*(time_apd1(end)/length(cent));

yyaxis left
plot(time_cent,int_lyso_emccd,'r','LineWidth',1.5)
xlabel('Time [sec]')
ylabel('Intensity [counts/sec]')
figureandaxiscolors('w','k',strcat(fname, ' APD2'))


yyaxis right
cent = spec_data_.raw_centroid;

plot(time_cent,cent,'b','LineWidth',1.5)

%%
outputDir = './output_two apd';
save_Tiff(cloud_, outputDir)
%%
hit_centroid = min(roundn(spec_data_.raw_centroid,-2)):0.01: ...
        max(roundn(spec_data_.raw_centroid,-2));

% plot video
cloud = zeros(size(cloud_));
hScatter = figure;
centroid_color = colormap(parula(length(hit_centroid)));

pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]
pos2 = [0.6, 0.5, 0.35, 0.35];
pos3 = [0.1, 0.1, 0.25, 0.25];
pos4 = [0.4, 0.1, 0.25, 0.25];
pos5 = [0.7, 0.1, 0.25, 0.25];

ax1 = axes('Position', pos1);
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([min(seg_pos(:,1)) max(seg_pos(:,1))]);
ylim([min(seg_pos(:,2)) max(seg_pos(:,2))]);
zlim([min(seg_pos(:,3)) max(seg_pos(:,3))]);
colorbar;
caxis([0 sum(seg_t)])
cjet=colormap(ax1, parula(length(spec_data_.raw_centroid)));
title('3D Photon Intensity Over Time');
view([1 1 1])
grid on
axis equal
hold on

 %%%%%%%%%%%%%%%%%%
ax2 = axes('Position', pos2);
% cjet=colormap(parula(length(hit_centroid)));
% cjlen=length(cjet);

time=(1:length(seg_pos(:,1)))/1000;
% figureandaxiscolors('w','k',dirname)
xlim([0 round(max(time))])
ylim([min(hit_centroid) max(hit_centroid)])
xlabel('Time(s)')
ylabel('Centroids')
grid on
hold on
% figTrajname=[dirname ' pH ratio' '.fig'];
% saveas(hSpec,figTrajname,'fig');


projection_xoy = squeeze(sum(cloud_, 3));
projection_xoz = squeeze(sum(cloud_, 2));
projection_yoz = squeeze(sum(cloud_, 1));

max_xoy = max(projection_xoy(:));
min_xoy = min(projection_xoy(:));
max_xoz = max(projection_xoz(:));
min_xoz = min(projection_xoz(:));
max_yoz = max(projection_yoz(:));
min_yoz = min(projection_yoz(:));

xRange = linspace(min(seg_pos(:,1)), max(seg_pos(:,1)), size(cloud,1));
yRange = linspace(min(seg_pos(:,2)), max(seg_pos(:,2)), size(cloud,2));
zRange = linspace(min(seg_pos(:,3)), max(seg_pos(:,3)), size(cloud,3));


[Xz, Yz] = ndgrid(xRange, yRange);
[Xy, Zy] = ndgrid(xRange, zRange);
[Yx, Zx] = ndgrid(yRange, zRange);

plasma_map = load('plasma_colormap.mat');
plasma_map = plasma_map.plasma;
ax3 = axes('Position', pos3);
xlabel('X');
ylabel('Y');
colorbar;
colormap(ax3,plasma_map);
caxis([min_xoy max_xoy])
grid on
hold on
axis tight

ax4 = axes('Position', pos4);
xlabel('X');
ylabel('Z');
colorbar;
colormap(ax4,plasma_map);
caxis([min_xoz max_xoz])
grid on
hold on
axis tight

ax5 = axes('Position', pos5);
xlabel('X');
ylabel('Z');
colorbar;
colormap(ax5,plasma_map);
caxis([min_yoz max_yoz])
grid on
hold on
axis tight

raw_centroids = spec_data_.raw_centroid;
cen_len = numel(raw_centroids);
len=length(seg_pos(:,1));
spec_time=time(round(linspace(1, len, cen_len)));
%

j = 1;

% video_name = strcat(dirname,'_MosaicImaging.mp4');
% outputVideo = VideoWriter(video_name, 'MPEG-4');
% open(outputVideo);

projection_xoy = squeeze(sum(cloud_, 3));
x = Xz((projection_xoy > 0));
y = Yz((projection_xoy > 0));
photons = projection_xoy(projection_xoy > 0);
scatter(ax3, x, y,15,photons,'filled', 'MarkerFaceAlpha',1); 

projection_xoz = squeeze(sum(cloud_, 2));
x = Xy((projection_xoz > 0));
z = Zy((projection_xoz > 0));
photons = projection_xoz(projection_xoz > 0);
scatter(ax4, x, z,15,photons,'filled', 'MarkerFaceAlpha',1); 

projection_yoz = squeeze(sum(cloud_, 1));
y = Yx((projection_yoz > 0));
z = Zx((projection_yoz > 0));
photons = projection_yoz(projection_yoz > 0);
scatter(ax5, y, z,15,photons,'filled', 'MarkerFaceAlpha',1); 

plot(ax2, spec_time(1,:), raw_centroids,'r','LineWidth',1)

for i = 1:size(spec_time, 2)
    
    seg=(floor((i-1)*length(seg_pos)/length(spec_time))+1):floor((i)*length(seg_pos)/length(spec_time));
    p1=plot3(ax1, seg_pos(seg,1),seg_pos(seg,2),seg_pos(seg,3));
    set(p1,'Color',cjet(i,:),'LineWidth',1);

    % seg_ = ((floor((j-1)*cen_len/length(pos)*10)+1):floor((j)*cen_len/length(pos)*10));
    cid=find(abs(hit_centroid-roundn(raw_centroids(i),-2)) < 1e-3);
    plot(ax3, seg_pos(seg,1),seg_pos(seg,2), 'Color',[centroid_color(cid,:) 1],'LineWidth',0.5);
    plot(ax4, seg_pos(seg,1),seg_pos(seg,3), 'Color',[centroid_color(cid,:) 1],'LineWidth',0.5);
    plot(ax5, seg_pos(seg,2),seg_pos(seg,3), 'Color',[centroid_color(cid,:) 1],'LineWidth',0.5);

%     %%%%%
%     drawnow;
%     pause(0.001);
%     currFrame = getframe(hScatter);
%     writeVideo(outputVideo, currFrame);
%     %%%%%%%%
end

saveas(hScatter ,strcat(fname, ' mosaicImaging'), 'fig')
% 
%%
figure
for i = 1:size(pos, 1)
    value = MM(:,:,:,i);
    imagesc(sum(value, 3));
    pause(0.1)
end
imagesc(sum(sum(MM(:,:,:,:), 4),3));
%%
% centroid_color = colormap(summer(length(hit_centroid)));


image_size = size(cloud_); 
x_min = min(seg_pos(:,1));
x_max = max(seg_pos(:,1));
y_min = min(seg_pos(:,2));
y_max = max(seg_pos(:,2));
z_min = min(seg_pos(:,3));
z_max = max(seg_pos(:,3));

% 2. 计算每个维度的像素尺寸
pix_sizeX = (x_max - x_min) / image_size(1);
pix_sizeY = (y_max - y_min) / image_size(2);
pix_sizeZ = (z_max - z_min) / image_size(3);

% 3. 将轨迹数据转换为图像的像素坐标
x_pixel = (seg_pos(:,1) - x_min) / pix_sizeX + 1;
y_pixel = (seg_pos(:,2) - y_min) / pix_sizeY + 1;
z_pixel = (seg_pos(:,3) - z_min) / pix_sizeZ + 1;

% 确保像素坐标不超出图像边界
x_pixel = max(min(x_pixel, image_size(1)), 1);
y_pixel = max(min(y_pixel, image_size(2)), 1);
z_pixel = max(min(z_pixel, image_size(3)), 1);

% 在不同平面上进行可视化
pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]
pos2 = [0.6, 0.5, 0.35, 0.35];
pos3 = [0.1, 0.1, 0.25, 0.25];
pos4 = [0.4, 0.1, 0.25, 0.25];
pos5 = [0.7, 0.1, 0.25, 0.25];

figure

data = load('plasma_colormap.mat');
plasma_map = data.plasma;

% XOY平面
ax3 = axes('Position', pos3);
xlabel('X');
ylabel('Y');
img_xy = squeeze(sum(cloud_, 3)); % 提取XY平面图像
% img_xy_resized = imresize(img_xy, output_size); % 缩放图像
imshow(img_xy, []);
colormap(ax3, plasma_map); % 设置自定义colormap为plasma
title('XOY Plane');
hold on;

% YOZ平面
ax4 = axes('Position', pos4);
xlabel('X');
ylabel('Z');
img_yz = squeeze(sum(cloud_, 1)); % 提取YZ平面图像
% img_yz_resized = imresize(img_yz, output_size); % 缩放图像
imshow(img_yz, []);
colormap(ax4, plasma_map); % 设置自定义colormap为plasma
title('YOZ Plane');
hold on;

% XOZ平面
ax5 = axes('Position', pos5);
xlabel('X');
ylabel('Z');
img_xz = squeeze(sum(cloud_, 2)); % 提取XZ平面图像
% img_xz_resized = imresize(img_xz, output_size); % 缩放图像
imshow(img_xz, []);
colormap(ax5, plasma_map); % 设置自定义colormap为plasma
title('XOZ Plane');
hold on;

% 绘制轨迹
for i = 1:size(spec_time, 2)
    seg = (floor((i-1)*length(seg_pos)/length(spec_time))+1):floor(i*length(seg_pos)/length(spec_time));
    cid = find(abs(hit_centroid-roundn(raw_centroids(i),-2)) < 1e-3);
    plot(ax3, y_pixel(seg), x_pixel(seg), 'Color', [centroid_color(cid,:) 1], 'LineWidth', 0.75);
    plot(ax4, z_pixel(seg), y_pixel(seg), 'Color', [centroid_color(cid,:) 1], 'LineWidth', 0.75);
    plot(ax5, z_pixel(seg), x_pixel(seg), 'Color', [centroid_color(cid,:) 1], 'LineWidth', 0.75);
end

%%

function [apd1_imData, apd2_imData] = plot_two_APD(apd1_imData, apd2_imData, tickTime, fname)


    ticksbtw=[1 diff(apd1_imData(:,4))']; % trans ticks to time
    satuposition=find(ticksbtw<0);
    ticksbtw(satuposition)=2^32+ticksbtw(satuposition)+1; % cycle point
    Totalticks=cumsum(ticksbtw); % accumulated times 
    apd1_imData(:,4)=Totalticks;
    a=ticksbtw==2;
    apd1_imData(a,:)=[];
    
    ticksbtw=[1 diff(apd2_imData(:,4))']; % trans ticks to time
    satuposition=find(ticksbtw<0);
    ticksbtw(satuposition)=2^32+ticksbtw(satuposition)+1; % cycle point
    Totalticks=cumsum(ticksbtw); % accumulated times 
    apd2_imData(:,4)=Totalticks;
    a=ticksbtw==2;
    apd2_imData(a,:)=[];
    
    [~,dirname,~]=fileparts(fname);
    
    timePrecision = 0.01;
    endTimeApd1 = max(apd1_imData(:,4) * tickTime);
    timeBins = 0:timePrecision:endTimeApd1; %ms
    [int_apd1, ~] = histcounts(apd1_imData(:,4) * tickTime, timeBins);
    int_apd1 = int_apd1 / timePrecision;
    
    endTimeApd2 = max(apd2_imData(:,4) * tickTime);
    timeBins = 0:timePrecision:endTimeApd2; %ms
    [int_apd2, ~] = histcounts(apd2_imData(:,4) * tickTime, timeBins);
    int_apd2 = int_apd2 / timePrecision;
    
    time_apd2=(1:length(int_apd2))*timePrecision;
    time_apd1=(1:length(int_apd1))*timePrecision;
    
    hALLinOne = figure;
    pos1 = [0.1, 0.55, 0.35, 0.35]; % [left bottom width height]
    pos2 = [0.6, 0.55, 0.35, 0.35];
    
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
    saveas(hALLinOne,strcat(dirname, ' two APD'), 'fig');

    apd1_imData(:,1)=[];
    apd2_imData(:,1)=[];

    
end

function [M, delta, seg_pos, seg_t, Int] = processData(imData, SegPhotons,  ...
    xkt, ykt, tickTime, TAGfreq, trackData, frame_time)
    M = [];
    delta = [];
    seg_pos = [];
    seg_t = [];
    Int = [];
    numSegments = floor(size(imData, 1) / SegPhotons);
    t1 = 0;
    
    for k = 1:numSegments+1
        segmentStart = (k-1) * SegPhotons + 1;
        segmentEnd = min(k * SegPhotons, size(imData, 1));
        segment = imData(segmentStart:segmentEnd, :);
        
        % 处理每个段
        [M_, delta_, seg_pos_, t_, int_] = calculateProbabilityMap(segment, tickTime, TAGfreq, ...
            xkt, ykt, trackData, frame_time,t1);

        M = cat(3, M, M_);
        delta = cat(1, delta, delta_);
        seg_pos = cat(1, seg_pos, seg_pos_);
        t1 = max(t_);

        
        seg_t = [seg_t max(t_)];
        Int =  cat(1, Int, int_);
        fprintf("%d/%d finished!\n", k, numSegments);
%         delta = cat(1, delta, [delta_x,delta_y,delta_z]);
%         resultData = [resultData; processedSegment];
    end



end

function [M, delta, seg_pos, t, int] = calculateProbabilityMap(ImData, tickTime, TAGfreq, ...
    xkt, ykt, trackData, frame_time,t1)
    Xk = xkt(ImData(:,1) + 1);
    Yk = ykt(ImData(:,1) + 1);

    % 对TAGphase进行时间修正
    TAGphase=ImData(:,2);
    cx1 = cumsum(628 * tickTime * TAGfreq * ones(size(TAGphase))); % 一个正弦周期对应
    cx2 = zeros(size(cx1));
    i = find(~~TAGphase);
    if ~isempty(i)
        cx2(i) = diff([0; cx1(i) - TAGphase(i)]);
    end
    x3 = mod(cx1 - cumsum(cx2)+ 116, 628);
    x3(~~TAGphase) = TAGphase(~~TAGphase);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phase_edges = linspace(0, 2*pi, 21);
    % 计算每个相位区间的 z 值范围
    z_values = sin(phase_edges);
    z_lengths = abs(diff(z_values));  % 每个区间的长度
    
    % 计算校正因子，将 z_lengths 归一化
    sampling_factors = z_lengths ./ max(z_lengths);
    

    phase_data = x3/100;
    n = length(phase_data);
    corrected_phase_data = NaN(n, 1);

    % 对于每一个实际相位数据，按照其所在区间的校正因子进行降采样
    for i = 1:length(phase_edges) - 1
        % 找到属于当前相位区间的实际数据
        idx = find(phase_data >= phase_edges(i) & phase_data < phase_edges(i + 1));
        
        if ~isempty(idx)
            % 计算保留的样本数
            num_samples = round(length(idx) * sampling_factors(i));
            
            % 随机选择样本进行保留
            if num_samples > 0
                sampled_idx = idx(randperm(length(idx), num_samples));
                corrected_phase_data(sampled_idx) = phase_data(sampled_idx);
            end
        end
    end
    reidx = find(isnan(corrected_phase_data));
    tmp_data = corrected_phase_data(~isnan(corrected_phase_data));
    corrected_phase_data(reidx) = tmp_data(randi(length(tmp_data), length(reidx), 1));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 最终数据准备    
    dataFinal(:,1) = Xk + 3; % Xk
    dataFinal(:,2) = Yk + 3;
    dataFinal(:,3) = floor((sin(corrected_phase_data)+1)*5)+1;
    dataFinal(:,4) = ImData(:,3); % Tick
    dataFinal(:,5) = ImData(:,4); % Time

    [M, delta, seg_pos, t, int] = calculateParticleInfo(trackData, dataFinal, frame_time, tickTime,t1);
end

function [M, delta, seg_pos, t, int] = calculateParticleInfo(trackData, dataFinal, frameTime, tickTime,t1)
    x = dataFinal(:,1);
    y = dataFinal(:,2);
    z = dataFinal(:,3);
    t = dataFinal(:,4) * tickTime; % time in seconds

    M = [];
    delta = [];
    seg_pos = [];%
    int = [];
    
    for i = t1: frameTime: max(t) % total number of frames
        seg = floor(i * 1e3+1) : min(floor((i + frameTime) * 1e3),size(trackData,1));
        position = find((t >= i) & (t < i + frameTime));     
%         per_frame_time = floor(t(position) * 1e3) + 1; % ms

        delta_x = mean(trackData(seg, 4)); % Assuming trackData has x, y, z columns as 4th, 5th, and 6th.
        delta_y = mean(trackData(seg, 5));
        delta_z = mean(trackData(seg, 6));

        x_seg = trackData(seg, 4);
        y_seg = trackData(seg, 5);
        z_seg = trackData(seg, 6);

        % Accumulate emission data to form probability map
        out = accumarray([x(position) y(position) z(position)], 1);
%         outNorm = accumarray([x(range) y(range) z(range)], 1);
%         ArrayFinal = out ./ outNorm;
% 
%         outNorm = accumarray([x(range) y(range) z(range)], 1);
        ArrayFinal = out;

        M = cat(3, M, ArrayFinal);
        delta = cat(1, delta, [delta_x,delta_y,delta_z]);
        seg_pos = cat(1, seg_pos, [x_seg,y_seg,z_seg]);

        int = cat(1,int,sum(out(:)/frameTime));

        disp(['Photon Emission Rate = ' num2str(sum(out,'all')/frameTime/1000) 'kHz'])
    end
end

function combinedVolume = combineVoxels(voxelData, centers, xySize, zSize)
    % voxelData: 4D array of size [5, 5, 11, 602]
    % centers: 3D coordinate array of size [602, 3]
    % xySize, zSize: physical dimensions of each voxel

    % Determine the bounds of the full volume
    xCoords = centers(:,1);
    yCoords = centers(:,2);
    zCoords = centers(:,3);

    minX = min(xCoords);
    maxX = max(xCoords);
    minY = min(yCoords);
    maxY = max(yCoords);
    minZ = min(zCoords);
    maxZ = max(zCoords);

    % Calculate the size of the full volume array
    volumeSizeX = ceil((maxX - minX + xySize * 5) / xySize);
    volumeSizeY = ceil((maxY - minY + xySize * 5) / xySize);
    volumeSizeZ = ceil((maxZ - minZ + zSize * 10) / zSize);

    % Initialize the full volume array
    combinedVolume = zeros(volumeSizeX, volumeSizeY, volumeSizeZ);

    % Half sizes of the voxel dimensions (assuming center is the middle point)
    halfVoxelX = xySize * 2.5;
    halfVoxelY = xySize * 2.5;
    halfVoxelZ = zSize * 5;

    % Iterate through each voxel and place it in the correct position
    for i = 1:size(centers, 1)
        % Calculate the starting index in the full volume
        startX = round((xCoords(i) - minX + halfVoxelX) / xySize) - 2;
        startY = round((yCoords(i) - minY + halfVoxelY) / xySize) - 2;
        startZ = round((zCoords(i) - minZ + halfVoxelZ) / zSize) - 5;

        % Place the voxel data into the full volume
        value = voxelData(:, :, :, i);
        raw_value = combinedVolume(startX + 1:startX + 5, startY + 1:startY + 5, startZ + 1:startZ + 10);
        
        mean_value = zeros(size(value));
        mean_value(raw_value~=0) = ((raw_value(raw_value~=0)+ value(raw_value~=0)))/2;
        mean_value(raw_value==0) = value(raw_value==0);
        
        combinedVolume(startX + 1:startX + 5, startY + 1:startY + 5, startZ + 1:startZ + 10) = max(value, raw_value);
        % combinedVolume(startX + 1:startX + 5, startY + 1:startY + 5, startZ + 1:startZ + 10) = mean_value;
    end

%     [~, ~, z] = ind2sub(size(combinedVolume), find(combinedVolume > 0));
%     % 计算维度的边界
%     minZ = min(z);
%     maxZ = max(z);
%     % 裁剪数组
%     combinedVolume = double(combinedVolume(:, :, minZ:maxZ));
%     combinedVolume = rescale(combinedVolume, 0, 255);
end

function plot_project(cloud)
    % 计算强度值投影
    % 计算 xoy 平面的投影 (在 z 方向上进行求和)
    projection_xoy = sum(cloud, 3);
    projection_xoy = (projection_xoy - min(projection_xoy(:))) / (max(projection_xoy(:)) - min(projection_xoy(:)));  % 归一化到 [0, 1]
    
    % 计算 xoz 平面的投影 (在 y 方向上进行求和)
    projection_xoz = sum(cloud, 2);
    projection_xoz = squeeze(projection_xoz);  % 去掉多余的维度
    projection_xoz = (projection_xoz - min(projection_xoz(:))) / (max(projection_xoz(:)) - min(projection_xoz(:)));  % 归一化到 [0, 1]
    
    % 计算 yoz 平面的投影 (在 x 方向上进行求和)
    projection_yoz = sum(cloud, 1);
    projection_yoz = squeeze(projection_yoz);  % 去掉多余的维度
    projection_yoz = (projection_yoz - min(projection_yoz(:))) / (max(projection_yoz(:)) - min(projection_yoz(:)));  % 归一化到 [0, 1]
    
    % 显示 xoy 平面的投影
    figure;
    imagesc(projection_xoy);
    colorbar;
    title('xoy平面强度值投影');
    xlabel('x');
    ylabel('y');
    axis equal;
    axis tight;
    caxis([0 1]);  % 将颜色轴限制在 [0, 1]
    set(gca, 'XTick', [], 'YTick', []);  % 移除 x 和 y 轴上的数字
    
    % 显示 xoz 平面的投影
    figure;
    imagesc(projection_xoz');
    colorbar;
    title('xoz平面强度值投影');
    xlabel('x');
    ylabel('z');
    axis equal;
    axis tight;
    caxis([0 1]);  % 将颜色轴限制在 [0, 1]
    set(gca, 'XTick', [], 'YTick', []);  % 移除 x 和 z 轴上的数字
    
    % 显示 yoz 平面的投影
    figure;
    imagesc(projection_yoz');
    colorbar;
    title('yoz平面强度值投影');
    xlabel('y');
    ylabel('z');
    axis equal;
    axis tight;
    caxis([0 1]);  % 将颜色轴限制在 [0, 1]
    set(gca, 'XTick', [], 'YTick', []);  % 移除 y 和 z 轴上的数字
end

function save_Tiff(cloud, outputDir)
    % 获取cloud的深度（z轴的大小）
    numSlices = size(cloud, 3);
    
    % 检查目录是否存在，如果不存在，则创建
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % 遍历每一层，并保存为 TIFF 文件
    for k = 1:numSlices
    %     文件名
        filename = fullfile(outputDir, sprintf('slice_%d.tif', k));
    %     提取当前层
        slice = uint16(cloud(:,:,k));
    %     保存为 TIFF 文件
        imwrite(slice, filename);
    end

end



