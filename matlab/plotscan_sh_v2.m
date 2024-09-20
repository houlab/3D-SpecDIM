clc
warning off

%%
[fname,pname]=uigetfile('*.*');
cd(pname);

figure(6)
int=load('DIO0.dat');
int(isinf(int))=0;
int(isnan(int))=0;
int=int';
mark=im2bw(int./max(int(:)),0.1);

imshow(flipud(rot90(int)),[])

%%
folder = './';  % 替换成你的文件夹路径
filePattern = fullfile(folder, '*.tdms');  % 指定后缀为 .txt 的文件
tdmsFiles = dir(filePattern);

fname = tdmsFiles(1).name;
%%
% fname = '../data/beads/488beads/230828 SM001 TR007.tdms';
tic
tdms_struct = TDMS_getStruct(fname);
disp('Data Loaded')
toc
fn = fieldnames(tdms_struct);
data=tdms_struct.(fn{2});


%% read data
H=65;
V=385;
for i=1:V
    if(i==1)
        image_z(:,:,i)=data.("Untitled").data;
    else
        image_z(:,:,i)=data.(['Untitled_',num2str(i-1)]).data;
    end
end

h_n=floor(size(image_z,2)./H);
image_ccd = zeros(65,385,h_n);

for i=1:h_n
    image_ccd(:,:,i) = image_z(1,(i-1)*H+1:i*H,:);
    % image_ccd(:,:,i)=image_tdms(1,44:108,64:448);
end

crop_x = 22;
crop_y = 0;
crop_w = 128;
crop_h = 64;

posImg = image_ccd(crop_y+1:crop_h, crop_x+1:crop_x+crop_w,:);
specImg = image_ccd(crop_y+1:crop_h, crop_x+crop_w+1:end, :);

%%
% figure(2)
% for i = 1:size(posImg,3)
%     subplot(121)
%     imshow(posImg(:,:,i),[]);
%     subplot(122)
%     imshow(specImg(:,:,i),[]);
%     title('image'+string(i))
%     pause(0.01);
% end

%%
ccdScanInfo = struct();
ccdScanInfo.frame = zeros(h_n,1);
ccdScanInfo.X = zeros(h_n,1);
ccdScanInfo.Y = zeros(h_n,1);
ccdScanInfo.Int = zeros(h_n,1);
ccdScanInfo.Sig = zeros(h_n,1);
ccdScanInfo.raw_centroid = zeros(h_n,1);
threshold = 5e2;

for i = 1:h_n
    trackPos = posImg(27:42,56:71,i);
    %trackPos = posImg(31:46,54:69,i);  %16*16
    signal = mean(trackPos(6:10,6:10),"all");
    bk = (mean(trackPos(1:4,1:4),"all") + mean(trackPos(13:16,1:4),"all") + ...
        mean(trackPos(1:4,13:16),"all") + mean(trackPos(13:16,13:16),"all"))/4;

    snr = 10 * log10(signal / bk);

    if signal < threshold || snr < 2
        continue
    end

    [cx,cy] = PSFfiting(trackPos);
    if (cx==0 && cy==0) 
        continue
    end
    
     rotatedImage = imrotate(specImg(:,:,i), 2.1, 'bilinear', 'crop');
%     channel488 = rotatedImage(31:46,137:160); % 512.5874-560.4332
%     channel561 = rotatedImage(31:46,110:129); % 646.3444-582.5684
%     channel640 = rotatedImage(31:46,88:107);  % 658.0592-745.2931

    channel488 = rotatedImage(27:42,137:160); % 512.5874-560.4332
    channel561 = rotatedImage(27:42,110:129); % 646.3444-582.5684
    channel640 = rotatedImage(27:42,88:107);  % 658.0592-745.2931

    trackSpec = [channel640, channel561, channel488];
    trackSpec = flip(trackSpec, 2);
    norm_trackPos = (trackPos - min(trackPos(:))) / (max(trackPos(:)) - min(trackPos(:)));
    norm_trackSpec = (trackSpec - min(trackSpec(:))) / (max(trackSpec(:)) - min(trackSpec(:)));

    trackImg = [norm_trackPos,norm_trackSpec];
    trackCurve = [normalize(mean(norm_trackPos, 1),'range'), ...
                  normalize(mean(norm_trackSpec, 1),'range')];
    raw_centroid = fitCentroid(trackImg,trackCurve,cx,cy);
    if raw_centroid == 0
        continue
    end

    ccdScanInfo.frame(i,1) = i;
    ccdScanInfo.X(i,1) = cx;
    ccdScanInfo.Y(i,1) = cy;
    Intensity = trackPos(round(cx),round(cy));
    ccdScanInfo.Int(i,1) = Intensity;
    ccdScanInfo.raw_centroid(i,1) = raw_centroid;

end

fields = fieldnames(ccdScanInfo); 
ind = find(ccdScanInfo.frame~=0);
for i = 1:numel(fields)
    field = fields{i};
    if isnumeric(ccdScanInfo.(field))
        % 如果字段是数组，则进行修改
        ccdScanInfo.(field) = ccdScanInfo.(field)(ind);
    end
end

%% read scan data
% load("int.mat")
% int=int';
scanspecImg = zeros(size(int));
for i=1:size(ccdScanInfo.frame,1)
    if ccdScanInfo.raw_centroid(i) == 0
        continue
    end
    frame = ccdScanInfo.frame(i);
    [scanX, scanY] = ind2sub(size(int), frame);
    scanspecImg(scanX, scanY) = ccdScanInfo.raw_centroid(i);

end

lowerLim = 510;
upperLim = 740;




% 设置全局参数
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultFigurePosition', [100, 100, 600, 600]);

% 创建一个1x3的子图网格
% hInt = figure;
% 设置全局参数
% set(0, 'DefaultAxesFontSize', 12);
% % 设置子图的大小为300x300
% subplotWidth = 300;
% subplotHeight = 300;

% 子图1
% subplot('Position', [0.05, 0.1, subplotWidth/800, subplotHeight/600]);
ax1 = figure;
imshow(flipud(rot90(scanspecImg)),[]);
num_colors = 1024;
colormap jet;
colorbar; caxis([lowerLim,upperLim]);

cmap = jet(num_colors);
caxis([lowerLim, upperLim]);
title('Spectrum mapping');

% 保存图像
[~,dirname,~]=fileparts(fname);
figTrajname = [dirname ' scan spec1' '.fig'];
saveas(ax1,figTrajname,'fig');

% 子图2
% ax2 = subplot('Position', [0.375, 0.1, subplotWidth/800, subplotHeight/600]);
ax2 = figure;
int = (int - min(min(int))) / (max(max(int)) - min(min(int)));
imshow(flipud(rot90(int)),[]);
colormap(ax2, 'gray'); 
title('Intensity image');

% 保存图像
[~,dirname,~]=fileparts(fname);
figTrajname = [dirname ' scan int1' '.fig'];
saveas(ax2,figTrajname,'fig');

% 子图3
% ax3 = subplot('Position', [0.7, 0.1, subplotWidth/800, subplotHeight/600]);
ax3 = figure;
scanspecImg(scanspecImg < lowerLim) = lowerLim; % adjust contrast
scanspecImg(scanspecImg > upperLim) = upperLim; % by cutoff and saturation
scanspecImg=(scanspecImg-lowerLim)/(upperLim-lowerLim);

indices = round(scanspecImg * (num_colors - 1)) + 1;
[r, c] = size(scanspecImg);
rgb_image = reshape(cmap(indices, :), r, c, 3);

ImageAdj = imadjust(int, stretchlim(int), []);
imgRatioColorMask =  rgb_image .* ImageAdj;
imshow(flipud(rot90(imgRatioColorMask)), [])
colormap(ax3, 'jet');
colorbar; caxis([lowerLim,upperLim]);

title('Mixup imaging');

[~,dirname,~]=fileparts(fname);
figTrajname = [dirname ' scan mix1' '.fig'];
saveas(ax3,figTrajname,'fig');

% 保存图像
% [~,dirname,~]=fileparts(fname);
% figTrajname = [dirname ' spec' '.fig'];
% saveas(hInt,figTrajname,'fig');

% 清除全局设置
set(0, 'DefaultAxesFontSize', 'remove');
set(0, 'DefaultFigurePosition', 'remove');
