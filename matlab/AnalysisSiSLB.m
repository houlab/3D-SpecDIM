%%
% read .dat data
fname = ["dppc_cl2.dat", "dppc.dat"];
pname = '/Volumes/shah_ssd/data/spms_track/manuscript/FigS4/';
% [fname,pname]=uigetfile('*.dat');

fid=fopen(strcat(pname,fname(1)));
data_1=fread(fid, Inf, 'int32', 0);  % 以uint32格式读取数据，'b'表示大端字节序
fclose(fid);

row = 65;
col = 385;
img_ccd_1 = reshape(data_1, row, col, []);

fid=fopen(strcat(pname,fname(2)));
data_2=fread(fid, Inf, 'int32', 0);  % 以uint32格式读取数据，'b'表示大端字节序
fclose(fid);

row = 65;
col = 385;
img_ccd_2 = reshape(data_2, row, col, []);

image_ccd = cat(3,img_ccd_1, img_ccd_2);

fname = fname(1);
cd(pname);
[~,dirname,~]=fileparts(fname);
mkdir(dirname);
cd(dirname);

save('image_ccd.mat', 'image_ccd','-v7.3')
%%
 [~,dirname,~]=fileparts(fname);
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
spec_data = AnalysisSpecImg_centroid(image_ccd, fname);
save("spec_data","spec_data");

%%
s1 = 1;
s2 = 1001;
e1 = 500;
e2 = 1500;

PosTrackPoint = [39, 34];
lx = PosTrackPoint(1) - 16 / 2;
ly = PosTrackPoint(2) - 16 / 2;
rx = PosTrackPoint(1) + 16 / 2 - 1;
ry = PosTrackPoint(2) + 16 / 2 - 1;

raw_centroid_1 = spec_data.raw_centroid(s1:e1);
raw_centroid_2 = spec_data.raw_centroid(s2:e2);
raw_int = squeeze(mean(spec_data.posImg(lx:rx,ly:ry,:),[1,2]));

% ind = find(spec_data.pos(:,1)==0);
% spec_data.raw_centroid(ind) = [];
spec_time=(1:length(cat(1,raw_centroid_1,raw_centroid_1)))*0.083;

hfig = figure;
pos1 = [0.1, 0.7, 0.32, 0.25]; % [left bottom width height]
pos2 = [0.45, 0.7, 0.1, 0.25];
pos3 = [0.65, 0.7, 0.3, 0.25];
%%%%%%%%%%%%%%%
ax1 = axes('Position', pos1);
dataVector = spec_data.raw_centroid;
plot(ax1, spec_time(1:e1-s1+1), raw_centroid_1, 'Color',"#0072BD", 'LineWidth',1.5); % 假设数据在 dataVector 变量中
hold on 

ylim([600 625])

cen_len = numel(raw_centroid_2);
plot(ax1, spec_time(1:e1-s1+1), raw_centroid_2, 'Color',"r", 'LineWidth',1.5); % 假设数据在 dataVector 变量
xlabel('Time [sec]')
ylabel('Spec Cent. [nm]')
grid on
set(gca, 'GridColor', [0.8 0.8 0.8]);


ax2 = axes('Position', pos2);
mu = mean(raw_centroid_1);
sigma = std(raw_centroid_1);
histogram(ax2, raw_centroid_1, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','#0072BD','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;
% 计算正态分布数据
x_values = linspace(min(raw_centroid_1), max(raw_centroid_1), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax2, pdf_values, x_values, 'Color','#0072BD', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));

%%%%%%%%%%%%%%%%%
mu = mean(raw_centroid_2);
sigma = std(raw_centroid_2);
histogram(ax2, raw_centroid_2, 'Orientation', 'horizontal', 'Normalization', 'pdf',...
    'FaceColor','r','FaceAlpha',0.3,'EdgeAlpha',0);
hold on;
ylim([600 625])
% 计算正态分布数据
x_values = linspace(min(raw_centroid_2), max(raw_centroid_2), 100);
pdf_values = normpdf(x_values, mu, sigma);

% 绘制正态分布曲线
plot(ax2, pdf_values, x_values, 'Color','r', 'LineWidth', 2);

% 添加均值和标准差的标注
text(max(pdf_values)*0.6, mu - sigma*1.4, sprintf('\\mu = %.2f nm', mu));
text(max(pdf_values)*0.6, mu + sigma*1.4, sprintf('\\sigma = %.2f nm', sigma));
axis off

ax3 = axes('Position', pos3);
plot(ax3,spec_time(1:e1-s1+1),raw_int(s1:e1),'Color','#0072BD','LineWidth',1.5)
hold on
plot(ax3,spec_time(1:e1-s1+1),raw_int(s2:e2),'r','LineWidth',1.5)

ylabel('Intensity [counts/sec]')
xlabel('Time [sec]')
figureandaxiscolors('w','k','')


