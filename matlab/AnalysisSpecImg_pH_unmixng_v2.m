%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%ouput spec centroid data%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spec_data=AnalysisSpecImg_pH(image_ccd, fname, mGoldHaloSpec)
  
    minVal = 0;
    h_n=size(image_ccd,3);
    crop_x = 54;
    crop_y = 0;
    crop_w = 64;
    crop_h = 64;

    if contains(fname, ["240223"])
        PosTrackPoint = [29, 36]; % 264 150  240-304  64-448
        SpecTrackPoint = [29,168]; % 264 340

    elseif contains(fname, ["240306","240308"])
        PosTrackPoint = [23, 24]; % 264 150  240-304  64-448
        SpecTrackPoint = [23,153]; % 264 340

    elseif contains(fname, ["240313"])
        PosTrackPoint = [26, 30]; % 264 150  240-304  64-448
        SpecTrackPoint = [25,158]; % 264 340
        
    elseif contains(fname, ["240315"])
        PosTrackPoint = [26, 30]; % 264 150  240-304  64-448
        SpecTrackPoint = [22,158]; % 264 340
    elseif contains(fname, ["240320"])
        PosTrackPoint = [27, 27]; % 264 150  240-304  64-448
        SpecTrackPoint = [32,153]; % 264 340
    elseif contains(fname, ["240322"])
        PosTrackPoint = [27, 19]; % 264 150  240-304  64-448
        SpecTrackPoint = [26,146]; % 264 340
    elseif contains(fname,["240129","240227"])
        PosTrackPoint = [26, 33]; % 264 150  240-304  64-448
        SpecTrackPoint = [26,156]; % 264 340    
    elseif contains(fname,  "240626")
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [41, 33]; % 264 150  240-304  64-448
        SpecTrackPoint = [42,102]; % 264 340
    elseif contains(fname,  ["240717"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [36, 40]; % 264 150  240-304  64-448
        SpecTrackPoint = [36,120]; % 264 340
    elseif contains(fname,  ["240719"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [38, 35]; % 264 150  240-304  64-448
        SpecTrackPoint = [38,115]; % 264 340
    elseif contains(fname,  ["240911"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [33, 33]; % 264 150  240-304  64-448
        SpecTrackPoint = [29,100]; % 264 340
    else
        PosTrackPoint = [29, 36]; % 264 150  240-304  64-448
        SpecTrackPoint = [29,168]; % 264 340
    end

    
    posImg = image_ccd(crop_y+1:crop_h, crop_x+1:crop_x+crop_w,:);
    specImg = image_ccd(crop_y+1:crop_h, crop_x+crop_w+1:end, :);
    
    %%
    
    cropArea = [16 80];

    spec_data = struct();
    spec_data.trackImg = zeros(cropArea(1),cropArea(2),h_n);
    spec_data.posImg = posImg;
    spec_data.frame = zeros(h_n,1);
    spec_data.raw_centroid = zeros(h_n,1);
    spec_data.pos = zeros(h_n,2);
    spec_data.snr = zeros(h_n,1);
    spec_data.mGold_int = zeros(h_n,1);
    spec_data.HaloTag_int = zeros(h_n,1);
    spec_data.Lyso_int = zeros(h_n,1);
    spec_data.trackCurve = zeros(h_n,cropArea(2));

    lx = PosTrackPoint(1) - cropArea(1) / 2;
    ly = PosTrackPoint(2) - cropArea(1) / 2;
    rx = PosTrackPoint(1) + cropArea(1) / 2 - 1;
    ry = PosTrackPoint(2) + cropArea(1) / 2 - 1;

    lx_s = SpecTrackPoint(1) - cropArea(1) / 2;
    rx_s = SpecTrackPoint(1) + cropArea(1) / 2-1;

    ly_s_640 = SpecTrackPoint(2) - 38;
    ry_s_640 = SpecTrackPoint(2) - 18; %21pix

    ly_s_561 = SpecTrackPoint(2) - 17;
    ry_s_561 = SpecTrackPoint(2) + 3; %21pix

    ly_s_488 = SpecTrackPoint(2) + 9;
    ry_s_488 = SpecTrackPoint(2) + 30; %22pix

    max_488 = max(mean(specImg(lx_s:rx_s,ly_s_488:ry_s_488,:),1),[],"all");
    min_488 = min(mean(specImg(lx_s:rx_s,ly_s_488:ry_s_488,:),1),[],"all");

    max_561 = max(mean(specImg(lx_s:rx_s,ly_s_561:ry_s_561,:),1),[],"all");
    min_561 = min(mean(specImg(lx_s:rx_s,ly_s_561:ry_s_561,:),1),[],"all");

    max_640 = max(mean(specImg(lx_s:rx_s,ly_s_640:ry_s_640,:),1),[],"all");
    min_640 = min(mean(specImg(lx_s:rx_s,ly_s_640:ry_s_640,:),1),[],"all");
    
    
    for i = 1:h_n
        
        trackPos = posImg(lx:rx,ly:ry,i); %16*16

        rotatedImage = imrotate(specImg(:,:,i), 1.1, 'bilinear', 'crop');
        
        channel488 = rotatedImage(lx_s:rx_s,ly_s_488:ry_s_488); % 512.5874-560.4332
        channel561 = rotatedImage(lx_s:rx_s,ly_s_561:ry_s_561); % 646.3444-582.5684
        channel640 = rotatedImage(lx_s:rx_s,ly_s_640:ry_s_640);  % 658.0592-745.2931
        
        trackSpec = [channel640, channel561, channel488];

        trackSpec = flip(trackSpec, 2);
        norm_trackPos = (trackPos - min(trackPos(:))) / (max(trackPos(:)) - min(trackPos(:)));
        norm_trackSpec = (trackSpec - min(trackSpec(:))) / (max(trackSpec(:)) - min(trackSpec(:)));
    
        trackImg = [norm_trackPos,norm_trackSpec];
        trackCurve = [mean(trackPos, 1), ...
                      mean(trackSpec, 1)];

        spec_data.trackImg(:,:,i) = trackImg;
        spec_data.trackCurve(i,:) = trackCurve;
        spec_data.frame(i) = i;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        signal = mean(trackPos(cropArea(1)/2-2:cropArea(1)/2+2, ...
            cropArea(1)/2-2:cropArea(1)/2+2),"all");
        bk = (mean(trackPos(1:4,1:4),"all") + mean(trackPos(end-3:end,1:4),"all") + ...
            mean(trackPos(1:4,end-3:end),"all") + mean(trackPos(13:16,end-3:end),"all"))/4;
    
        snr = 10 * log10(signal / bk);
        

        [cx,cy] = PSFfiting(trackPos);
        
        % raw_centroid = mean(channel488,"all") / mean(channel561,"all");
        % raw_centroid = max(mean(channel488, 1)) / max(mean(channel561,1));
        raw_centroid = calc_unmixing_spec(trackCurve, mGoldHaloSpec);

        spec_data.pos(i,:) = [cx,cy];
        spec_data.snr(i) = snr;
        spec_data.raw_centroid(i) = raw_centroid;
        spec_data.mGold_int(i) = (max(mean(channel488, 1)) - min_488) / (max_488 - min_488);
        spec_data.HaloTag_int(i) = (max(mean(channel561, 1)) - min_561) / (max_561 - min_561);
        spec_data.Lyso_int(i) = (max(mean(channel640, 1)) - min_640) / (max_640 - min_640);

    end
    spec_data.raw_centroid(find(isnan(spec_data.raw_centroid)))=0;
    spec_data.raw_centroid = fillZerosWithClosestNonZero(spec_data.raw_centroid);

    %%
%     [~,dirname,~]=fileparts(fname);
%     tiff_fileName = strrep(dirname, ' ', '_')+".tiff";
%     
%     t = Tiff(tiff_fileName,'w');
%     tagstruct.ImageLength = size(spec_data.trackImg, 1);
%     tagstruct.ImageWidth = size(spec_data.trackImg, 2);
%     tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
%     tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
%     tagstruct.BitsPerSample = 64;
%     tagstruct.RowsPerStrip = 16;
%     tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%     t.setTag(tagstruct);
%     
%     % 循环遍历图像序列，并将每个图像写入TIFF文件
%     for i = 1:size(spec_data.trackImg, 3)
%         img = spec_data.trackImg(:, :, i);
%         t.setTag(tagstruct);
%         t.write(img);
%         t.writeDirectory();
%     end
end



%%

function spec_ratio = calc_unmixing_spec(spec_curve, mGoldHaloSpec)
    % 定义时间点和波长范围
    mGold_spec = normalize(mGoldHaloSpec.mGold,'range')';
    JF549_spec = normalize(movmean(mGoldHaloSpec.JF549', 40),'range')';

    filter_func = @(x, center, width) exp(-((x-center).^2)/(2*(width/2.355)^2));
    mGold_spec_filter = mGold_spec .* (1-filter_func(500:0.1:740, 488, 20)) ...
                .* (1-filter_func(500:0.1:740, 561, 20)) ...
                .* (1-filter_func(500:0.1:740, 640, 20));

    JF549_spec_filter = JF549_spec .* (1-filter_func(500:0.1:740, 488, 20)) ...
                .* (1-filter_func(500:0.1:740, 561, 20)) ...
                .* (1-filter_func(500:0.1:740, 640, 20));

    mGold_spec_filter = mGold_spec_filter(1:37:end);
    JF549_spec_filter = JF549_spec_filter(1:37:end);
    
    
    max_JF549 = max(spec_curve(1,39:59));
    min_JF549 = min(spec_curve(1,21:59));
    norm_spec_549 = (spec_curve(1,21:end) - min_JF549) / (max_JF549 - min_JF549);



    M = [mGold_spec_filter(1:60); JF549_spec_filter(1:60)]';
    b_vec = norm_spec_549';

    % 使用线性代数方法解丰度系数
    x = M \ b_vec;  % 解线性方程 Mx = b
    spec_ratio = x(1) / x(2);
                          


end

%%


function data = fillZerosWithClosestNonZero(data)
    n = length(data); % 获取数据长度
    for i = 1:n
        if data(i) == 0
            % 初始化距离和值
            distance = inf;
            value = 0;
            
            % 向前查找非零值
            for j = i-1:-1:1
                if data(j) ~= 0
                    if (i-j) < distance
                        distance = i-j;
                        value = data(j);
                    end
                    break;
                end
            end
            
            % 向后查找非零值
            for j = i+1:n
                if data(j) ~= 0
                    if (j-i) < distance
                        distance = j-i;
                        value = data(j);
                    end
                    break;
                end
            end
            
            % 替换为最近的非零值
            data(i) = value;
        end
    end
end







