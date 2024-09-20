%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%分析光谱数据，输入为tiff文件%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spec_data=AnalysisSpecImg_centroid(image_ccd, fname)    
    h_n=size(image_ccd,3);    
    shift_640=0;
    crop_x = 54;
    crop_y = 0;
    crop_w = 64;
    crop_h = 64;
    if contains(fname,["230828"])
        PosTrackPoint = [34, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [34,157]; % 264 340
        shift_640 = -20;
    elseif contains(fname,"240311")
        PosTrackPoint = [26, 27]; % 264 150  240-304  64-448
        SpecTrackPoint = [26,153]; % 264 340
        shift_640 = -20;
    elseif contains(fname,["dppc"])
        PosTrackPoint = [39, 34]; % 264 150  240-304  64-448
        SpecTrackPoint = [38,161]; % 264 340
        shift_640 = 0;
    elseif contains(fname,'240318')
        PosTrackPoint = [27, 24]; % 264 150  240-304  64-448
        SpecTrackPoint = [29,153]; % 264 340
        shift_640 = 0;
    elseif contains(fname, ["231111"])
        crop_x = 119;
        crop_y = 68;
        PosTrackPoint = [32, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [32,153]; % 264 340
    elseif contains(fname, ["231116", "231121", "231114"])
        crop_x = 119;
        crop_y = 118;
        PosTrackPoint = [32, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [32,162]; % 264 340
    elseif contains(fname, ["231128 TR003", "231128 TR004","231128 TR006", ...
            "231128 TR008", "231128 TR009", "231128 TR010","231128 TR005"])
        crop_x = 118;
        crop_y = 142;
        PosTrackPoint = [32, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [28,150]; % 264 340
    elseif contains(fname, ["231128 TR013", "231128 TR014","231128 TR016", ...
            "231128 TR017"])
        PosTrackPoint = [32, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [32,149]; % 264 340
    elseif contains(fname, "240419")
        PosTrackPoint = [41, 31]; % 264 150  240-304  64-448
        SpecTrackPoint = [18,171]; % 264 340
    elseif contains(fname, ["240504", "240514"])
        crop_x = 100;
        crop_y = 0;
        PosTrackPoint = [36, 32]; % 264 150  240-304  64-448
        SpecTrackPoint = [32,105]; % 264 340
    elseif contains(fname,  ["240628","240701","240702"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [38, 35]; % 264 150  240-304  64-448
        SpecTrackPoint = [38,102]; % 264 340     
    elseif contains(fname,  ["240708","240718", "240719"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [38, 35]; % 264 150  240-304  64-448
        SpecTrackPoint = [38,115]; % 264 340
    elseif contains(fname,  ["240725","240801"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [36, 36]; % 264 150  240-304  64-448
        SpecTrackPoint = [36,115]; % 264 340

    elseif contains(fname,  ["240706"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [38, 33]; % 264 150  240-304  64-448
        SpecTrackPoint = [34,115]; % 264 340     

    elseif contains(fname,  ["240902"])
        crop_x = 13;
        crop_y = 0;
        PosTrackPoint = [34, 34]; % 264 150  240-304  64-448
        SpecTrackPoint = [22,105]; % 264 340  
    elseif contains(fname, 'simulation')
        crop_x = 0;
        crop_y = 0;
        PosTrackPoint = [9, 9];
        SpecTrackPoint = [9,32];
        crop_w = 16;
        crop_h = 16;
    end
    
    posImg = image_ccd(crop_y+1:crop_y+crop_h, crop_x+1:crop_x+crop_w,:);
    specImg = image_ccd(crop_y+1:crop_y+crop_h, crop_x+crop_w+1:end, :);
    
    %%
    
    cropArea = [16 80];

    spec_data = struct();
    spec_data.trackImg = zeros(cropArea(1),cropArea(2),h_n);
    spec_data.posImg = posImg;
    spec_data.frame = zeros(h_n,1);
    spec_data.raw_centroid = zeros(h_n,1);
    spec_data.pos = zeros(h_n,2);
    spec_data.snr = zeros(h_n,1);
    spec_data.trackCurve = zeros(h_n,cropArea(2));

    lx = PosTrackPoint(1) - cropArea(1) / 2;
    ly = PosTrackPoint(2) - cropArea(1) / 2;
    rx = PosTrackPoint(1) + cropArea(1) / 2 - 1;
    ry = PosTrackPoint(2) + cropArea(1) / 2 - 1;

    lx_s = SpecTrackPoint(1) - cropArea(1) / 2;
    rx_s = SpecTrackPoint(1) + cropArea(1) / 2-1;

    ly_s_640 = SpecTrackPoint(2) - 38+shift_640;
    ry_s_640 = SpecTrackPoint(2) - 18+shift_640; %21pix

    ly_s_561 = SpecTrackPoint(2) - 17;
    ry_s_561 = SpecTrackPoint(2) + 3; %21pix

    ly_s_488 = SpecTrackPoint(2) + 9;
    ry_s_488 = SpecTrackPoint(2) + 30; %22pix

    if contains(fname, "simulation")
        ly_s_640 = 44;
        ry_s_640 = 64; %21pix
    
        ly_s_561 = 23;
        ry_s_561 = 43; %21pix
    
        ly_s_488 = 1;
        ry_s_488 = 22; %22pix
    end

    
    for i = 1:h_n
        
        trackPos = posImg(lx:rx,ly:ry,i); %16*16
        
        if ~contains(fname, "simulation")
            rotatedImage = imrotate(specImg(:,:,i), 2.1, 'bilinear', 'crop');
            
        else
            rotatedImage = specImg(:,:,i);
        end

        channel488 = rotatedImage(lx_s:rx_s,ly_s_488:ry_s_488); % 512.5874-560.4332
        channel561 = rotatedImage(lx_s:rx_s,ly_s_561:ry_s_561); % 646.3444-582.5684
        channel640 = rotatedImage(lx_s:rx_s,ly_s_640:ry_s_640);  % 658.0592-745.2931
        
       
        if ~contains(fname, "simulation")
            trackSpec = [channel640, channel561, channel488];
            trackSpec = flip(trackSpec, 2);
        else
            trackSpec = [channel488, channel561, channel640];
        end

        norm_trackPos = (trackPos - min(trackPos(:))) / (max(trackPos(:)) - min(trackPos(:)));
        norm_trackSpec = (trackSpec - min(trackSpec(:))) / (max(trackSpec(:)) - min(trackSpec(:)));
    
        trackImg = [norm_trackPos,norm_trackSpec];
        trackCurve = [normalize(mean(norm_trackPos, 1),'range'), ...
                      normalize(mean(norm_trackSpec, 1),'range')];

        spec_data.trackImg(:,:,i) = trackImg;
        spec_data.frame(i) = i;
        spec_data.trackCurve(i,:) = trackCurve;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        signal = mean(trackPos(cropArea(1)/2-2:cropArea(1)/2+2, ...
            cropArea(1)/2-2:cropArea(1)/2+2),"all");
        bk = (mean(trackPos(1:4,1:4),"all") + mean(trackPos(end-3:end,1:4),"all") + ...
            mean(trackPos(1:4,end-3:end),"all") + mean(trackPos(13:16,end-3:end),"all"))/4;
    
        snr = 10 * log10(signal / bk);
        
        if snr <0.1
            continue
        end
%     
        [cx,cy] = PSFfiting(trackPos);
        if (cx==0 && cy==0) 
            continue
        end

        raw_centroid = fitCentroid_v2(trackCurve,cx);
        if raw_centroid == 0
            continue
        end

        spec_data.pos(i,:) = [cx,cy];
        spec_data.snr(i) = snr;
        if contains(fname,["230828", "240311"])
            spec_data.raw_centroid(i) = raw_centroid -15;
        elseif contains(fname,["240801"])
            spec_data.raw_centroid(i) = raw_centroid -10;
        elseif contains(fname,["dppc"])
            spec_data.raw_centroid(i) = raw_centroid +7;
        else
            spec_data.raw_centroid(i) = raw_centroid;
        end
    
    end
    spec_data.raw_centroid = fillZerosWithClosestNonZero(spec_data.raw_centroid);
    spec_data.snr = fillZerosWithClosestNonZero(spec_data.snr);
    % spec_data.raw_centroid(find(spec_data.raw_centroid==0))=minVal;
    %%
    [~,dirname,~]=fileparts(fname);
    tiff_fileName = strrep(dirname, ' ', '_')+".tiff";
    
    t = Tiff(tiff_fileName,'w');
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
    end
    
end

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

function centroid=fitCentroid_v2(specCurve, cx)
    % 自定义双峰高斯分布模型
    gaussianModel = @(p, x) p(1) * exp(-((x - p(2)) / p(3)).^2);

    % 初始参数猜测
    initialGuess_psf = [1, 8, 0.5];   
    psf_curve = specCurve(1:16);
    spec_curve = specCurve(17:80);

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

    if pos > 16 || pos < 1 || spec > 80 || spec < 16 ||...
            pix_dis > 80 || pix_dis < 0 || fitRMSE_spec > 0.4 || fitRMSE_psf > 0.3
        centroid = 0;
        return
    end

    f_sp = @(x)(0.02215*x.^2+3.077*x+591.6);
    centroid = f_sp(pix_dis-37);
    
end
