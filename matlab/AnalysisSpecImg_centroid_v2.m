%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%The spectral data was analyzed and input was tiff file%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spec_data=AnalysisSpecImg_centroid_v2(image_ccd, fname)    
    h_n=size(image_ccd,3); 
    % image_ccd = image_ccd(:,60:end,:); % 230828 TR012

    PosTrackPoint = locatePSF(mean(image_ccd(:,1:size(image_ccd,1),:), 3));
  
    mean_spec_img = mean(imgaussfilt(image_ccd(:,size(image_ccd,1)+1:end,:), 2), [2,3]);
    [~, spec_x] = max(mean_spec_img);

    if contains(fname, "251503")
        spec_500nm = 172; % pix coor calibrated 175
    elseif contains(fname, ["241224", "241226", "241227", "241216", "241217"])
        spec_500nm = 174; % pix coor calibrated 175
    elseif contains(fname, "230828")
        spec_500nm = 229;
    elseif contains(fname, "250211")
        spec_500nm = 192;
    else
        spec_500nm = 175; % pix coor calibrated 175
    end

    
    % spec_500nm = 229; % 230828 TR012

    % spec_x = 35; % no_tracking comparision
    % PosTrackPoint(2) = PosTrackPoint(2)+5; % no_tracking comparision
    % spec_500nm = 170; % no_tracking comparision

    spec_y = spec_500nm + PosTrackPoint(2) - size(image_ccd,1);
    SpecTrackPoint = [spec_x, spec_y];

    posImg = image_ccd(:, 1:size(image_ccd,1),:);
    specImg = image_ccd(:, size(image_ccd,1)+1:end, :);
    
    %%
    
    cropArea = [16 85];

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
    ly_s = SpecTrackPoint(2) - cropArea(2) + cropArea(1) + 1; 
    ry_s = SpecTrackPoint(2);      
    
    for i = 1:h_n
        
        trackPos = posImg(lx:rx,ly:ry,i); %16*16
        
        if ~contains(fname, "simulation")
            rotatedImage = imrotate(specImg(:,:,i), 1.5, 'bilinear', 'crop');
        else
            rotatedImage = specImg(:,:,i);
        end
        
        trackSpec = rotatedImage(lx_s:rx_s,ly_s:ry_s);
        trackSpec = flip(trackSpec, 2);
        
        trackSpec_crop64 = [rotatedImage(lx_s:rx_s,ry_s-21:ry_s), rotatedImage(lx_s:rx_s,ly_s:41)];
        trackSpec_crop64 = flip(trackSpec_crop64, 2);
       

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
        spec_data.raw_centroid(i) = raw_centroid;
    end
    % spec_data.raw_centroid(spec_data.raw_centroid==0) = [];
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
    
    % Loop through the image sequence and write each image to the TIFF file
    for i = 1:size(spec_data.trackImg, 3)
        img = spec_data.trackImg(:, :, i);
        t.setTag(tagstruct);
        t.write(img);
        t.writeDirectory();
    end
    
end

function data = fillZerosWithClosestNonZero(data)
    n = length(data);
    for i = 1:n
        if data(i) == 0
            distance = inf;
            value = 0;
            
            for j = i-1:-1:1
                if data(j) ~= 0
                    if (i-j) < distance
                        distance = i-j;
                        value = data(j);
                    end
                    break;
                end
            end

            for j = i+1:n
                if data(j) ~= 0
                    if (j-i) < distance
                        distance = j-i;
                        value = data(j);
                    end
                    break;
                end
            end

            data(i) = value;
        end
    end
end

function centroid=fitCentroid_v2(specCurve, cx)
    % Custom Gaussian distribution model
    gaussianModel = @(p, x) p(1) * exp(-((x - p(2)) / p(3)).^2);

    % Initial parameter guessing
    initialGuess_psf = [1, 8, 1];   
    psf_curve = specCurve(1:16);
    spec_curve = specCurve(17:end);

    [~,init_value]= max(spec_curve);
    initialGuess_spec = [1, init_value, 1];

    try
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

    if pos > 16 || pos < 1 || spec > 85 || spec < 16 ||...
            pix_dis > 85 || pix_dis < 0 || fitRMSE_spec > 0.3 || fitRMSE_psf > 0.3
        centroid = 0;
        return
    end
    
    f_sp = @(x)(0.0342*x^2+0.9157*x+498.1);


    % f_sp = @(x)(0.02215*x.^2+3.077*x+591.6);
    centroid = f_sp(pix_dis-8);
    
end

function psf_coords = locatePSF(image)
% locatePSF - Locate the point spread function (PSF) center pixel coordinates in the image
%
% Input：
%   image      - Input image matrix (grayscale image or RGB image)
%
% Output：
%   psf_coords - Pixel coordinates of PSF [row, col]
%

    blurred_image = imgaussfilt(image, 2);

    max_value = max(blurred_image(:));
    [row, col] = find(blurred_image == max_value, 1);

    window_size = 11; 
    half_size = floor(window_size / 2);
    [rows, cols] = size(image);

    row_min = max(row - half_size, 1);
    row_max = min(row + half_size, rows);
    col_min = max(col - half_size, 1);
    col_max = min(col + half_size, cols);

    roi = double(image(row_min:row_max, col_min:col_max));

    [x, y] = meshgrid(col_min:col_max, row_min:row_max);

    assert(all(size(x) == size(roi)), 'x and roi sizes do not match');
    assert(all(size(y) == size(roi)), 'y and roi sizes do not match');


    amp_init = max(roi(:));
    x0_init = col; 
    y0_init = row; 
    sigma_init = 2; 
    offset_init = min(roi(:)); 
    init_params = [amp_init, x0_init, y0_init, sigma_init, sigma_init, offset_init];

    gauss2d = @(p, x, y) p(1) * exp(-((x - p(2)).^2 / (2 * p(4)^2) + (y - p(3)).^2 / (2 * p(5)^2))) + p(6);

    opt_fun = @(p) sum((gauss2d(p, x, y) - roi).^2, 'all'); % 返回标量

    options = optimset('Display', 'off');
    params_fit = fminsearch(opt_fun, init_params, options);

    psf_coords = [round(params_fit(3)), round(params_fit(2))]; % [row, col]
end