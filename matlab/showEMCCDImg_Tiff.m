function image_ccd = showEMCCDImg_Tiff(trackTime,decrate, fname, ~)
    %%
    warning off
    fname_ = fname;
    info = imfinfo(fname_);  % 替换'yourfile.tif'为你的文件名
    num_frames = numel(info);
    image_ccd_ = zeros(65,233,num_frames);  % 创建一个单元数组来存储所有帧
    for k = 1:num_frames
        image_ccd_(:,:,k) = imread(fname_, k);  % 读取第k帧
    end

    exposure_time = trackTime / num_frames; % 10ms-25.191

    h_n = ceil(num_frames / decrate);
    image_ccd = zeros(65,233,h_n);

    index = 1;
    for i=1:decrate:num_frames
        image_ccd(:,:,index) = image_ccd_(:,:,i);
        index = index + 1;
    end

    tiff_fileName = strrep(strcat(dirname,'_',string(exposure_time)), ' ', '_')+".tiff";

    [~,dirname,~]=fileparts(fname);
    tiff_fileName = strrep(strcat(dirname,'_',string(exposure_time)), ' ', '_')+".tiff";
    
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
    for i = 1:decrate:size(image_ccd,3)
        img = uint16(image_ccd(:,:,i));
        t.setTag(tagstruct);
        t.write(img);
        t.writeDirectory();
        
        if mod(i, 500)==0
            disp("processing "+string(i) + " / "+string(h_n));
        end

    end

    save('image_ccd.mat', 'image_ccd','-v7.3')

   
    

    
    


 