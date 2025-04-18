function image_ccd = showEMCCDImg_tdms(trackTime,decrate, fname, pname)
    %%
    % read .dat data
    warning off
    % [fname,pname]=uigetfile('*.tdms');
    
%     [~,dirname,~]=fileparts(fname);
%     cd(dirname);
    
    tdms_struct = tdmsread(fullfile(pname,fname));
    disp('Data Loaded')
    data=tdms_struct{1,1};

   %% read data
    H=65;
    V=233; 
   
    image_z = table2array(data)';

    h_n=floor(size(image_z,2)./H);
    exposure_time = trackTime / h_n; % 10ms-25.191
    

    h_n = ceil(h_n / decrate);
    image_ccd = zeros(H,V,h_n);
    index = 1;
    for i=1:decrate:floor(size(image_z,2)./H)
        image_ccd(:,:,index) = image_z(:,(i-1)*H+1:i*H)';
        index = index + 1;
        % image_ccd(:,:,i)=image_tdms(1,44:108,64:448);
    end
    
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