function image_ccd = readTiff2mat(pname, fileList)
    fnames = {};
    for j = 1:length(fileList)
        
        folderPath = strcat(pname,fileList(j));
        files = dir(folderPath);
        files = files(~[files.isdir]);
        for i = 1:length(files)
            fileName = files(i).name;
            if contains(fileName, '.tdms') && ~contains(fileName, 'SM')
                % 将符合条件的文件名添加到数组中
                fnames{end+1} = strcat(folderPath,'/',fileName); % 将符合条件的文件名添加到selectedFiles
            end
        end
    end
    
    for j = 1:length(fnames)
        close all
        
        [pname, fname, ~] = fileparts(fnames{j});
        cd(pname)
        cd(fname)
        fname = strcat(fname,'.tdms');
        disp(fname)
    
        files = dir("./");
        files = files(~[files.isdir]);
    
         for i = 1:length(files)
            fileName = files(i).name;
            if contains(fileName, '.tif') && ~contains(fileName, '._')
               info = imfinfo(fileName);  % 替换'yourfile.tif'为你的文件名
                num_frames = numel(info);
                image_ccd = zeros(65,233,num_frames);  % 创建一个单元数组来存储所有帧
                for k = 1:num_frames
                    image_ccd(:,:,k) = imread(fileName, k);  % 读取第k帧
                end
                save('image_ccd.mat', 'image_ccd','-v7.3')
            end
        end
        
    end


    % fname = "./240701 TR050 Gain200 5ms.tif";
    % info = imfinfo(fname);  % 替换'yourfile.tif'为你的文件名
    % num_frames = numel(info);
    % image_ccd = zeros(65,233,num_frames);  % 创建一个单元数组来存储所有帧
    % for k = 1:num_frames
    %     image_ccd(:,:,k) = imread(fname, k);  % 读取第k帧
    % end
    % save('image_ccd.mat', 'image_ccd','-v7.3')

end