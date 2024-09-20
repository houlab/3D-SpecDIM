function dataStruct = readDataFolders(FolderPath)
    dataStruct = struct();

    fnames = {}; 
    files = dir(FolderPath);
    files = files(~[files.isdir]);

    for i = 1:length(files)
        fileName = files(i).name;
        if contains(fileName, '.tdms') && ~contains(fileName, 'SM') &&  ~contains(fileName, 'IM') 
            % 将符合条件的文件名添加到数组中
            fnames{end+1} = strcat(files(i).folder,'/',fileName); % 将符合条件的文件名添加到selectedFiles
        end
    end

    
    for j = 1:length(fnames)
        close all
        [pname, fname, ~] = fileparts(fnames{j});
        disp(fname)
        fname = strcat(fname,'.tdms');
        
        [~,dirname,~]=fileparts(fname);
        cd(fullfile(pname,dirname));  
        load('spec_data.mat');
        load(strcat(dirname,'.mat'));

        spec_struct = struct();
        spec_struct.spec_data = spec_data;
        spec_struct.traj_data = trackDataFilt;
        dataStruct.(strcat('key_',strrep(dirname,' ', '_'))) = spec_struct;
        disp(fname + " done!")
    end

end