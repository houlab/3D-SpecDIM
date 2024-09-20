function [trackData,trackDataFilt]=trajLoadTDMSCalibrated_galvo_v2(decrate,fname,pname)
warning off
%Analyze tracking data from tdms directly
%trackData - raw data
%trackDataFilt - decimated 100 times, removed first and last 0.1 second
%First column - x_k (nm)
%Second column - y_k (nm)
%Third column - z_k (nm)
%Fourth column - x in microns
%Fifth column - y in microns
%Sixth column - z in microns
%Seventh column - x bit
%Eighth column - y bit
%Ninth column - empty
%Tenth column - empty
%Eleventh column - Intensity 1 (counts/sec)
%Twelfth - Intensity 2 (counts/sec)
% 2024年5月30日 by WangZhong
%Go to directory

close all

if contains(fname,'IM')
   fname = fname([1:6,14:end]);
end

cd(pname);
tic
tdms_struct = TDMS_getStruct(fname);
disp([fname,' Data Loaded'])
toc

fn = fieldnames(tdms_struct);
data=tdms_struct.(fn{2});

visibility='on';
%Get tracking file size
elementsTotal=length(data.Index.data(1:decrate:end));

%Reject short Trajectories
if elementsTotal<10000
    disp(['Warning: ' fname ' : Trajectory too short']);
else

    %Constants
    xconv=(1/32767*76.249); %%%% change
    yconv=(1/32767*76.403);
    zconv=(1/32767*66.342);

    trackPos=zeros(elementsTotal,5);
    trackPos(:,1)=double(data.X_readout__bits_.data(1:decrate:end))*xconv; %x piezo readout
    trackPos(:,2)=double(data.Y_readout__bits_.data(1:decrate:end))*yconv; %y piezo readout
    trackPos(:,3)=double(data.Z_readout__bits_.data(1:decrate:end))*zconv; %z piezo readout
    trackPos(:,4)=double(data.X_control__bits_.data(1:decrate:end))*xconv; %x piezo control
    trackPos(:,5)=double(data.Y_control__bits_.data(1:decrate:end))*yconv; %y piezo control
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%

    %%%%%find the x, y readout position where control=50;
    if max(trackPos(:,4))>46 & min(trackPos(:,4))<=46
      [~,x46]=min(abs(trackPos(:,4)-46)); 
      px46=trackPos(x46,1);
    else
       px46=70.4;
    end
    if max(trackPos(:,5))>46 & min(trackPos(:,5))<=46
      [~,y46]=min(abs(trackPos(:,5)-46));  
      py46=trackPos(y46,2);
    else
       py46=68.8;
    end
    %%%%%%%%%%%
    for i=1:elementsTotal%%%%%Calibrate axis
      if trackPos(i,4)>46
        trackPos(i,1)=px46+(trackPos(i,4)-46)*(-0.01408/2*(trackPos(i,4)+46)+1.83);  
      end
    
      if trackPos(i,5)>46
        trackPos(i,2)=py46+(trackPos(i,5)-46)*(-0.01354/2*(trackPos(i,5)+46)+1.792);    
      end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Track Data Converted')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

    APD_12 = isfield(data,'Intensity_1__Hz_');
    if APD_12
        trackInt(1,:) = double(data.Intensity_1__Hz_.data(1:decrate:end)); % APD 1 intensity
        trackInt(2,:) = double(data.Intensity_2__Hz_.data(1:decrate:end)); % APD 2 intensity
        disp('2个APD的数据')
    else
        trackInt=double(data.Intensity__Hz_.data(1:decrate:end)); %intensity
        disp('1个APD数据')
    end
    x_k=double(data.x_k__nm_.data(1:decrate:end)); %x_k
    y_k=double(data.y_k__nm_.data(1:decrate:end)); %y_k
    z_k=double(data.z_k__nm_.data(1:decrate:end)); %z_k

    %     stddev=double(data.std_dev__nm_.data(1:decrate:end)); %std dev
    xencoder=double(zeros(size(data.x_k__nm_.data(1:decrate:end)))); %place holder
    yencoder=double(zeros(size(data.x_k__nm_.data(1:decrate:end)))); %place holder

    trackData=[x_k' y_k' z_k' trackPos xencoder' yencoder' trackInt'];
    trackIntFilt=dectrackdata(trackInt',100);
    trackDataFilt=dectrackdata(trackData,100);

    disp('Track Data Decimated')

    %Create Directory for File Storage
    [~,dirname,~]=fileparts(fname);
    mkdir(dirname);
    cd(dirname);

    %Intensity and Instantaneous Velocity Figure
    time=(1:length(trackDataFilt(:,1)))/1000*decrate;
    int=trackIntFilt(:,1);       

    %Save variables
    saveDataName=[dirname '.mat'];
    save(saveDataName,'trackDataFilt','trackData');


    hALLinOne = figure;

    set(hALLinOne,'Visible',visibility);
    set(hALLinOne,'Renderer','OpenGL')
    set(0,'CurrentFigure',hALLinOne)

    pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]

    if APD_12
        pos2 = [0.6, 0.75, 0.35, 0.2];
        pos2_2 = [0.6, 0.48, 0.35, 0.2];
    else
        pos2 = [0.6, 0.5, 0.35, 0.35];
    end
    pos3 = [0.1, 0.1, 0.25, 0.25];
    pos4 = [0.4, 0.1, 0.25, 0.25];
    pos5 = [0.7, 0.1, 0.25, 0.25];

    ax1 = axes('Position', pos1);

    set(ax1,'Visible',visibility);
    caxis([0 length(trackDataFilt(:,1))/1000*decrate]);
    h = colorbar;
    set( h,'fontsize',10, 'Color', 'k');
    % hrange = 0:length(trackDataFilt(:,1))/5000:length(trackDataFilt(:,1))/1000;
    % set( h,'ticks',hrange,'fontsize',12,'ticklabels',{hrange}, 'Color', 'w');
    h.Label.String = 'time(s) ';%添加单位
    set(h,'fontsize',10,'FontWeight','bold','Color', 'k');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 -0.1 ],'Rotation',360);

    traj3D_colormap(trackDataFilt(:,4),trackDataFilt(:,5),trackDataFilt(:,6));
    view([1 1 1]);
    axis image
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    figureandaxiscolors('w','k',dirname)


    ax2 = axes('Position', pos2);
    set(ax2,'Visible',visibility);
    %Intensity and Instantaneous Velocity Figure
    plot(ax2,time,int,'r')
    xlabel('Time [sec]')
    ylabel('Intensity APD 1 [counts/sec]')
    figureandaxiscolors('w','k',dirname)

    ax3 = axes('Position', pos3);
    plot(ax3,time,trackDataFilt(:,4));
    axis square
    xlabel('Time [sec]');
    ylabel('X [\mum]');
    figureandaxiscolors('w','k',dirname)

    ax4 = axes('Position', pos4);
    plot(ax4,time,trackDataFilt(:,5));
    axis square
    xlabel('Time [sec]');
    ylabel('Y [\mum]');
    figureandaxiscolors('w','k',dirname)

    ax5 = axes('Position', pos5);
    plot(ax5,time,trackDataFilt(:,6));
    axis square
    xlabel('Time [sec]');
    ylabel('Z [\mum]');
    figureandaxiscolors('w','k',dirname)

    if APD_12
        ax6 = axes('Position',pos2_2);
        plot(ax6,time,trackIntFilt(:,2),'r');
        xlabel('Time [sec]')
        ylabel('Intensity ADP 2 [counts/sec]')
        figureandaxiscolors('w','k','')
    end


    saveas(hALLinOne,dirname,'fig');


end

%Decimates trackdata
function tempDataFilt=dectrackdata(tempData,factor)
s=size(tempData);
for j=1:s(2)
    tempDataFilt(:,j)=decimate(tempData(:,j),factor);
end
