function [trackData]=load_trajdata(fname)

% load trajectory data
% 
tic
tdms_struct = TDMS_getStruct(fname);
disp('TracData Loaded')
toc

fn = fieldnames(tdms_struct);
data=tdms_struct.(fn{2});

%Get tracking file size
elementsTotal=length(data.Intensity__Hz_.data); % 3067500

%Reject short Trajectories
if elementsTotal<10000
    error(['Error. ' fname ' : Trajectory too short']);
end
    
%Constants
xconv=(1/32767*76.249);
yconv=(1/32767*76.403);
zconv=(1/32767*66.342);

trackPos=zeros(elementsTotal,5);
trackPos(:,1)=double(data.X_readout__bits_.data)*xconv; %x piezo readout
trackPos(:,2)=double(data.Y_readout__bits_.data)*yconv; %y piezo readout
trackPos(:,3)=double(data.Z_readout__bits_.data)*zconv; %z piezo readout
trackPos(:,4)=double(data.X_control__bits_.data)*xconv; %x piezo control
trackPos(:,5)=double(data.Y_control__bits_.data)*yconv; %y piezo control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
x46=0;
y46=0;
%%%%%find the x, y readout position where control=46;
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

disp('Track Data Converted')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trackInt=double(data.Intensity__Hz_.data); %intensity
x_k=double(data.x_k__nm_.data); %x_k
y_k=double(data.y_k__nm_.data); %y_k
z_k=double(data.z_k__nm_.data); %z_k

stddev=double(data.std_dev__nm_.data); %std dev
xencoder=double(zeros(size(data.x_k__nm_.data))); %place holder
yencoder=double(zeros(size(data.x_k__nm_.data))); %place holder

trackData=[trackInt' x_k' y_k' trackPos xencoder' yencoder' z_k' stddev'];




