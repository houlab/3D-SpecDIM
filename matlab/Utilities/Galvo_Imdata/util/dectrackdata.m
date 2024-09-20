function tempDataFilt=dectrackdata(tempData,factor)
s=size(tempData);
for j=1:s(2);
    tempDataFilt(:,j)=decimate(tempData(:,j),factor);
end