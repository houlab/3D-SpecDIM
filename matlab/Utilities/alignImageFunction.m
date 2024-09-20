function [out,ia,ib,repKTX,repKTY,repTimeKT]=alignImageFunction(offset,tempImData,XYsize,KTBinTime,tickTime)

    xkt=[1,2,1,3,5,4,5,3,4,5,3,1,2,4,5,4,2,1,3,5,4,2,3,2,1]-3;
    ykt=[5,3,1,2,1,3,5,4,2,4,5,4,2,1,3,5,4,2,1,2,4,5,3,1,3]-3;
    %xkt=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]-3;
    %ykt=[5,5,5,5,5,4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,1,1,1,1,1]-3;
    tempXData=XYsize*xkt(tempImData(:,1)+1)';
    tempYData=XYsize*ykt(tempImData(:,1)+1)';

    expXKT=kron(xkt,ones(1,round(KTBinTime/tickTime)));
    expYKT=kron(ykt,ones(1,round(KTBinTime/tickTime)));
    timeKT=(1:length(expXKT));

    NoofKT=floor(tempImData(1,3)/max(timeKT));
    time=tempImData(:,3)-NoofKT*max(timeKT);

    repKTX=repmat(expXKT',round(max(time)/max(timeKT))+2,1);
    repKTY=repmat(expYKT',round(max(time)/max(timeKT))+2,1);
    repTimeKT=(1:length(repKTX));

    [~,ia,ib]=intersect(int32(time+offset),int32(repTimeKT'));
    out=sum((double(tempXData(ia))-repKTX(ib)*XYsize).^2);



  

