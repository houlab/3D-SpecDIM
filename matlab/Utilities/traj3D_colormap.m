function traj3D_colormap(x,y,z)
%figure;
len=length(x);
cjet=colormap(parula(round(len/33)));

cjlen=length(cjet);
hold on;
for i=1:cjlen-1
    seg=(floor((i-1)*len/cjlen)+1):floor((i+1)*len/cjlen);
    p1=plot3(x(seg),y(seg),z(seg));
    %%%%%
%     drawnow;
%     pause(0.1);
%     M(i) = getframe;
%     %%%%%%%%
    set(p1,'Color',cjet(i,:),'LineWidth',1);
% set(p1,'Color',cjet(i,:),'LineWidth',2);
end
% movie 
%view(3);
%adjustPlotRange(gcf);