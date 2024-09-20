function plot_spec(traj_int,x,y,z,spec_data,dirname)
    time=(1:length(traj_int(:,1)))/1000;
    len=length(x);
    visibility='on';
    
    hScatter=figure;
    set(hScatter,'Visible',visibility);
    set(hScatter,'Renderer','OpenGL')
    set(0,'CurrentFigure',hScatter)
    
    pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]
    pos2 = [0.6, 0.5, 0.35, 0.35];
    pos3 = [0.1, 0.1, 0.25, 0.25];
    pos4 = [0.4, 0.1, 0.25, 0.25];
    pos5 = [0.7, 0.1, 0.25, 0.25];
    
    ax1 = axes('Position', pos1);
    
    caxis([0 length(x)/1000]);
    h = colorbar;
    set( h,'fontsize',10, 'Color', 'w');
    h.Label.String = 'time(s) ';%添加单位
    set(h,'fontsize',10,'Color', 'w');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 -0.8 ],'Rotation',360);
    
    n = 1; % 设置n的值
    numGroups = floor(numel(spec_data.raw_centroid) / n);  % 计算能整除的最大组数
    centroids = spec_data.raw_centroid(1:numGroups * n);  % 截取能整除的部分数据
    centroids = mean(reshape(centroids,n,[]),1);
    
    
    view([1 1 1]);
    axis image
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    
    % hit_centroid = roundn(min(spec_data.raw_centroid),-2):0.01: ...
    % roundn(max(spec_data.raw_centroid),-2);
    hit_centroid = roundn(min(spec_data.raw_centroid),-2):0.01: ...
     roundn(max(spec_data.raw_centroid),-2);
    cjet=colormap(parula(length(hit_centroid)));
    cjlen=length(cjet);
    [sortedCentroid,sortIndexes]=sort(centroids);
    
    caxis([min(hit_centroid) max(hit_centroid)]);
    h = colorbar;
    % 添加颜色条，并设置字体颜色和范围
    set( h,'fontsize',10, 'Color', 'k');
    h.Label.String = 'centroid';%添加单位
    set(h,'fontsize',10,'Color', 'k');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 min(spec_data.raw_centroid)-0.15 ],'Rotation',360);
    
    % axis tight
    % axis equal
    figureandaxiscolors('w','k',strcat(dirname,' Spec'))
    
    xlim([min(x),max(x)])
    ylim([min(y),max(y)])
    zlim([min(z),max(z)])
    
    hold on
    
    
    %%%%%%%%%%%%%%%%%%
    ax2 = axes('Position', pos2);
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))])
    ylim([min(hit_centroid) max(hit_centroid)])
    xlabel('Time(s)')
    ylabel('Centroids')
    hold on
    % figTrajname=[dirname ' pH ratio' '.fig'];
    % saveas(hSpec,figTrajname,'fig');
    
    ax3 = axes('Position', pos3);
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))])
    ylim([min(x) max(x)])
    xlabel('Time(s)')
    ylabel('x (um)')
    hold on
    
    ax4 = axes('Position', pos4);
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))])
    ylim([min(y) max(y)])
    xlabel('Time(s)')
    ylabel('y (um)')
    hold on
    
    ax5 = axes('Position', pos5);
    figureandaxiscolors('w','k',dirname)
    xlim([0 round(max(time))])
    ylim([min(z) max(z)])
    xlabel('Time(s)')
    ylabel('z (um)')
    hold on
    
    disp_rate = 1; % EMCCD Frame
    cjlen = floor(numel(spec_data.raw_centroid)/disp_rate);
    
    cen_len = numel(spec_data.raw_centroid);
    spec_time=time(round(linspace(1, len, cen_len)));
    
    pos_imgs = spec_data.trackImg(:,1:16,:);
    spec_imgs = spec_data.trackImg(:,16:end,:);
    
    % video_name = strcat(dirname,'.mp4');
    % outputVideo = VideoWriter(video_name, 'MPEG-4');
    % open(outputVideo);
    
    for i=1:cjlen-1
    
        seg=(floor((i-1)*len/cjlen)+1):floor((i)*len/cjlen);
    
        p1=plot3(ax1, x(seg),y(seg),z(seg));
        tolerance = 1e-6;  
        cid=find(abs(hit_centroid-roundn(spec_data.raw_centroid(i),-2)) < tolerance);
        set(p1,'Color',cjet(cid,:),'LineWidth',1);
    
        plot(ax3, time(seg), x(seg),'Color',"#0072BD",'LineWidth',1);
        plot(ax4, time(seg), y(seg),'Color',"#0072BD",'LineWidth',1);
        plot(ax5, time(seg), z(seg),'Color',"#0072BD",'LineWidth',1);
    
        seg = ((i-1)*disp_rate+1:(i+1)*disp_rate);
        plot(ax2, spec_time(seg), spec_data.raw_centroid(seg),'r','LineWidth',1)    
    
    %     %%%%%
    %     drawnow;
    %     pause(0.01);
    %     currFrame = getframe(hScatter);
    %     writeVideo(outputVideo, currFrame);
    %     %%%%%%%%
    
    end
    % close(outputVideo);
    % saveas(hScatter,strcat(dirname, ' spec'), 'fig');

end