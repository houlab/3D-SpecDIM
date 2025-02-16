function plot_traj(trackDataFilt,dirname)
    visibility = 'on';
    decrate = 1;

    hALLinOne = figure;

    set(hALLinOne,'Visible',visibility);
    set(hALLinOne,'Renderer','OpenGL')
    set(0,'CurrentFigure',hALLinOne)
    
    pos1 = [0.1, 0.5, 0.35, 0.35]; % [left bottom width height]
    pos2 = [0.6, 0.5, 0.35, 0.35];
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
    set(h,'fontsize',10,'Color', 'k');
    h_text = h.Label;%将“cm”的句柄赋值给h_text
    set(h_text,'Position',[ 0.5 -0.1 ],'Rotation',360);

    traj3D_colormap(trackDataFilt(:,4),trackDataFilt(:,5),trackDataFilt(:,6));
    view([1 1 1]);
    
    xlabel('X [\mum]');
    ylabel('Y [\mum]');
    zlabel('Z [\mum]');
    figureandaxiscolors('w','k',dirname)
    axis image

    ax2 = axes('Position', pos2);
    set(ax2,'Visible',visibility);
    %Intensity and Instantaneous Velocity Figure
    int=trackDataFilt(:,11);
    time=(1:length(trackDataFilt(:,1)))/1000*decrate;
    plot(ax2,time,int,'r','LineWidth',1.5)
    xlabel('Time [sec]')
    ylabel('Intensity [counts/sec]')
    figureandaxiscolors('w','k',dirname)
    axis tight
    % ylim([0, 2e4])
    

    ax3 = axes('Position', pos3);
    plot(ax3,time,trackDataFilt(:,4),'LineWidth',1.5);
    axis square
    xlabel('Time [sec]');
    ylabel('X [\mum]');
    figureandaxiscolors('w','k',dirname)

    ax4 = axes('Position', pos4);
    plot(ax4,time,trackDataFilt(:,5),'LineWidth',1.5);
    axis square
    xlabel('Time [sec]');
    ylabel('Y [\mum]');
    figureandaxiscolors('w','k',dirname)

    ax5 = axes('Position', pos5);
    plot(ax5,time,trackDataFilt(:,6),'LineWidth',1.5);
    axis square
    xlabel('Time [sec]');
    ylabel('Z [\mum]');
    figureandaxiscolors('w','k',dirname)
end