function para = fig_1us_v1(data_interval,para)
num_data_interval = length(data_interval);
disp_num = ['以',num2str(para.times_fit_interval*1e6),'us 间隔分割的数据总数',num2str(num_data_interval)];
disp(disp_num)
%%
stratp = input('开始次数 = ');
endp = input('结束次数 = ');

if stratp > num_data_interval
    stratp = 1;
    disp('输入开始次数过大，调整为1')
end

if (endp - stratp) > num_data_interval
endp = num_data_interval - stratp - 1;
disp(['输入间隔过大，调整到',num2str(endp)]);
end

para.stratp = stratp;
para.endp = endp;

TrackData_us = data_interval(stratp:endp,:);
times_fit_interval = para.times_fit_interval;
dirname = para.dirname;
%
hALLinOne = figure;
visibility='on';
set(hALLinOne,'Visible',visibility);
set(hALLinOne,'Renderer','OpenGL')
set(0,'CurrentFigure',hALLinOne)

ax1 = axes;

set(ax1,'Visible',visibility);
% clim([TrackData_us(1,4) TrackData_us(end,4)]);
h = colorbar;
set( h,'fontsize',10, 'Color', 'w');
h.Label.String = 'time(s) ';%添加单位
set(h,'fontsize',10,'FontWeight','bold','Color', 'w');
h_text = h.Label;%将“cm”的句柄赋值给h_text
set(h_text,'Position',[ 0.5 -0.1 ],'Rotation',360);

traj3D_colormap(TrackData_us(:,1),TrackData_us(:,2),TrackData_us(:,3));
view([1 1 1]);
axis image
xlabel('X [\mum]');
ylabel('Y [\mum]');
zlabel('Z [\mum]');
title3Dname = {[dirname,' 时间间隔 = ', num2str(times_fit_interval*1e6),' us'],...
    [' 时间从 ',num2str(round(data_interval(stratp,4)*1e3)),...
    'ms ~ ',num2str(round(data_interval(endp,4)*1e3)),' ms']};
figureandaxiscolors('k','w',title3Dname)

% save_file_path_name = [para.save_file_path,dirname];
% saveas(hALLinOne,save_file_path_name,'fig');

end