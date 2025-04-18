function [time,msd,D,r]=msdcalc(x,y,z,samplingRate)
xyz=[x y z];
nData = size(xyz,1); %# number of data points
numberOfDeltaT = floor(nData/4); 
%# for MSD, dt should be up to 1/4 of number of data points

msd = zeros(numberOfDeltaT,3); %# We'll store [mean, std, n]
% hWait=waitbar(0,'MSD','Color','w');
% set(hWait,'Position',[249 260.25 270 56.25])
%# calculate msd for all deltaplotT's
for dt = 1:numberOfDeltaT
   deltaCoords = xyz(1+dt:end,:) - xyz(1:end-dt,:);
   squaredDisplacement = sum(deltaCoords.^2,2); %# dx^2+dy^2+dz^2

   msd(dt,1) = mean(squaredDisplacement); %# average
   msd(dt,2) = std(squaredDisplacement); %# std
   msd(dt,3) = length(squaredDisplacement); %# 
   % waitbar(dt/numberOfDeltaT,hWait)
end
time=(1:length(msd(:,1)))/samplingRate;

%Diffusion coefficient in micron^2/sec
options = optimoptions('lsqcurvefit', 'Display', 'off');
D = lsqcurvefit(@slopeonly,6,time',msd(:,1), [], [], options)/6;

%Particle radius in nm
r=(1/((D*1e-12)*6*pi/1.381e-23/293.15*0.17970))*1e9; 
% 90 wt% -> 0.17970 85 wt% -> 0.090987 0 wt% -> 1.05e-3


hMSD=figure;
set(0,'CurrentFigure',hMSD);
upperEnvelope = msd(:,1) + msd(:,2)./sqrt(msd(:,3)); % 上包络
lowerEnvelope = msd(:,1) - msd(:,2)./sqrt(msd(:,3)); % 下包络
fill([time, fliplr(time)], [upperEnvelope', fliplr(lowerEnvelope')], ...
     [0.8, 0.8, 0.8], 'EdgeColor', 'none'); % 灰色包络，边缘无颜色
hold on
plot(time, msd(:,1), 'b-', 'LineWidth', 0.5);

SS_res = sum((msd(:,1)' - 6 * D * time).^2);
SS_tot = sum((msd(:,1) - mean(msd(:,1))).^2);
R2 = 1 - (SS_res / SS_tot);

hold('on');
% errorbar(time,msd(:,1),msd(:,2)./sqrt(msd(:,3)));
plot(time,6*D*time,'r--','LineWidth', 4);
lims=[min(time) max(time) min(msd(:,1)) max(msd(:,1))];
text(0.1*lims(2),0.9*lims(4),sprintf('D = %g',D),'Color','k');
text(0.1*lims(2),0.8*lims(4),sprintf('r = %g',r),'Color','k');
text(0.1*lims(2),0.7*lims(4),sprintf('R^2 = %g',R2),'Color','k');
xlabel('Lag time (s)');
ylabel('MSD (\mum^2)');
grid on
axis tight