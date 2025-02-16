KT_Size = [ % KT Pos Array
-2 -1 -2 0 2 1 2 0 1 2 0 -2 -1 1 2 1 -1 -2 0 2 1 -1 0 -1 -2;
2 0 -2 -1 -2 0 2 1 -1 1 2 1 -1 -2 0 2 1 -1 -2 -1 1 2 0 -2 0
] + 3;


input_start_time = 284.85;
input_end_time = 343.01;
Find_start_time = find(data2(:,12)>input_start_time,1);
Find_end_time = length(find(data2(:,12)<input_end_time));

data2_ = data2(Find_start_time:Find_end_time,:);
data2_ = data2_(data2_(:,3)==1,:);
time_interv = 1; %s

MM = zeros(5,5,floor((input_end_time-input_start_time)/time_interv));

fwhm_x = zeros(floor((input_end_time-input_start_time)/time_interv),1);
fwhm_y = zeros(floor((input_end_time-input_start_time)/time_interv),1);

k = 1;
figure
for i = 1:time_interv:(input_end_time-input_start_time)

    Find_dt_1s = find((data2_(:,12)-data2_(1,12))>time_interv,1);
    data_1s_kt = data2_(1:Find_dt_1s,2);
    data2_ = data2_(Find_dt_1s:end,:);

    kt_counts = histcounts(data_1s_kt, 0:25);
    kt_data = zeros(5,5);
    for j = 1:25
        kt_data(KT_Size(1,j),KT_Size(2,j)) = kt_counts(j);
    end

    MM(:,:,k) = kt_data;
    % imagesc(kt_data);
    % title(k)
    % pause(1)

    x = 0.1:0.2:1;
    y = sum(kt_data,1);
    xq = linspace(min(x), max(x), 1000);  
    yq = interp1(x, y, xq, 'linear');  
    
    [maxVal, maxIndex] = max(yq);
    halfMax = (maxVal + min(yq)) / 2;
    
    leftIndex = find(yq(1:maxIndex) <= halfMax, 1, 'last');
    rightIndex = find(yq(maxIndex:end) <= halfMax, 1, 'first') + maxIndex - 1;
    
    if isempty(rightIndex)
        rightIndex = length(xq);
    end
    if isempty(leftIndex)
        leftIndex = 1;
    end

    fwhm_x(k) = xq(rightIndex) - xq(leftIndex);

    y = sum(kt_data,2)';
    xq = linspace(min(x), max(x), 1000);  
    yq = interp1(x, y, xq, 'linear');  
    pd = fitdist(normalize(yq,'range')', 'Normal');
    FWHM_y = 2 * sqrt(2 * log(2)) * pd.sigma;

    
    [maxVal, maxIndex] = max(yq);
    halfMax = (maxVal + min(yq)) / 2;

    leftIndex = find(yq(1:maxIndex) <= halfMax, 1, 'last');
    rightIndex = find(yq(maxIndex:end) <= halfMax, 1, 'first') + maxIndex - 1;

    if isempty(rightIndex)
        rightIndex = length(xq);
    end
    if isempty(leftIndex)
        leftIndex = 1;
    end

    fwhm_y(k) = xq(rightIndex) - xq(leftIndex);
    size_mito(k) = mean([fwhm_x(k),fwhm_y(k)]);
    k = k + 1;
    
end
%%
data = MM(:,:,5); 
figure;

subplot(3, 3, [4, 5, 7, 8]);
imagesc(data);
% colormap("hot")
colorbar;
title('Original Data');
xlabel('X-axis');
ylabel('Y-axis');
hold on;

x = 0.1:0.2:0.9;
y = sum(data,1);
xq = linspace(min(x), max(x), 1000);  
yq = interp1(x, y, xq, 'linear'); 


[maxVal, maxIndex] = max(yq);
halfMax = (maxVal + min(yq)) / 2;

leftIndex = find(yq(1:maxIndex) <= halfMax, 1, 'last');
rightIndex = find(yq(maxIndex:end) <= halfMax, 1, 'first') + maxIndex - 1;

FWHM_x = xq(rightIndex) - xq(leftIndex);

pd = fitdist(normalize(yq,'range')', 'Normal');
y_norm = pdf(pd, xq);

subplot(3, 3, [1, 2]);
plot(xq, y_norm, 'r', 'LineWidth', 2);
title('Gaussian Fit (X-axis)');
ylabel('Intensity');
xline((pd.mu - FWHM_x / 2), '--b');
xline((pd.mu + FWHM_x / 2), '--b');
text(pd.mu, max(yq), sprintf('FWHM = %.2f', FWHM_x), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

subplot(3, 3, [6, 9]);

y = sum(data,2)';
xq = linspace(min(x), max(x), 1000);  
yq = interp1(x, y, xq, 'linear'); 

[maxVal, maxIndex] = max(yq);
halfMax = (maxVal + min(yq)) / 2;

leftIndex = find(yq(1:maxIndex) <= halfMax, 1, 'last');
rightIndex = find(yq(maxIndex:end) <= halfMax, 1, 'first') + maxIndex - 1;

FWHM_y = xq(rightIndex) - xq(leftIndex);

pd = fitdist(normalize(yq,'range')', 'Normal');
y_norm = pdf(pd, xq);
plot(y_norm, xq, 'r', 'LineWidth', 2);
title('Gaussian Fit (Y-axis)');
xlabel('Intensity');
yline((mean(xq) - FWHM_y / 2), '--b');
yline((mean(xq) + FWHM_y / 2), '--b');
text(max(yq), mean(xq), sprintf('FWHM = %.2f', FWHM_y), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
set(gca, 'YDir', 'reverse'); 

axis tight;
hold off;

figure
plot((1:length(fwhm_y)).*time_interv,fwhm_y);
hold on
plot((1:length(fwhm_x)).*time_interv,fwhm_x);
% ylim([0.4 0.7])
grid on

