%%
function [cx,cy] = PSFfiting(trackPos)
    cx = 0;
    cy = 0;

    [height, width] = size(trackPos);
    % 定义高斯函数模型
    gaussianModel = @(p, x) p(1) * exp(-((x(:, 1) - p(2)).^2 + (x(:, 2) - p(3)).^2) / (2 * p(4)^2));
    % 创建网格坐标
    [x1, x2] = meshgrid(1:size(trackPos, 2), 1:size(trackPos, 1));
    x = [x1(:), x2(:)];
    % 初始化参数猜测值
    p0 = [max(trackPos(:)), size(trackPos, 2)/2, size(trackPos, 1)/2, 10];
    
    try
        % 使用 lsqcurvefit 进行拟合
        options = optimset('Display', 'off');
        params = lsqcurvefit(gaussianModel, p0, x, trackPos(:),[],[],options);
    catch exception
        disp(exception.message); 
        return
    end

    centerX = params(3);
    centerY = params(2);
    
    if centerX>1 && centerY>1 && centerX <= height && centerY <= width
        cx = centerX;
        cy = centerY;
    else
        cx = 0;
        cy = 0;
    end

end