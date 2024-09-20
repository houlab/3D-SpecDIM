function centroid=fitCentroid(specImg, specCurve, cx, cy)
    
    % 自定义双峰高斯分布模型
    gaussianModel = @(p, x) p(1) * exp(-((x - p(2)) / p(3)).^2);

    % 初始参数猜测
    initialGuess_psf = [1, 8, 0.5];   
    psf_curve = specCurve(1:16);
    spec_curve = specCurve(17:80);

    [~,init_value]= max(spec_curve);
    initialGuess_spec = [1, init_value, 0.5];

    try
        % 使用 fit 函数进行拟合
        nlModel_psf = fitnlm((1:length(psf_curve))',psf_curve',gaussianModel,initialGuess_psf);
        fitRMSE_psf = nlModel_psf.RMSE;

        nlModel_spec  = fitnlm((1:length(spec_curve))',spec_curve',gaussianModel,initialGuess_spec);
        fitRMSE_spec = nlModel_spec.RMSE;
    catch
        centroid = 0;
        % disp("can not fit the curve!")
        return
    end
    pos = nlModel_psf.Coefficients.Estimate(2);
    spec = nlModel_spec.Coefficients.Estimate(2)+16;

    pix_dis = spec - pos;
    % peak2_coordinate = fittedParams(5);

    if pos > 16 || pos < 1 || spec > 80 || spec < 16 ||...
            pix_dis > 80 || pix_dis < 0 || fitRMSE_spec > 0.4 || fitRMSE_psf > 0.1
        centroid = 0;
        return
    end

    f_sp = @(x)(0.02215*x.^2+3.077*x+591.6);
    if spec >=16 && spec < 40
        centroid = f_sp(pix_dis-42);
    elseif spec >=40 && spec < 60
        centroid = f_sp(pix_dis-35);
    elseif spec >=60 && spec < 80
        centroid = f_sp(pix_dis-33);
    else
        centroid = 0;
    end
    

% %         if centroid > 600
%         figure(12)
%         subplot(121)
%         imshow(specImg,[]);
%         hold on
%         plot(cy,cx,'.','MarkerSize',30);
%         subplot(122)
%         plot(specCurve)
% % 
% %             % 显示拟合结果
%         figure(13)
%         plot((1:length(specCurve)), specCurve, 'b.');
%         hold on;
%         plot(17:80,predict(nlModel_spec,(1:64)'), 'r-');
%         plot(predict(nlModel_psf,(1:16)'), 'r-');
%         hold off;
%         legend('Data', 'Fitted Double Gaussian');
       % end

end
