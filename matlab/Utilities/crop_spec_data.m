function spec_data_crop = crop_spec_data(spec_data, startframe, endframe)
    spec_data_crop.trackImg = spec_data.trackImg(:,:,startframe:endframe);
    spec_data_crop.posImg = spec_data.posImg(:,:,startframe:endframe);
    spec_data_crop.frame = spec_data.frame(startframe:endframe,:);
    spec_data_crop.raw_centroid = spec_data.raw_centroid(startframe:endframe,:);
    spec_data_crop.pos = spec_data.pos(startframe:endframe,:);
    spec_data_crop.snr = spec_data.snr(startframe:endframe,:);
    spec_data_crop.trackCurve = spec_data.trackCurve(startframe:endframe,:);
end