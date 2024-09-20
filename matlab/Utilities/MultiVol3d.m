% MultiVol3d
function MultiVol3d(DataArrayStructure)
setappdata(gcf,'index',2)
setappdata(gcf,'DataArray',DataArrayStructure)
vol3d('cdata',smooth3(DataArrayStructure(1).FrameData), ...
    'delta', DataArrayStructure(1).delta, ...
    'frame_id', DataArrayStructure(1).frame_id)
axis image
end


