function kImWrite(im,label,varargin)
root='D:\Data';
appRoot=[root '\' date2ser];
mkdir(appRoot)

imwrite(im,[appRoot '\' date2ser ' ' label],varargin{:})

