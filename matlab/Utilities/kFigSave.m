function kFigSave(hFig,label)
figure(hFig)
set(hFig,'WindowStyle','normal')
set(hFig,'PaperPositionMode','auto')
set(hFig,'Units','inches')
pause(1)
set(hFig,'Position',[1 1 4 4])
root='D:\Data';
appRoot=[root '\' date2ser];
mkdir(appRoot)

saveas(hFig,[appRoot '\' date2ser ' ' label])

exportfig(hFig,[appRoot '\' date2ser ' ' label '.png'],'Format','png','Resolution',300,'Color','cmyk')
%export_fig([appRoot '\' date2ser ' ' label],'-png','-r300')%,'-transparent')
%export_fig([appRoot '\' date2ser ' ' label],'-eps','-r300')%,'-transparent')
set(hFig,'Renderer','Painters')
exportfig(hFig,[appRoot '\' date2ser ' ' label '.eps'],'Format','eps','Resolution',300,'Color','cmyk')
set(hFig,'WindowStyle','docked')
