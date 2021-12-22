function handle = plotPsychometricCurve(perfPsych,cl,linestyle)
plotWhat = 'bin';
ca = gca;
plotFit = 1;
sh = cl;
applyDefaults = true;
try
  axes(ca);
catch
  figure(ca);
end

switch plotWhat
  case 'all'
    x    = perfPsych.perfPsych_xaxis;
    y    = perfPsych.perfPsychJ;
    l    = -perfPsych.perfPsychJSEM(:,1)+y;
    u    = perfPsych.perfPsychJSEM(:,2)-y;
    xl   = [-perfPsych.perfPsych_xaxis(end)-1 perfPsych.perfPsych_xaxis(end)+1];
    if plotFit
      fitx = perfPsych.fitAll.xaxis;
      fity = perfPsych.fitAll.curve;
    end
  case 'bin'
    x    = perfPsych.perfPsych_xaxisBins;
    y    = perfPsych.perfPsychJ_binned;
    l    = -perfPsych.perfPsychJ_binnedSEM(:,1)+y;
    u    = perfPsych.perfPsychJ_binnedSEM(:,2)-y;
    xl   = [-perfPsych.perfPsych_xaxisBins(end)-1 perfPsych.perfPsych_xaxisBins(end)+1];
    if plotFit
      fitx = perfPsych.fitAll.xaxis;
      fity = perfPsych.fitAll.curve;
    end
end
% y(x==0)=nan;
hold on
plot([0 0],[0 100],'--','color',[.7 .7 .7],'linewidth',0.8)
plot([x(1)-.5 x(end)+.5],[50 50],'--','color',[.7 .7 .7],'linewidth',0.8)
%if plotFit(1)
  handle.err = errorbar(x,y*100,l*100,u*100,'o','color',cl,'markersize',4,'markerfacecolor',cl);
  handle.fit = plot(fitx,fity*100,linestyle,'color',sh,'linewidth',2);
% else
%   y(x==0) = [];
%   l(x==0) = [];
%   u(x==0) = [];
%   x(x==0) = [];
%   if numel(plotFit) > 1
%     if plotFit(2)
%       handle  = errorbar(x,y*100,l*100,u*100,'o-','color',cl,'markersize',4,'markerfacecolor',cl);
%     else
%       handle  = errorbar(x,y*100,l*100,u*100,'o','color',cl,'markersize',4,'markerfacecolor',cl);
%     end
%   else
%     handle  = errorbar(x,y*100,l*100,u*100,'o-','color',cl,'markersize',4,'markerfacecolor',cl);
%   end
% end
if sum(isnan(x))== 1
    x(isnan(x))=[];
    box off;
    xlim([x(1)-1 x(end)+1]);
else
    box off
    xlim([x(1)-1 x(end)+1]);
end
ylim([0 100])
if applyDefaults
  set(gca,'fontsize',24,'xcolor','k','ycolor','k','ytick',0:25:100,'linewidth',0.8)
  ylabel('went contra (%)','fontsize',24,'color','k')
  xlabel('#contra - #ipsi towers','fontsize',24,'color','k')
end