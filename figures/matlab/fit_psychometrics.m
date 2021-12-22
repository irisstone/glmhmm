function ax = fit_psychometrics(colors,plot_title,file_path,save_path)
%% Plot laser on vs laser off data
%clf;
data = load(file_path);
data_cell = struct2cell(data); 
linestyles=["-","--"];


ax= [];
figure('Renderer', 'painters', 'Position', [10 10 290 320])

inits = [20,20];
lb = [[1e-6 1e-6 -inf -inf];[1e-6 1e-6 -inf -inf]];
ub = [[0.75 0.75 inf inf];[0.75 0.75 inf inf]];
psychBins = -16:4:16;

% Laser OFF
choice = data_cell{2};
diffT = data_cell{1};
perfPsych = psychometricFit(choice,diffT,psychBins,inits(1),lb(1,:),ub(1,:));
h = plotPsychometricCurve(perfPsych,colors(1,:),linestyles(1));
ax(1) = h.fit;

% Laser ON
choice = data_cell{4};
diffT = data_cell{3};
perfPsych = psychometricFit(choice,diffT,psychBins,inits(2),lb(2,:),ub(2,:));
h = plotPsychometricCurve(perfPsych,colors(2,:),linestyles(2));
ax(2) = h.fit;

%legend(ax,'laser off','laser on','Location','northwest');
title(plot_title,'FontSize',24, 'fontweight','normal');
set(gcf,'color','w');

saveas(gcf,save_path)


