function plot_far_frr_bars(FAR, FRR, titleStr, models)

nModels = numel(models);

data = [
    FAR(:,1), FRR(:,1), ...
    FAR(:,2), FRR(:,2)
];

figure('Color','w','Position',[200 200 900 500]);

b = bar(data, 'grouped');
set(b,'LineWidth',1.2);

pcColor = [0.20 0.40 0.80];
vrColor = [0.85 0.33 0.20];

b(1).FaceColor = pcColor;
b(2).FaceColor = pcColor;
b(3).FaceColor = vrColor;
b(4).FaceColor = vrColor;

b(1).FaceAlpha = 1.0;
b(2).FaceAlpha = 0.45;
b(3).FaceAlpha = 1.0;
b(4).FaceAlpha = 0.45;

set(gca, ...
    'XTick', 1:nModels, ...
    'XTickLabel', models, ...
    'FontSize', 11, ...
    'LineWidth', 1.2);

xtickangle(15);
ylabel('Percentage (%)');
%title(titleStr);

ylim([0 65]);
grid on;

legend({'FAR (PC)','FRR (PC)','FAR (VR)','FRR (VR)'}, ...
       'Location','northwest');

% -------- SAVE FIGURE --------
fname = regexprep(titleStr,'[^a-zA-Z0-9]','_');
exportgraphics(gcf, [fname '.png'], 'Resolution', 300);

end
