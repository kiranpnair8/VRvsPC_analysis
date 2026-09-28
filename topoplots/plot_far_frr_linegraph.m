%% ================================
% FAR–FRR Trade-off Plots (Organized Legend)
% X = FAR, Y = FRR
% 3 Figures: Same-Env, Cross-Env, Mixed Training
% Legend order: Model-wise pairing (PC then VR)
% ================================

clc; clear; close all;

%% ----------------
% Model names
% ----------------
models = {'Reve+MLP','Reve+SVM','Reve+KNN','Reve+RF','EEGNet'};
nModels = numel(models);

% Colors (one per model)
colors = lines(nModels);

% Markers
pc_marker = 'o';
vr_marker = 's';

%% ----------------
% Data (FAR, FRR)  [PC, VR]
% ----------------

% ---------- SAME ENV ----------
FAR_same = [
    12.24 15.01;
    18.52 20.11;
    19.58 20.83;
    34.39 33.93;
    9.39  14.02
];

FRR_same = [
    9.92  12.90;
    18.06 17.86;
    50.20 51.59;
    32.14 30.75;
    6.55  16.87
];

% ---------- CROSS ENV ----------
FAR_cross = [
    16.07 13.69;
    20.63 21.49;
    17.00 22.55;
    30.75 43.25;
    13.82 16.67
];

FRR_cross = [
    40.08 44.05;
    34.13 30.36;
    59.52 58.13;
    43.06 39.68;
    42.46 47.62
];

% ---------- MIXED TRAIN (PC+VR) ----------
FAR_mixed = [
    8.80  9.99;
    15.81 18.58;
    19.05 20.44;
    34.19 30.82;
    9.59  7.34
];

FRR_mixed = [
    9.52  7.74;
    17.66 17.46;
    49.01 52.78;
    30.95 32.54;
    7.54  5.16
];

%% ================================
% Helper: plot one condition with paired legend
% ================================
plot_condition(FAR_same,  FRR_same,  'Same-Environment FAR–FRR Trade-off',  models, colors, pc_marker, vr_marker);
plot_condition(FAR_cross, FRR_cross, 'Cross-Environment FAR–FRR Trade-off', models, colors, pc_marker, vr_marker);
plot_condition(FAR_mixed, FRR_mixed, 'Mixed Training (PC + VR) FAR–FRR Trade-off', models, colors, pc_marker, vr_marker);

%% ================================
% Local function (MATLAB allows local functions at end of script)
% ================================
function plot_condition(FAR, FRR, titleStr, models, colors, pc_marker, vr_marker)

nModels = numel(models);

figure('Color','w','Position',[200 200 750 600]);
hold on;

hPC = gobjects(nModels,1);
hVR = gobjects(nModels,1);

for i = 1:nModels
    % PC point
    hPC(i) = plot(FAR(i,1), FRR(i,1), pc_marker, ...
        'Color', colors(i,:), ...
        'MarkerFaceColor', colors(i,:), ...
        'MarkerSize', 9, ...
        'LineWidth', 1.8);

    % VR point
    hVR(i) = plot(FAR(i,2), FRR(i,2), vr_marker, ...
        'Color', colors(i,:), ...
        'MarkerSize', 9, ...
        'LineWidth', 1.8);
end

title(titleStr);
xlabel('FAR (%)');
ylabel('FRR (%)');

grid on;
axis equal;
xlim([0 60]);
ylim([0 60]);

% ---- Legend: pair per model (PC then VR) ----
legendHandles = reshape([hPC hVR].', [], 1);

legendLabels = cell(2*nModels,1);
k = 1;
for i = 1:nModels
    legendLabels{k}   = [models{i} ' (PC)']; k = k + 1;
    legendLabels{k}   = [models{i} ' (VR)']; k = k + 1;
end

legend(legendHandles, legendLabels, 'Location','northwest');

end
