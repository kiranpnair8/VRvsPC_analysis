%% ================================
% FAR–FRR Grouped Bar Plots
% ================================

clc; clear; close all;

models = {'REVE+MLP','REVE+SVM','REVE+k-NN','REVE+RF','EEGNet'};

%% ---------- SAME ENV ----------
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

plot_far_frr_bars(FAR_same, FRR_same, ...
    'Same-Environment FAR–FRR Comparison', models);

%% ---------- CROSS ENV ----------
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

plot_far_frr_bars(FAR_cross, FRR_cross, ...
    'Cross-Environment FAR–FRR Comparison', models);

%% ---------- MIXED TRAIN ----------
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

plot_far_frr_bars(FAR_mixed, FRR_mixed, ...
    'Mixed Training (PC + VR) FAR–FRR Comparison', models);