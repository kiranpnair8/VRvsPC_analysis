clear
clc
close all
% plot_topos_from_table.m
% Assumes: kl_values.csv, mmd_values.csv, eloc32_BIOSEMI.txt are in pwd or path

% --------- user config ----------
channels16 = {"Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"};

kl_csv = 'kl_values.csv';
mmd_csv = 'mmd_values.csv';

% Choose subjects to plot: either subject row index (1-based) OR subject name.
% Example: pick subject rows 1 and 21 (change as needed)
subjects_to_plot = [13, 2, 4, 3, 5, 6];  % indices of rows in the CSV table
% ---------------------------------

% Read tables
Tkl  = readtable(kl_csv);    % columns: Subjects, Fp1, Fp2, ...
Tmmd = readtable(mmd_csv);

% canonicalize variable names (remove dots/spaces, lowercase)
varnames = Tkl.Properties.VariableNames;
varnames_clean = cellfun(@(s) lower(strrep(strrep(s,'.',''),' ','')), varnames, 'UniformOutput', false);

% canonicalize desired channel names
channels16 = {"Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"};
channels16_clean = cellfun(@(s) lower(strrep(s,'.','')), channels16, 'UniformOutput', false);

% find best matches
chanIdx = nan(1,numel(channels16));
for k = 1:numel(channels16_clean)
    % exact match first
    idx = find(strcmp(channels16_clean{k}, varnames_clean), 1);
    if isempty(idx)
        % fallback: contains
        idx = find(contains(varnames_clean, channels16_clean{k}), 1);
    end
    if ~isempty(idx)
        chanIdx(k) = idx;
    else
        warning('Channel %s not found in table variable names.', channels16{k});
    end
end

%% ----- group-average topoplots (KL and MMD) -----
% Assumptions:
% - Tkl and Tmmd are readtables with the same ordering of rows (subjects)
% - chanIdx is the numeric index mapping channels16 -> table columns (1x16)
% - eloc16_file is the 16-channel eloc file path (created earlier)
% - channels16 is the 1x16 cell array of channel names in the same order

eloc16_file = 'eloc16_BIOSEMI.loc';  % or whichever you created
% -------- get numeric matrices (subjects x 16) --------
kl_mat = Tkl{:, chanIdx};   % may be numeric or cell
mmd_mat = Tmmd{:, chanIdx};

% If readtable returned cells/strings convert to numeric (robust)
if iscell(kl_mat)
    kl_mat = cellfun(@(c) str2double(string(c)), kl_mat);
end
if iscell(mmd_mat)
    mmd_mat = cellfun(@(c) str2double(string(c)), mmd_mat);
end

% kl_mat and mmd_mat should now be [nSubjects x 16]
[nsub, nch] = size(kl_mat);

% Compute per-channel mean ignoring NaNs (this drops blanks)
mean_kl = nanmean(kl_mat, 1);   % 1 x 16
mean_mmd = nanmean(mmd_mat, 1); % 1 x 16

% Compute per-channel N (non-NaN counts) and SEM if you want
N_kl = sum(isfinite(kl_mat), 1);           % number of subjects with data for each channel
N_mmd = sum(isfinite(mmd_mat), 1);
std_kl = nanstd(kl_mat, 0, 1);
std_mmd = nanstd(mmd_mat, 0, 1);
sem_kl = std_kl ./ sqrt(max(N_kl,1));
sem_mmd = std_mmd ./ sqrt(max(N_mmd,1));

% Optional: warn if any channel has very few subjects
minN = 5; % choose threshold you consider acceptable
if any(N_kl < minN)
    warning('Some KL channels have < %d subjects: %s', minN, mat2str(find(N_kl < minN)));
end
if any(N_mmd < minN)
    warning('Some MMD channels have < %d subjects: %s', minN, mat2str(find(N_mmd < minN)));
end

% Convert to column vectors in channels16 order (for topoplot)
Vl16_mean_kl  = mean_kl(:);
Vl16_mean_mmd = mean_mmd(:);

% Optionally mask channels with 0 valid subjects (leave NaN so topoplot doesn't show)
Vl16_mean_kl(N_kl == 0)   = NaN;
Vl16_mean_mmd(N_mmd == 0) = NaN;

% Determine plotting scale (consistent colorbar)
all_vals = [Vl16_mean_kl(:); Vl16_mean_mmd(:)];
all_vals = all_vals(isfinite(all_vals));
if isempty(all_vals)
    vabs = 1;
else
    vabs = max(abs(all_vals));
end

% Plot group maps
figure('Color','w','Position',[150 150 1100 480]);
subplot(1,2,1);
try
    topoplot(Vl16_mean_kl, eloc16_file, 'interplimits','electrodes', 'maplimits', [-vabs vabs], 'style','both');
catch
    topoplot1(Vl16_mean_kl, eloc16_file, 'interplimits','electrodes', 'maplimits', [-vabs vabs], 'style','both');
end
title(sprintf('Group mean KL divergence', nsub));
colorbar; caxis([-vabs vabs]);

subplot(1,2,2);
try
    topoplot(Vl16_mean_mmd, eloc16_file, 'interplimits','electrodes', 'maplimits', [0 vabs], 'style','both');
catch
    topoplot1(Vl16_mean_mmd, eloc16_file, 'interplimits','electrodes', 'maplimits', [0 vabs], 'style','both');
end
title('Group mean MMD');
colorbar; caxis([0 vabs]);

% Save figure
saveas(gcf, 'group_topos_KL_MMD.png');
fprintf('Saved group_topos_KL_MMD.png\n');

% Optional: write the per-channel means, N and SEM to CSV for reporting
out = table(channels16(:), mean_kl(:), sem_kl(:), N_kl(:), mean_mmd(:), sem_mmd(:), N_mmd(:), ...
    'VariableNames', {'Channel','Mean_KL','SEM_KL','N_KL','Mean_MMD','SEM_MMD','N_MMD'});
writetable(out, 'group_channel_stats.csv');
fprintf('Saved group_channel_stats.csv\n');
