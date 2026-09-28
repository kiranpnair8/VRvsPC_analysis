clear
clc
close all
% plot_topos_from_table.m
% Assumes: kl_values.csv, mmd_values.csv, eloc32_BIOSEMI.txt are in pwd or path

% --------- user config ----------
channels16 = {"Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"};
eloc_file = 'eloc16_BIOSEMI.loc';   % your 32-eloc file
kl_csv = 'kl_values.csv';
mmd_csv = 'mmd_values.csv';

% Choose subjects to plot: either subject row index (1-based) OR subject name.
% Example: pick subject rows 1 and 21 (change as needed)
subjects_to_plot = [13, 2, 4, 3, 5, 6];  % indices of rows in the CSV table
% ---------------------------------

% Read tables
Tkl  = readtable(kl_csv);    % columns: Subjects, Fp1, Fp2, ...
Tmmd = readtable(mmd_csv);

% Get channel columns (ensure they exist)
% If your CSV has exact channel names, this will work. Otherwise adapt names.
chanNamesInTable = channels16; % cell array
for c = 1:numel(chanNamesInTable)
    if ~ismember(chanNamesInTable{c}, Tkl.Properties.VariableNames)
        error('Channel %s not found in KL table columns.', chanNamesInTable{c});
    end
end

% Read eloc file into channel info using readlocs (EEGLAB helper)
% readlocs can accept a filename or return structure [eloc] = readlocs(file)
elocs = readlocs(eloc_file);   % returns struct array with fields .labels .theta .radius etc.

% Normalize eloc labels to plain text without trailing dots, spaces
elocLabels = cellstr({elocs.labels});
elocLabelsClean = cellfun(@(s) strtrim(strrep(s, '.', '')), elocLabels, 'UniformOutput', false);

% Map our 16 channel names -> indices in elocLabelsClean
map16_to_32 = nan(size(channels16));
for k = 1:numel(channels16)
    idx = find(strcmpi(channels16{k}, elocLabelsClean), 1);
    if isempty(idx)
        % Try alternative matching (e.g. Fp1 vs Fp1.)
        idx = find(contains(elocLabelsClean, channels16{k}, 'IgnoreCase', true), 1);
    end
    if ~isempty(idx)
        map16_to_32(k) = idx;
    else
        warning('Channel %s not found in eloc file. It will be left as NaN.', channels16{k});
    end
end




%%%%%DEBUG
%% Debug / robust column mapping for channels
% show what readtable actually produced
fprintf('KL table variable names:\n');
disp(Tkl.Properties.VariableNames');

% If the table header includes extra rows or weird names, show first few rows
disp('First few rows (KL table):');
disp(Tkl(1:min(5,height(Tkl)), :));

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

%% --- Build vals_all robustly using numeric column indices (chanIdx) ---
vals_all = [];
for s = 1:height(Tkl)
    rowvals = Tkl{ s, chanIdx };   % 1 x 16 (positional indexing)
    % convert if cell
    if iscell(rowvals)
        rownum = nan(1,numel(rowvals));
        for k = 1:numel(rowvals)
            v = rowvals{k};
            if isnumeric(v)
                rownum(k) = double(v);
            else
                numv = str2double(string(v));
                if ~isnan(numv)
                    rownum(k) = numv;
                else
                    rownum(k) = NaN;
                end
            end
        end
        rowvals = rownum;
    end
    vals_all = [vals_all; rowvals(:)']; %#ok<AGROW>
end

% compute global vmax (symmetric) from both KL and MMD sheets if present
% use chanIdx to extract correct columns
kl_vals = Tkl{:, chanIdx};
mmd_vals = Tmmd{:, chanIdx};

% ensure numeric
if iscell(kl_vals), kl_vals = cellfun(@(c) str2double(string(c)), kl_vals); end
if iscell(mmd_vals), mmd_vals = cellfun(@(c) str2double(string(c)), mmd_vals); end

all_vals = [kl_vals(:); mmd_vals(:)];
all_vals = all_vals(isfinite(all_vals));
if isempty(all_vals)
    vmax = 1;
else
    vmax = max(abs(all_vals));
end
fprintf('Found vmax = %g\n', vmax);

% -------- plotting subjects (use subjects_to_plot) ----------
for sidx = 1:numel(subjects_to_plot)
    row = subjects_to_plot(sidx);
    % get a readable subject name (handle numeric or string)
    if ismember('Subjects', Tkl.Properties.VariableNames)
        subjval = Tkl.Subjects(row);
        if isnumeric(subjval)
            subjname = sprintf('subject_%d', subjval);
        else
            subjname = string(subjval);
        end
    else
        subjname = sprintf('row%d', row);
    end

    % extract the KL and MMD channel values using positional indices
    kl_row = Tkl{row, chanIdx};   % 1x16 (may be numeric or cell)
    mmd_row = Tmmd{row, chanIdx};

    % convert cells to numeric if needed
    if iscell(kl_row)
        kl_row = cellfun(@(v) double(v).*(isnumeric(v)) + str2double(string(v)).*(~isnumeric(v)), kl_row);
    end
    if iscell(mmd_row)
        mmd_row = cellfun(@(v) double(v).*(isnumeric(v)) + str2double(string(v)).*(~isnumeric(v)), mmd_row);
    end

    % Build 32-length vectors (NaN default) and map the 16 channels into 32 eloc positions
    Vl32_kl = nan(numel(elocs),1);
    Vl32_mmd = nan(numel(elocs),1);
    for c = 1:numel(channels16)
        idx32 = map16_to_32(c);
        if ~isnan(idx32)
            Vl32_kl(idx32) = kl_row(c);
            Vl32_mmd(idx32) = mmd_row(c);
        end
    end

    % clamp negative MMD for display (optional)
    Vl32_mmd(Vl32_mmd < 0) = 0;

    % Plot side-by-side topoplots
    fig = figure('Position',[100 100 1200 520],'Color','w');
    subplot(1,2,1);
    try
        topoplot(Vl32_kl, eloc_file, 'maplimits', [-vmax vmax], 'style', 'both', 'gridscale', 67);
    catch
        topoplot1(Vl32_kl, eloc_file, 'maplimits', [-vmax vmax], 'style', 'both', 'gridscale', 67);
    end
    title(sprintf('KL divergence — %s', char(subjname)),'Interpreter','none');
    colorbar; caxis([-vmax vmax]);

    subplot(1,2,2);
    try
        topoplot(Vl32_mmd, eloc_file, 'maplimits', [0 vmax], 'style', 'both', 'gridscale', 67);
    catch
        topoplot1(Vl32_mmd, eloc_file, 'maplimits', [0 vmax], 'style', 'both', 'gridscale', 67);
    end
    title(sprintf('MMD — %s', char(subjname)),'Interpreter','none');
    colorbar; caxis([0 vmax]);

    outname = sprintf('topo_subj_row%d_%s.png', row, char(subjname));
    saveas(fig, outname);
    fprintf('Saved %s\n', outname);
    close(fig);
end