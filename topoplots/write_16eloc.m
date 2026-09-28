close all
clear 
clc
% make_eloc16_from_32.m
channels16 = {"Fp1","Fp2","Fc5","Fz","Fc6","T7","Cz","T8","P7","P3","Pz","P4","P8","O1","Oz","O2"};
eloc32_file = 'eloc32_BIOSEMI.loc'; % or .loc whatever you have
out16_file = 'eloc16_BIOSEMI.loc';  % output

% read 32-eloc (EEGLAB helper)
elocs32 = readlocs(eloc32_file);    % returns structure array with .labels .theta .radius

% canonicalize labels (remove dots/spaces)
labels32 = cellstr({elocs32.labels});
labels32_clean = cellfun(@(s) strtrim(strrep(s,'.','')), labels32, 'UniformOutput', false);

keep_idx = nan(1,numel(channels16));
for k = 1:numel(channels16)
    idx = find(strcmpi(channels16{k}, labels32_clean), 1);
    if isempty(idx)
        idx = find(contains(labels32_clean, channels16{k}, 'IgnoreCase', true), 1);
    end
    keep_idx(k) = idx;
end

if any(isnan(keep_idx))
    error('Some channels not found in eloc32. Missing: %s', strjoin(channels16(isnan(keep_idx)), ', '));
end

eloc16 = elocs32(keep_idx); % keep ordering corresponding to channels16

% Write a simple ASCII eloc file (angle radius label) like your original format
fid = fopen(out16_file,'w');
for i = 1:numel(eloc16)
    % follow format: <index> <degrees> <radius> <label>
    fprintf(fid, '%d\t%g\t%g\t%s\n', i, eloc16(i).theta, eloc16(i).radius, strrep(eloc16(i).labels,'.',''));
end
fclose(fid);
fprintf('Wrote %s\n', out16_file);

% Example: using topoplot with new eloc file
% Suppose Vl16 is a 16x1 vector in channels16 order:
% topoplot(Vl16, out16_file, 'interplimits','electrodes','maplimits',[-v v],'style','both');