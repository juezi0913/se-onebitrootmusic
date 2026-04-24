function [merged_results, merged_mat_path, output_dir] = merge_se_rootmusic_with_baselines(new_method_mat, fig12_baseline_mat, fig38_baseline_mat, output_dir)
% Merge an SE-RootMUSIC-only run with previously saved baseline MAT files.
%
% Defaults:
%   new_method_mat      -> newest results_*.mat in results_paper8_v3
%   fig12_baseline_mat  -> results_20260319_140411.mat in current folder
%   fig38_baseline_mat  -> 20260324_103936/results_20260324_103936.mat
%   output_dir          -> merged_seroot_<timestamp>

    if nargin < 1 || isempty(new_method_mat)
        files = dir(fullfile(pwd, 'results_paper8_v3', 'results_*.mat'));
        if isempty(files)
            error('No new-method results_*.mat found in results_paper8_v3.');
        end
        [~, idx] = max([files.datenum]);
        new_method_mat = fullfile(files(idx).folder, files(idx).name);
    end
    if nargin < 2 || isempty(fig12_baseline_mat)
        fig12_baseline_mat = fullfile(pwd, 'results_20260319_140411.mat');
    end
    if nargin < 3 || isempty(fig38_baseline_mat)
        fig38_baseline_mat = fullfile(pwd, '20260324_103936', 'results_20260324_103936.mat');
    end
    if nargin < 4 || isempty(output_dir)
        output_dir = fullfile(pwd, ['merged_seroot_' datestr(now, 'yyyymmdd_HHMMSS')]);
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    Snew = load(new_method_mat, 'results');
    S12 = load(fig12_baseline_mat, 'results');
    S38 = load(fig38_baseline_mat, 'results');

    new_results = Snew.results;
    base12 = S12.results;
    base38 = S38.results;

    fig_fields = { ...
        'fig1_snr_3src_uncorr', ...
        'fig2_snapshots_3src_uncorr', ...
        'fig3_snr_4src_uncorr_close', ...
        'fig4_snapshots_4src_uncorr_close', ...
        'fig5_antennas_4src_uncorr_close', ...
        'fig6_separation_4src_uncorr', ...
        'fig7_snr_4src_corr_close', ...
        'fig8_snapshots_4src_corr_close'};

    merged_results = struct();
    merged_results.mode = 'merged_seroot';
    merged_results.output_dir = output_dir;
    merged_results.timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    merged_results.new_method_source = new_method_mat;
    merged_results.fig12_baseline_source = fig12_baseline_mat;
    merged_results.fig38_baseline_source = fig38_baseline_mat;

    for i = 1:numel(fig_fields)
        field_name = fig_fields{i};
        if i <= 2
            base_exp = base12.(field_name);
        else
            base_exp = base38.(field_name);
        end
        new_exp = new_results.(field_name);
        merged_results.(field_name) = merge_single_experiment(base_exp, new_exp);
    end

    merged_mat_path = fullfile(output_dir, sprintf('results_%s.mat', merged_results.timestamp));
    results = merged_results; %#ok<NASGU>
    save(merged_mat_path, 'results', '-v7.3');

    replot_saved_results_from_mat(merged_mat_path, fullfile(output_dir, ['replot_' merged_results.timestamp]));
end

function exp_out = merge_single_experiment(base_exp, new_exp)
    [base_algos, keep_idx, oracle_insert_idx] = remove_old_main_method(base_exp.algorithms);
    [new_algo, new_idx] = find_new_main_method(new_exp.algorithms);

    exp_out = base_exp;
    exp_out.algorithms = insert_algo_cell(base_algos, new_algo, oracle_insert_idx);

    field_list = { ...
        'mse_db', ...
        'mse_db_raw', ...
        'mse_linear_rad2', ...
        'mse_linear_deg2', ...
        'rmse_deg', ...
        'success_rate', ...
        'avg_runtime'};

    for i = 1:numel(field_list)
        fn = field_list{i};
        base_val = base_exp.(fn);
        base_val = base_val(keep_idx, :);
        new_val = new_exp.(fn)(new_idx, :);
        exp_out.(fn) = insert_row_block(base_val, new_val, oracle_insert_idx);
    end
end

function [algos_out, keep_idx, oracle_insert_idx] = remove_old_main_method(algos_in)
    keep_idx = true(1, numel(algos_in));
    oracle_insert_idx = numel(algos_in) + 1;
    for i = 1:numel(algos_in)
        name_i = get_algo_field(algos_in{i}, 'name', '');
        short_i = get_algo_field(algos_in{i}, 'short', '');
        if strcmpi(short_i, 'fgd') || strcmpi(name_i, 'FGD') || strcmpi(name_i, 'FGD-v3') ...
                || strcmpi(short_i, 'cl_rootmusic') || strcmpi(short_i, 'se_rootmusic') ...
                || strcmpi(short_i, 'tgu_rootmusic')
            keep_idx(i) = false;
        end
        if contains(lower(name_i), 'oracle') && oracle_insert_idx == numel(algos_in) + 1
            oracle_insert_idx = sum(keep_idx(1:(i-1))) + 1;
        end
    end
    keep_idx = find(keep_idx);
    algos_out = algos_in(keep_idx);
end

function [algo, idx] = find_new_main_method(algos_in)
    idx = [];
    for i = 1:numel(algos_in)
        name_i = get_algo_field(algos_in{i}, 'name', '');
        short_i = get_algo_field(algos_in{i}, 'short', '');
        if strcmpi(short_i, 'se_rootmusic') || strcmpi(name_i, 'SE-RootMUSIC')
            idx = i;
            break;
        end
    end
    if isempty(idx)
        error('SE-RootMUSIC was not found in the new-method result file.');
    end
    algo = algos_in{idx};
end

function cell_out = insert_algo_cell(cell_in, algo, insert_idx)
    insert_idx = min(max(insert_idx, 1), numel(cell_in) + 1);
    cell_out = [cell_in(1:(insert_idx-1)), {algo}, cell_in(insert_idx:end)];
end

function Y = insert_row_block(Ybase, ynew, insert_idx)
    insert_idx = min(max(insert_idx, 1), size(Ybase, 1) + 1);
    Y = [Ybase(1:(insert_idx-1), :); ynew; Ybase(insert_idx:end, :)];
end

function val = get_algo_field(algo, field_name, default_val)
    if isstruct(algo) && isfield(algo, field_name)
        val = algo.(field_name);
    else
        val = default_val;
    end
end
