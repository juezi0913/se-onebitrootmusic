function stats = run_seroot_globalrefine_stats(mode_str)
% run_seroot_globalrefine_stats
% -------------------------------------------------------------
% Collect trigger / accept statistics of the gated global refinement
% in the current clean-fixed SE-RootMUSIC paper-version.
% -------------------------------------------------------------

    if nargin < 1 || isempty(mode_str)
        mode_str = 'full';
    end

    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);

    rng_seed = 20260316;
    rng(rng_seed);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_dir = fullfile(script_dir, sprintf('seroot_globalrefine_stats_%s', timestamp));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    [mc_times, rootloop_opts, metric_opts] = get_mode_defaults(mode_str);
    scene_defs = get_scene_definitions(mc_times);

    stats = struct();
    stats.timestamp = timestamp;
    stats.mode = mode_str;
    stats.output_dir = output_dir;
    stats.rng_seed = rng_seed;
    stats.rootloop_opts = rootloop_opts;
    stats.scene_names = cell(1, numel(scene_defs));
    stats.trigger_rate = nan(1, numel(scene_defs));
    stats.accept_rate = nan(1, numel(scene_defs));
    stats.accept_given_trigger = nan(1, numel(scene_defs));
    stats.mean_shift_deg = nan(1, numel(scene_defs));

    for is = 1:numel(scene_defs)
        sc = scene_defs{is};
        fprintf('\n========================================================\n');
        fprintf('Running global-refine stats for %s\n', sc.label);
        fprintf('========================================================\n');

        trig_cnt = 0;
        acc_cnt = 0;
        shift_sum = 0;
        shift_cnt = 0;
        trial_cnt = 0;

        xvals = sc.cfg.scan_values(:).';
        xmask = sc.xmask(xvals);
        xvals = xvals(xmask);

        for ix = 1:numel(xvals)
            val = xvals(ix);
            [theta_true_deg, M, N, snr_db] = resolve_scan_setting(sc.cfg, val, metric_opts);
            fprintf('  x = %g | M=%d N=%d SNR=%g\n', val, M, N, snr_db);

            data0 = generate_doa_data(theta_true_deg, M, N, snr_db, sc.cfg.rho12, sc.cfg.amp_vec);
            for mc = 1:sc.cfg.mc_times
                if mc == 1
                    data = data0;
                else
                    data = generate_doa_data(theta_true_deg, M, N, snr_db, sc.cfg.rho12, sc.cfg.amp_vec);
                end

                est = se_rootmusic_cleanproj_1bit_doa_estimator(data.Y_onebit, sc.cfg.K, rootloop_opts, data);
                info = est.global_refine;
                trial_cnt = trial_cnt + 1;
                if info.triggered
                    trig_cnt = trig_cnt + 1;
                    if info.accepted
                        acc_cnt = acc_cnt + 1;
                        shift_sum = shift_sum + mean(abs(info.theta_shift_deg));
                        shift_cnt = shift_cnt + 1;
                    end
                end
            end
        end

        stats.scene_names{is} = sc.label;
        stats.trigger_rate(is) = trig_cnt / max(trial_cnt, 1);
        stats.accept_rate(is) = acc_cnt / max(trial_cnt, 1);
        stats.accept_given_trigger(is) = acc_cnt / max(trig_cnt, 1);
        if shift_cnt > 0
            stats.mean_shift_deg(is) = shift_sum / shift_cnt;
        else
            stats.mean_shift_deg(is) = 0;
        end
    end

    T = table( ...
        string(stats.scene_names(:)), ...
        stats.trigger_rate(:), ...
        stats.accept_rate(:), ...
        stats.accept_given_trigger(:), ...
        stats.mean_shift_deg(:), ...
        'VariableNames', {'Scene', 'TriggerRate', 'AcceptRate', 'AcceptGivenTrigger', 'MeanShiftDeg'});
    stats.summary_table = T;

    writetable(T, fullfile(output_dir, sprintf('seroot_globalrefine_stats_%s.csv', timestamp)));
    write_markdown_summary(T, fullfile(output_dir, sprintf('seroot_globalrefine_stats_%s.md', timestamp)));
    plot_stats_bar(T, fullfile(output_dir, sprintf('seroot_globalrefine_stats_%s.png', timestamp)));

    save(fullfile(output_dir, sprintf('seroot_globalrefine_stats_%s.mat', timestamp)), 'stats', '-v7.3');
    fprintf('\nGlobal refinement statistics saved to:\n%s\n', output_dir);
end

function [mc_times, rootloop_opts, metric_opts] = get_mode_defaults(mode_str)
    switch lower(mode_str)
        case 'quick'
            mc_times = 40;
            rootloop_opts = struct( ...
                'max_outer_iter', 4, 'use_arcsine', true, 'toeplitz_project', true, 'psd_floor', 1e-6, ...
                'covariance_shrink', 0.025, 'diag_loading', 7e-4, 'fb_avg', true, 'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, 'smooth_sep_threshold_deg', 3.5, 'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8, 'use_adaptive_opts', false, ...
                'subspace_enhance_enable', true, 'subspace_enhance_mode', 'clean', ...
                'global_refine_enable', true, 'global_force_refine', false, ...
                'reweight_floor', 0.20, 'residual_power', 1.0, 'residual_tol', 3e-3, 'theta_tol_deg', 0.05, ...
                'noise_floor_scale', 1.0, 'subspace_clean_blend_rho', 0.40, 'cov_update_step', 0.65, ...
                'cov_update_inner', 2, 'model_consistency_weight', 0.55, ...
                'global_trigger_sep_deg', 4.0, 'global_trigger_boundary_ratio', 0.22, ...
                'global_trigger_quant_residual', 0.18, 'global_refine_max_shift_deg', 0.7, ...
                'global_refine_min_sep_deg', 0.15, 'global_refine_anchor_weight', 0.08, ...
                'global_refine_lambda_reg', 1e-3, 'global_refine_max_iter', 16, 'global_refine_tol', 1e-4, ...
                'global_refine_joint_iter', 4, 'global_refine_fd_deg', 0.03, 'global_refine_step_size', 0.22, ...
                'global_accept_tol', 1e-4, 'global_accept_max_shift_deg', 0.4, ...
                'global_accept_mean_shift_deg', 0.22, 'global_accept_min_improve_ratio', 2e-4);
        case 'full'
            mc_times = 100;
            rootloop_opts = struct( ...
                'max_outer_iter', 6, 'use_arcsine', true, 'toeplitz_project', true, 'psd_floor', 1e-6, ...
                'covariance_shrink', 0.015, 'diag_loading', 5e-4, 'fb_avg', true, 'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, 'smooth_sep_threshold_deg', 3.5, 'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8, 'use_adaptive_opts', false, ...
                'subspace_enhance_enable', true, 'subspace_enhance_mode', 'clean', ...
                'global_refine_enable', true, 'global_force_refine', false, ...
                'reweight_floor', 0.18, 'residual_power', 1.0, 'residual_tol', 2e-3, 'theta_tol_deg', 0.03, ...
                'noise_floor_scale', 1.0, 'subspace_clean_blend_rho', 0.40, 'cov_update_step', 0.65, ...
                'cov_update_inner', 2, 'model_consistency_weight', 0.55, ...
                'global_trigger_sep_deg', 4.0, 'global_trigger_boundary_ratio', 0.20, ...
                'global_trigger_quant_residual', 0.18, 'global_refine_max_shift_deg', 0.7, ...
                'global_refine_min_sep_deg', 0.15, 'global_refine_anchor_weight', 0.08, ...
                'global_refine_lambda_reg', 1e-3, 'global_refine_max_iter', 20, 'global_refine_tol', 1e-4, ...
                'global_refine_joint_iter', 4, 'global_refine_fd_deg', 0.03, 'global_refine_step_size', 0.22, ...
                'global_accept_tol', 1e-4, 'global_accept_max_shift_deg', 0.4, ...
                'global_accept_mean_shift_deg', 0.22, 'global_accept_min_improve_ratio', 2e-4);
        otherwise
            error('mode_str must be quick or full');
    end
    metric_opts = struct('theta_global_offset_deg', 0.37);
end

function scenes = get_scene_definitions(mc_times)
    cfg1 = struct('id',1,'name','fig1_snr_3src_uncorr','scan_type','snr', ...
        'scan_values',-10:2:20,'K',3,'theta_true_deg',[-22 10 37],'M',20,'N',40,'snr_db',[],'rho12',0,'amp_vec',ones(1,3),'mc_times',mc_times);
    cfg3 = struct('id',3,'name','fig3_snr_4src_uncorr_close','scan_type','snr', ...
        'scan_values',-10:2:20,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',40,'snr_db',[],'rho12',0,'amp_vec',ones(1,4),'mc_times',mc_times);
    cfg6 = struct('id',6,'name','fig6_separation_4src_uncorr','scan_type','separation', ...
        'scan_values',1:9,'K',4,'theta_true_deg',[],'theta_ref_deg',-20,'M',20,'N',40,'snr_db',0,'rho12',0,'amp_vec',ones(1,4),'mc_times',mc_times);
    cfg7 = struct('id',7,'name','fig7_snr_4src_corr_close','scan_type','snr', ...
        'scan_values',-10:2:20,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',40,'snr_db',[],'rho12',0.95,'amp_vec',ones(1,4),'mc_times',mc_times);

    scenes = {
        struct('label', '普通三源非相关', 'cfg', cfg1, 'xmask', @(x) true(size(x))), ...
        struct('label', '非相关近邻源', 'cfg', cfg3, 'xmask', @(x) true(size(x))), ...
        struct('label', '高相关近邻源', 'cfg', cfg7, 'xmask', @(x) true(size(x))), ...
        struct('label', '大角间隔普通源', 'cfg', cfg6, 'xmask', @(x) x >= 5)
    };
end

function [theta_true_deg, M, N, snr_db] = resolve_scan_setting(cfg, val, metric_opts)
    theta_offset_deg = metric_opts.theta_global_offset_deg;
    switch lower(cfg.scan_type)
        case 'snr'
            theta_true_deg = sort(cfg.theta_true_deg + theta_offset_deg);
            M = cfg.M; N = cfg.N; snr_db = val;
        case 'separation'
            pair_center_deg = cfg.theta_ref_deg + theta_offset_deg;
            theta_true_deg = sort([pair_center_deg - val/2, pair_center_deg + val/2, 10 + theta_offset_deg, 37 + theta_offset_deg]);
            M = cfg.M; N = cfg.N; snr_db = cfg.snr_db;
        otherwise
            error('Unsupported scan type');
    end
end

function data = generate_doa_data(theta_true_deg, M, N, snr_db, rho12, amp_vec)
    d_lambda = 0.5;
    sigma2noise = 1.0;
    idxR = (0:M-1).';
    K = numel(theta_true_deg);
    theta_rad = deg2rad(theta_true_deg(:).');
    A = zeros(M, K);
    for k = 1:K
        A(:, k) = exp(-1j * 2 * pi * d_lambda * sin(theta_rad(k)) * idxR);
    end

    amp_vec = amp_vec(:);
    P_base = diag(amp_vec.^2);
    if rho12 > 0 && K >= 2
        P_base(1, 2) = rho12 * amp_vec(1) * amp_vec(2);
        P_base(2, 1) = conj(P_base(1, 2));
    end

    sigma2source = 10^(0.1 * snr_db) * sigma2noise;
    P = sigma2source * P_base;
    Lc = chol((P + P') / 2 + 1e-12 * eye(K), 'lower');
    S = Lc * ((randn(K, N) + 1j * randn(K, N)) / sqrt(2));
    noise = sqrt(sigma2noise / 2) * (randn(M, N) + 1j * randn(M, N));
    X_full = A * S + noise;
    Y_onebit = sign(real(X_full)) + 1j * sign(imag(X_full));
    Y_onebit(real(Y_onebit) == 0) = 1 + 1j * imag(Y_onebit(real(Y_onebit) == 0));
    Y_onebit(imag(Y_onebit) == 0) = real(Y_onebit(imag(Y_onebit) == 0)) + 1j;

    data = struct('theta_true_deg', theta_true_deg, 'M', M, 'N', N, 'K', K, ...
        'A', A, 'S', S, 'P', P, 'sigma2noise', sigma2noise, 'X_full', X_full, ...
        'Y_onebit', Y_onebit, 'rho12', rho12);
end

function write_markdown_summary(T, save_path)
    fid = fopen(save_path, 'w');
    fprintf(fid, '# SE-RootMUSIC Global Refinement Statistics\n\n');
    fprintf(fid, '| Scene | Trigger rate | Accept rate | Accept given trigger | Mean shift (deg) |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|\n');
    for i = 1:height(T)
        fprintf(fid, '| %s | %.4f | %.4f | %.4f | %.4f |\n', ...
            char(T.Scene(i)), T.TriggerRate(i), T.AcceptRate(i), T.AcceptGivenTrigger(i), T.MeanShiftDeg(i));
    end
    fclose(fid);
end

function plot_stats_bar(T, save_path)
    fig = figure('Visible', 'off', 'Position', [100, 100, 980, 420]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    bar(categorical(T.Scene), [T.TriggerRate, T.AcceptRate], 'grouped');
    ylabel('Rate');
    title('Global refinement trigger / accept rate');
    legend({'Trigger rate', 'Accept rate'}, 'Location', 'northwest');
    grid on;

    nexttile;
    bar(categorical(T.Scene), T.MeanShiftDeg);
    ylabel('Mean angular shift (deg)');
    title('Accepted global refinement shift');
    grid on;

    saveas(fig, save_path);
    close(fig);
end
