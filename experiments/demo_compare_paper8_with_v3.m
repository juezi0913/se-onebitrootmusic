function results = demo_compare_paper8_with_v3(mode_str, fig_ids, selected_shorts)
% Compare Fig.1-Fig.8 paper experiments with FGD replacing AdaBoost lines.

    if nargin < 1 || isempty(mode_str), mode_str = 'quick'; end
    if nargin < 2 || isempty(fig_ids), fig_ids = 1:8; end
    if nargin < 3, selected_shorts = []; end
    rng_seed = 20260316;
    rng(rng_seed);

    output_dir = fullfile(pwd, 'results_paper8_clean_fixed');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    [mc_times, rootloop_opts, fgd_opts, baseline_opts, metric_opts] = get_mode_defaults(mode_str);
    method_opts = struct('rootloop', rootloop_opts, 'fgd', fgd_opts, 'baseline', baseline_opts);
    algos = get_algorithm_list();
    algos = filter_algorithm_list(algos, selected_shorts);
    exps = get_experiment_configs(fig_ids, mc_times);

    results = struct();
    results.mode = mode_str;
    results.output_dir = output_dir;
    results.timestamp = timestamp;
    results.method_opts = method_opts;
    results.metric_opts = metric_opts;
    results.algorithms = algos;
    results.method_version = 'SE-RootMUSIC-clean-fixed';
    results.subspace_enhance_mode = rootloop_opts.subspace_enhance_mode;
    results.use_adaptive_opts = rootloop_opts.use_adaptive_opts;
    results.rng_seed = rng_seed;

    for ie = 1:numel(exps)
        cfg = exps{ie};
        fprintf('\n========================================================\n');
        fprintf('Running %s\n', cfg.name);
        fprintf('========================================================\n');
        results.(cfg.name) = run_one_experiment(cfg, algos, method_opts, metric_opts, output_dir, timestamp);
    end

    save(fullfile(output_dir, sprintf('results_%s.mat', timestamp)), 'results', '-v7.3');
    fprintf('\nAll selected experiments finished. Results saved to:\n%s\n', output_dir);
end

function algos = filter_algorithm_list(algos, selected_shorts)
    if isempty(selected_shorts)
        return;
    end
    if ischar(selected_shorts) || isstring(selected_shorts)
        selected_shorts = cellstr(selected_shorts);
    end
    keep_mask = false(1, numel(algos));
    for i = 1:numel(algos)
        keep_mask(i) = any(strcmpi(algos{i}.short, selected_shorts)) || any(strcmpi(algos{i}.name, selected_shorts));
    end
    algos = algos(keep_mask);
    if isempty(algos)
        error('No algorithms matched selected_shorts.');
    end
end

function [mc_times, rootloop_opts, fgd_opts, baseline_opts, metric_opts] = get_mode_defaults(mode_str)
    switch lower(mode_str)
        case 'quick'
            mc_times = 40;
            rootloop_opts = struct( ...
                'max_outer_iter', 4, ...
                'use_arcsine', true, ...
                'toeplitz_project', true, ...
                'psd_floor', 1e-6, ...
                'covariance_shrink', 0.025, ...
                'diag_loading', 7e-4, ...
                'fb_avg', true, ...
                'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, ...
                'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8, ...
                'use_adaptive_opts', false, ...
                'subspace_enhance_enable', true, ...
                'subspace_enhance_mode', 'clean', ...
                'global_refine_enable', true, ...
                'global_force_refine', false, ...
                'reweight_floor', 0.20, ...
                'residual_power', 1.0, ...
                'residual_tol', 3e-3, ...
                'theta_tol_deg', 0.05, ...
                'noise_floor_scale', 1.0, ...
                'subspace_clean_blend_rho', 0.40, ...
                'cov_update_step', 0.65, ...
                'cov_update_inner', 2, ...
                'model_consistency_weight', 0.55, ...
                'local_refine_enable', true, ...
                'local_trigger_sep_deg', 4.0, ...
                'local_trigger_boundary_ratio', 0.22, ...
                'local_refine_rounds', 2, ...
                'local_refine_half_width_deg', [0.6 0.2], ...
                'local_refine_steps_deg', [0.2 0.05], ...
                'local_refine_max_shift_deg', 0.8, ...
                'local_refine_min_sep_deg', 0.15, ...
                'local_refine_anchor_weight', 0.12, ...
                'local_refine_lambda_reg', 1e-3, ...
                'local_refine_max_iter', 16, ...
                'local_refine_tol', 1e-4, ...
                'local_accept_tol', 1e-4, ...
                'local_trigger_quant_residual', 0.18);
            fgd_opts = struct( ...
                'R', 20, ...
                'theta_grid_deg', -90:1:90, ...
                'lambda_reg', 1e-3, ...
                'refit_max_iter', 30, ...
                'refit_tol', 1e-5, ...
                'active_extra', 4, ...
                'support_pick_min_sep_deg', 0.5, ...
                'final_pick_min_sep_deg', 0.75, ...
                'refine_anchor_weight', 0.05, ...
                'verbose', false);
            baseline_opts = struct( ...
                'use_arcsine', true, ...
                'covariance_shrink', 0.05, ...
                'diag_loading', 1e-3, ...
                'fb_avg', true, ...
                'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, ...
                'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8);
        case 'full'
            mc_times = 100;
            rootloop_opts = struct( ...
                'max_outer_iter', 6, ...
                'use_arcsine', true, ...
                'toeplitz_project', true, ...
                'psd_floor', 1e-6, ...
                'covariance_shrink', 0.015, ...
                'diag_loading', 5e-4, ...
                'fb_avg', true, ...
                'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, ...
                'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8, ...
                'use_adaptive_opts', false, ...
                'subspace_enhance_enable', true, ...
                'subspace_enhance_mode', 'clean', ...
                'global_refine_enable', true, ...
                'global_force_refine', false, ...
                'reweight_floor', 0.18, ...
                'residual_power', 1.0, ...
                'residual_tol', 2e-3, ...
                'theta_tol_deg', 0.03, ...
                'noise_floor_scale', 1.0, ...
                'subspace_clean_blend_rho', 0.40, ...
                'cov_update_step', 0.65, ...
                'cov_update_inner', 2, ...
                'model_consistency_weight', 0.55, ...
                'local_refine_enable', true, ...
                'local_trigger_sep_deg', 4.0, ...
                'local_trigger_boundary_ratio', 0.20, ...
                'local_refine_rounds', 2, ...
                'local_refine_half_width_deg', [0.7 0.25], ...
                'local_refine_steps_deg', [0.2 0.05], ...
                'local_refine_max_shift_deg', 0.9, ...
                'local_refine_min_sep_deg', 0.15, ...
                'local_refine_anchor_weight', 0.10, ...
                'local_refine_lambda_reg', 1e-3, ...
                'local_refine_max_iter', 20, ...
                'local_refine_tol', 1e-4, ...
                'local_accept_tol', 1e-4, ...
                'local_trigger_quant_residual', 0.18);
            fgd_opts = struct( ...
                'R', 40, ...
                'theta_grid_deg', -90:1:90, ...
                'lambda_reg', 1e-3, ...
                'refit_max_iter', 80, ...
                'refit_tol', 1e-5, ...
                'active_extra', 4, ...
                'support_pick_min_sep_deg', 0.5, ...
                'final_pick_min_sep_deg', 0.75, ...
                'refine_anchor_weight', 0.05, ...
                'verbose', false);
            baseline_opts = struct( ...
                'use_arcsine', true, ...
                'covariance_shrink', 0.03, ...
                'diag_loading', 8e-4, ...
                'fb_avg', true, ...
                'spatial_smoothing', true, ...
                'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, ...
                'smooth_boundary_ratio', 0.22, ...
                'smoothing_margin', 8);
        otherwise
            error('mode_str must be quick or full');
    end

    metric_opts = struct( ...
        'theta_global_offset_deg', 0.37, ...
        'mse_plot_floor_rad2', 1e-10);
end

function algos = get_algorithm_list()
    algos = {
        struct('name','One-bit MUSIC','short','ob_music','runner',@run_onebit_music), ...
        struct('name','OGIR','short','ogir','runner',@run_ogir), ...
        struct('name','CBIHT','short','cbiht','runner',@run_cbiht), ...
        struct('name','Gr-SBL','short','grsbl','runner',@run_grsbl), ...
        struct('name','One-bit root-MUSIC','short','ob_rootmusic','runner',@run_onebit_rootmusic), ...
        struct('name','CL-RootMUSIC','short','cl_rootmusic','runner',@run_cl_rootmusic), ...
        struct('name','SE-RootMUSIC','short','se_rootmusic','runner',@run_se_rootmusic), ...
        struct('name','TGU-RootMUSIC','short','tgu_rootmusic','runner',@run_tgu_rootmusic), ...
        struct('name','HP root-MUSIC (oracle)','short','hp_rootmusic','runner',@run_hp_rootmusic) ...
    };
end

function exps = get_experiment_configs(fig_ids, mc_times)
    exps = {};
    for fig_id = fig_ids
        switch fig_id
            case 1
                cfg = struct('id',1,'name','fig1_snr_3src_uncorr','title','Fig.1-like','scan_type','snr', ...
                    'scan_values',-10:2:20,'K',3,'theta_true_deg',[-22 10 37],'M',20,'N',40,'snr_db',[],'rho12',0,'amp_vec',ones(1,3),'success_tol_deg',2.0,'mc_times',mc_times);
            case 2
                cfg = struct('id',2,'name','fig2_snapshots_3src_uncorr','title','Fig.2-like','scan_type','snapshots', ...
                    'scan_values',10:10:80,'K',3,'theta_true_deg',[-22 10 37],'M',20,'N',[],'snr_db',5,'rho12',0,'amp_vec',ones(1,3),'success_tol_deg',2.0,'mc_times',mc_times);
            case 3
                cfg = struct('id',3,'name','fig3_snr_4src_uncorr_close','title','Fig.3-like','scan_type','snr', ...
                    'scan_values',-10:2:20,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',40,'snr_db',[],'rho12',0,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            case 4
                cfg = struct('id',4,'name','fig4_snapshots_4src_uncorr_close','title','Fig.4-like','scan_type','snapshots', ...
                    'scan_values',10:10:80,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',[],'snr_db',5,'rho12',0,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            case 5
                cfg = struct('id',5,'name','fig5_antennas_4src_uncorr_close','title','Fig.5-like','scan_type','antennas', ...
                    'scan_values',10:10:80,'K',4,'theta_true_deg',[-22 -20 10 37],'M',[],'N',40,'snr_db',5,'rho12',0,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            case 6
                cfg = struct('id',6,'name','fig6_separation_4src_uncorr','title','Fig.6-like','scan_type','separation', ...
                    'scan_values',1:9,'K',4,'theta_true_deg',[],'theta_ref_deg',-20,'M',20,'N',40,'snr_db',0,'rho12',0,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            case 7
                cfg = struct('id',7,'name','fig7_snr_4src_corr_close','title','Fig.7-like','scan_type','snr', ...
                    'scan_values',-10:2:20,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',40,'snr_db',[],'rho12',0.95,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            case 8
                cfg = struct('id',8,'name','fig8_snapshots_4src_corr_close','title','Fig.8-like','scan_type','snapshots', ...
                    'scan_values',10:10:80,'K',4,'theta_true_deg',[-22 -20 10 37],'M',40,'N',[],'snr_db',5,'rho12',0.95,'amp_vec',ones(1,4),'success_tol_deg',2.0,'mc_times',mc_times);
            otherwise
                error('Unknown figure id');
        end
        exps{end+1} = cfg; %#ok<AGROW>
    end
end

function out = run_one_experiment(cfg, algos, method_opts, metric_opts, output_dir, timestamp)
    xvals = cfg.scan_values(:).';
    n_algo = numel(algos);
    mse_db = nan(n_algo, numel(xvals));
    mse_db_raw = nan(n_algo, numel(xvals));
    mse_linear_rad2 = nan(n_algo, numel(xvals));
    mse_linear_deg2 = nan(n_algo, numel(xvals));
    rmse_deg = nan(n_algo, numel(xvals));
    success_rate = nan(n_algo, numel(xvals));
    avg_runtime = nan(n_algo, numel(xvals));
    crb_mse_db = nan(1, numel(xvals));
    crb_mse_db_raw = nan(1, numel(xvals));
    crb_mse_rad2 = nan(1, numel(xvals));
    theta_true_deg_scan = cell(1, numel(xvals));
    example_saved = false;
    has_cl_rootmusic = any(cellfun(@(a) strcmp(a.short, 'cl_rootmusic'), algos));

    for ix = 1:numel(xvals)
        val = xvals(ix);
        [theta_true_deg, M, N, snr_db] = resolve_scan_setting(cfg, val, metric_opts);
        theta_true_deg_scan{ix} = theta_true_deg;
        fprintf('  x = %g | M=%d N=%d SNR=%g\n', val, M, N, snr_db);

        sq_err_sum = zeros(n_algo, 1);
        succ_cnt = zeros(n_algo, 1);
        t_sum = zeros(n_algo, 1);

        data0 = generate_doa_data(theta_true_deg, M, N, snr_db, cfg.rho12, cfg.amp_vec);
        crb_mse_rad2(ix) = hp_crb_stochastic_ula(theta_true_deg, data0.P, data0.sigma2noise, M, N);
        crb_mse_db_raw(ix) = 10*log10(crb_mse_rad2(ix) + eps);
        crb_mse_db(ix) = 10*log10(max(crb_mse_rad2(ix), metric_opts.mse_plot_floor_rad2));

        for mc = 1:cfg.mc_times
            if mc == 1
                data = data0;
            else
                data = generate_doa_data(theta_true_deg, M, N, snr_db, cfg.rho12, cfg.amp_vec);
            end

            for ia = 1:n_algo
                t0 = tic;
                doa_est = algos{ia}.runner(data, cfg.K, method_opts);
                t_sum(ia) = t_sum(ia) + toc(t0);

                if any(~isfinite(doa_est)) || numel(doa_est) < cfg.K
                    sq_err_mean_deg2 = inf;
                    max_abs_err = inf;
                else
                    [sq_err_mean_deg2, max_abs_err] = doa_error_best_match(sort(doa_est(:).'), theta_true_deg);
                end

                sq_err_sum(ia) = sq_err_sum(ia) + sq_err_mean_deg2;
                if max_abs_err <= cfg.success_tol_deg
                    succ_cnt(ia) = succ_cnt(ia) + 1;
                end
            end

            if has_cl_rootmusic && ~example_saved && mc == 1 && ix == ceil(numel(xvals) / 2)
                est_main = run_cl_rootmusic(data, cfg.K, method_opts, true);
                save_cl_rootmusic_example(est_main, theta_true_deg, cfg, output_dir, timestamp);
                example_saved = true;
            end
        end

        for ia = 1:n_algo
            mse_linear_deg2(ia, ix) = sq_err_sum(ia) / cfg.mc_times;
            mse_linear_rad2(ia, ix) = mse_linear_deg2(ia, ix) * (pi/180)^2;
            mse_db_raw(ia, ix) = 10*log10(mse_linear_rad2(ia, ix) + eps);
            mse_db(ia, ix) = 10*log10(max(mse_linear_rad2(ia, ix), metric_opts.mse_plot_floor_rad2));
            rmse_deg(ia, ix) = sqrt(mse_linear_deg2(ia, ix));
            success_rate(ia, ix) = succ_cnt(ia) / cfg.mc_times;
            avg_runtime(ia, ix) = t_sum(ia) / cfg.mc_times;
        end
    end

    mse_plot_opts = get_curve_plot_options(cfg, 'mse');
    success_plot_opts = get_curve_plot_options(cfg, 'success');
    runtime_plot_opts = get_curve_plot_options(cfg, 'runtime');
    crb_for_plot = [];
    if mse_plot_opts.show_crb
        crb_for_plot = crb_mse_db;
    end

    plot_compare_curve( ...
        xvals, mse_db, algos, xlabel_from_scan(cfg.scan_type), 'MSE (dB)', ...
        sprintf('%s | MSE', cfg.title), ...
        fullfile(output_dir, sprintf('%s_mse_%s.png', cfg.name, timestamp)), ...
        mse_plot_opts, crb_for_plot);
    plot_compare_curve( ...
        xvals, success_rate, algos, xlabel_from_scan(cfg.scan_type), 'Success rate', ...
        sprintf('%s | Success rate', cfg.title), ...
        fullfile(output_dir, sprintf('%s_success_%s.png', cfg.name, timestamp)), ...
        success_plot_opts);
    plot_compare_curve( ...
        xvals, avg_runtime, algos, xlabel_from_scan(cfg.scan_type), 'Runtime (sec)', ...
        sprintf('%s | Runtime', cfg.title), ...
        fullfile(output_dir, sprintf('%s_runtime_%s.png', cfg.name, timestamp)), ...
        runtime_plot_opts);

    out = struct( ...
        'cfg', cfg, ...
        'xvals', xvals, ...
        'theta_true_deg_scan', {theta_true_deg_scan}, ...
        'mse_db', mse_db, ...
        'mse_db_raw', mse_db_raw, ...
        'mse_linear_rad2', mse_linear_rad2, ...
        'mse_linear_deg2', mse_linear_deg2, ...
        'crb_mse_db', crb_mse_db, ...
        'crb_mse_db_raw', crb_mse_db_raw, ...
        'crb_mse_rad2', crb_mse_rad2, ...
        'rmse_deg', rmse_deg, ...
        'success_rate', success_rate, ...
        'avg_runtime', avg_runtime, ...
        'algorithms', {algos});
end

function [theta_true_deg, M, N, snr_db] = resolve_scan_setting(cfg, val, metric_opts)
    theta_offset_deg = metric_opts.theta_global_offset_deg;
    switch lower(cfg.scan_type)
        case 'snr'
            theta_true_deg = sort(cfg.theta_true_deg + theta_offset_deg);
            M = cfg.M;
            N = cfg.N;
            snr_db = val;
        case 'snapshots'
            theta_true_deg = sort(cfg.theta_true_deg + theta_offset_deg);
            M = cfg.M;
            N = val;
            snr_db = cfg.snr_db;
        case 'antennas'
            theta_true_deg = sort(cfg.theta_true_deg + theta_offset_deg);
            M = val;
            N = cfg.N;
            snr_db = cfg.snr_db;
        case 'separation'
            pair_center_deg = cfg.theta_ref_deg + theta_offset_deg;
            theta_true_deg = sort([pair_center_deg - val/2, pair_center_deg + val/2, 10 + theta_offset_deg, 37 + theta_offset_deg]);
            M = cfg.M;
            N = cfg.N;
            snr_db = cfg.snr_db;
        otherwise
            error('Unknown scan type');
    end
end

function s = xlabel_from_scan(scan_type)
    switch lower(scan_type)
        case 'snr'
            s = 'SNR (dB)';
        case 'snapshots'
            s = 'Number of snapshots';
        case 'antennas'
            s = 'Number of antennas';
        case 'separation'
            s = 'Angular separation \Delta\theta (deg)';
        otherwise
            s = 'x';
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

    if K >= 2
        min_sep_deg = min(diff(sort(theta_true_deg)));
    else
        min_sep_deg = inf;
    end

    data = struct( ...
        'theta_true_deg', theta_true_deg, ...
        'M', M, ...
        'N', N, ...
        'K', K, ...
        'A', A, ...
        'S', S, ...
        'P', P, ...
        'sigma2noise', sigma2noise, ...
        'X_full', X_full, ...
        'Y_onebit', Y_onebit, ...
        'rho12', rho12, ...
        'min_sep_deg', min_sep_deg);
end

function doa = run_onebit_music(data, K, method_opts)
    [G, M_eff] = onebit_noise_subspace(data.Y_onebit, K, method_opts.baseline, data);
    doa = music(G, K, M_eff);
end

function doa = run_onebit_rootmusic(data, K, method_opts)
    [G, M_eff] = onebit_noise_subspace(data.Y_onebit, K, method_opts.baseline, data);
    doa = root_music_doa(G, M_eff, K);
end

function varargout = run_cl_rootmusic(data, K, method_opts, return_struct)
    if nargin < 4, return_struct = false; end
    local_opts = adapt_cl_rootmusic_opts(method_opts.rootloop, data);
    est = cl_rootmusic_1bit_doa_estimator(data.Y_onebit, K, local_opts, data);
    if return_struct
        varargout{1} = est;
    else
        varargout{1} = est.doa_est_deg;
    end
end

function varargout = run_se_rootmusic(data, K, method_opts, return_struct)
    if nargin < 4, return_struct = false; end
    if isfield(method_opts.rootloop, 'use_adaptive_opts') && method_opts.rootloop.use_adaptive_opts
        local_opts = adapt_cl_rootmusic_opts(method_opts.rootloop, data);
    else
        local_opts = method_opts.rootloop;
    end
    if isfield(local_opts, 'subspace_enhance_mode') && strcmpi(local_opts.subspace_enhance_mode, 'clean')
        est = se_rootmusic_cleanproj_1bit_doa_estimator(data.Y_onebit, K, local_opts, data);
    else
        est = se_rootmusic_1bit_doa_estimator(data.Y_onebit, K, local_opts, data);
    end
    if return_struct
        varargout{1} = est;
    else
        varargout{1} = est.doa_est_deg;
    end
end

function varargout = run_tgu_rootmusic(data, K, method_opts, return_struct)
    if nargin < 4, return_struct = false; end
    local_opts = adapt_tgu_rootmusic_opts(method_opts.rootloop, data);
    est = teacher_guided_unfolded_rootmusic_estimator(data.Y_onebit, K, local_opts, data);
    if return_struct
        varargout{1} = est;
    else
        varargout{1} = est.doa_est_deg;
    end
end

function doa = run_hp_rootmusic(data, K, ~)
    G = hp_noise_subspace(data.X_full, K);
    doa = root_music_doa(G, data.M, K);
end

function doa = run_cbiht(data, K, ~)
    doa = cbiht(data.Y_onebit, 360, K, data.M);
end

function doa = run_grsbl(data, K, ~)
    doa = Gr_SBL_1Bit(data.Y_onebit, K, data.M, data.N, 10, 1, 0, 361);
end

function doa = run_ogir(data, K, ~)
    doa = OGIR_1Bit(data.Y_onebit, 361, data.M, data.N, 300, 10, 1e-4, 10, 1e-5, K);
end

function varargout = run_fgd(data, K, method_opts, return_struct)
    if nargin < 4, return_struct = false; end
    local_opts = adapt_fgd_opts(method_opts.fgd, data);
    est = fgd_1bit_doa_estimator(data.Y_onebit, K, local_opts);
    if return_struct
        varargout{1} = est;
    else
        varargout{1} = est.doa_est_deg;
    end
end

function rootloop_opts = adapt_cl_rootmusic_opts(base_opts, data)
    rootloop_opts = base_opts;
    obs_meta = estimate_observation_scene(data.Y_onebit, data.K, base_opts);

    if data.K >= 4
        rootloop_opts.max_outer_iter = max(get_field_or_default(rootloop_opts, 'max_outer_iter', 5), 5);
        rootloop_opts.base_data_blend = max(get_field_or_default(rootloop_opts, 'base_data_blend', 0.18), 0.18);
        rootloop_opts.residual_blend_strength = max(get_field_or_default(rootloop_opts, 'residual_blend_strength', 0.62), 0.62);
    end

    if obs_meta.min_sep_est_deg <= 3.8 || obs_meta.boundary_ratio >= 0.20
        rootloop_opts.max_outer_iter = max(get_field_or_default(rootloop_opts, 'max_outer_iter', 5), get_field_or_default(base_opts, 'max_outer_iter', 5) + 1);
        rootloop_opts.covariance_shrink = min(get_field_or_default(rootloop_opts, 'covariance_shrink', 0.02), 0.02);
        rootloop_opts.base_data_blend = max(get_field_or_default(rootloop_opts, 'base_data_blend', 0.18), 0.22);
        rootloop_opts.residual_blend_strength = max(get_field_or_default(rootloop_opts, 'residual_blend_strength', 0.62), 0.72);
        rootloop_opts.quant_feedback_step = max(get_field_or_default(rootloop_opts, 'quant_feedback_step', 0.95), 1.00);
        rootloop_opts.update_relaxation = max(get_field_or_default(rootloop_opts, 'update_relaxation', 0.72), 0.78);
        rootloop_opts.local_refine_half_width_deg = [0.9 0.3];
        rootloop_opts.local_refine_steps_deg = [0.2 0.05];
        rootloop_opts.local_refine_max_shift_deg = max(get_field_or_default(rootloop_opts, 'local_refine_max_shift_deg', 0.8), 1.0);
        rootloop_opts.local_refine_max_iter = max(get_field_or_default(rootloop_opts, 'local_refine_max_iter', 18), 24);
        rootloop_opts.local_trigger_sep_deg = max(get_field_or_default(rootloop_opts, 'local_trigger_sep_deg', 4.0), 4.5);
    end

    if data.M >= 60
        rootloop_opts.diag_loading = max(get_field_or_default(rootloop_opts, 'diag_loading', 5e-4), 8e-4);
        rootloop_opts.local_refine_half_width_deg = [1.0 0.35];
        rootloop_opts.local_refine_max_shift_deg = max(get_field_or_default(rootloop_opts, 'local_refine_max_shift_deg', 0.9), 1.2);
        rootloop_opts.local_refine_anchor_weight = min(get_field_or_default(rootloop_opts, 'local_refine_anchor_weight', 0.10), 0.08);
    end
end

function tgu_opts = adapt_tgu_rootmusic_opts(base_opts, data)
    tgu_opts = struct();
    tgu_opts.num_stages = max(5, get_field_or_default(base_opts, 'max_outer_iter', 5));
    tgu_opts.use_arcsine = get_field_or_default(base_opts, 'use_arcsine', true);
    tgu_opts.toeplitz_project = get_field_or_default(base_opts, 'toeplitz_project', true);
    tgu_opts.psd_floor = get_field_or_default(base_opts, 'psd_floor', 1e-6);
    tgu_opts.covariance_shrink = max(0.015, get_field_or_default(base_opts, 'covariance_shrink', 0.02));
    tgu_opts.diag_loading = get_field_or_default(base_opts, 'diag_loading', 5e-4);
    tgu_opts.fb_avg = get_field_or_default(base_opts, 'fb_avg', true);
    tgu_opts.spatial_smoothing = get_field_or_default(base_opts, 'spatial_smoothing', true);
    tgu_opts.smooth_boundary_ratio = get_field_or_default(base_opts, 'smooth_boundary_ratio', 0.22);
    tgu_opts.smooth_sep_threshold_deg = get_field_or_default(base_opts, 'smooth_sep_threshold_deg', 3.5);
    tgu_opts.smoothing_margin = get_field_or_default(base_opts, 'smoothing_margin', 8);
    tgu_opts.feedback_step = min(1.0, get_field_or_default(base_opts, 'quant_feedback_step', 0.95));
    tgu_opts.reweight_floor = get_field_or_default(base_opts, 'reweight_floor', 0.18);
    tgu_opts.residual_power = get_field_or_default(base_opts, 'residual_power', 1.0);
    tgu_opts.residual_tol = get_field_or_default(base_opts, 'residual_tol', 2e-3);
    tgu_opts.theta_tol_deg = get_field_or_default(base_opts, 'theta_tol_deg', 0.03);
    tgu_opts.noise_floor_scale = get_field_or_default(base_opts, 'noise_floor_scale', 1.0);
    tgu_opts.root_fallback_grid_deg = -90:0.2:90;
    if data.K >= 4
        tgu_opts.num_stages = max(tgu_opts.num_stages, 5);
    end
end

function fgd_opts = adapt_fgd_opts(base_opts, data)
    fgd_opts = base_opts;

    if data.K >= 4
        fgd_opts.active_extra = max(get_field_or_default(fgd_opts, 'active_extra', 4), 6);
        fgd_opts.refine_rounds = max(get_field_or_default(fgd_opts, 'refine_rounds', 2), 2);
        fgd_opts.refine_refit_max_iter = max(get_field_or_default(fgd_opts, 'refine_refit_max_iter', 20), 24);
        fgd_opts.refine_half_width_deg = max(get_field_or_default(fgd_opts, 'refine_half_width_deg', 1.2), 1.4);
    end

    if data.min_sep_deg <= 3.5 || data.rho12 >= 0.70 || data.M >= 60
        fgd_opts.theta_grid_deg = densify_theta_grid(get_field_or_default(fgd_opts, 'theta_grid_deg', -90:1:90), 0.5);
        fgd_opts.R = max(get_field_or_default(fgd_opts, 'R', 30), get_field_or_default(base_opts, 'R', 30) + 10);
        fgd_opts.active_extra = max(get_field_or_default(fgd_opts, 'active_extra', 4), 8);
        fgd_opts.support_pick_min_sep_deg = min(get_field_or_default(fgd_opts, 'support_pick_min_sep_deg', 0.5), 0.5);
        fgd_opts.final_pick_min_sep_deg = min(get_field_or_default(fgd_opts, 'final_pick_min_sep_deg', 0.75), 0.75);
        fgd_opts.refine_rounds = max(get_field_or_default(fgd_opts, 'refine_rounds', 2), 3);
        fgd_opts.refine_half_width_deg = max(get_field_or_default(fgd_opts, 'refine_half_width_deg', 1.2), 1.6);
        fgd_opts.refine_refit_max_iter = max(get_field_or_default(fgd_opts, 'refine_refit_max_iter', 20), 30);
        fgd_opts.refine_anchor_weight = max(get_field_or_default(fgd_opts, 'refine_anchor_weight', 0.05), 0.08);
        fgd_opts.refine_max_shift_deg = max(get_field_or_default(fgd_opts, 'refine_max_shift_deg', 1.2), 1.2);
        fgd_opts.refine_min_sep_deg = min(get_field_or_default(fgd_opts, 'refine_min_sep_deg', 0.15), 0.15);
    end
end

function theta_grid_deg = densify_theta_grid(theta_grid_deg, target_step_deg)
    theta_grid_deg = theta_grid_deg(:).';
    if numel(theta_grid_deg) < 2
        return;
    end
    current_step_deg = min(diff(theta_grid_deg));
    if current_step_deg > target_step_deg
        theta_grid_deg = theta_grid_deg(1):target_step_deg:theta_grid_deg(end);
    end
end

function [G, M_eff] = onebit_noise_subspace(Y_onebit, K, baseline_opts, ~)
    [M, N] = size(Y_onebit);
    Rq = (Y_onebit * Y_onebit') / N;
    R = build_onebit_covariance_surrogate(Rq, baseline_opts);
    R = apply_fb_avg_if_needed(R, baseline_opts.fb_avg);
    obs_meta = estimate_covariance_difficulty_from_surrogate(R, K);

    use_smoothing = baseline_opts.spatial_smoothing ...
        && M > (K + 2) ...
        && (obs_meta.boundary_ratio >= baseline_opts.smooth_boundary_ratio || obs_meta.min_sep_est_deg <= baseline_opts.smooth_sep_threshold_deg);
    if use_smoothing
        L = choose_smoothing_subarray(M, K, baseline_opts.smoothing_margin);
        if L < M
            R = spatial_smooth_covariance(R, L);
            R = apply_fb_avg_if_needed(R, baseline_opts.fb_avg);
        end
    end

    R = apply_covariance_regularization(R, baseline_opts.covariance_shrink, baseline_opts.diag_loading);
    M_eff = size(R, 1);
    [Q, D] = eig((R + R') / 2);
    [~, idx] = sort(real(diag(D)), 'ascend');
    Q = Q(:, idx);
    G = Q(:, 1:(M_eff - K));
end

function meta = estimate_observation_scene(Y_onebit, K, opts)
    [~, N] = size(Y_onebit);
    Rq = (Y_onebit * Y_onebit') / max(N, 1);
    R = build_onebit_covariance_surrogate(Rq, opts);
    R = apply_fb_avg_if_needed(R, get_field_or_default(opts, 'fb_avg', true));
    R = apply_covariance_regularization(R, get_field_or_default(opts, 'covariance_shrink', 0.02), get_field_or_default(opts, 'diag_loading', 5e-4));
    meta = estimate_covariance_difficulty_from_surrogate(R, K);
end

function meta = estimate_covariance_difficulty_from_surrogate(R, K)
    meta = struct('boundary_ratio', 0, 'min_sep_est_deg', inf);
    eig_desc = sort(real(eig((R + R') / 2)), 'descend');
    if numel(eig_desc) >= (K + 1)
        meta.boundary_ratio = eig_desc(K + 1) / max(eig_desc(K), eps);
    end

    try
        [Q, D] = eig((R + R') / 2);
        [~, idx] = sort(real(diag(D)), 'ascend');
        Q = Q(:, idx);
        G = Q(:, 1:(size(R, 1) - K));
        theta_est = root_music_doa(G, size(R, 1), K);
        theta_est = sort(real(theta_est(:).'));
        if numel(theta_est) >= 2
            meta.min_sep_est_deg = min(diff(theta_est));
        end
    catch
        meta.min_sep_est_deg = inf;
    end
end

function R = build_onebit_covariance_surrogate(Rq, baseline_opts)
    M = size(Rq, 1);
    diag_scale = sqrt(max(real(diag(Rq)), eps));
    scale_mat = diag_scale * diag_scale.';
    Rn = Rq ./ scale_mat;
    Rn = clip_complex_unit_interval(Rn);

    if baseline_opts.use_arcsine
        R = sin((pi / 2) * real(Rn)) + 1j * sin((pi / 2) * imag(Rn));
    else
        R = Rn;
    end

    R(1:(M+1):end) = 1;
    R = (R + R') / 2;
end

function R = clip_complex_unit_interval(R)
    R = max(min(real(R), 1 - 1e-6), -1 + 1e-6) + 1j * max(min(imag(R), 1 - 1e-6), -1 + 1e-6);
end

function R = apply_fb_avg_if_needed(R, enable_fb_avg)
    if ~enable_fb_avg
        return;
    end
    M = size(R, 1);
    J = fliplr(eye(M));
    R = 0.5 * (R + J * conj(R) * J);
end

function R = apply_covariance_regularization(R, shrinkage, diag_loading)
    M = size(R, 1);
    tau = real(trace(R)) / M;
    R = (1 - shrinkage) * R + shrinkage * tau * eye(M);
    if diag_loading > 0
        R = R + diag_loading * tau * eye(M);
    end
    R = (R + R') / 2;
end

function L = choose_smoothing_subarray(M, K, smoothing_margin)
    L = max(K + 2, M - smoothing_margin);
    L = min(L, M);
end

function Rss = spatial_smooth_covariance(R, L)
    M = size(R, 1);
    num_sub = M - L + 1;
    Rss = zeros(L, L);
    for start_idx = 1:num_sub
        sub_idx = start_idx:(start_idx + L - 1);
        Rss = Rss + R(sub_idx, sub_idx);
    end
    Rss = Rss / num_sub;
    Rss = (Rss + Rss') / 2;
end

function G = hp_noise_subspace(X_full, K)
    [M, N] = size(X_full);
    R = (X_full * X_full') / N;
    [Q, D] = eig((R + R') / 2);
    [~, idx] = sort(real(diag(D)), 'ascend');
    Q = Q(:, idx);
    G = Q(:, 1:(M - K));
end

function [sq_err_mean_deg2, max_abs_err] = doa_error_best_match(theta_hat_deg, theta_true_deg)
    theta_hat_deg = theta_hat_deg(:).';
    theta_true_deg = theta_true_deg(:).';
    K = numel(theta_true_deg);
    if numel(theta_hat_deg) < K
        sq_err_mean_deg2 = inf;
        max_abs_err = inf;
        return;
    end

    perms_idx = perms(1:K);
    best_sq = inf;
    best_max = inf;
    for i = 1:size(perms_idx, 1)
        th = theta_hat_deg(perms_idx(i, :));
        err = th - theta_true_deg;
        sqv = mean(err.^2);
        maxv = max(abs(err));
        if sqv < best_sq
            best_sq = sqv;
            best_max = maxv;
        end
    end
    sq_err_mean_deg2 = best_sq;
    max_abs_err = best_max;
end

function save_cl_rootmusic_example(est, theta_true_deg, cfg, output_dir, timestamp)
    fig = figure('Visible', 'off', 'Position', [100, 100, 960, 420]);

    subplot(1, 2, 1);
    iters = 1:numel(est.quant_residual_hist);
    plot(iters, est.quant_residual_hist, 'b-o', 'LineWidth', 1.2, 'MarkerSize', 6); hold on;
    plot(iters, est.update_gap_hist, 'r-s', 'LineWidth', 1.2, 'MarkerSize', 6);
    xlabel('Iteration');
    ylabel('Magnitude');
    title(sprintf('%s | Closed-loop residuals', cfg.title));
    legend({'Quantized residual', 'Covariance update gap'}, 'Location', 'northeast');
    grid on;

    subplot(1, 2, 2);
    theta_hist = est.theta_hist_deg;
    if ~isempty(theta_hist)
        plot(1:size(theta_hist, 1), theta_hist, 'LineWidth', 1.2); hold on;
    end
    for k = 1:numel(theta_true_deg)
        yline(theta_true_deg(k), '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.0);
    end
    xlabel('Iteration');
    ylabel('Angle (deg)');
    title(sprintf('%s | DOA trajectory', cfg.title));
    grid on;

    saveas(fig, fullfile(output_dir, sprintf('%s_clroot_diag_%s.png', cfg.name, timestamp)));
    close(fig);
end

function plot_opts = get_curve_plot_options(cfg, curve_kind)
    plot_opts = struct('legend_location', 'best', 'ylim', [], 'show_crb', strcmp(curve_kind, 'mse'));

    if strcmp(curve_kind, 'mse')
        if cfg.id == 2
            plot_opts.legend_location = 'northeast';
            plot_opts.ylim = [-78, -36];
        elseif cfg.id == 6
            plot_opts.legend_location = 'northeast';
            plot_opts.show_crb = false;
        end
    elseif strcmp(curve_kind, 'success')
        plot_opts.ylim = [0, 1];
    end
end

function plot_compare_curve(xvals, Y, algos, xlabel_str, ylabel_str, title_str, save_path, plot_opts, crb_y)
    if nargin < 8 || isempty(plot_opts)
        plot_opts = struct('legend_location', 'best', 'ylim', [], 'show_crb', true);
    end
    if nargin < 9
        crb_y = [];
    end

    fig = figure('Visible', 'off', 'Position', [100, 100, 880, 660]);
    hold on;
    styles = {'k-*','r-v','r--o',[0.9290 0.6940 0.1250],'g-s','b-d','m-^'};
    for ia = 1:numel(algos)
        st = styles{ia};
        if ischar(st)
            plot(xvals, Y(ia, :), st, 'LineWidth', 1.4, 'MarkerSize', 7, 'DisplayName', algos{ia}.name);
        else
            plot(xvals, Y(ia, :), '-<', 'Color', st, 'LineWidth', 1.4, 'MarkerSize', 7, 'DisplayName', algos{ia}.name);
        end
    end

    if plot_opts.show_crb && ~isempty(crb_y)
        plot(xvals, crb_y, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, 'DisplayName', 'HP-CRB (oracle)');
    end

    xlabel(xlabel_str);
    ylabel(ylabel_str);
    title(title_str);
    grid on;
    if ~isempty(plot_opts.ylim)
        ylim(plot_opts.ylim);
    end
    legend('show', 'Location', plot_opts.legend_location);
    saveas(fig, save_path);
    close(fig);
end

function val = get_field_or_default(s, field_name, default_value)
    if isstruct(s) && isfield(s, field_name)
        val = s.(field_name);
    else
        val = default_value;
    end
end
