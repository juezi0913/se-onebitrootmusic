function results = run_seroot_multiclose_snr(mode_str)
% run_seroot_multiclose_snr
% -------------------------------------------------------------
% Multi-close-source supplementary experiment for the current
% SE-RootMUSIC paper-version.
%
% Scenario: two close pairs
%   theta = [-22, -20, 10, 12] + 0.37 deg
% We compare:
%   1) one-bit root-MUSIC
%   2) closed-loop only
%   3) CL + clean subspace projection
%   4) full SE-RootMUSIC
% -------------------------------------------------------------

    if nargin < 1 || isempty(mode_str)
        mode_str = 'full';
    end

    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);

    rng_seed = 20260316;
    rng(rng_seed);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_dir = fullfile(script_dir, sprintf('seroot_multiclose_%s', timestamp));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    [mc_times, rootloop_opts, baseline_opts, metric_opts] = get_mode_defaults(mode_str);
    variants = get_variants(rootloop_opts, baseline_opts);
    cfg = get_multiclose_config(mc_times);

    results = struct();
    results.timestamp = timestamp;
    results.mode = mode_str;
    results.output_dir = output_dir;
    results.rng_seed = rng_seed;
    results.variants = variants;
    results.cfg = cfg;

    xvals = cfg.scan_values(:).';
    n_var = numel(variants);
    mse_db = nan(n_var, numel(xvals));
    success_rate = nan(n_var, numel(xvals));
    avg_runtime = nan(n_var, numel(xvals));
    theta_true_deg_scan = cell(1, numel(xvals));

    for ix = 1:numel(xvals)
        snr_db = xvals(ix);
        [theta_true_deg, M, N] = resolve_setting(cfg, metric_opts);
        theta_true_deg_scan{ix} = theta_true_deg;
        fprintf('  SNR = %g | M=%d N=%d\n', snr_db, M, N);

        sq_err_sum = zeros(n_var, 1);
        succ_cnt = zeros(n_var, 1);
        t_sum = zeros(n_var, 1);

        data0 = generate_doa_data(theta_true_deg, M, N, snr_db, cfg.rho12, cfg.amp_vec);
        for mc = 1:cfg.mc_times
            if mc == 1
                data = data0;
            else
                data = generate_doa_data(theta_true_deg, M, N, snr_db, cfg.rho12, cfg.amp_vec);
            end
            for iv = 1:n_var
                t0 = tic;
                doa_est = variants{iv}.runner(data, cfg.K);
                t_sum(iv) = t_sum(iv) + toc(t0);

                if any(~isfinite(doa_est)) || numel(doa_est) < cfg.K
                    sq_err_mean_deg2 = inf;
                    max_abs_err = inf;
                else
                    [sq_err_mean_deg2, max_abs_err] = doa_error_best_match(sort(doa_est(:).'), theta_true_deg);
                end
                sq_err_sum(iv) = sq_err_sum(iv) + sq_err_mean_deg2;
                if max_abs_err <= cfg.success_tol_deg
                    succ_cnt(iv) = succ_cnt(iv) + 1;
                end
            end
        end

        for iv = 1:n_var
            mse_linear_rad2 = (sq_err_sum(iv) / cfg.mc_times) * (pi/180)^2;
            mse_db(iv, ix) = 10 * log10(max(mse_linear_rad2, metric_opts.mse_plot_floor_rad2));
            success_rate(iv, ix) = succ_cnt(iv) / cfg.mc_times;
            avg_runtime(iv, ix) = t_sum(iv) / cfg.mc_times;
        end
    end

    plot_compare_curve(xvals, mse_db, variants, 'SNR (dB)', 'MSE (dB)', ...
        'Multi-close-source | MSE', ...
        fullfile(output_dir, sprintf('multiclose_snr_mse_%s.png', timestamp)));
    plot_compare_curve(xvals, success_rate, variants, 'SNR (dB)', 'Success rate', ...
        'Multi-close-source | Success', ...
        fullfile(output_dir, sprintf('multiclose_snr_success_%s.png', timestamp)));
    plot_compare_curve(xvals, avg_runtime, variants, 'SNR (dB)', 'Runtime (sec)', ...
        'Multi-close-source | Runtime', ...
        fullfile(output_dir, sprintf('multiclose_snr_runtime_%s.png', timestamp)));

    results.xvals = xvals;
    results.theta_true_deg_scan = theta_true_deg_scan;
    results.mse_db = mse_db;
    results.success_rate = success_rate;
    results.avg_runtime = avg_runtime;

    save(fullfile(output_dir, sprintf('seroot_multiclose_%s.mat', timestamp)), 'results', '-v7.3');
    fprintf('\nMulti-close-source results saved to:\n%s\n', output_dir);
end

function cfg = get_multiclose_config(mc_times)
    cfg = struct( ...
        'name', 'multiclose_snr_two_pairs', ...
        'scan_values', -10:2:20, ...
        'K', 4, ...
        'theta_true_deg', [-22 -20 10 12], ...
        'M', 40, ...
        'N', 40, ...
        'rho12', 0, ...
        'amp_vec', ones(1,4), ...
        'success_tol_deg', 2.0, ...
        'mc_times', mc_times);
end

function variants = get_variants(rootloop_opts, baseline_opts)
    cl_only_opts = rootloop_opts;
    cl_only_opts.use_adaptive_opts = false;
    cl_only_opts.subspace_enhance_enable = false;
    cl_only_opts.global_refine_enable = false;
    cl_only_opts.local_refine_enable = false;

    cl_se_opts = rootloop_opts;
    cl_se_opts.use_adaptive_opts = false;
    cl_se_opts.subspace_enhance_enable = true;
    cl_se_opts.subspace_enhance_mode = 'clean';
    cl_se_opts.global_refine_enable = false;
    cl_se_opts.local_refine_enable = false;

    full_opts = rootloop_opts;
    full_opts.use_adaptive_opts = false;
    full_opts.subspace_enhance_enable = true;
    full_opts.subspace_enhance_mode = 'clean';
    full_opts.global_refine_enable = true;

    variants = {
        struct('name', 'One-bit root-MUSIC', 'short', 'ob_rootmusic', ...
               'runner', @(data, K) run_onebit_rootmusic(data, K, baseline_opts)), ...
        struct('name', 'Closed-loop only', 'short', 'cl_only', ...
               'runner', @(data, K) run_seroot_variant(data, K, cl_only_opts)), ...
        struct('name', 'CL + SE projection', 'short', 'cl_se', ...
               'runner', @(data, K) run_seroot_variant(data, K, cl_se_opts)), ...
        struct('name', 'CL + SE + gated global refinement', 'short', 'full_seroot', ...
               'runner', @(data, K) run_seroot_variant(data, K, full_opts)) ...
    };
end

function doa = run_onebit_rootmusic(data, K, baseline_opts)
    [G, M_eff] = onebit_noise_subspace(data.Y_onebit, K, baseline_opts);
    doa = root_music_doa(G, M_eff, K);
end

function varargout = run_seroot_variant(data, K, rootloop_opts, return_struct)
    if nargin < 4, return_struct = false; end
    est = se_rootmusic_cleanproj_1bit_doa_estimator(data.Y_onebit, K, rootloop_opts, data);
    if return_struct
        varargout{1} = est;
    else
        varargout{1} = est.doa_est_deg;
    end
end

function [mc_times, rootloop_opts, baseline_opts, metric_opts] = get_mode_defaults(mode_str)
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
            baseline_opts = struct( ...
                'use_arcsine', true, 'covariance_shrink', 0.05, 'diag_loading', 1e-3, ...
                'fb_avg', true, 'spatial_smoothing', true, 'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, 'smooth_boundary_ratio', 0.22, 'smoothing_margin', 8);
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
            baseline_opts = struct( ...
                'use_arcsine', true, 'covariance_shrink', 0.03, 'diag_loading', 8e-4, ...
                'fb_avg', true, 'spatial_smoothing', true, 'smooth_corr_threshold', 0.70, ...
                'smooth_sep_threshold_deg', 3.5, 'smooth_boundary_ratio', 0.22, 'smoothing_margin', 8);
        otherwise
            error('mode_str must be quick or full');
    end
    metric_opts = struct('theta_global_offset_deg', 0.37, 'mse_plot_floor_rad2', 1e-10);
end

function [theta_true_deg, M, N] = resolve_setting(cfg, metric_opts)
    theta_true_deg = sort(cfg.theta_true_deg + metric_opts.theta_global_offset_deg);
    M = cfg.M;
    N = cfg.N;
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

function [G, M_eff] = onebit_noise_subspace(Y_onebit, K, baseline_opts)
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
    R(1:(M + 1):end) = 1;
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

function plot_compare_curve(xvals, Y, variants, xlabel_str, ylabel_str, title_str, save_path)
    fig = figure('Visible', 'off', 'Position', [100, 100, 900, 660]);
    hold on;
    styles = {'g-s', 'k-*', 'b-d', 'm-^'};
    for iv = 1:numel(variants)
        st = styles{iv};
        plot(xvals, Y(iv, :), st, 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', variants{iv}.name);
    end
    xlabel(xlabel_str);
    ylabel(ylabel_str);
    title(title_str);
    grid on;
    legend('show', 'Location', 'best');
    saveas(fig, save_path);
    close(fig);
end
