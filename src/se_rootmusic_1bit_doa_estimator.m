function result = se_rootmusic_1bit_doa_estimator(Y_onebit, K, opts, data)
% se_rootmusic_1bit_doa_estimator
% -------------------------------------------------------------
% Subspace-enhanced closed-loop one-bit root-MUSIC with:
%   1) quantization-consistent covariance feedback
%   2) Toeplitz/PSD structural projection
%   3) eigen-gap/rank-restoration projection
%   4) model-consistent covariance reconstruction
%   5) residual reweighting
%   6) confidence-gated global one-bit consistency refinement
%
% The method alternates between:
%   latent covariance -> root-MUSIC -> model covariance fit
%   -> quantized-domain residual correction -> structural projection
% -------------------------------------------------------------

    if nargin < 2
        error('At least Y_onebit and K are required.');
    end
    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if nargin < 4 || isempty(data)
        data = struct();
    end

    opts = fill_default_opts(opts, K);

    [M, N] = size(Y_onebit);

    Rq = (Y_onebit * Y_onebit') / max(N, 1);
    Rn_q = normalize_quantized_covariance(Rq);
    C_obs = inverse_onebit_map(Rn_q, opts.use_arcsine);
    C_obs = project_structured_correlation(C_obs, opts);

    C_latent = C_obs;
    theta_prev_deg = [];
    theta_hist_deg = nan(opts.max_outer_iter, K);
    quant_residual_hist = nan(opts.max_outer_iter, 1);
    update_gap_hist = nan(opts.max_outer_iter, 1);
    smooth_dim_hist = nan(opts.max_outer_iter, 1);
    model_rank_hist = nan(opts.max_outer_iter, 1);

    for it = 1:opts.max_outer_iter
        [R_sub, M_eff] = prepare_rootmusic_covariance(C_latent, K, opts);
        smooth_dim_hist(it) = M_eff;

        G = noise_subspace_from_covariance(R_sub, K);
        theta_est_deg = safe_rootmusic_readout(G, M_eff, K);
        theta_est_deg = stabilize_theta_order(theta_est_deg, theta_prev_deg);
        theta_hist_deg(it, :) = theta_est_deg(:).';

        [R_model, fit_info] = fit_covariance_from_doa(C_latent, theta_est_deg, opts);
        model_rank_hist(it) = fit_info.source_rank;
        C_model = covariance_to_correlation(R_model);
        C_model = project_structured_correlation(C_model, opts);

        Rn_pred = forward_onebit_map(C_model);
        quant_residual = Rn_q - Rn_pred;
        quant_residual(1:(M+1):end) = 0;
        quant_residual_hist(it) = norm(quant_residual, 'fro') / sqrt(numel(quant_residual));

        weight_mat = build_residual_weights(quant_residual, opts);
        Rn_feedback = clip_complex_unit_interval(Rn_pred + opts.quant_feedback_step * weight_mat .* quant_residual);
        C_feedback = inverse_onebit_map(Rn_feedback, opts.use_arcsine);

        residual_level = abs(quant_residual);
        residual_level = residual_level / max(max(residual_level), eps);
        blend_mat = opts.base_data_blend + opts.residual_blend_strength * residual_level;
        blend_mat = min(max(blend_mat, 0), 1);
        blend_mat(1:(M+1):end) = 1;

        C_fused = (1 - blend_mat) .* C_model + blend_mat .* C_feedback;
        C_fused = (1 - opts.update_relaxation) * C_latent + opts.update_relaxation * C_fused;
        C_next = project_structured_correlation(C_fused, opts);
        C_next = apply_subspace_enhancement_if_needed(C_next, K, opts);
        C_next = project_structured_correlation(C_next, opts);

        update_gap_hist(it) = norm(C_next - C_latent, 'fro') / max(norm(C_latent, 'fro'), eps);
        C_latent = C_next;
        theta_prev_deg = theta_est_deg;

        if it >= 2
            theta_delta = max(abs(theta_hist_deg(it, :) - theta_hist_deg(it-1, :)));
            if quant_residual_hist(it) < opts.residual_tol && theta_delta < opts.theta_tol_deg
                break;
            end
        end
    end

    valid_it = find(isfinite(quant_residual_hist), 1, 'last');
    if isempty(valid_it)
        valid_it = 1;
        theta_hist_deg(1, :) = safe_rootmusic_readout(noise_subspace_from_covariance(C_obs, K), M, K);
        quant_residual_hist(1) = nan;
        update_gap_hist(1) = nan;
        smooth_dim_hist(1) = M;
        model_rank_hist(1) = nan;
    end

    theta_seed_deg = sort(theta_hist_deg(valid_it, :));
    [R_sub_final, M_eff_final] = prepare_rootmusic_covariance(C_latent, K, opts);
    final_quant_residual = quant_residual_hist(valid_it);
    [should_global_refine, refine_meta] = should_run_global_refinement(theta_seed_deg, R_sub_final, K, opts, final_quant_residual);
    if should_global_refine
        Y_R = [real(Y_onebit); imag(Y_onebit)];
        refine_result = global_onebit_refinement(Y_R, theta_seed_deg, refine_meta, opts);
        if refine_result.accepted
            theta_seed_deg = refine_result.theta_refined_deg;
        end
    else
        refine_result = empty_global_result(theta_seed_deg);
        refine_result.boundary_ratio = refine_meta.boundary_ratio;
        refine_result.min_sep_est_deg = refine_meta.min_sep_est_deg;
    end

    result = struct();
    result.doa_est_deg = sort(theta_seed_deg);
    result.theta_hist_deg = theta_hist_deg(1:valid_it, :);
    result.quant_residual_hist = quant_residual_hist(1:valid_it);
    result.update_gap_hist = update_gap_hist(1:valid_it);
    result.smooth_dim_hist = smooth_dim_hist(1:valid_it);
    result.model_rank_hist = model_rank_hist(1:valid_it);
    result.Rq = Rq;
    result.C_obs = C_obs;
    result.C_final = C_latent;
    result.theta_seed_deg = sort(theta_hist_deg(valid_it, :));
    result.global_refine = refine_result;
    result.local_refine = refine_result;
    result.boundary_ratio = refine_result.boundary_ratio;
    result.M_eff_final = M_eff_final;
    result.opts = opts;
end

function opts = fill_default_opts(opts, K)
    if ~isfield(opts, 'max_outer_iter'),           opts.max_outer_iter = 5; end
    if ~isfield(opts, 'use_arcsine'),              opts.use_arcsine = true; end
    if ~isfield(opts, 'toeplitz_project'),         opts.toeplitz_project = true; end
    if ~isfield(opts, 'psd_floor'),                opts.psd_floor = 1e-6; end
    if ~isfield(opts, 'covariance_shrink'),        opts.covariance_shrink = 0.02; end
    if ~isfield(opts, 'diag_loading'),             opts.diag_loading = 5e-4; end
    if ~isfield(opts, 'fb_avg'),                   opts.fb_avg = true; end
    if ~isfield(opts, 'spatial_smoothing'),        opts.spatial_smoothing = true; end
    if ~isfield(opts, 'smooth_corr_threshold'),    opts.smooth_corr_threshold = 0.70; end
    if ~isfield(opts, 'smooth_sep_threshold_deg'), opts.smooth_sep_threshold_deg = 3.5; end
    if ~isfield(opts, 'smooth_boundary_ratio'),    opts.smooth_boundary_ratio = 0.22; end
    if ~isfield(opts, 'smoothing_margin'),         opts.smoothing_margin = 8; end
    if ~isfield(opts, 'base_data_blend'),          opts.base_data_blend = 0.18; end
    if ~isfield(opts, 'residual_blend_strength'),  opts.residual_blend_strength = 0.62; end
    if ~isfield(opts, 'quant_feedback_step'),      opts.quant_feedback_step = 0.95; end
    if ~isfield(opts, 'update_relaxation'),        opts.update_relaxation = 0.72; end
    if ~isfield(opts, 'reweight_floor'),           opts.reweight_floor = 0.20; end
    if ~isfield(opts, 'residual_power'),           opts.residual_power = 1.00; end
    if ~isfield(opts, 'residual_tol'),             opts.residual_tol = 2e-3; end
    if ~isfield(opts, 'theta_tol_deg'),            opts.theta_tol_deg = 0.03; end
    if ~isfield(opts, 'noise_floor_scale'),        opts.noise_floor_scale = 1.0; end
    if ~isfield(opts, 'root_fallback_grid_deg'),   opts.root_fallback_grid_deg = -90:0.2:90; end
    if ~isfield(opts, 'min_signal_rank'),          opts.min_signal_rank = K; end
    if ~isfield(opts, 'local_refine_enable'),      opts.local_refine_enable = true; end
    if ~isfield(opts, 'local_force_refine'),       opts.local_force_refine = false; end
    if ~isfield(opts, 'subspace_enhance_enable'),  opts.subspace_enhance_enable = true; end
    if ~isfield(opts, 'subspace_gap_target'),      opts.subspace_gap_target = 1.45; end
    if ~isfield(opts, 'subspace_signal_boost'),    opts.subspace_signal_boost = 0.18; end
    if ~isfield(opts, 'subspace_noise_flatten'),   opts.subspace_noise_flatten = 0.35; end
    if ~isfield(opts, 'subspace_max_blend'),       opts.subspace_max_blend = 0.42; end
    if ~isfield(opts, 'local_trigger_sep_deg'),    opts.local_trigger_sep_deg = 4.0; end
    if ~isfield(opts, 'local_trigger_boundary_ratio'), opts.local_trigger_boundary_ratio = 0.22; end
    if ~isfield(opts, 'local_refine_rounds'),      opts.local_refine_rounds = 2; end
    if ~isfield(opts, 'local_refine_half_width_deg'), opts.local_refine_half_width_deg = [0.6 0.2]; end
    if ~isfield(opts, 'local_refine_steps_deg'),   opts.local_refine_steps_deg = [0.2 0.05]; end
    if ~isfield(opts, 'local_refine_max_shift_deg'), opts.local_refine_max_shift_deg = 0.8; end
    if ~isfield(opts, 'local_refine_min_sep_deg'), opts.local_refine_min_sep_deg = 0.15; end
    if ~isfield(opts, 'local_refine_anchor_weight'), opts.local_refine_anchor_weight = 0.12; end
    if ~isfield(opts, 'local_refine_lambda_reg'),  opts.local_refine_lambda_reg = 1e-3; end
    if ~isfield(opts, 'local_refine_max_iter'),    opts.local_refine_max_iter = 18; end
    if ~isfield(opts, 'local_refine_tol'),         opts.local_refine_tol = 1e-4; end
    if ~isfield(opts, 'local_accept_tol'),         opts.local_accept_tol = 1e-4; end
    if ~isfield(opts, 'local_trigger_quant_residual'), opts.local_trigger_quant_residual = 0.18; end
    if ~isfield(opts, 'local_trigger_extension_factor'), opts.local_trigger_extension_factor = 1.15; end
    if ~isfield(opts, 'local_trigger_margin_boundary'),  opts.local_trigger_margin_boundary = 0.06; end
    if ~isfield(opts, 'local_trigger_margin_residual'),  opts.local_trigger_margin_residual = 0.02; end
    if ~isfield(opts, 'local_accept_pair_shift_scale'),  opts.local_accept_pair_shift_scale = 0.18; end
    if ~isfield(opts, 'local_accept_pair_shift_cap_deg'), opts.local_accept_pair_shift_cap_deg = 0.35; end
    if ~isfield(opts, 'local_accept_min_improve_ratio'), opts.local_accept_min_improve_ratio = 2e-4; end
    if ~isfield(opts, 'global_refine_enable'),     opts.global_refine_enable = true; end
    if ~isfield(opts, 'global_force_refine'),      opts.global_force_refine = false; end
    if ~isfield(opts, 'global_trigger_sep_deg'),   opts.global_trigger_sep_deg = 4.0; end
    if ~isfield(opts, 'global_trigger_boundary_ratio'), opts.global_trigger_boundary_ratio = 0.22; end
    if ~isfield(opts, 'global_trigger_quant_residual'), opts.global_trigger_quant_residual = 0.18; end
    if ~isfield(opts, 'global_refine_rounds'),     opts.global_refine_rounds = 2; end
    if ~isfield(opts, 'global_refine_half_width_deg'), opts.global_refine_half_width_deg = [0.5 0.15]; end
    if ~isfield(opts, 'global_refine_steps_deg'),  opts.global_refine_steps_deg = [0.2 0.05]; end
    if ~isfield(opts, 'global_refine_max_shift_deg'), opts.global_refine_max_shift_deg = 0.7; end
    if ~isfield(opts, 'global_refine_min_sep_deg'), opts.global_refine_min_sep_deg = 0.15; end
    if ~isfield(opts, 'global_refine_anchor_weight'), opts.global_refine_anchor_weight = 0.08; end
    if ~isfield(opts, 'global_refine_lambda_reg'), opts.global_refine_lambda_reg = opts.local_refine_lambda_reg; end
    if ~isfield(opts, 'global_refine_max_iter'),   opts.global_refine_max_iter = opts.local_refine_max_iter; end
    if ~isfield(opts, 'global_refine_tol'),        opts.global_refine_tol = opts.local_refine_tol; end
    if ~isfield(opts, 'global_accept_tol'),        opts.global_accept_tol = opts.local_accept_tol; end
    if ~isfield(opts, 'global_accept_max_shift_deg'), opts.global_accept_max_shift_deg = 0.4; end
    if ~isfield(opts, 'global_accept_mean_shift_deg'), opts.global_accept_mean_shift_deg = 0.22; end
    if ~isfield(opts, 'global_accept_min_improve_ratio'), opts.global_accept_min_improve_ratio = 2e-4; end
end

function val = get_data_field(data, field_name, default_val)
    if isstruct(data) && isfield(data, field_name)
        val = data.(field_name);
    else
        val = default_val;
    end
end

function Rn = normalize_quantized_covariance(Rq)
    diag_scale = sqrt(max(real(diag(Rq)), eps));
    scale_mat = diag_scale * diag_scale.';
    Rn = Rq ./ scale_mat;
    Rn = clip_complex_unit_interval(Rn);
    M = size(Rn, 1);
    Rn(1:(M+1):end) = 1;
    Rn = (Rn + Rn') / 2;
end

function C = inverse_onebit_map(Rn_q, use_arcsine)
    if use_arcsine
        C = sin((pi / 2) * real(Rn_q)) + 1j * sin((pi / 2) * imag(Rn_q));
    else
        C = Rn_q;
    end
    C = covariance_to_correlation(C);
end

function Rn_pred = forward_onebit_map(C)
    C = covariance_to_correlation(C);
    C = clip_complex_unit_interval(C);
    Rn_pred = (2 / pi) * asin(real(C)) + 1j * (2 / pi) * asin(imag(C));
    M = size(Rn_pred, 1);
    Rn_pred(1:(M+1):end) = 1;
    Rn_pred = (Rn_pred + Rn_pred') / 2;
end

function C = covariance_to_correlation(R)
    R = (R + R') / 2;
    d = sqrt(max(real(diag(R)), eps));
    C = R ./ (d * d.');
    C = clip_complex_unit_interval(C);
    M = size(C, 1);
    C(1:(M+1):end) = 1;
    C = (C + C') / 2;
end

function X = clip_complex_unit_interval(X)
    X = max(min(real(X), 1 - 1e-6), -1 + 1e-6) + 1j * max(min(imag(X), 1 - 1e-6), -1 + 1e-6);
end

function C = project_structured_correlation(C, opts)
    C = (C + C') / 2;
    if opts.toeplitz_project
        C = project_to_toeplitz_hermitian(C);
    end
    C = project_to_psd(C, opts.psd_floor);
    C = covariance_to_correlation(C);
    if opts.toeplitz_project
        C = project_to_toeplitz_hermitian(C);
        C = project_to_psd(C, opts.psd_floor);
        C = covariance_to_correlation(C);
    end
end

function T = project_to_toeplitz_hermitian(C)
    M = size(C, 1);
    T = zeros(M, M);
    for k = 0:(M-1)
        diag_vals = diag(C, k);
        avg_val = mean(diag_vals);
        T = T + diag(avg_val * ones(M-k, 1), k);
        if k > 0
            T = T + diag(conj(avg_val) * ones(M-k, 1), -k);
        end
    end
    T = (T + T') / 2;
end

function R = project_to_psd(R, floor_val)
    [U, D] = eig((R + R') / 2);
    d = real(diag(D));
    d = max(d, floor_val);
    R = U * diag(d) * U';
    R = (R + R') / 2;
end

function [R_sub, M_eff] = prepare_rootmusic_covariance(C_latent, K, opts)
    R_sub = apply_covariance_regularization(C_latent, opts.covariance_shrink, opts.diag_loading);
    R_sub = apply_fb_avg_if_needed(R_sub, opts.fb_avg);

    prep_meta = estimate_covariance_difficulty(R_sub, K);
    use_smoothing = opts.spatial_smoothing ...
        && size(R_sub, 1) > (K + 2) ...
        && (prep_meta.boundary_ratio >= opts.smooth_boundary_ratio || prep_meta.min_sep_est_deg <= opts.smooth_sep_threshold_deg);
    if use_smoothing
        L = choose_smoothing_subarray(size(R_sub, 1), K, opts.smoothing_margin);
        if L < size(R_sub, 1)
            R_sub = spatial_smooth_covariance(R_sub, L);
            R_sub = apply_fb_avg_if_needed(R_sub, opts.fb_avg);
            R_sub = apply_covariance_regularization(R_sub, opts.covariance_shrink, opts.diag_loading);
        end
    end

    R_sub = project_to_psd((R_sub + R_sub') / 2, opts.psd_floor);
    R_sub = apply_subspace_enhancement_if_needed(R_sub, K, opts);
    M_eff = size(R_sub, 1);
end

function R = apply_subspace_enhancement_if_needed(R, K, opts)
    if ~opts.subspace_enhance_enable || size(R, 1) <= K
        return;
    end

    R0 = (R + R') / 2;
    trace0 = real(trace(R0));
    [U, D] = eig(R0);
    [d, idx] = sort(real(diag(D)), 'descend');
    U = U(:, idx);
    d = max(d, opts.psd_floor);
    if numel(d) <= K
        return;
    end

    noise_idx = (K + 1):numel(d);
    noise_mean = max(mean(d(noise_idx)), opts.psd_floor);
    gap_now = d(K) / max(d(K + 1), opts.psd_floor);
    ambiguity = max(0, opts.subspace_gap_target - gap_now) / max(opts.subspace_gap_target, eps);
    blend = min(opts.subspace_max_blend, ambiguity);
    if blend <= 0
        return;
    end

    d_target = d;
    min_signal_edge = opts.subspace_gap_target * max(d(K + 1), noise_mean);
    d_target(1:K) = max(d_target(1:K), min_signal_edge);
    d_target(1:K) = d_target(1:K) .* (1 + opts.subspace_signal_boost * ambiguity);

    d_noise = d_target(noise_idx);
    d_target(noise_idx) = (1 - opts.subspace_noise_flatten * ambiguity) .* d_noise ...
        + (opts.subspace_noise_flatten * ambiguity) .* noise_mean;

    d_new = (1 - blend) .* d + blend .* d_target;
    d_new = max(d_new, opts.psd_floor);
    if trace0 > 0
        d_new = d_new * trace0 / max(sum(d_new), eps);
    end

    R = U * diag(d_new) * U';
    R = (R + R') / 2;
    R = covariance_to_correlation(R);
end

function meta = estimate_covariance_difficulty(R_sub, K)
    meta = struct('boundary_ratio', 0, 'min_sep_est_deg', inf);
    eig_desc = sort(real(eig((R_sub + R_sub') / 2)), 'descend');
    if numel(eig_desc) >= (K + 1)
        meta.boundary_ratio = eig_desc(K + 1) / max(eig_desc(K), eps);
    end

    try
        G = noise_subspace_from_covariance(R_sub, K);
        theta_est_deg = safe_rootmusic_readout(G, size(R_sub, 1), K);
        theta_est_deg = sort(theta_est_deg(:).');
        if numel(theta_est_deg) >= 2
            meta.min_sep_est_deg = min(diff(theta_est_deg));
        end
    catch
        meta.min_sep_est_deg = inf;
    end
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
    tau = real(trace(R)) / max(M, 1);
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
    Rss = Rss / max(num_sub, 1);
    Rss = (Rss + Rss') / 2;
end

function G = noise_subspace_from_covariance(R, K)
    [Q, D] = eig((R + R') / 2);
    [~, idx] = sort(real(diag(D)), 'ascend');
    Q = Q(:, idx);
    G = Q(:, 1:(size(R, 1) - K));
end

function theta_deg = safe_rootmusic_readout(G, M_eff, K)
    theta_deg = root_music_doa(G, M_eff, K);
    theta_deg = theta_deg(:).';
    if numel(theta_deg) ~= K || any(~isfinite(theta_deg)) || any(abs(theta_deg) > 90)
        theta_deg = fallback_music_readout(G, K, M_eff);
    end
    theta_deg = sort(real(theta_deg));
end

function theta_deg = fallback_music_readout(G, K, M_eff)
    theta_grid_deg = -90:0.2:90;
    theta_rad = deg2rad(theta_grid_deg);
    idxR = (0:(M_eff-1)).';
    gg = zeros(size(theta_rad));
    for ii = 1:numel(theta_rad)
        a = exp(-1j * pi * sin(theta_rad(ii)) * idxR);
        gg(ii) = 1 / max(abs((a') * G * (G') * a), 1e-12);
    end
    [~, sort_idx] = sort(gg, 'descend');
    picks = [];
    for ii = 1:numel(sort_idx)
        cand = theta_grid_deg(sort_idx(ii));
        if isempty(picks) || all(abs(cand - theta_grid_deg(picks)) >= 0.5)
            picks(end+1) = sort_idx(ii); %#ok<AGROW>
            if numel(picks) >= K
                break;
            end
        end
    end
    picks = picks(1:min(K, numel(picks)));
    theta_deg = sort(theta_grid_deg(picks));
    if numel(theta_deg) < K
        theta_deg = [theta_deg, theta_grid_deg(sort_idx(1:(K - numel(theta_deg))))];
        theta_deg = sort(theta_deg);
    end
end

function theta_deg = stabilize_theta_order(theta_deg, theta_prev_deg)
    theta_deg = sort(theta_deg(:).');
    if isempty(theta_prev_deg) || numel(theta_prev_deg) ~= numel(theta_deg)
        return;
    end
    perm_idx = perms(1:numel(theta_deg));
    best_cost = inf;
    best_theta = theta_deg;
    for i = 1:size(perm_idx, 1)
        trial = theta_deg(perm_idx(i, :));
        cost = norm(trial - theta_prev_deg, 2);
        if cost < best_cost
            best_cost = cost;
            best_theta = trial;
        end
    end
    theta_deg = best_theta;
end

function [R_model, info] = fit_covariance_from_doa(C_target, theta_deg, opts)
    M = size(C_target, 1);
    A = steering_matrix_ula(theta_deg, M);
    eigvals = sort(real(eig((C_target + C_target') / 2)), 'ascend');
    num_noise = max(M - numel(theta_deg), 1);
    sigma2 = opts.noise_floor_scale * max(mean(eigvals(1:num_noise)), opts.psd_floor);

    S_est = C_target - sigma2 * eye(M);
    A_pinv = pinv(A);
    P_est = A_pinv * S_est * A_pinv';
    P_est = (P_est + P_est') / 2;
    [U, D] = eig(P_est);
    d = max(real(diag(D)), 0);
    P_est = U * diag(d) * U';
    P_est = (P_est + P_est') / 2;

    R_model = A * P_est * A' + sigma2 * eye(M);
    R_model = (R_model + R_model') / 2;

    info = struct();
    info.sigma2 = sigma2;
    info.source_rank = sum(d > 1e-8);
end

function A = steering_matrix_ula(theta_deg, M)
    theta_rad = deg2rad(theta_deg(:).');
    idx = (0:(M-1)).';
    A = zeros(M, numel(theta_rad));
    for k = 1:numel(theta_rad)
        A(:, k) = exp(-1j * pi * sin(theta_rad(k)) * idx) / sqrt(M);
    end
end

function W = build_residual_weights(E, opts)
    A = abs(E);
    A = A / max(max(A), eps);
    W = opts.reweight_floor + (1 - opts.reweight_floor) * (A .^ opts.residual_power);
    M = size(W, 1);
    W(1:(M+1):end) = 0;
end

function [tf, meta] = should_run_global_refinement(theta_seed_deg, R_sub, K, opts, quant_residual_level)
    tf = false;
    meta = struct('boundary_ratio', 0, 'min_sep_est_deg', inf, ...
        'quant_residual', quant_residual_level, 'trigger_mode', 'none');
    if ~opts.global_refine_enable || isempty(theta_seed_deg)
        return;
    end

    theta_seed_deg = sort(theta_seed_deg(:).');
    if numel(theta_seed_deg) >= 2
        meta.min_sep_est_deg = min(diff(theta_seed_deg));
    end

    if opts.global_force_refine
        tf = true;
        meta.trigger_mode = 'forced';
        return;
    end

    eig_desc = sort(real(eig((R_sub + R_sub') / 2)), 'descend');
    if numel(eig_desc) >= (K + 1)
        meta.boundary_ratio = eig_desc(K + 1) / max(eig_desc(K), eps);
    else
        meta.boundary_ratio = 0;
    end

    strong_close_scene = meta.min_sep_est_deg <= 0.9 * opts.global_trigger_sep_deg;
    weak_boundary = meta.boundary_ratio >= opts.global_trigger_boundary_ratio;
    residual_high = isfinite(quant_residual_level) && quant_residual_level >= opts.global_trigger_quant_residual;

    if strong_close_scene
        tf = true;
        meta.trigger_mode = 'close';
        return;
    end

    if weak_boundary && residual_high
        tf = true;
        meta.trigger_mode = 'ambiguous';
    end
end

function refine_result = global_onebit_refinement(Y_R, theta_seed_deg, refine_meta, opts)
    theta_seed_deg = sort(theta_seed_deg(:).');
    K = numel(theta_seed_deg);
    M = size(Y_R, 1) / 2;
    N = size(Y_R, 2);

    B_seed = build_support_dictionary(theta_seed_deg, M);
    C0 = zeros(2 * K, N);
    [C_ref, F_ref, ~] = logistic_refit_global(Y_R, B_seed, C0, opts.global_refine_lambda_reg, opts.global_refine_max_iter, opts.global_refine_tol);
    obj_base = global_refine_objective(Y_R, F_ref, C_ref, theta_seed_deg, theta_seed_deg, opts);

    theta_ref = theta_seed_deg;
    obj_best = obj_base;
    accepted_update = false;

    for rr = 1:opts.global_refine_rounds
        half_width = pick_round_value(opts.global_refine_half_width_deg, rr);
        step_deg = pick_round_value(opts.global_refine_steps_deg, rr);

        for kk = 1:K
            [theta_trial_best, C_trial_best, F_trial_best, obj_trial_best] = ...
                coordinate_global_search(theta_ref, theta_seed_deg, kk, half_width, step_deg, ...
                    Y_R, C_ref, M, opts);

            if obj_trial_best < obj_best
                theta_ref = theta_trial_best;
                C_ref = C_trial_best;
                F_ref = F_trial_best;
                obj_best = obj_trial_best;
                accepted_update = true;
            end
        end
    end

    refine_result = empty_global_result(theta_seed_deg);
    refine_result.triggered = true;
    refine_result.theta_refined_deg = theta_ref;
    refine_result.obj_before = obj_base;
    refine_result.obj_after = obj_best;
    refine_result.improvement = obj_base - obj_best;
    theta_shift = abs(theta_ref - theta_seed_deg);
    mean_shift = mean(theta_shift);
    max_shift = max(theta_shift);
    min_required_improve = max(opts.global_accept_tol, opts.global_accept_min_improve_ratio * abs(obj_base));
    shift_ok = max_shift <= opts.global_accept_max_shift_deg && mean_shift <= opts.global_accept_mean_shift_deg;
    improve_ok = (obj_base - obj_best) > min_required_improve;
    refine_result.accepted = accepted_update && improve_ok && shift_ok;
    refine_result.theta_shift_deg = theta_ref - theta_seed_deg;
    refine_result.trigger_mode = refine_meta.trigger_mode;
    refine_result.max_allowed_shift_deg = opts.global_accept_max_shift_deg;
    refine_result.boundary_ratio = refine_meta.boundary_ratio;
    refine_result.min_sep_est_deg = refine_meta.min_sep_est_deg;
    if ~refine_result.accepted
        refine_result.theta_refined_deg = theta_seed_deg;
    end
end

function [theta_best, C_best, F_best, obj_best] = coordinate_global_search(theta_ref, theta_anchor, kk, half_width, step_deg, Y_R, C_warm, M, opts)
    theta_best = theta_ref;
    C_best = C_warm;
    F_best = build_support_dictionary(theta_ref, M) * C_warm;
    obj_best = inf;

    lb = max(-90, theta_ref(kk) - half_width);
    ub = min( 90, theta_ref(kk) + half_width);
    lb = max(lb, theta_anchor(kk) - opts.global_refine_max_shift_deg);
    ub = min(ub, theta_anchor(kk) + opts.global_refine_max_shift_deg);

    if kk > 1
        lb = max(lb, theta_ref(kk - 1) + opts.global_refine_min_sep_deg);
    end
    if kk < numel(theta_ref)
        ub = min(ub, theta_ref(kk + 1) - opts.global_refine_min_sep_deg);
    end

    if lb >= ub
        theta_best = theta_ref;
        B_now = build_support_dictionary(theta_ref, M);
        [C_best, F_best, ~] = logistic_refit_global(Y_R, B_now, C_warm, opts.global_refine_lambda_reg, opts.global_refine_max_iter, opts.global_refine_tol);
        obj_best = global_refine_objective(Y_R, F_best, C_best, theta_ref, theta_anchor, opts);
        return;
    end

    grid = unique([lb:step_deg:ub, theta_ref(kk)]);
    for ii = 1:numel(grid)
        theta_trial = theta_ref;
        theta_trial(kk) = grid(ii);
        if ~is_theta_order_valid(theta_trial, opts.global_refine_min_sep_deg)
            continue;
        end

        B_trial = build_support_dictionary(theta_trial, M);
        [C_trial, F_trial, ~] = logistic_refit_global(Y_R, B_trial, C_warm, opts.global_refine_lambda_reg, opts.global_refine_max_iter, opts.global_refine_tol);
        obj_trial = global_refine_objective(Y_R, F_trial, C_trial, theta_trial, theta_anchor, opts);
        if obj_trial < obj_best
            obj_best = obj_trial;
            theta_best = theta_trial;
            C_best = C_trial;
            F_best = F_trial;
        end
    end

    if ~isfinite(obj_best)
        theta_best = theta_ref;
        B_now = build_support_dictionary(theta_ref, M);
        [C_best, F_best, ~] = logistic_refit_global(Y_R, B_now, C_warm, opts.global_refine_lambda_reg, opts.global_refine_max_iter, opts.global_refine_tol);
        obj_best = global_refine_objective(Y_R, F_best, C_best, theta_ref, theta_anchor, opts);
    end
end

function tf = is_theta_order_valid(theta_deg, min_sep_deg)
    theta_deg = sort(theta_deg(:).');
    if numel(theta_deg) < 2
        tf = true;
        return;
    end
    tf = all(diff(theta_deg) >= min_sep_deg - 1e-12);
end

function obj = global_refine_objective(Y_R, F, C, theta_trial, theta_anchor, opts)
    obj = objective_value(Y_R, F, C, opts.global_refine_lambda_reg);
    delta = theta_trial(:) - theta_anchor(:);
    obj = obj + opts.global_refine_anchor_weight * sum(delta(:).^2);
end

function val = pick_round_value(v, idx)
    v = v(:).';
    val = v(min(idx, numel(v)));
end

function result = empty_global_result(theta_seed_deg)
    result = struct();
    result.triggered = false;
    result.accepted = false;
    result.theta_refined_deg = theta_seed_deg;
    result.theta_shift_deg = zeros(size(theta_seed_deg));
    result.obj_before = NaN;
    result.obj_after = NaN;
    result.improvement = 0;
    result.boundary_ratio = NaN;
    result.min_sep_est_deg = NaN;
    result.trigger_mode = 'none';
    result.max_allowed_shift_deg = NaN;
end

function B = build_support_dictionary(theta_list_deg, M)
    L = numel(theta_list_deg);
    B = zeros(2 * M, 2 * L);
    for l = 1:L
        cols = (2 * l - 1):(2 * l);
        B(:, cols) = build_real_atom_block(theta_list_deg(l), M);
    end
end

function B = build_real_atom_block(theta_deg, M)
    theta_rad = deg2rad(theta_deg);
    m = (0:(M - 1)).';
    a = exp(-1j * pi * sin(theta_rad) * m) / sqrt(M);
    u = [real(a); imag(a)];
    v = [-imag(a); real(a)];
    B = [u, v];
end

function [C, F, info] = logistic_refit_global(Y_R, B, C0, lambda_reg, max_iter, tol)
    C = C0;
    F = B * C;
    step = 1.0;
    loss_prev = objective_value(Y_R, F, C, lambda_reg);

    for it = 1:max_iter
        T = clip_value(Y_R .* F, -50, 50);
        dLdF = -Y_R ./ (1 + exp(T));
        gradC = B.' * dLdF + lambda_reg * C;
        grad_norm_sq = sum(gradC(:).^2);
        if grad_norm_sq < 1e-14
            break;
        end

        step_local = step;
        accepted = false;
        for bt = 1:20
            C_trial = C - step_local * gradC;
            F_trial = B * C_trial;
            loss_trial = objective_value(Y_R, F_trial, C_trial, lambda_reg);
            if loss_trial <= loss_prev - 1e-4 * step_local * grad_norm_sq
                accepted = true;
                break;
            end
            step_local = 0.5 * step_local;
        end

        if ~accepted
            break;
        end

        rel_change = norm(C_trial(:) - C(:)) / max(1, norm(C(:)));
        C = C_trial;
        F = F_trial;
        loss_prev = loss_trial;
        step = min(1.0, 1.2 * step_local);
        if rel_change < tol
            break;
        end
    end

    info = struct('num_iter', it, 'final_step', step, 'final_loss', loss_prev);
end

function val = objective_value(Y_R, F, C, lambda_reg)
    Z = clip_value(-Y_R .* F, -50, 50);
    val = sum(log1p(exp(Z)), 'all');
    if ~isempty(C)
        val = val + 0.5 * lambda_reg * sum(C(:).^2);
    end
end

function X = clip_value(X, xmin, xmax)
    X = min(max(X, xmin), xmax);
end
