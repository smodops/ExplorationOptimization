###############################################################################
#  TS_Policy.jl — Thompson Sampling Policy
#
#  Two entry points:
#    TS_Policy(params, inst, ts_prior)             — cold start
#    TS_Policy(params, inst, ts_prior, init)       — warm start (initialized)
#
#  Both dispatch to `_run_TS`.
###############################################################################

"""
    TS_Policy(params, inst, ts_prior [, init]) -> PolicyResult

Run the Thompson Sampling policy for `params.R` replications.

At every `MCU` customers, the policy re-estimates the MNL parameters via a
bootstrapped MLE (loglike_TS). See the details of the bootstrapping 
procedure in the electronic companion (Section EC.2) of the paper. 

Arguments
---------
- `params`    : `SimParams` — simulation hyper-parameters.
- `inst`      : `InstanceData` — instance data.
- `ts_prior` : `TSPrior` — prior mean and covariance.
- `init`      : Optional `InitData` for warm-start; `nothing` for cold start.

Returns
-------
- `PolicyResult` with mean, lb, ub, and per-replication runtimes.
"""
function TS_Policy(params::SimParams, inst::InstanceData,
                   ts_prior::TSPrior,
                   init::Union{InitData,Nothing}=nothing)::PolicyResult
    return _run_TS(params, inst, ts_prior, init)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Internal runner
# ─────────────────────────────────────────────────────────────────────────────

function _run_TS(params::SimParams, inst::InstanceData,
                 ts_prior::TSPrior,
                 init::Union{InitData,Nothing})::PolicyResult
    T = params.T;  R = params.R;  Spac = params.Spac;  MCU = params.MCU
    N = inst.N;    M = inst.M;    D = inst.D
    mu = ts_prior.mu;  sigma = ts_prior.sigma
    Lngth = Int(ceil(T / Spac))

    J      = zeros(Lngth, R)
    time_r = zeros(R)

    for r in 1:R
        # ── Per-replication initialisation ───────────────────────────────────
        T_init   = isnothing(init) ? 0 : init.T_init
        n        = isnothing(init) ? zeros(N, M) : reshape(copy(init.s), N, M)
        Et       = 0.0
        Er       = zeros(Lngth)

        hist_zt  = zeros(T, N+1)
        hist_st  = zeros(T, N)
        class    = zeros(T)

        mle_ind  = 0
        t_mle    = 0
        Sexploit = zeros(N, M)
        f        = zeros(1, M)

        data   = Matrix{Float64}(undef, 0, N+2)
        assort = Matrix{Float64}(undef, 0, N)

        start = time()

        for t in 1:T
            # ── MLE update trigger ───────────────────────────────────────────
            if t - t_mle >= MCU || mle_ind == 0
                try
                    # Build data
                    if t == 1 && !isnothing(init)
                        data, assort = init.purchase, init.assort
                    else
                        data, assort = build_data(hist_zt, class, hist_st, t)
                        if !isnothing(init)
                            data, assort = prepend_init(data, assort, init)
                        end
                    end

                    # Bootstrap resample
                    total_obs = size(data, 1)
                    if total_obs > 0
                        ix     = rand(1:total_obs, total_obs)
                        data   = data[ix, :]
                        assort = assort[ix, :]
                    end

                    # Draw Beta_prior from the TS prior
                    Beta_prior = reshape(rand(MvNormal(mu, sigma), 1), N, D)

                    # Penalised MLE
                    Beta_0 = fill(-1.0, N, D)
                    obj(B) = loglike_TS(B, inst, data, assort, Beta_prior, sigma)
                    res    = Optim.optimize(obj, Beta_0;
                                           method=ConjugateGradient(),
                                           show_trace=false)
                    vhat   = exp.(Optim.minimizer(res) * inst.X)

                    for m in 1:M
                        f[m], Sexploit[:, m] = StaticSol(vhat[:, m], inst)
                    end

                    mle_ind = 1
                    t_mle   = t
                    log_solved("TS MLE", t)
                catch e
                    log_failed("TS MLE", t, e)
                end
            end

            # ── Customer arrival ─────────────────────────────────────────────
            It       = sample_profile(inst.pcum, M)
            class[t] = It

            # ── Assortment decision ──────────────────────────────────────────
            St = (mle_ind == 0) ? random_assortment(N, C) : float(Sexploit[:, It])

            # ── Purchase simulation ──────────────────────────────────────────
            rt, eSt = simulate_purchase(St, inst.eU[:, It], inst.w, N)
            Et         += inst.fopt[It] - eSt
            n[:, It]   .+= vec(St)
            hist_zt[t, rt] = 1.0
            hist_st[t, :]  = vec(St)

            if mod(t, Spac) == 0
                Er[div(t, Spac)] = Et
            end
        end

        time_r[r] = time() - start
        J[:, r]   = Er
        log_replication(r, time_r[r])
    end

    Jm, Jlb, Jub = regret_ci(J)
    return PolicyResult(Jm, Jlb, Jub, time_r)
end
